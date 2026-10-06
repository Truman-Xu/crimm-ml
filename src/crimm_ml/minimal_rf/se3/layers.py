"""SE(3)-equivariant layers, with DGL replaced by scatter ops.

Adapted from `se3_transformer/model/layers/{linear,norm,convolution,attention}.py`,
merged into one module. Every parameter name, submodule name and tensor layout
is preserved exactly, so `Base_ckpt.pt`'s state dict loads into these classes
with the same keys it uses for the upstream ones -- that is the invariant the
whole rewrite is checked against.

What changed:
  * `graph: DGLGraph` -> `graph: EdgeGraph`; `graph.edges()` -> `.src`/`.dst`.
  * `dgl.ops.copy_e_sum` / `e_dot_v` / `edge_softmax` -> `graph.py` equivalents.
  * `nvtx_range` profiling context managers removed (they are NVIDIA-profiler
    annotations that also pull in `torch.cuda` on a CPU-only build).

What was dropped as unreachable, with the evidence:
  * `ConvSE3FuseLevel.FULL` selection, and with it `basis['fully_fused']`. FULL
    is only ever *requested* when `SE3Transformer(tensor_cores=True)`, and
    `SE3TransformerWrapper` never passes it, so `fuse_level` arrives as PARTIAL
    and `PARTIAL >= FULL` is false. The enum value survives because
    `VersatileConvSE3` still uses it as a per-instance flag.
  * `GPooling`, the only other DGL user in the upstream package. Reachable only
    via `SE3Transformer(pooling=...)`, which this model never sets.

Fuse levels are plain ints rather than an `Enum`: they are read off `self` in
`forward` and compared, and an int is unambiguously a traceable constant.
"""
from itertools import product
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .basis import degree_to_dim
from .fiber import Fiber, aggregate_residual, unfuse_features
from .graph import EdgeGraph, copy_e_sum, e_dot_v, edge_softmax

# Maximum level of fusing applied to a TFN convolution. Higher is faster and
# uses more memory. See the module docstring for why FULL is never selected.
FUSE_NONE = 0
FUSE_PARTIAL = 1
FUSE_FULL = 2


class RadialProfile(nn.Module):
    """Maps invariant edge features (edge length included) to the weights that
    combine the basis matrices into a convolution kernel. $R^{l,k}$ in TFN
    notation."""

    def __init__(self, num_freq: int, channels_in: int, channels_out: int,
                 edge_dim: int = 1, mid_dim: int = 32, use_layer_norm: bool = False):
        super().__init__()
        modules = [
            nn.Linear(edge_dim, mid_dim),
            nn.LayerNorm(mid_dim) if use_layer_norm else None,
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.LayerNorm(mid_dim) if use_layer_norm else None,
            nn.ReLU(),
            nn.Linear(mid_dim, num_freq * channels_in * channels_out, bias=False),
        ]
        self.net = nn.Sequential(*[m for m in modules if m is not None])

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class VersatileConvSE3(nn.Module):
    """One TFN convolution, used for pairwise, partially fused and fully fused
    variants alike."""

    def __init__(self, freq_sum: int, channels_in: int, channels_out: int,
                 edge_dim: int, use_layer_norm: bool, fuse_level: int):
        super().__init__()
        self.freq_sum = freq_sum
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.fuse_level = fuse_level
        self.radial_func = RadialProfile(num_freq=freq_sum, channels_in=channels_in,
                                         channels_out=channels_out, edge_dim=edge_dim,
                                         use_layer_norm=use_layer_norm)

    def forward(self, features: Tensor, invariant_edge_feats: Tensor, basis: Optional[Tensor]):
        num_edges = features.shape[0]
        in_dim = features.shape[2]
        radial_weights = self.radial_func(invariant_edge_feats) \
            .view(-1, self.channels_out, self.channels_in * self.freq_sum)

        if basis is not None:
            # einsum 'n i l, n o i f, n l f k -> n o k', written as two matmuls
            out_dim = basis.shape[-1]
            if self.fuse_level != FUSE_FULL:
                # Upstream correction for a padded basis. With the pad trick
                # dropped this is a no-op: every basis reaching here has an odd
                # final dimension (2*d_out+1), so the adjustment is +0. Kept
                # verbatim rather than deleted, so numerics cannot drift if a
                # future fiber configuration changes which basis is passed.
                out_dim += out_dim % 2 - 1
            basis_view = basis.view(num_edges, in_dim, -1)
            tmp = (features @ basis_view).view(num_edges, -1, basis.shape[-1])
            return (radial_weights @ tmp)[:, :, :out_dim]
        # k = l = 0 non-fused case
        return radial_weights @ features


class ConvSE3(nn.Module):
    """SE(3)-equivariant graph convolution (a Tensor Field Network convolution).

    Maps an arbitrary input Fiber to an arbitrary output Fiber. With
    `pool=False` the result stays on edges instead of being summed into nodes.
    Input edge features of degree > 0 are concatenated onto the source node
    features before the kernel is applied.
    """

    def __init__(self, fiber_in: Fiber, fiber_out: Fiber, fiber_edge: Fiber,
                 pool: bool = True, use_layer_norm: bool = False,
                 self_interaction: bool = False, max_degree: int = 4,
                 fuse_level: int = FUSE_PARTIAL, allow_fused_output: bool = False):
        super().__init__()
        self.pool = pool
        self.fiber_in = fiber_in
        self.fiber_out = fiber_out
        self.self_interaction = self_interaction
        self.max_degree = max_degree
        self.allow_fused_output = allow_fused_output

        # channels_in accounts for the concatenated edge features
        channels_in_set = set([f.channels + fiber_edge[f.degree] * (f.degree > 0) for f in self.fiber_in])
        channels_out_set = set([f.channels for f in self.fiber_out])
        unique_channels_in = (len(channels_in_set) == 1)
        unique_channels_out = (len(channels_out_set) == 1)
        degrees_up_to_max = list(range(max_degree + 1))
        common_args = dict(edge_dim=fiber_edge[0] + 1, use_layer_norm=use_layer_norm)

        if fuse_level >= FUSE_PARTIAL and unique_channels_in and fiber_in.degrees == degrees_up_to_max:
            # Convolutions fused per output degree
            self.used_fuse_level = FUSE_PARTIAL
            self.conv_out = nn.ModuleDict()
            for d_out, c_out in fiber_out:
                sum_freq = sum([degree_to_dim(min(d_out, d)) for d in fiber_in.degrees])
                self.conv_out[str(d_out)] = VersatileConvSE3(
                    sum_freq, list(channels_in_set)[0], c_out,
                    fuse_level=self.used_fuse_level, **common_args)

        elif fuse_level >= FUSE_PARTIAL and unique_channels_out and fiber_out.degrees == degrees_up_to_max:
            # Convolutions fused per input degree
            self.used_fuse_level = FUSE_PARTIAL
            self.conv_in = nn.ModuleDict()
            for d_in, c_in in fiber_in:
                sum_freq = sum([degree_to_dim(min(d_in, d)) for d in fiber_out.degrees])
                # FUSE_FULL here is the per-instance flag described in
                # VersatileConvSE3, not a fuse-level selection: the in-fused
                # basis has an even final dimension, so the padding correction
                # must not fire. Matches upstream.
                self.conv_in[str(d_in)] = VersatileConvSE3(
                    sum_freq, c_in, list(channels_out_set)[0],
                    fuse_level=FUSE_FULL, **common_args)
        else:
            # Pairwise TFN convolutions
            self.used_fuse_level = FUSE_NONE
            self.conv = nn.ModuleDict()
            for (degree_in, channels_in), (degree_out, channels_out) in (self.fiber_in * self.fiber_out):
                dict_key = f"{degree_in},{degree_out}"
                channels_in_new = channels_in + fiber_edge[degree_in] * (degree_in > 0)
                sum_freq = degree_to_dim(min(degree_in, degree_out))
                self.conv[dict_key] = VersatileConvSE3(
                    sum_freq, channels_in_new, channels_out,
                    fuse_level=self.used_fuse_level, **common_args)

        if self_interaction:
            self.to_kernel_self = nn.ParameterDict()
            for degree_out, channels_out in fiber_out:
                if fiber_in[degree_out]:
                    self.to_kernel_self[str(degree_out)] = nn.Parameter(
                        torch.randn(channels_out, fiber_in[degree_out]) / np.sqrt(fiber_in[degree_out]))

    def forward(self, node_feats: Dict[str, Tensor], edge_feats: Dict[str, Tensor],
                graph: EdgeGraph, basis: Dict[str, Tensor]):
        invariant_edge_feats = edge_feats["0"].squeeze(-1)
        src, dst = graph.src, graph.dst
        out = {}
        in_features = []

        # Gather input features from the source node of each edge
        for degree_in in self.fiber_in.degrees:
            src_node_features = node_feats[str(degree_in)][src]
            if degree_in > 0 and str(degree_in) in edge_feats:
                src_node_features = torch.cat([src_node_features, edge_feats[str(degree_in)]], dim=1)
            in_features.append(src_node_features)

        if self.used_fuse_level == FUSE_PARTIAL and hasattr(self, "conv_out"):
            in_features_fused = torch.cat(in_features, dim=-1)
            for degree_out in self.fiber_out.degrees:
                out[str(degree_out)] = self.conv_out[str(degree_out)](
                    in_features_fused, invariant_edge_feats, basis[f"out{degree_out}_fused"])

        elif self.used_fuse_level == FUSE_PARTIAL and hasattr(self, "conv_in"):
            out = 0
            for degree_in, feature in zip(self.fiber_in.degrees, in_features):
                out = out + self.conv_in[str(degree_in)](
                    feature, invariant_edge_feats, basis[f"in{degree_in}_fused"])
            if not self.allow_fused_output or self.self_interaction or self.pool:
                out = unfuse_features(out, self.fiber_out.degrees)
        else:
            for degree_out in self.fiber_out.degrees:
                out_feature = 0
                for degree_in, feature in zip(self.fiber_in.degrees, in_features):
                    dict_key = f"{degree_in},{degree_out}"
                    out_feature = out_feature + self.conv[dict_key](
                        feature, invariant_edge_feats, basis.get(dict_key, None))
                out[str(degree_out)] = out_feature

        for degree_out in self.fiber_out.degrees:
            if self.self_interaction and str(degree_out) in self.to_kernel_self:
                dst_features = node_feats[str(degree_out)][dst]
                kernel_self = self.to_kernel_self[str(degree_out)]
                out[str(degree_out)] = out[str(degree_out)] + kernel_self @ dst_features

            if self.pool:
                if isinstance(out, dict):
                    out[str(degree_out)] = copy_e_sum(graph, out[str(degree_out)])
                else:
                    out = copy_e_sum(graph, out)
        return out


class LinearSE3(nn.Module):
    """Equivariant 1x1 convolution: mixes channels within each degree, never
    across degrees."""

    def __init__(self, fiber_in: Fiber, fiber_out: Fiber):
        super().__init__()
        self.weights = nn.ParameterDict({
            str(degree_out): nn.Parameter(
                torch.randn(channels_out, fiber_in[degree_out]) / np.sqrt(fiber_in[degree_out]))
            for degree_out, channels_out in fiber_out
        })

    def forward(self, features: Dict[str, Tensor], *args, **kwargs) -> Dict[str, Tensor]:
        return {degree: self.weights[degree] @ features[degree] for degree in self.weights}


def _feature_norm(feat: Tensor) -> Tensor:
    """L2 norm over the last (m) axis, keeping it.

    Degree-0 features have an m axis of length 1, where the norm is `|x|`, and
    that case is written as `abs()`. PATCH (vs se3_transformer): upstream calls
    `.norm(dim=-1, keepdim=True)` for every degree, which exports as ONNX
    `ReduceL2`. ONNX Runtime's CUDA kernel (1.23.2) returns its input unchanged
    when every reduced axis has length 1, sign included, so negative features
    got a negative "norm", the clamp below turned it into 2**-24, and the
    division blew the features up by ~1e7. CPU was unaffected. `abs()` is the
    same value (sqrt(x*x) == |x| in binary floating point) and exports as `Abs`.
    """
    if feat.shape[-1] == 1:
        return feat.abs()
    return feat.norm(dim=-1, keepdim=True)


class NormSE3(nn.Module):
    """Norm-based equivariant nonlinearity: transform the per-feature norm,
    keep the direction."""

    NORM_CLAMP = 2 ** -24  # smallest positive subnormal for FP16

    def __init__(self, fiber: Fiber, nonlinearity: nn.Module = nn.ReLU()):
        super().__init__()
        self.fiber = fiber
        self.nonlinearity = nonlinearity

        if len(set(fiber.channels)) == 1:
            # Fuse all the layer norms into one group norm
            self.group_norm = nn.GroupNorm(num_groups=len(fiber.degrees),
                                           num_channels=sum(fiber.channels))
        else:
            self.layer_norms = nn.ModuleDict({
                str(degree): nn.LayerNorm(channels) for degree, channels in fiber
            })

    def forward(self, features: Dict[str, Tensor], *args, **kwargs) -> Dict[str, Tensor]:
        output = {}
        if hasattr(self, "group_norm"):
            norms = [_feature_norm(features[str(d)]).clamp(min=self.NORM_CLAMP)
                     for d in self.fiber.degrees]
            fused_norms = torch.cat(norms, dim=-2)
            new_norms = self.nonlinearity(self.group_norm(fused_norms.squeeze(-1))).unsqueeze(-1)
            new_norms = torch.chunk(new_norms, chunks=len(self.fiber.degrees), dim=-2)
            for norm, new_norm, d in zip(norms, new_norms, self.fiber.degrees):
                output[str(d)] = features[str(d)] / norm * new_norm
        else:
            for degree, feat in features.items():
                norm = _feature_norm(feat).clamp(min=self.NORM_CLAMP)
                new_norm = self.nonlinearity(self.layer_norms[degree](norm.squeeze(-1)).unsqueeze(-1))
                output[degree] = new_norm * feat / norm
        return output


class AttentionSE3(nn.Module):
    """Multi-headed sparse graph self-attention, SE(3)-equivariant."""

    def __init__(self, num_heads: int, key_fiber: Fiber, value_fiber: Fiber):
        super().__init__()
        self.num_heads = num_heads
        self.key_fiber = key_fiber
        self.value_fiber = value_fiber

    def forward(self, value: Union[Tensor, Dict[str, Tensor]],
                key: Union[Tensor, Dict[str, Tensor]],
                query: Dict[str, Tensor], graph: EdgeGraph):
        if isinstance(key, Tensor):
            # all degrees fused
            key = key.reshape(key.shape[0], self.num_heads, -1)
            out = torch.cat([query[str(d)] for d in self.key_fiber.degrees], dim=-1)
            query = out.reshape(list(query.values())[0].shape[0], self.num_heads, -1)
        else:
            key = self.key_fiber.to_attention_heads(key, self.num_heads)
            query = self.key_fiber.to_attention_heads(query, self.num_heads)

        edge_weights = e_dot_v(graph, key, query).squeeze(-1)
        edge_weights = edge_weights / np.sqrt(self.key_fiber.num_features)
        edge_weights = edge_softmax(graph, edge_weights)
        edge_weights = edge_weights[..., None, None]

        if isinstance(value, Tensor):
            v = value.view(value.shape[0], self.num_heads, -1, value.shape[-1])
            weights = edge_weights * v
            feat_out = copy_e_sum(graph, weights)
            feat_out = feat_out.view(feat_out.shape[0], -1, feat_out.shape[-1])  # merge heads
            return unfuse_features(feat_out, self.value_fiber.degrees)

        out = {}
        for degree, channels in self.value_fiber:
            v = value[str(degree)].view(-1, self.num_heads, channels // self.num_heads,
                                        degree_to_dim(degree))
            weights = edge_weights * v
            res = copy_e_sum(graph, weights)
            out[str(degree)] = res.view(-1, channels, degree_to_dim(degree))  # merge heads
        return out


class AttentionBlockSE3(nn.Module):
    """Attention + skip connection + linear projection."""

    def __init__(self, fiber_in: Fiber, fiber_out: Fiber, fiber_edge: Optional[Fiber] = None,
                 num_heads: int = 4, channels_div: int = 2, use_layer_norm: bool = False,
                 max_degree: int = 4, fuse_level: int = FUSE_PARTIAL, **kwargs):
        super().__init__()
        if fiber_edge is None:
            fiber_edge = Fiber({})
        self.fiber_in = fiber_in
        # same structure as fiber_out but with channels divided by channels_div
        value_fiber = Fiber([(degree, channels // channels_div) for degree, channels in fiber_out])
        # same structure as fiber_out, restricted to degrees present in fiber_in
        key_query_fiber = Fiber([(fe.degree, fe.channels) for fe in value_fiber
                                 if fe.degree in fiber_in.degrees])

        self.to_key_value = ConvSE3(fiber_in, value_fiber + key_query_fiber, pool=False,
                                    fiber_edge=fiber_edge, use_layer_norm=use_layer_norm,
                                    max_degree=max_degree, fuse_level=fuse_level,
                                    allow_fused_output=True)
        self.to_query = LinearSE3(fiber_in, key_query_fiber)
        self.attention = AttentionSE3(num_heads, key_query_fiber, value_fiber)
        self.project = LinearSE3(value_fiber + fiber_in, fiber_out)

    def forward(self, node_features: Dict[str, Tensor], edge_features: Dict[str, Tensor],
                graph: EdgeGraph, basis: Dict[str, Tensor]):
        fused_key_value = self.to_key_value(node_features, edge_features, graph, basis)
        key, value = self._get_key_value_from_fused(fused_key_value)
        query = self.to_query(node_features)
        z = self.attention(value, key, query, graph)
        z_concat = aggregate_residual(node_features, z, "cat")
        return self.project(z_concat)

    def _get_key_value_from_fused(self, fused_key_value):
        if isinstance(fused_key_value, Tensor):
            value, key = torch.chunk(fused_key_value, chunks=2, dim=-2)
        else:
            key, value = {}, {}
            for degree, feat in fused_key_value.items():
                if int(degree) in self.fiber_in.degrees:
                    value[degree], key[degree] = torch.chunk(feat, chunks=2, dim=-2)
                else:
                    value[degree] = feat
        return key, value
