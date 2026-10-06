"""`SE3Transformer`, taking an `EdgeGraph` instead of a `DGLGraph`.

Adapted from `se3_transformer/model/transformer.py`. The forward pass is
unchanged in substance; what differs is that every tensor it needs now arrives
as a tensor rather than being read off an opaque graph object, so the whole
module is traceable end to end.

Dropped, with the evidence:
  * `pooling` / `GPooling` -- `SE3TransformerWrapper` never passes `pooling`,
    and `GPooling` is the last DGL import in the upstream package.
  * `tensor_cores`, `low_memory`, and the pad trick / fully-fused basis they
    gate. Never enabled here; see `layers.py`.
  * `SE3TransformerPooled` and `add_argparse_args` -- CLI/QM9 scaffolding.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .basis import get_all_clebsch_gordon, get_basis, update_basis_with_fused
from .fiber import Fiber
from .graph import EdgeGraph
from .layers import FUSE_PARTIAL, AttentionBlockSE3, ConvSE3, NormSE3


class Sequential(nn.Sequential):
    """`nn.Sequential` that forwards extra args and kwargs (the graph and the
    basis) to every module."""

    def forward(self, input, *args, **kwargs):
        for module in self:
            input = module(input, *args, **kwargs)
        return input


def get_populated_edge_features(relative_pos: Tensor,
                                edge_features: Optional[Dict[str, Tensor]] = None):
    """Append each edge's length to its degree-0 features.

    This is the step that failed under tracing while DGL was in the way: the
    length comes from `rel_pos`, which used to live on `graph.edata`, so it
    never reached the traced graph and the radial MLP received 32 features
    instead of 33.
    """
    edge_features = edge_features.copy() if edge_features else {}
    r = relative_pos.norm(dim=-1, keepdim=True)
    if "0" in edge_features:
        edge_features["0"] = torch.cat([edge_features["0"], r[..., None]], dim=1)
    else:
        edge_features["0"] = r[..., None]
    return edge_features


class SE3Transformer(nn.Module):
    def __init__(self, num_layers: int, fiber_in: Fiber, fiber_hidden: Fiber,
                 fiber_out: Fiber, num_heads: int, channels_div: int,
                 fiber_edge: Fiber = Fiber({}), return_type: Optional[int] = None,
                 norm: bool = True, use_layer_norm: bool = True, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.fiber_edge = fiber_edge
        self.num_heads = num_heads
        self.channels_div = channels_div
        self.return_type = return_type
        self.max_degree = max(*fiber_in.degrees, *fiber_hidden.degrees, *fiber_out.degrees)

        graph_modules = []
        for _ in range(num_layers):
            graph_modules.append(AttentionBlockSE3(fiber_in=fiber_in, fiber_out=fiber_hidden,
                                                   fiber_edge=fiber_edge, num_heads=num_heads,
                                                   channels_div=channels_div,
                                                   use_layer_norm=use_layer_norm,
                                                   max_degree=self.max_degree,
                                                   fuse_level=FUSE_PARTIAL))
            if norm:
                graph_modules.append(NormSE3(fiber_hidden))
            fiber_in = fiber_hidden

        graph_modules.append(ConvSE3(fiber_in=fiber_in, fiber_out=fiber_out,
                                     fiber_edge=fiber_edge, self_interaction=True,
                                     use_layer_norm=use_layer_norm,
                                     max_degree=self.max_degree))
        self.graph_modules = Sequential(*graph_modules)

        # Clebsch-Gordan coefficients are constants -- functions of the degrees
        # alone, never of the data. Materialised once here and registered as
        # non-persistent buffers so that (a) they move with `.to(device)`,
        # (b) they never appear in the traced forward, and (c) `persistent=False`
        # keeps them out of `state_dict()`, so `Base_ckpt.pt` still loads with
        # `strict=True`. Upstream reached for an `lru_cache`d e3nn call inside
        # the forward instead, which dynamo re-traced on every call.
        self._cg_names = []
        for idx, K_Js in enumerate(get_all_clebsch_gordon(self.max_degree, torch.device('cpu'))):
            names = []
            for freq_idx, Q_J in enumerate(K_Js):
                name = f'cg_{idx}_{freq_idx}'
                self.register_buffer(name, Q_J.float(), persistent=False)
                names.append(name)
            self._cg_names.append(names)

    def forward(self, graph: EdgeGraph, node_feats: Dict[str, Tensor],
                edge_feats: Optional[Dict[str, Tensor]] = None,
                basis: Optional[Dict[str, Tensor]] = None):
        if basis is None:
            clebsch_gordon = [[getattr(self, n) for n in names] for names in self._cg_names]
            basis = get_basis(graph.rel_pos, max_degree=self.max_degree,
                              compute_gradients=False, clebsch_gordon=clebsch_gordon)
        basis = update_basis_with_fused(basis, self.max_degree)
        edge_feats = get_populated_edge_features(graph.rel_pos, edge_feats)
        node_feats = self.graph_modules(node_feats, edge_feats, graph=graph, basis=basis)

        if self.return_type is not None:
            return node_feats[str(self.return_type)]
        return node_feats
