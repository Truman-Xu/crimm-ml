"""The entire DGL surface of the SE(3)-Transformer, as plain tensor ops.

NVIDIA's `se3_transformer` reaches for DGL in exactly four places, and this
module replaces all four. That is the whole dependency: a `DGLGraph` here is
just an edge list plus the relative positions along those edges.

  DGL                            | replacement
  -------------------------------|---------------------------------------------
  `graph.edges()`                | `EdgeGraph.src`, `EdgeGraph.dst`
  `graph.edata['rel_pos']`       | `EdgeGraph.rel_pos`
  `dgl.ops.copy_e_sum(g, e)`     | `copy_e_sum`  -> `scatter_add_` at dst
  `dgl.ops.e_dot_v(g, e, v)`     | `e_dot_v`     -> `(e * v[dst]).sum(-1)`
  `dgl.ops.edge_softmax(g, w)`   | `edge_softmax`-> segment softmax over dst

DGL's message-passing convention, which these mirror: an edge runs src -> dst,
messages are *aggregated at dst*, and `edge_softmax` normalises over the edges
sharing a destination. Verified directly against DGL 2.4.0 on random graphs
(`examples/verify_no_dgl.py`): `copy_e_sum` reproduces bit-exactly, `e_dot_v`
and `edge_softmax` to ~5e-7 and ~1e-8 respectively -- fused-kernel reduction
order, not a semantic difference.

`dgl.graph((src, dst))` preserves the edge order it is given, and `g.edges()`
returns it unchanged (also verified), so an `EdgeGraph` built from the same
`(src, dst)` tensors indexes edges identically to the DGL graph it replaces.

DETERMINISM -- a real difference from DGL, not just a rounding one. DGL's
`copy_e_sum` is a segment reduction over a sorted CSR and is bit-reproducible
on CUDA. `index_add_` is implemented with atomics, so its accumulation order
varies between calls: measured at 1.2e-5 max self-difference over 5 calls of
one op (E=5700, N=76), which the 36-block trunk amplifies to ~6e-5 relative in
`pair` and ~4e-4 Angstrom in the output coordinates. Negligible for structure
prediction, but it does mean the model is no longer bit-reproducible run to run
by default, which the DGL version was.

`torch.use_deterministic_algorithms(True)` restores exact repeatability
(verified: 0.0 over 5 calls) because torch routes `index_add_` to a
sort-based kernel. This module deliberately does not set that flag -- it is
global process state and belongs to the caller -- so the ops here simply
inherit whatever the caller asked torch for. The verification scripts enable
it; sorting the edges by destination does *not* help, since the atomics remain.

Why this matters beyond removing a dependency: a `DGLGraph` is an opaque Python
object holding tensors, so `torch.compile` cannot trace through it and
`torch.export` cannot represent it at all. An `EdgeGraph` is a NamedTuple of
tensors -- a pytree -- so it flows through dynamo and export like any other
argument, which is what makes an ahead-of-time compiled artifact possible.
"""
from typing import NamedTuple

import torch
from torch import Tensor


class EdgeGraph(NamedTuple):
    """A directed graph as three tensors and a node count.

    `src`/`dst` are int64 `(E,)` node indices; edge `e` runs `src[e] -> dst[e]`.
    `rel_pos` is `(E, 3)`, the vector from the source node to the destination
    node, and is what the spherical-harmonic basis is built from.

    `num_nodes` is a plain `int` on purpose: it is `B * L`, fixed once the
    sequence length is known, so tracing should burn it in as a constant rather
    than carry it as a symbolic size. Only `E` is data-dependent.
    """

    src: Tensor
    dst: Tensor
    rel_pos: Tensor
    num_nodes: int

    def to(self, device):
        return EdgeGraph(self.src.to(device), self.dst.to(device),
                         self.rel_pos.to(device), self.num_nodes)

    @property
    def num_edges(self) -> int:
        return self.src.shape[0]


def _dst_index_like(graph: EdgeGraph, x: Tensor) -> Tensor:
    """`graph.dst` broadcast to `x`'s shape, for `scatter_add_`/`scatter_reduce_`.

    `expand` is stride-0, so this allocates nothing.
    """
    return graph.dst.reshape((-1,) + (1,) * (x.dim() - 1)).expand_as(x)


def copy_e_sum(graph: EdgeGraph, edge_feats: Tensor) -> Tensor:
    """Sum edge features into their destination nodes. `dgl.ops.copy_e_sum`.

    `(E, ...) -> (num_nodes, ...)`. Destinations with no incoming edge stay
    zero, which is what DGL produces too.

    `scatter_add_`, NOT the more natural `index_add_`, and this is not a style
    preference -- it is a correctness fix for ONNX Runtime.

    ORT's lowering of `index_add_` parallelises over the inner dimension and
    **races on duplicate destination indices**, which every graph aggregation
    has by construction. Measured on the real refiner graph (N=76, E=5322):
    width 32 or less is correct, width 64 or more is wrong by ~15-28% relative
    *and unstable run to run* -- three runs of the same session gave three
    answers. The type-1 features (width 6) came out fine while the type-0
    features (width 64) were garbage, from the same call, which is what made
    this so slow to localise.

    `scatter_add_` is bit-exact against `index_add_` in eager and stable and
    exact through ORT. It needs the index broadcast to the full shape, but
    `expand` is stride-0 so that costs nothing.
    """
    out = edge_feats.new_zeros((graph.num_nodes,) + edge_feats.shape[1:])
    return out.scatter_add_(0, _dst_index_like(graph, edge_feats), edge_feats)


def e_dot_v(graph: EdgeGraph, edge_feats: Tensor, node_feats: Tensor) -> Tensor:
    """Dot each edge feature with its *destination* node feature. `dgl.ops.e_dot_v`.

    `(E, H, C)` dot `(N, H, C)` -> `(E, H, 1)`, keeping DGL's trailing 1.
    """
    return (edge_feats * node_feats[graph.dst]).sum(-1, keepdim=True)


def edge_softmax(graph: EdgeGraph, edge_weights: Tensor) -> Tensor:
    """Softmax over the edges sharing a destination node. `dgl.ops.edge_softmax`.

    Subtracts the per-destination max before exponentiating, as DGL does, so a
    large attention logit cannot overflow fp32/fp16.

    `scatter_reduce_` rather than `index_reduce_`: same result, but it is the
    op with an ONNX lowering (opset 18 `ScatterElements` with `reduction="max"`),
    and this has to survive export.
    """
    dst = graph.dst
    index = _dst_index_like(graph, edge_weights)

    maxes = edge_weights.new_full((graph.num_nodes,) + edge_weights.shape[1:], float("-inf"))
    maxes = maxes.scatter_reduce_(0, index, edge_weights, reduce="amax", include_self=True)
    shifted = (edge_weights - maxes[dst]).exp()

    # scatter_add_ rather than index_add_ for the same ORT-race reason as
    # copy_e_sum. Attention weights are narrow (num_heads = 4) so this one was
    # never actually hit, but the two must not diverge.
    denom = shifted.new_zeros((graph.num_nodes,) + shifted.shape[1:])
    denom = denom.scatter_add_(0, index, shifted)
    return shifted / denom[dst]
