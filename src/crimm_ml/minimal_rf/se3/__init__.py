"""A DGL-free, e3nn-free SE(3)-Transformer.

Vendored from NVIDIA's `se3_transformer` (the package RFdiffusion installs from
`env/SE3Transformer`) with its two non-torch runtime dependencies removed:

  * **DGL** -- was used for an edge list, per-edge vectors, and three scatter
    ops. All four are plain tensor operations; see `graph.py`.
  * **e3nn** -- was used for Wigner 3j symbols (constants, now baked into
    `_wigner_3j_data.py`) and spherical harmonics (a polynomial, vendored into
    `spherical_harmonics.py`).

`torch` is the only import left. That is what makes the model exportable: a
`DGLGraph` is an opaque Python object that `torch.export` cannot represent at
all, and it was the reason the SE(3) region -- about two thirds of the forward
pass -- had to stay behind a `torch._dynamo.disable` fence.

Parameter names and tensor layouts are unchanged, so `Base_ckpt.pt` loads
without remapping. Numerical equivalence against the DGL/e3nn originals is
checked by the RFdiffusion fork's `crimm_integration/examples/verify_no_dgl.py`.
"""
from .fiber import Fiber
from .graph import EdgeGraph, copy_e_sum, e_dot_v, edge_softmax
from .transformer import SE3Transformer

__all__ = ["Fiber", "EdgeGraph", "SE3Transformer", "copy_e_sum", "e_dot_v", "edge_softmax"]
