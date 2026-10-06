"""`Fiber`, copied from `se3_transformer/model/fiber.py`.

Unchanged except for the import of `degree_to_dim` and the removal of the
unused `from_features`/`combine_*` helpers' dependence on the upstream package.
Pure Python bookkeeping over (degree, channels) pairs -- it describes tensor
layouts and never touches DGL, so there is nothing to rewrite here. It is
vendored only so this package has no import path back into `se3_transformer`.
"""
from collections import namedtuple
from itertools import product
from typing import Dict

import torch
from torch import Tensor

from .basis import degree_to_dim

FiberEl = namedtuple("FiberEl", ["degree", "channels"])


class Fiber(dict):
    """Describes a set of features split by type (degree). A degree-k feature
    has dimension 2k+1; the 'channels' are how many such features there are."""

    def __init__(self, structure):
        if isinstance(structure, dict):
            structure = [FiberEl(int(d), int(m))
                         for d, m in sorted(structure.items(), key=lambda x: x[1])]
        elif not isinstance(structure[0], FiberEl):
            structure = list(map(lambda t: FiberEl(*t), sorted(structure, key=lambda x: x[1])))
        self.structure = structure
        super().__init__({d: m for d, m in self.structure})

    @property
    def degrees(self):
        return sorted([t.degree for t in self.structure])

    @property
    def channels(self):
        return [self[d] for d in self.degrees]

    @property
    def num_features(self):
        """Size of the tensor if all features were concatenated."""
        return sum(t.channels * degree_to_dim(t.degree) for t in self.structure)

    @staticmethod
    def create(num_degrees: int, num_channels: int):
        """Degrees 0..num_degrees-1, all with the same multiplicity."""
        return Fiber([(degree, num_channels) for degree in range(num_degrees)])

    @staticmethod
    def from_features(feats: Dict[str, Tensor]):
        structure = {}
        for k, v in feats.items():
            degree = int(k)
            assert len(v.shape) == 3, "Feature shape should be (N, C, 2D+1)"
            assert v.shape[-1] == degree_to_dim(degree)
            structure[degree] = v.shape[-2]
        return Fiber(structure)

    def __getitem__(self, degree: int):
        return dict(self.structure).get(degree, 0)

    def __iter__(self):
        return iter(self.structure)

    def __mul__(self, other):
        if isinstance(other, Fiber):
            return product(self.structure, other.structure)
        elif isinstance(other, int):
            return Fiber({t.degree: t.channels * other for t in self.structure})

    def __add__(self, other):
        if isinstance(other, Fiber):
            return Fiber({t.degree: t.channels + other[t.degree] for t in self.structure})
        elif isinstance(other, int):
            return Fiber({t.degree: t.channels + other for t in self.structure})

    def __repr__(self):
        return str(self.structure)

    @staticmethod
    def combine_max(f1, f2):
        new_dict = dict(f1.structure)
        for k, m in f2.structure:
            new_dict[k] = max(new_dict.get(k, 0), m)
        return Fiber(list(new_dict.items()))

    @staticmethod
    def combine_selectively(f1, f2):
        new_dict = dict(f1.structure)
        for k in f1.degrees:
            if k in f2.degrees:
                new_dict[k] += f2[k]
        return Fiber(list(new_dict.items()))

    def to_attention_heads(self, tensors: Dict[str, Tensor], num_heads: int):
        # dict(N, num_channels, 2d+1) -> (N, num_heads, -1)
        fibers = [tensors[str(degree)].reshape(*tensors[str(degree)].shape[:-2], num_heads, -1)
                  for degree in self.degrees]
        return torch.cat(fibers, -1)


def aggregate_residual(feats1, feats2, method: str):
    """Add or concatenate two fiber features. Degrees follow feats2."""
    if method in ["add", "sum"]:
        return {k: (v + feats1[k]) if k in feats1 else v for k, v in feats2.items()}
    elif method in ["cat", "concat"]:
        return {k: torch.cat([v, feats1[k]], dim=1) if k in feats1 else v
                for k, v in feats2.items()}
    raise ValueError("Method must be add/sum or cat/concat")


def unfuse_features(features: Tensor, degrees) -> Dict[str, Tensor]:
    return dict(zip(map(str, degrees),
                    features.split([degree_to_dim(deg) for deg in degrees], dim=-1)))
