import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# PATCH (vs rfdiffusion): `from opt_einsum import contract as einsum`.
# `opt_einsum.contract` is a third-party Python function, so torch.compile's
# dynamo tracer cannot trace into it and breaks the graph at every call site --
# ~26 per IterBlock, 936 breaks across the model. `torch.einsum` is an ATen op
# that traces straight through. Every einsum expression in this package is a
# two-operand contraction with a single possible contraction path, so the path
# search opt_einsum provides is irrelevant here; representative ops were
# checked and agree bit-for-bit. See README, "torch.compile" section.
from torch import einsum
import copy
# PATCH (vs rfdiffusion): `import dgl` removed. The graph builders below
# now return `se3.graph.EdgeGraph` -- the same edge list, relative positions
# and node count, as a NamedTuple of tensors instead of an opaque DGLGraph.
# See `se3/graph.py` for the op-by-op correspondence.
from .se3.graph import EdgeGraph
from .util import base_indices, RTs_by_torsion, xyzs_in_base_frame, rigid_from_3_points


def pairwise_distance(a, b=None):
    """Exact Euclidean distance matrix, replacing `torch.cdist`.

    PATCH (vs rfdiffusion). Two independent reasons, both measured:

    1. **`torch.cdist` does not export to ONNX correctly.** There is no ONNX
       `cdist` operator, so the exporter emits the ||a||^2 - 2ab + ||b||^2
       expansion -- the classic catastrophic-cancellation form -- which differs
       from torch's own result by up to 7.8e-3 Angstrom. `rbf()` is a Gaussian
       basis, so it amplifies that to ~1e-2 *relative* in the pair features,
       which then compounds through 36 blocks. Passing
       `compute_mode='donot_use_mm_for_euclid_dist'` does not help: that form
       fails to export at all.

    2. **`torch.cdist`'s default mode is already inaccurate in eager.** For more
       than 25 points it uses the same matmul expansion, and the error is
       visible in the one entry whose value is known exactly: the diagonal, a
       point's distance to itself, comes out as 1.1e-2 Angstrom instead of 0.
       Stock RFdiffusion has carried this all along.

    Computing the differences explicitly is bit-exact, and exports bit-exactly.
    It costs an (..., L, L, 3) intermediate rather than (..., L, L) -- 69 KB at
    L=76, 3 MB at L=500 -- which is why the fast path exists at all, and is not
    a meaningful cost at the sizes this model runs at.

    The trade this makes: torch and ONNX now agree *exactly*, at the price of
    differing from stock RFdiffusion's distance features by ~1e-2 Angstrom.
    That is far below structural significance (the `rbf` bins are ~0.5 Angstrom
    wide) and it is a move toward the correct answer, not away from it.
    """
    if b is None:
        b = a
    d = a.unsqueeze(-2) - b.unsqueeze(-3)
    # clamp guards sqrt'(0) = inf if anything ever backprops through this;
    # it puts 1e-6 on the diagonal instead of 0, against cdist's 1.1e-2.
    return torch.sqrt((d * d).sum(-1).clamp(min=1e-12))


def find_breaks(ix, thresh=35):
    # finds positions in ix where the jump is greater than 100
    breaks = np.where(np.diff(ix) > thresh)[0]
    return np.array(breaks)+1


def init_lecun_normal(module):
    def truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2):
        normal = torch.distributions.normal.Normal(0, 1)

        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma

        alpha_normal_cdf = normal.cdf(torch.tensor(alpha))
        p = alpha_normal_cdf + (normal.cdf(torch.tensor(beta)) - alpha_normal_cdf) * uniform

        v = torch.clamp(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
        x = mu + sigma * np.sqrt(2) * torch.erfinv(v)
        x = torch.clamp(x, a, b)

        return x

    def sample_truncated_normal(shape):
        stddev = np.sqrt(1.0/shape[-1])/.87962566103423978  # shape[-1] = fan_in
        return stddev * truncated_normal(torch.rand(shape))

    module.weight = torch.nn.Parameter( (sample_truncated_normal(module.weight.shape)) )
    return module

def init_lecun_normal_param(weight):
    def truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2):
        normal = torch.distributions.normal.Normal(0, 1)

        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma

        alpha_normal_cdf = normal.cdf(torch.tensor(alpha))
        p = alpha_normal_cdf + (normal.cdf(torch.tensor(beta)) - alpha_normal_cdf) * uniform

        v = torch.clamp(2 * p - 1, -1 + 1e-8, 1 - 1e-8)
        x = mu + sigma * np.sqrt(2) * torch.erfinv(v)
        x = torch.clamp(x, a, b)

        return x

    def sample_truncated_normal(shape):
        stddev = np.sqrt(1.0/shape[-1])/.87962566103423978  # shape[-1] = fan_in
        return stddev * truncated_normal(torch.rand(shape))

    weight = torch.nn.Parameter( (sample_truncated_normal(weight.shape)) )
    return weight

# for gradient checkpointing
def create_custom_forward(module, **kwargs):
    def custom_forward(*inputs):
        return module(*inputs, **kwargs)
    return custom_forward


class Dropout(nn.Module):
    # Dropout entire row or column
    def __init__(self, broadcast_dim=None, p_drop=0.15):
        super(Dropout, self).__init__()
        # give ones with probability of 1-p_drop / zeros with p_drop
        self.sampler = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-p_drop]))
        self.broadcast_dim=broadcast_dim
        self.p_drop=p_drop
    def forward(self, x):
        if not self.training: # no drophead during evaluation mode
            return x
        shape = list(x.shape)
        if not self.broadcast_dim == None:
            shape[self.broadcast_dim] = 1
        mask = self.sampler.sample(shape).to(x.device).view(shape)

        x = mask * x / (1.0 - self.p_drop)
        return x

def rbf(D):
    # Distance radial basis function
    D_min, D_max, D_count = 0., 20., 36
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu[None,:]
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    return RBF

def get_seqsep(idx, cyclic=None):
    '''
    Input:
        - idx: residue indices of given sequence (B,L)
    Output:
        - seqsep: sequence separation feature with sign (B, L, L, 1)
                  Sergey found that having sign in seqsep features helps a little
    '''
    seqsep = idx[:,None,:] - idx[:,:,None]
    sign = torch.sign(seqsep)
    neigh = torch.abs(seqsep)
    neigh[neigh > 1] = 0.0 # if bonded -- 1.0 / else 0.0
    neigh = sign * neigh

    # PATCH (vs rfdiffusion/util_module.py): the `breaks`/`chainids` block
    # below used to run unconditionally, immediately before this guard. Its
    # only consumer is the loop inside `if cyclic is not None`, so with no
    # cyclic peptide -- the loop-building case -- every line of it was computed
    # and discarded. It is not free: two `.cpu().numpy()` calls force a
    # GPU->CPU synchronisation each, and `get_seqsep` runs once per `Str2Str`,
    # i.e. 40 times per forward pass. It also cost ~10% of eager runtime and
    # was the largest single source of torch.compile graph breaks (78 of 194),
    # because dynamo lowers `find_breaks`' `np.where` to `aten.nonzero`, a
    # data-dependent-output-shape op. Moving it inside the guard that already
    # gates its only reader is behaviour-preserving by construction, and was
    # verified bit-exact. See README, "torch.compile" section.
    if (cyclic is not None):
        # add cyclic edges
        breaks = find_breaks(idx.squeeze().cpu().numpy())
        chainids = np.zeros_like(idx.squeeze().cpu().numpy())
        for i, b in enumerate(breaks):
            chainids[b:] = i+1
        chainids = torch.from_numpy(chainids).to(device=idx.device)

        # add cyclic edges with multiple chains
        for chid in torch.unique(chainids):
            is_chid = chainids==chid
            cur_cyclic = cyclic*is_chid
            cur_cres = cur_cyclic.nonzero()

            if cur_cyclic.sum()>=2:
                neigh[:,cur_cres[-1],cur_cres[0]] = 1
                neigh[:,cur_cres[0],cur_cres[-1]] = -1

    return neigh.unsqueeze(-1)

# PATCH (vs rfdiffusion/util_module.py): DGL removed, and the `@torch._dynamo.
# disable` fence that used to wrap both of these functions narrowed to the one
# line that actually needs it.
#
# What these used to do: pack the edge list into a `dgl.DGLGraph` and hang the
# per-edge vectors off `G.edata['rel_pos']`. A DGLGraph is an opaque Python
# object, so dynamo could not trace it and `torch.export` cannot represent it at
# all -- which is why the entire SE(3) region, about two thirds of the forward
# pass, had to run eager. They now return an `EdgeGraph`: the identical
# `(src, dst, rel_pos, num_nodes)`, as a NamedTuple of tensors.
#
# `torch.where(cond)` with a single argument is `nonzero`, and dynamo can never
# give its result a static shape: the output length is unbacked by construction,
# whatever the mask happens to contain. Hence the fence. But only ONE of the two
# graph builders is genuinely value-dependent, and the distinction decides which
# breaks can be removed:
#
#   make_topk_graph -- genuinely value-dependent, still fenced. Its mask is the
#       union of the k nearest neighbours with the sequence-local pairs. Both
#       sets have shape-fixed sizes (L*k and a function of idx alone), but their
#       *overlap* moves as the backbone moves, so the union does too: measured
#       5319/5319/5318/5317 edges across the four refinement blocks of a single
#       forward, at identical input shapes.
#
#   make_full_graph -- NOT value-dependent. Its mask is `sep.abs() > 0`, which
#       has no distance term at all; for unique `idx` it is exactly the
#       off-diagonal mask, so the edge count is exactly L*(L-1). It only looked
#       untraceable because it was written through `nonzero` and inherited the
#       same opacity. It now builds its indices arithmetically and does not
#       break the graph -- 36 of the 40 fence crossings per forward, and 7 of
#       the 9 compiled graph breaks, used to be here.
#
# Fencing just the `nonzero` call, rather than the whole builder, leaves the
# distance map, top-k selection and the gather of `rel_pos`/`pair` inside the
# compiled graph. The RFdiffusion fork's
# `crimm_integration/examples/graph_break_probe.py` reproduces all of the above.
#
# For ahead-of-time export the fence is not enough -- an exported artifact
# cannot contain a graph break either. The edge list has to be built outside the
# exported module and passed in as `src`/`dst`/`rel_pos`, which is exactly the
# shape of this interface. See README, "Removing DGL".
def _edge_index_impl(cond):
    """The one data-dependent-output-shape op in the model: `(B,L,L)` boolean
    mask -> the `(b, i, j)` indices of its true entries."""
    return torch.where(cond)


#: The fenced version, used by default. `torch.compile` cannot represent a
#: data-dependent output shape, so without the fence it fails outright instead
#: of breaking cleanly.
_edge_index_from_cond = torch._dynamo.disable(_edge_index_impl)


def set_export_mode(enabled: bool):
    """Remove (or restore) the dynamo fence around the edge-list construction.

    `torch.export` *can* represent `nonzero` -- it gives the result an unbacked
    symbolic dimension -- and ONNX has a native `NonZero` operator. The fence
    exists only for `torch.compile`, and an exported artifact cannot contain a
    graph break, so it has to come off before exporting.

    Call `set_export_mode(True)` before `torch.export` / `torch.onnx.export`,
    and `set_export_mode(False)` to go back to the compile-friendly form.
    """
    global _edge_index_from_cond
    _edge_index_from_cond = _edge_index_impl if enabled \
        else torch._dynamo.disable(_edge_index_impl)


def _full_edge_index(B, L, device):
    """The `(b, i, j)` indices of every off-diagonal entry of a `(B, L, L)`
    mask, in exactly the order `torch.where`/`nonzero` returns them.

    `nonzero` emits indices in row-major order -- sorted by `b`, then `i`, then
    `j` -- and `make_full_graph` gathers `pair[b,i,j]` in that same order, so
    reproducing the order is part of the contract, not a nicety.

    For row `i` the columns are `0..L-1` with `i` removed. Taking
    `arm = arange(L-1)` and adding `(arm >= i)` shifts every value at or past
    `i` up by one, which is that list, already ascending. Everything here is a
    function of shapes alone, so dynamo traces straight through it.
    """
    ar = torch.arange(L, device=device)
    arm = torch.arange(L - 1, device=device)
    j_row = arm.unsqueeze(0) + (arm.unsqueeze(0) >= ar.unsqueeze(1)).to(arm.dtype)
    i_row = ar.unsqueeze(1).expand(L, L - 1)

    E = L * (L - 1)
    i = i_row.reshape(1, E).expand(B, E).reshape(-1)
    j = j_row.reshape(1, E).expand(B, E).reshape(-1)
    b = torch.arange(B, device=device).unsqueeze(1).expand(B, E).reshape(-1)
    return b, i, j


def full_graph_precondition_holds(idx):
    """Does `make_full_graph`'s static edge list agree with the original mask?

    The rewrite assumes `sep.abs() > 0` selects exactly the off-diagonal
    entries, which holds iff no value repeats within a row of `idx`. That is
    true of every input this pipeline builds -- a single chain carries strictly
    increasing residue numbers, and RFdiffusion offsets additional chains by
    +200 -- but it is an assumption about values, so it is worth being able to
    check rather than trust.

    Deliberately NOT called from the model: comparing tensor values in the
    forward pass would reintroduce exactly the data-dependent control flow this
    rewrite removes. Call it from a test, or once at setup, on the same `idx`
    the model will be given.
    """
    sep = idx[:, None, :] - idx[:, :, None]
    L = idx.shape[1]
    offdiag = ~torch.eye(L, dtype=torch.bool, device=idx.device)
    return bool(torch.equal(sep.abs() > 0, offdiag.unsqueeze(0).expand_as(sep)))


def make_full_graph(xyz, pair, idx, top_k=64, kmin=9):
    '''
    Input:
        - xyz: current backbone cooordinates (B, L, 3, 3)
        - pair: pair features from Trunk (B, L, L, E)
        - idx: residue index from ground truth pdb (unused -- see below)
    Output:
        - G: EdgeGraph over B*L nodes

    PATCH (vs rfdiffusion/util_module.py): the edge list is built
    arithmetically by `_full_edge_index` instead of with
    `torch.where(sep.abs() > 0)`. The mask here carries no distance term, so
    for unique `idx` it is just the off-diagonal mask and the result is
    identical -- but it no longer goes through `nonzero`, so it no longer
    breaks the compiled graph. `idx` is now unused; the parameter stays because
    `Str2Str.forward` passes it positionally alongside `make_topk_graph`, which
    does need it. `full_graph_precondition_holds(idx)` checks the assumption.
    '''

    B, L = xyz.shape[:2]

    b, i, j = _full_edge_index(B, L, xyz.device)

    src = b*L+i
    tgt = b*L+j
    rel_pos = (xyz[b,j,:] - xyz[b,i,:]).detach()  # no gradient through basis function
    G = EdgeGraph(src=src, dst=tgt, rel_pos=rel_pos, num_nodes=B*L)

    return G, pair[b,i,j][...,None]


def make_topk_graph(xyz, pair, idx, top_k=64, kmin=32, eps=1e-6):
    '''
    Input:
        - xyz: current backbone cooordinates (B, L, 3, 3)
        - pair: pair features from Trunk (B, L, L, E)
        - idx: residue index from ground truth pdb
    Output:
        - G: EdgeGraph over B*L nodes
    '''

    B, L = xyz.shape[:2]
    device = xyz.device

    # distance map from current CA coordinates
    D = pairwise_distance(xyz) + torch.eye(L, device=device).unsqueeze(0)*999.9  # (B, L, L)
    # seq sep
    sep = idx[:,None,:] - idx[:,:,None]
    sep = sep.abs() + torch.eye(L, device=device).unsqueeze(0)*999.9
    D = D + sep*eps

    # get top_k neighbors
    D_neigh, E_idx = torch.topk(D, min(top_k, L), largest=False) # shape of E_idx: (B, L, top_k)
    topk_matrix = torch.zeros((B, L, L), device=device)
    topk_matrix.scatter_(2, E_idx, 1.0)

    # put an edge if any of the 3 conditions are met:
    #   1) |i-j| <= kmin (connect sequentially adjacent residues)
    #   2) top_k neighbors
    cond = torch.logical_or(topk_matrix > 0.0, sep < kmin)
    b,i,j = _edge_index_from_cond(cond)

    src = b*L+i
    tgt = b*L+j
    rel_pos = (xyz[b,j,:] - xyz[b,i,:]).detach()  # no gradient through basis function
    G = EdgeGraph(src=src, dst=tgt, rel_pos=rel_pos, num_nodes=B*L)

    return G, pair[b,i,j][...,None]

def make_rotX(angs, eps=1e-6):
    B,L = angs.shape[:2]
    NORM = torch.linalg.norm(angs, dim=-1) + eps

    RTs = torch.eye(4,  device=angs.device).repeat(B,L,1,1)

    RTs[:,:,1,1] = angs[:,:,0]/NORM
    RTs[:,:,1,2] = -angs[:,:,1]/NORM
    RTs[:,:,2,1] = angs[:,:,1]/NORM
    RTs[:,:,2,2] = angs[:,:,0]/NORM
    return RTs

# rotate about the z axis
def make_rotZ(angs, eps=1e-6):
    B,L = angs.shape[:2]
    NORM = torch.linalg.norm(angs, dim=-1) + eps

    RTs = torch.eye(4,  device=angs.device).repeat(B,L,1,1)

    RTs[:,:,0,0] = angs[:,:,0]/NORM
    RTs[:,:,0,1] = -angs[:,:,1]/NORM
    RTs[:,:,1,0] = angs[:,:,1]/NORM
    RTs[:,:,1,1] = angs[:,:,0]/NORM
    return RTs

# rotate about an arbitrary axis
def make_rot_axis(angs, u, eps=1e-6):
    B,L = angs.shape[:2]
    NORM = torch.linalg.norm(angs, dim=-1) + eps

    RTs = torch.eye(4,  device=angs.device).repeat(B,L,1,1)

    ct = angs[:,:,0]/NORM
    st = angs[:,:,1]/NORM
    u0 = u[:,:,0]
    u1 = u[:,:,1]
    u2 = u[:,:,2]

    RTs[:,:,0,0] = ct+u0*u0*(1-ct)
    RTs[:,:,0,1] = u0*u1*(1-ct)-u2*st
    RTs[:,:,0,2] = u0*u2*(1-ct)+u1*st
    RTs[:,:,1,0] = u0*u1*(1-ct)+u2*st
    RTs[:,:,1,1] = ct+u1*u1*(1-ct)
    RTs[:,:,1,2] = u1*u2*(1-ct)-u0*st
    RTs[:,:,2,0] = u0*u2*(1-ct)-u1*st
    RTs[:,:,2,1] = u1*u2*(1-ct)+u0*st
    RTs[:,:,2,2] = ct+u2*u2*(1-ct)
    return RTs

class ComputeAllAtomCoords(nn.Module):
    def __init__(self):
        super(ComputeAllAtomCoords, self).__init__()

        self.base_indices = nn.Parameter(base_indices, requires_grad=False)
        self.RTs_in_base_frame = nn.Parameter(RTs_by_torsion, requires_grad=False)
        self.xyzs_in_base_frame = nn.Parameter(xyzs_in_base_frame, requires_grad=False)

    def forward(self, seq, xyz, alphas, non_ideal=False, use_H=True):
        B,L = xyz.shape[:2]

        Rs, Ts = rigid_from_3_points(xyz[...,0,:],xyz[...,1,:],xyz[...,2,:], non_ideal=non_ideal)

        RTF0 = torch.eye(4).repeat(B,L,1,1).to(device=Rs.device)

        # bb
        RTF0[:,:,:3,:3] = Rs
        RTF0[:,:,:3,3] = Ts

        # omega
        RTF1 = torch.einsum(
            'brij,brjk,brkl->bril',
            RTF0, self.RTs_in_base_frame[seq,0,:], make_rotX(alphas[:,:,0,:]))

        # phi
        RTF2 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF0, self.RTs_in_base_frame[seq,1,:], make_rotX(alphas[:,:,1,:]))

        # psi
        RTF3 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF0, self.RTs_in_base_frame[seq,2,:], make_rotX(alphas[:,:,2,:]))

        # CB bend
        basexyzs = self.xyzs_in_base_frame[seq]
        NCr = 0.5*(basexyzs[:,:,2,:3]+basexyzs[:,:,0,:3])
        CAr = (basexyzs[:,:,1,:3])
        CBr = (basexyzs[:,:,4,:3])
        CBrotaxis1 = (CBr-CAr).cross(NCr-CAr)
        CBrotaxis1 /= torch.linalg.norm(CBrotaxis1, dim=-1, keepdim=True)+1e-8
        
        # CB twist
        NCp = basexyzs[:,:,2,:3] - basexyzs[:,:,0,:3]
        NCpp = NCp - torch.sum(NCp*NCr, dim=-1, keepdim=True)/ torch.sum(NCr*NCr, dim=-1, keepdim=True) * NCr
        CBrotaxis2 = (CBr-CAr).cross(NCpp)
        CBrotaxis2 /= torch.linalg.norm(CBrotaxis2, dim=-1, keepdim=True)+1e-8
        
        CBrot1 = make_rot_axis(alphas[:,:,7,:], CBrotaxis1 )
        CBrot2 = make_rot_axis(alphas[:,:,8,:], CBrotaxis2 )
        
        RTF8 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF0, CBrot1,CBrot2)
        
        # chi1 + CG bend
        RTF4 = torch.einsum(
            'brij,brjk,brkl,brlm->brim', 
            RTF8, 
            self.RTs_in_base_frame[seq,3,:], 
            make_rotX(alphas[:,:,3,:]), 
            make_rotZ(alphas[:,:,9,:]))

        # chi2
        RTF5 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF4, self.RTs_in_base_frame[seq,4,:],make_rotX(alphas[:,:,4,:]))

        # chi3
        RTF6 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF5,self.RTs_in_base_frame[seq,5,:],make_rotX(alphas[:,:,5,:]))

        # chi4
        RTF7 = torch.einsum(
            'brij,brjk,brkl->bril', 
            RTF6,self.RTs_in_base_frame[seq,6,:],make_rotX(alphas[:,:,6,:]))

        RTframes = torch.stack((
            RTF0,RTF1,RTF2,RTF3,RTF4,RTF5,RTF6,RTF7,RTF8
        ),dim=2)

        xyzs = torch.einsum(
            'brtij,brtj->brti', 
            RTframes.gather(2,self.base_indices[seq][...,None,None].repeat(1,1,1,4,4)), basexyzs
        )

        if use_H:
            return RTframes, xyzs[...,:3]
        else:
            return RTframes, xyzs[...,:14,:3]
