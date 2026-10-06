"""crimm-facing helper functions shared by both RFdiffusion backends.

Everything here operates on a crimm `PolymerChain` strictly through its
public API (`get_pdb_str`, `.gaps`, `.reported_res`, `.copy()`, `.add()`,
`.sort_residues()`, `ResidueTopologySet`). Nothing under crimm/ is edited by
this module.
"""
import warnings

import numpy as np

from crimm.IO.PDBString import get_pdb_str
from crimm.Modeller import ResidueTopologySet

from crimm_ml.minimal_rf.chemical import aa2long

warnings.warn(
    "crimm_ml.crimm_utils is deprecated and will be removed in the next release: building loops "
    "into crimm chains now lives in crimm itself (crimm.Modeller.RFLoopBuilder), "
    "which rebuilds full residues rather than glycine backbones.",
    FutureWarning, stacklevel=2)

_GLY_TOPOLOGY = None


def _gly_topology():
    # crimm's own GLY topology (crimm/Data/toppar/prot.rtf, RESI GLY) has no
    # CB atom -- see build_gly_residues below for why that matters.
    global _GLY_TOPOLOGY
    if _GLY_TOPOLOGY is None:
        _GLY_TOPOLOGY = ResidueTopologySet("protein")["GLY"]
    return _GLY_TOPOLOGY


def build_contig_string(chain, gap):
    """Build an RFdiffusion contig string that fixes the residues actually
    present immediately around `gap`, and diffuses exactly `len(gap)` new
    residues in its place.

    `gap` must be one of the sets in `chain.gaps` (crimm's grouped-missing-
    residue representation, 1-indexed and continuous). Only the two islands
    immediately flanking `gap` are described. Critically, those islands stop
    at the nearest *other* gap in the chain, not at the chain's overall
    first/last reported residue -- `get_pdb_str` only emits currently
    *present* atoms, so a fixed-motif segment that reached across another,
    unrelated gap would reference residues that simply aren't in the PDB
    text handed to RFdiffusion. One gap per call; a caller filling every gap
    in a chain calls this once per gap.
    """
    chain_id = str(chain.id)
    if len(chain_id) != 1:
        raise ValueError(
            f"RFdiffusion contig strings need a single-character chain id, got {chain_id!r}"
        )
    first_res = chain.reported_res[0][0]
    last_res = chain.reported_res[-1][0]
    gap_start, gap_end = min(gap), max(gap)
    loop_len = len(gap)

    # Clamp the flanking islands to the nearest other gap's boundary, so we
    # never claim a residue as "fixed" that isn't actually present.
    before_bound = first_res
    after_bound = last_res
    for other in chain.gaps:
        if other == gap:
            continue
        other_end = max(other)
        if other_end < gap_start and other_end + 1 > before_bound:
            before_bound = other_end + 1
        other_start = min(other)
        if other_start > gap_end and other_start - 1 < after_bound:
            after_bound = other_start - 1

    segments = []
    if gap_start > before_bound:
        segments.append(f"{chain_id}{before_bound}-{gap_start - 1}")
    segments.append(str(loop_len))
    if gap_end < after_bound:
        segments.append(f"{chain_id}{gap_end + 1}-{after_bound}")

    if len(segments) == 1:
        raise ValueError(
            "Gap has no present residues on either side to use as a fixed motif "
            "(it's flanked entirely by other gaps or chain termini)."
        )
    return "/".join(segments)


def chain_to_pdb_lines(chain):
    """Return the chain's current atoms as PDB-format lines, suitable for
    `rfdiffusion.inference.utils.parse_pdb_lines`. No disk I/O."""
    return get_pdb_str(chain).splitlines(keepends=True)


def _kabsch_transform(mobile, target):
    """Return a function mapping points from `mobile`'s frame onto
    `target`'s frame (both `(N,3)`), via the same SVD-based rigid alignment
    `rfdiffusion.util.calc_rmsd` uses internally -- that function only
    returns the resulting RMSD scalar, so this re-derives the same
    rotation/translation in a form we can actually apply to other points.
    """
    mobile_centroid = mobile.mean(axis=0)
    target_centroid = target.mean(axis=0)
    mobile_c = mobile - mobile_centroid
    target_c = target - target_centroid

    C = target_c.T @ mobile_c
    V, S, W = np.linalg.svd(C)
    d = np.ones((3, 3))
    d[:, -1] = np.sign(np.linalg.det(V) * np.linalg.det(W))
    U = (d * V) @ W

    def transform(points):
        return (points - mobile_centroid) @ U + target_centroid

    return transform


def _as_numpy(x):
    """NumPy view of `x`, which may be a NumPy array (MinimalLoopSampler) or a
    torch tensor (FullPipelineLoopBuilder). Duck-typed so torch is never
    imported here."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def extract_new_loop_coords(contig_map, target_feats, x_final, px0_final=None):
    """Slice out RFdiffusion's output for exactly the newly-diffused
    positions (`contig_map.ref[i] == ("_", "_")`), in ascending sequence
    order (matches `sorted(gap)`, since contig segments are emitted
    left-to-right).

    RFdiffusion's reverse-diffusion pipeline does not keep the output in
    the input structure's absolute coordinate frame: the motif's internal
    geometry is preserved, but the whole output (motif + generated region)
    is typically translated/rotated as a rigid body relative to the input.
    Before slicing anything out, this realigns the output onto the true
    input frame using the motif's own CA atoms (output position vs. true
    `target_feats["xyz_27"]` position), via `_kabsch_transform` -- without
    this, a newly-generated loop's coordinates would not actually connect
    to the rest of the untouched crimm chain.

    Returns the FULL, untruncated per-position atom14 array -- not just the
    4 backbone atoms `scripts/run_inference.py` keeps -- plus a `named` view
    keyed by atom name, so a caller can see what RFdiffusion's mask token
    (`aa2long[21]`) actually reconstructs for a diffused position: an
    idealized CB in addition to N/CA/C/O, nothing beyond that.
    """
    diffused_idx0 = [i for i, ref in enumerate(contig_map.ref) if ref == ("_", "_")]
    atom_names_14 = [name.strip() if name else None for name in aa2long[21][:14]]

    motif_output_ca = _as_numpy(x_final)[contig_map.hal_idx0, 1]
    motif_true_ca = _as_numpy(target_feats["xyz_27"])[contig_map.ref_idx0, 1]
    to_input_frame = _kabsch_transform(motif_output_ca, motif_true_ca)

    def _realign(t):
        arr = _as_numpy(t)
        shape = arr.shape
        return to_input_frame(arr.reshape(-1, 3)).reshape(shape)

    coords_np = _realign(x_final)
    raw_atom14 = coords_np[diffused_idx0]

    raw_atom14_px0 = None
    if px0_final is not None:
        raw_atom14_px0 = _realign(px0_final)[diffused_idx0]

    named = [
        {
            name: coords_np[idx, slot]
            for slot, name in enumerate(atom_names_14)
            if name
        }
        for idx in diffused_idx0
    ]

    return {
        "diffused_idx0": diffused_idx0,
        "raw_atom14": raw_atom14,
        "raw_atom14_px0": raw_atom14_px0,
        "atom_names_14": atom_names_14,
        "named": named,
    }


def build_gly_residues(named_coords, gap):
    """Turn RFdiffusion's per-position named coordinates into real crimm
    `Residue` objects (resname GLY -- RFdiffusion never outputs anything
    else for a diffused position), ready for `insert_residues`.

    Only {'N','CA','C','O'} are used: crimm's GLY topology has no CB atom,
    even though RFdiffusion's mask token places an idealized one -- see
    `extract_new_loop_coords`'s `named`/`raw_atom14` for that value if
    you need it; passing 'CB' into `create_residue_from_coord_dict` here
    would KeyError against crimm's real GLY definition.
    """
    resseqs = sorted(gap)
    if len(named_coords) != len(resseqs):
        raise ValueError(
            f"RFdiffusion returned {len(named_coords)} new positions for a "
            f"{len(resseqs)}-residue gap {resseqs}"
        )
    gly_def = _gly_topology()
    residues = []
    for resseq, coord_dict in zip(resseqs, named_coords):
        backbone_only = {
            name: np.asarray(coord_dict[name])
            for name in ("N", "CA", "C", "O")
            if name in coord_dict
        }
        residues.append(
            gly_def.create_residue_from_coord_dict(backbone_only, resseq=resseq)
        )
    return residues


def insert_residues(chain, gap, residues, inplace=False):
    """Insert newly-built `residues` (matched 1:1 to `sorted(gap)`) into
    `chain`, filling that gap. Performs the same handful of operations
    crimm's own `ChainLoopBuilder._copy_gap_residues` does, written
    directly against `Chain`'s public API so this module has no
    dependency on crimm internals.
    """
    if not inplace:
        chain = chain.copy()
    for resseq, residue in zip(sorted(gap), residues):
        hetflag, _, icode = residue.id
        residue.id = (hetflag, resseq, icode)
        chain.add(residue)
    chain.sort_residues()
    return chain


def resolve_gap(chain, gap):
    """Pick the gap a `fill_gap` call should act on: the explicit `gap` if
    given (validated against `chain.gaps`), else the chain's sole gap."""
    if gap is not None:
        if gap not in chain.gaps:
            raise ValueError(f"{gap} is not one of this chain's gaps: {chain.gaps}")
        return gap
    if len(chain.gaps) != 1:
        raise ValueError(
            f"Chain has {len(chain.gaps)} gaps; pass `gap=` to pick one "
            f"(gaps: {chain.gaps})"
        )
    return chain.gaps[0]


def fill_gap_minimal(
    chain,
    model,
    gap=None,
    num_designs=1,
    design_index=0,
    T=50,
    final_step=1,
    rfdiffusion_repo_dir=None,
    deterministic=False,
    inplace=False,
    progress=None,
):
    """crimm-facing convenience wrapper for
    `crimm_ml.minimal_loop_sampler.MinimalLoopSampler`, mirroring
    `FullPipelineLoopBuilder.fill_gap`'s signature/return shape so the two
    backends are interchangeable. Kept here (not in minimal_loop_sampler.py)
    so that module stays free of any `import crimm`.

    `model` is loaded once by the caller with
    `crimm_ml.models.load_model` (PyTorch checkpoint or ONNX export)
    and can be reused across calls. `progress` is passed to
    `MinimalLoopSampler.sample` (see `crimm_ml.progress`).
    """
    from crimm_ml.minimal_loop_sampler import MinimalLoopSampler

    gap = resolve_gap(chain, gap)
    contig = build_contig_string(chain, gap)
    pdb_lines = chain_to_pdb_lines(chain)

    sampler = MinimalLoopSampler(
        pdb_lines,
        contig,
        model,
        T=T,
        rfdiffusion_repo_dir=rfdiffusion_repo_dir,
    )
    raw_results = sampler.sample(
        num_designs=num_designs, final_step=final_step, deterministic=deterministic,
        progress=progress,
    )

    for result in raw_results:
        result["coords"] = extract_new_loop_coords(
            result["contig_map"],
            result["target_feats"],
            result["x_final"],
            result.get("px0_final"),
        )

    chosen = raw_results[design_index]
    new_residues = build_gly_residues(chosen["coords"]["named"], gap)
    new_chain = insert_residues(chain, gap, new_residues, inplace=inplace)
    return new_chain, raw_results
