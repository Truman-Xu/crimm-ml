"""A standalone, crimm-integrated pipeline built on the stripped Phase 2
backend (`minimal_loop_sampler.MinimalLoopSampler`) -- structured the same
way `full_pipeline_loop_builder.py` wraps the real `Sampler`/
`SelfConditioning` pipeline, so the two are direct counterparts. Lives
entirely in this repo; nothing under crimm/ is modified.

Adds the three things `crimm_utils.fill_gap_minimal` alone doesn't cover:
  - structure acquisition: `fetch_chain` (crimm.Fetchers.fetch_rcsb) and
    `parse_mmcif_chain` (crimm.IO.MMCIFParser), so a caller can go straight
    from a PDB ID or a local mmCIF file to a chain ready to repair
  - gap reporting: `list_gaps`, a thin wrapper around
    crimm.Modeller.LoopBuilder.report_loops -- gives anchor/target
    coordinates and true residue names per gap, not just `chain.gaps`'s
    bare index sets
  - filling every gap in a chain in one call: `fill_all_gaps`

Every actual diffusion call still goes through `MinimalLoopSampler` --
never the real `Sampler`.

The network is loaded once, outside, with
`crimm_ml.models.load_model` (a PyTorch checkpoint or an ONNX
export) and passed in; every `fill_gap` call reuses it, so filling several
gaps no longer reloads the weights each time.
"""
import os
import warnings

from crimm_ml import crimm_utils

warnings.warn(
    "crimm_ml.minimal_pipeline_loop_builder is deprecated and will be removed in the next release: building loops "
    "into crimm chains now lives in crimm itself (crimm.Modeller.RFLoopBuilder), "
    "which rebuilds full residues rather than glycine backbones.",
    FutureWarning, stacklevel=2)


def _as_model(entity):
    """Normalize a crimm Structure OR Model to a Model -- the same level
    check crimm.IO.PDBString.get_pdb_str uses internally (Structure.level
    == 'S' -> its first Model is at .child_list[0])."""
    if entity.level == "S":
        return entity.child_list[0]
    return entity


def _residue_or_none(chain, resseq):
    # `resseq in chain` is NOT safe here: crimm/Biopython's __contains__
    # checks the bare id against child_dict's *tuple* keys without the
    # int -> (hetflag, resseq, icode) translation __getitem__ does via
    # Chain._translate_id, so it would spuriously read as "absent" even
    # for residues that are present. try/except on __getitem__ is correct.
    try:
        return chain[resseq]
    except KeyError:
        return None


def _report_loops_safe(chain):
    """Same per-gap shape as crimm.Modeller.LoopBuilder.report_loops
    (anchor/target flanking coordinates, loop_res_ids, loop resnames),
    built from the same underlying data (`chain.gaps`, `chain.missing_res`)
    -- but tolerant of a gap touching the chain terminus.

    report_loops itself (LoopBuilder.py:117-132) unconditionally looks up
    an anchor residue at `min(gap) - 1` and a target at `max(gap) + 1`; for
    a gap that starts at residue 1 or ends at the chain's last residue
    (very common in real structures -- unresolved termini), one of those
    lookups doesn't exist and it raises KeyError for the *entire* chain,
    not just that one gap. `anchor`/`target` are `None` here instead, for
    whichever side has no residue to anchor on.
    """
    res_dict = dict(chain.missing_res)
    loops = []
    for gap in chain.gaps:
        res_ids = sorted(gap)
        missing_res = [res_dict[res_id] for res_id in res_ids]

        anchor = None
        if (anchor_res := _residue_or_none(chain, min(gap) - 1)) is not None:
            anchor = {"C": anchor_res["C"].coord, "CA": anchor_res["CA"].coord}

        target = None
        if (target_res := _residue_or_none(chain, max(gap) + 1)) is not None:
            target = {"N": target_res["N"].coord, "CA": target_res["CA"].coord}

        loops.append(
            {
                "anchor": anchor,
                "loop_res_ids": res_ids,
                "loop": missing_res,
                "target": target,
            }
        )
    return loops


class MinimalPipelineLoopBuilder:
    """End-to-end crimm -> RFdiffusion loop-building pipeline, built on the
    stripped `MinimalLoopSampler` backend.

    Structure acquisition: `fetch_chain` / `parse_mmcif_chain`.
    Gap reporting: `list_gaps`.
    Filling: `fill_gap` (one gap) / `fill_all_gaps` (every gap in the
    chain, sequentially) -- both drive `MinimalLoopSampler` only.
    Output: `write_pdb`.
    """

    def __init__(self, model, T=50, rfdiffusion_repo_dir=None):
        """`model`: an `RFModel` from `crimm_ml.models.load_model`."""
        self.model = model
        self.rfdiffusion_repo_dir = rfdiffusion_repo_dir
        self.T = T

    # ---------------- structure acquisition ----------------

    @staticmethod
    def fetch_chain(pdb_id, chain_id, **fetch_kwargs):
        """Fetch `pdb_id` from RCSB via `crimm.Fetchers.fetch_rcsb` and
        return the requested chain. Extra kwargs (e.g. `use_bio_assembly`,
        `include_solvent`, `local_entry`) are forwarded to `fetch_rcsb`
        unchanged -- see its docstring for the full list."""
        from crimm.Fetchers import fetch_rcsb

        model = fetch_rcsb(pdb_id, **fetch_kwargs)
        return _as_model(model)[chain_id]

    @staticmethod
    def parse_mmcif_chain(mmcif_path, chain_id, **parser_kwargs):
        """Parse a local mmCIF file via `crimm.IO.MMCIFParser` and return
        the requested chain. Extra kwargs (e.g. `use_bio_assembly`,
        `include_solvent`) are forwarded to `MMCIFParser.__init__`."""
        from crimm.IO.MMCIFParser import MMCIFParser

        parser = MMCIFParser(**parser_kwargs)
        structure_id = os.path.splitext(os.path.basename(mmcif_path))[0]
        structure = parser.get_structure(mmcif_path, structure_id=structure_id)
        return _as_model(structure)[chain_id]

    # ---------------- gap reporting ----------------

    @staticmethod
    def list_gaps(chain):
        """Every gap in `chain`, in the same shape
        `crimm.Modeller.LoopBuilder.report_loops` returns: each entry
        carries `'anchor'`/`'target'` flanking coordinates, the missing
        residues' true names (`'loop'`), and their sequence numbers
        (`'loop_res_ids'`). RFdiffusion only ever consumes the *length* of
        `'loop_res_ids'` (see minimal_loop_sampler.py's module docstring)
        -- the anchor/target/resname info is here for the caller's own use
        (deciding which gaps are worth filling, sanity-checking span
        length against the anchor distance, etc.), not fed to the model.

        Uses `_report_loops_safe` rather than calling `report_loops`
        directly: that function crashes outright (for the whole chain, not
        just one gap) on any gap touching the chain terminus -- see
        `_report_loops_safe`'s docstring. Same data source
        (`chain.gaps`/`chain.missing_res`), same output shape, just
        tolerant of that case.
        """
        return _report_loops_safe(chain)

    # ---------------- filling (MinimalLoopSampler only) ----------------

    def fill_gap(
        self,
        chain,
        gap=None,
        num_designs=1,
        design_index=0,
        final_step=1,
        deterministic=False,
        inplace=False,
        progress=None,
    ):
        """Fill exactly one gap in `chain` using `MinimalLoopSampler`.

        `gap` accepts a `set[int]` (as in `chain.gaps`), or a
        `report_loops`-style dict (its `'loop_res_ids'` is used) -- so the
        output of `list_gaps` can be passed straight in. Same return
        contract as `full_pipeline_loop_builder.FullPipelineLoopBuilder.
        fill_gap`: `(new_chain, raw_results)`.
        """
        gap = self._coerce_gap(chain, gap)
        return crimm_utils.fill_gap_minimal(
            chain,
            self.model,
            gap=gap,
            num_designs=num_designs,
            design_index=design_index,
            T=self.T,
            final_step=final_step,
            rfdiffusion_repo_dir=self.rfdiffusion_repo_dir,
            deterministic=deterministic,
            inplace=inplace,
            progress=progress,
        )

    def fill_all_gaps(
        self,
        chain,
        num_designs=1,
        design_index=0,
        final_step=1,
        deterministic=False,
        progress=None,
    ):
        """Fill every gap in `chain`, one `MinimalLoopSampler` run per gap
        (see `FullPipelineLoopBuilder`'s docstring for why this stays
        one-gap-per-call rather than one multi-segment contig). Always
        works on a copy -- `chain` itself is never mutated, regardless of
        how many gaps get filled.

        Later gaps see earlier ones as part of the fixed motif once
        they're filled (their contig's flanking islands are recomputed
        fresh from the working chain's *current* `.gaps` each call) --
        that's intentional: by the second gap, the first gap's generated
        backbone is real structural context, not something to route
        around.

        Returns `(new_chain, results_by_gap)`, where `results_by_gap` maps
        `tuple(sorted(gap))` -> that gap's `raw_results` list.
        """
        gaps = [set(g) for g in chain.gaps]
        working_chain = chain.copy()
        if not gaps:
            return working_chain, {}

        results_by_gap = {}
        for gap in gaps:
            working_chain, raw_results = self.fill_gap(
                working_chain,
                gap=gap,
                num_designs=num_designs,
                design_index=design_index,
                final_step=final_step,
                deterministic=deterministic,
                inplace=True,
                progress=progress,
            )
            results_by_gap[tuple(sorted(gap))] = raw_results
        return working_chain, results_by_gap

    @staticmethod
    def _coerce_gap(chain, gap):
        if isinstance(gap, dict):
            gap = set(gap["loop_res_ids"])
        return crimm_utils.resolve_gap(chain, gap)

    # ---------------- output ----------------

    @staticmethod
    def write_pdb(chain, path=None):
        """Return `chain`'s current atoms as a PDB-format string, via
        crimm's own `get_pdb_str`; also writes it to `path` if given."""
        pdb_text = "".join(crimm_utils.chain_to_pdb_lines(chain))
        if path is not None:
            with open(path, "w") as f:
                f.write(pdb_text)
        return pdb_text
