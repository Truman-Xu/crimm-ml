"""`PotentialManager`, reduced to the no-op this pipeline always constructs.

RFdiffusion's guiding potentials steer the reverse diffusion with an extra
gradient term (substrate contacts, monomer ROG, oligomer contacts, ...). They
are configured by `potentials.guiding_potentials`, which is `null` in
`base.yaml` and which `MinimalLoopSampler` deliberately does not expose -- loop
building inside an existing structure has no use for them.

With no potentials configured, `Denoise.get_next_pose` short-circuits on
`self.potential_manager.is_empty()` and never calls anything else here
(`inference/utils.py:387`). So the manager exists only to satisfy that
interface, and the ~470 lines of potential implementations it used to dispatch
to were unreachable. They were removed along with `potentials/potentials.py`;
the originals remain in `rfdiffusion/potentials/` if they are ever wanted back,
and restoring them means restoring `parse_potential_string`,
`initialize_all_potentials` and `compute_all_potentials` too.

This keeps the exact constructor signature and the two methods that are
actually reached, so `Denoise` needs no change.
"""


class PotentialManager:
    """A manager that is always empty, because nothing can configure it."""

    def __init__(self, potentials_config, ppi_config, diffuser_config,
                 inference_config, hotspot_0idx, binderlen):
        self.potentials_config = potentials_config
        self.ppi_config = ppi_config
        self.inference_config = inference_config
        self.guide_scale = potentials_config.guide_scale
        self.guide_decay = potentials_config.guide_decay
        self.T = diffuser_config.T

        # The one branch that could ever produce a non-empty manager. Asserted
        # rather than handled: silently ignoring a requested potential would be
        # worse than failing loudly, and this pipeline has no way to set it.
        assert potentials_config.guiding_potentials is None, (
            "guiding potentials are not supported by this stripped build; "
            "see the module docstring for how to restore them"
        )
        self.potentials_to_apply = []

    def is_empty(self):
        """Always True here -- `Denoise.get_next_pose` returns early on this."""
        return True

    def get_guide_scale(self, t):
        """Scale factor for the guiding gradient at timestep `t`.

        Unreachable while `is_empty()` is True, but kept because it is part of
        the interface `Denoise` holds, and it depends only on config.
        """
        decays = {
            "constant": lambda t: self.guide_scale,
            "linear": lambda t: t / self.T * self.guide_scale,
            "quadratic": lambda t: t ** 2 / self.T ** 2 * self.guide_scale,
            "cubic": lambda t: t ** 3 / self.T ** 3 * self.guide_scale,
        }
        if self.guide_decay not in decays:
            raise ValueError(
                f"guide_decay must be one of {sorted(decays)}; got {self.guide_decay!r}"
            )
        return decays[self.guide_decay](t)
