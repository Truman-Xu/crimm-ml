"""What every model behind `MinimalLoopSampler` provides. No torch here."""

#: Input and output names of the network, in `forward` order. The ONNX export
#: (`examples/export_onnx.py`) names its graph inputs and outputs with these.
INPUT_NAMES = ["msa_latent", "msa_full", "seq", "xyz", "idx", "t1d", "t2d", "xyz_t", "alpha_t"]
OUTPUT_NAMES = ["msa0", "pair", "xyz_out", "state", "alpha", "logits_aa", "pred_lddt"]

#: The checkpoint sections the sampler reads (diffusion schedule,
#: preprocessing flags, and the architecture they belong to).
CONFIG_SECTIONS = ("model", "diffuser", "preprocess")

#: ONNX model metadata key holding those sections as JSON, so an ONNX model
#: needs no checkpoint file to be used.
CONFIG_METADATA_KEY = "crimm_rf.config_dict"


class RFModel:
    """The network, loaded one way or another.

    Call it with the nine arrays `MinimalLoopSampler._preprocess` builds
    (NumPy, float32 except `idx`, which is int64); it returns the seven
    outputs named in `OUTPUT_NAMES` as NumPy arrays on the CPU.

    Attributes:
        backend: `"torch"` or `"onnx"`.
        path: the file it was loaded from.
        device: where it runs, e.g. `"cpu"` or `"cuda:0"`.
        config: `{section: dict}` for `CONFIG_SECTIONS`, taken from the
            checkpoint (PyTorch) or the file's metadata (ONNX).
    """

    backend = None
    path = None
    device = None
    config = None

    def __call__(self, *inputs):
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}(path={str(self.path)!r}, device={self.device!r})"
