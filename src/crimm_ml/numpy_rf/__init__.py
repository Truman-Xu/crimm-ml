"""NumPy ports of the RFdiffusion code that runs around the network.

`MinimalLoopSampler` uses these, so sampling needs no torch when the network
itself runs in ONNX Runtime (`crimm_ml.models`). Each function is a
line-by-line port of its torch original in `crimm_ml.minimal_rf`,
keeping torch's float32/float64 choices so the numbers agree to float
rounding:

    geometry.py   frames, torsions, all-atom coordinates, template features,
                  and the residue tables  (util.py, util_module.py, kinematics.py)
    igso3.py      IGSO(3) tables, SO(3) exponential map         (igso3.py)
    diffusion.py  noise schedules, forward diffusion of frames  (diffusion.py)
    denoise.py    one reverse-diffusion step                    (inference/utils.py)

Randomness: the rotation draws use NumPy's global `np.random`, as upstream
does; the translation draws, which upstream takes from torch, come from a
`numpy.random.Generator` passed in as `noise_rng`. See `diffusion.py`.

The torch originals stay in `minimal_rf/` because the network uses some of
them and the equivalence tests use all of them as the reference.
"""
