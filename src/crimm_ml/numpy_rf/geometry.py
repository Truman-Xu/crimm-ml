"""Backbone geometry, torsions, all-atom coordinates and template features.

NumPy ports of the torch functions the sampler calls around the network:

    rfdiffusion util.py         rigid_from_3_points, generate_Cbeta, th_ang_v,
                                th_dih(_v), make_frame, get_tor_mask,
                                get_torsions, and the residue tables built at
                                import (torsion_indices, torsion_can_flip,
                                reference_angles, base_indices,
                                xyzs_in_base_frame, RTs_by_torsion)
    rfdiffusion util_module.py  make_rotX, make_rotZ, make_rot_axis,
                                ComputeAllAtomCoords -> compute_all_atom_coords
    rfdiffusion kinematics.py   get_pair_dist, get_ang, get_dih, xyz_to_c6d,
                                xyz_to_t2d, dist_to_onehot, get_init_xyz

Arrays are float32 wherever the torch originals ran in torch's default dtype,
so the numbers match them to float32 rounding; `tests` for that live in
`models/ln_probe/np_port/`. Only the code paths the sampler reaches are
ported: `rigid_from_3_points(non_ideal=True)` and `get_torsions(mask_in=...)`
raise instead.
"""
import numpy as np

from crimm_ml.minimal_rf.chemical import (
    INIT_CRDS, aa2long, aa2longalt, aa2num, ideal_coords, torsions,
)

F32 = np.float32


# ---------------------------------------------------------------- frames
def rigid_from_3_points(N, Ca, C, non_ideal=False, eps=1e-8):
    """[B, L, 3] x3 -> rotation [B, L, 3, 3] and origin (= Ca)."""
    if non_ideal:
        raise NotImplementedError("non_ideal frames are not used by the sampler")
    v1 = C - Ca
    v2 = N - Ca
    e1 = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + eps)
    u2 = v2 - np.sum(e1 * v2, axis=-1)[..., None] * e1
    e2 = u2 / (np.linalg.norm(u2, axis=-1, keepdims=True) + eps)
    e3 = np.cross(e1, e2)
    R = np.concatenate([e1[..., None], e2[..., None], e3[..., None]], axis=-1)
    return R, Ca


def generate_Cbeta(N, Ca, C):
    b = Ca - N
    c = C - Ca
    a = np.cross(b, c)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca


def _th_norm(x, eps=1e-8):
    return np.sqrt(np.sum(np.square(x), axis=-1, keepdims=True) + eps)


def _th_N(x, alpha=0.0):
    return x / (_th_norm(x) + alpha)


def th_ang_v(ab, bc, eps=1e-8):
    ab, bc = _th_N(ab), _th_N(bc)
    cos_angle = np.clip(np.sum(ab * bc, axis=-1), -1, 1)
    sin_angle = np.sqrt(1 - np.square(cos_angle) + eps)
    return np.stack((cos_angle, sin_angle), axis=-1)


def th_dih_v(ab, bc, cd):
    def cross(a, b):
        a, b = np.broadcast_arrays(a, b)
        return np.cross(a, b)

    ab, bc, cd = _th_N(ab), _th_N(bc), _th_N(cd)
    n1 = _th_N(cross(ab, bc))
    n2 = _th_N(cross(bc, cd))
    sin_angle = np.sum(cross(n1, bc) * n2, axis=-1)
    cos_angle = np.sum(n1 * n2, axis=-1)
    return np.stack((cos_angle, sin_angle), axis=-1)


def th_dih(a, b, c, d):
    return th_dih_v(a - b, b - c, c - d)


def make_frame(X, Y):
    Xn = X / np.linalg.norm(X)
    Y = Y - np.dot(Y, Xn) * Xn
    Yn = Y / np.linalg.norm(Y)
    Z = np.cross(Xn, Yn)
    Zn = Z / np.linalg.norm(Z)
    return np.stack((Xn, Yn, Zn), axis=-1)


# ------------------------------------------------ residue tables (util.py)
def _build_tables():
    tors_idx = np.zeros((22, 4, 4), dtype=np.int64)
    can_flip = np.zeros((22, 10), dtype=bool)
    for i in range(22):
        i_l, i_a = aa2long[i], aa2longalt[i]
        for j in range(4):
            if torsions[i][j] is None:
                continue
            for k in range(4):
                a = torsions[i][j][k]
                tors_idx[i, j, k] = i_l.index(a)
                if i_l.index(a) != i_a.index(a):
                    can_flip[i, 3 + j] = True
    can_flip[8, 4] = False  # HIS is a special case

    base_idx = np.zeros((22, 27), dtype=np.int64)
    xyz_base = np.ones((22, 27, 4), dtype=F32)
    RTs = np.tile(np.eye(4, dtype=F32), (22, 7, 1, 1))
    ref_angles = np.ones((22, 3, 2), dtype=F32)
    ex = np.array([1.0, 0.0, 0.0], dtype=F32)
    for i in range(22):
        i_l = aa2long[i]
        for name, base, coords in ideal_coords[i]:
            idx = i_l.index(name)
            base_idx[i, idx] = base
            xyz_base[i, idx, :3] = np.array(coords, dtype=F32)

        RTs[i, 0, :3, :3] = np.eye(3, dtype=F32)
        RTs[i, 0, :3, 3] = 0
        RTs[i, 1, :3, :3] = make_frame(xyz_base[i, 0, :3] - xyz_base[i, 1, :3], ex)
        RTs[i, 1, :3, 3] = xyz_base[i, 0, :3]
        RTs[i, 2, :3, :3] = make_frame(xyz_base[i, 2, :3] - xyz_base[i, 1, :3],
                                       xyz_base[i, 1, :3] - xyz_base[i, 0, :3])
        RTs[i, 2, :3, 3] = xyz_base[i, 2, :3]
        if torsions[i][0] is not None:
            a0, a1, a2 = tors_idx[i, 0, 0:3]
            RTs[i, 3, :3, :3] = make_frame(xyz_base[i, a2, :3] - xyz_base[i, a1, :3],
                                           xyz_base[i, a0, :3] - xyz_base[i, a1, :3])
            RTs[i, 3, :3, 3] = xyz_base[i, a2, :3]
        for j in range(1, 4):
            if torsions[i][j] is not None:
                a2 = tors_idx[i, j, 2]
                if (i == 18 and j == 2) or (i == 8 and j == 2):
                    a0, a1 = tors_idx[i, j, 0:2]
                    RTs[i, 3 + j, :3, :3] = make_frame(xyz_base[i, a2, :3] - xyz_base[i, a1, :3],
                                                       xyz_base[i, a0, :3] - xyz_base[i, a1, :3])
                else:
                    RTs[i, 3 + j, :3, :3] = make_frame(xyz_base[i, a2, :3], -ex)
                RTs[i, 3 + j, :3, 3] = xyz_base[i, a2, :3]

        NCr = 0.5 * (xyz_base[i, 0, :3] + xyz_base[i, 2, :3])
        CAr = xyz_base[i, 1, :3]
        CBr = xyz_base[i, 4, :3]
        CGr = xyz_base[i, 5, :3]
        ref_angles[i, 0, :] = th_ang_v(CBr - CAr, NCr - CAr)
        NCp = xyz_base[i, 2, :3] - xyz_base[i, 0, :3]
        NCpp = NCp - np.dot(NCp, NCr) / np.dot(NCr, NCr) * NCr
        ref_angles[i, 1, :] = th_ang_v(CBr - CAr, NCpp)
        ref_angles[i, 2, :] = th_ang_v(CGr, -ex)
    return tors_idx, can_flip, base_idx, xyz_base, RTs, ref_angles


(torsion_indices, torsion_can_flip, base_indices,
 xyzs_in_base_frame, RTs_by_torsion, reference_angles) = _build_tables()


# ---------------------------------------------------------------- torsions
def get_tor_mask(seq, torsion_indices, mask_in=None):
    if mask_in is not None:
        raise NotImplementedError("mask_in is not used by the sampler")
    B, L = seq.shape[:2]
    tors_mask = np.ones((B, L, 10), dtype=bool)
    tors_mask[..., 3:7] = torsion_indices[seq, :, -1] > 0
    tors_mask[:, 0, 1] = False
    tors_mask[:, -1, 0] = False
    tors_mask[:, :, 7] = seq != aa2num["GLY"]
    tors_mask[:, :, 8] = seq != aa2num["GLY"]
    m9 = np.logical_and(seq != aa2num["GLY"], seq != aa2num["ALA"])
    m9 = np.logical_and(m9, seq != aa2num["UNK"])
    tors_mask[:, :, 9] = np.logical_and(m9, seq != aa2num["MAS"])
    return tors_mask


def get_torsions(xyz_in, seq, torsion_indices, torsion_can_flip, ref_angles, mask_in=None):
    """xyz_in [B, L, 27, 3] float32 (NaN for absent atoms), seq [B, L] int."""
    B, L = xyz_in.shape[:2]
    tors_mask = get_tor_mask(seq, torsion_indices, mask_in)
    tors_planar = np.zeros((B, L, 10), dtype=bool)
    tors_planar[:, :, 5] = seq == aa2num["TYR"]

    with np.errstate(invalid="ignore"):
        xyz = xyz_in.copy()
        Rs, Ts = rigid_from_3_points(xyz[..., 0, :], xyz[..., 1, :], xyz[..., 2, :])
        Nideal = np.array([-0.5272, 1.3593, 0.000], dtype=F32)
        Cideal = np.array([1.5233, 0.000, 0.000], dtype=F32)
        xyz[..., 0, :] = Rs @ Nideal + Ts
        xyz[..., 2, :] = Rs @ Cideal + Ts

        tors = np.zeros((B, L, 10, 2), dtype=F32)
        tors[:, 0, 1, 0] = 1.0
        tors[:, -1, 0, 0] = 1.0
        tors[:, :-1, 0, :] = th_dih(xyz[:, :-1, 1, :], xyz[:, :-1, 2, :], xyz[:, 1:, 0, :], xyz[:, 1:, 1, :])
        tors[:, 1:, 1, :] = th_dih(xyz[:, :-1, 2, :], xyz[:, 1:, 0, :], xyz[:, 1:, 1, :], xyz[:, 1:, 2, :])
        tors[:, :, 2, :] = -1 * th_dih(xyz[:, :, 0, :], xyz[:, :, 1, :], xyz[:, :, 2, :], xyz[:, :, 3, :])

        ti = [np.take_along_axis(xyz, np.repeat(torsion_indices[seq, :, k, None], 3, axis=-1), axis=2)
              for k in range(4)]
        tors[:, :, 3:7, :] = th_dih(*ti)

        NC = 0.5 * (xyz[:, :, 0, :3] + xyz[:, :, 2, :3])
        CA = xyz[:, :, 1, :3]
        CB = xyz[:, :, 4, :3]
        t = th_ang_v(CB - CA, NC - CA)
        t0 = ref_angles[seq][..., 0, :]
        tors[:, :, 7, :] = np.stack((np.sum(t * t0, axis=-1),
                                     t[..., 0] * t0[..., 1] - t[..., 1] * t0[..., 0]), axis=-1)

        NCCA = NC - CA
        NCp = xyz[:, :, 2, :3] - xyz[:, :, 0, :3]
        NCpp = NCp - np.sum(NCp * NCCA, axis=-1, keepdims=True) / np.sum(NCCA * NCCA, axis=-1, keepdims=True) * NCCA
        t = th_ang_v(CB - CA, NCpp)
        t0 = ref_angles[seq][..., 1, :]
        tors[:, :, 8, :] = np.stack((np.sum(t * t0, axis=-1),
                                     t[..., 0] * t0[..., 1] - t[..., 1] * t0[..., 0]), axis=-1)

        CG = xyz[:, :, 5, :3]
        t = th_ang_v(CG - CB, CA - CB)
        t0 = ref_angles[seq][..., 2, :]
        tors[:, :, 9, :] = np.stack((np.sum(t * t0, axis=-1),
                                     t[..., 0] * t0[..., 1] - t[..., 1] * t0[..., 0]), axis=-1)

    nan0 = np.isnan(tors[..., 0])
    nan1 = np.isnan(tors[..., 1])
    tors[..., 0][nan0] = 1.0
    tors[..., 1][nan1] = 0.0

    tors_alt = tors.copy()
    tors_alt[torsion_can_flip[seq, :]] *= -1
    return tors, tors_alt, tors_mask, tors_planar


# ------------------------------------------------- all-atom (util_module)
def _eye4(B, L):
    return np.tile(np.eye(4, dtype=F32), (B, L, 1, 1))


def make_rotX(angs, eps=1e-6):
    B, L = angs.shape[:2]
    NORM = np.linalg.norm(angs, axis=-1) + eps
    RTs = _eye4(B, L)
    RTs[:, :, 1, 1] = angs[:, :, 0] / NORM
    RTs[:, :, 1, 2] = -angs[:, :, 1] / NORM
    RTs[:, :, 2, 1] = angs[:, :, 1] / NORM
    RTs[:, :, 2, 2] = angs[:, :, 0] / NORM
    return RTs


def make_rotZ(angs, eps=1e-6):
    B, L = angs.shape[:2]
    NORM = np.linalg.norm(angs, axis=-1) + eps
    RTs = _eye4(B, L)
    RTs[:, :, 0, 0] = angs[:, :, 0] / NORM
    RTs[:, :, 0, 1] = -angs[:, :, 1] / NORM
    RTs[:, :, 1, 0] = angs[:, :, 1] / NORM
    RTs[:, :, 1, 1] = angs[:, :, 0] / NORM
    return RTs


def make_rot_axis(angs, u, eps=1e-6):
    B, L = angs.shape[:2]
    NORM = np.linalg.norm(angs, axis=-1) + eps
    RTs = _eye4(B, L)
    ct = angs[:, :, 0] / NORM
    st = angs[:, :, 1] / NORM
    u0, u1, u2 = u[:, :, 0], u[:, :, 1], u[:, :, 2]
    RTs[:, :, 0, 0] = ct + u0 * u0 * (1 - ct)
    RTs[:, :, 0, 1] = u0 * u1 * (1 - ct) - u2 * st
    RTs[:, :, 0, 2] = u0 * u2 * (1 - ct) + u1 * st
    RTs[:, :, 1, 0] = u0 * u1 * (1 - ct) + u2 * st
    RTs[:, :, 1, 1] = ct + u1 * u1 * (1 - ct)
    RTs[:, :, 1, 2] = u1 * u2 * (1 - ct) - u0 * st
    RTs[:, :, 2, 0] = u0 * u2 * (1 - ct) - u1 * st
    RTs[:, :, 2, 1] = u1 * u2 * (1 - ct) + u0 * st
    RTs[:, :, 2, 2] = ct + u2 * u2 * (1 - ct)
    return RTs


def compute_all_atom_coords(seq, xyz, alphas):
    """`ComputeAllAtomCoords.forward(seq, xyz, alphas)` with its defaults
    (`non_ideal=False, use_H=True`). seq [B, L] int, xyz [B, L, >=3, 3],
    alphas [B, L, 10, 2]. Returns (RTframes [B, L, 9, 4, 4], xyz [B, L, 27, 3])."""
    B, L = xyz.shape[:2]
    Rs, Ts = rigid_from_3_points(xyz[..., 0, :], xyz[..., 1, :], xyz[..., 2, :])
    RTF0 = _eye4(B, L)
    RTF0[:, :, :3, :3] = Rs
    RTF0[:, :, :3, 3] = Ts

    rts = RTs_by_torsion[seq]  # [B, L, 7, 4, 4]
    RTF1 = RTF0 @ rts[:, :, 0] @ make_rotX(alphas[:, :, 0, :])
    RTF2 = RTF0 @ rts[:, :, 1] @ make_rotX(alphas[:, :, 1, :])
    RTF3 = RTF0 @ rts[:, :, 2] @ make_rotX(alphas[:, :, 2, :])

    basexyzs = xyzs_in_base_frame[seq]
    NCr = 0.5 * (basexyzs[:, :, 2, :3] + basexyzs[:, :, 0, :3])
    CAr = basexyzs[:, :, 1, :3]
    CBr = basexyzs[:, :, 4, :3]
    CBrotaxis1 = np.cross(CBr - CAr, NCr - CAr)
    CBrotaxis1 /= np.linalg.norm(CBrotaxis1, axis=-1, keepdims=True) + 1e-8

    NCp = basexyzs[:, :, 2, :3] - basexyzs[:, :, 0, :3]
    NCpp = NCp - np.sum(NCp * NCr, axis=-1, keepdims=True) / np.sum(NCr * NCr, axis=-1, keepdims=True) * NCr
    CBrotaxis2 = np.cross(CBr - CAr, NCpp)
    CBrotaxis2 /= np.linalg.norm(CBrotaxis2, axis=-1, keepdims=True) + 1e-8

    CBrot1 = make_rot_axis(alphas[:, :, 7, :], CBrotaxis1)
    CBrot2 = make_rot_axis(alphas[:, :, 8, :], CBrotaxis2)
    RTF8 = RTF0 @ CBrot1 @ CBrot2

    RTF4 = RTF8 @ rts[:, :, 3] @ make_rotX(alphas[:, :, 3, :]) @ make_rotZ(alphas[:, :, 9, :])
    RTF5 = RTF4 @ rts[:, :, 4] @ make_rotX(alphas[:, :, 4, :])
    RTF6 = RTF5 @ rts[:, :, 5] @ make_rotX(alphas[:, :, 5, :])
    RTF7 = RTF6 @ rts[:, :, 6] @ make_rotX(alphas[:, :, 6, :])

    RTframes = np.stack((RTF0, RTF1, RTF2, RTF3, RTF4, RTF5, RTF6, RTF7, RTF8), axis=2)
    idx = np.broadcast_to(base_indices[seq][..., None, None], (B, L, 27, 4, 4))
    frames = np.take_along_axis(RTframes, idx, axis=2)            # [B, L, 27, 4, 4]
    xyzs = (frames @ basexyzs[..., None])[..., 0]                 # [B, L, 27, 4]
    return RTframes, xyzs[..., :3]


# --------------------------------------------- template features (kinematics)
PARAMS = {"DMIN": 2.0, "DMAX": 20.0, "DBINS": 36, "ABINS": 36}


def get_pair_dist(a, b):
    """`torch.cdist(a, b, p=2)` as torch computes it. Above 25 points per set
    torch uses matrix products rather than explicit differences, which is
    less exact; that is reproduced rather than replaced, because the result
    is binned into the template features and this is what the model was run
    with."""
    if a.shape[-2] <= 25 and b.shape[-2] <= 25:
        return np.sqrt(np.sum((a[..., :, None, :] - b[..., None, :, :]) ** 2, axis=-1))
    a_norm = np.sum(a ** 2, axis=-1, keepdims=True)
    b_norm = np.sum(b ** 2, axis=-1, keepdims=True)
    a_ = np.concatenate([a * -2, np.ones_like(a_norm)], axis=-1)
    b_ = np.concatenate([b, b_norm], axis=-1)
    res = a_ @ np.swapaxes(b_, -1, -2)
    res = res + a_norm
    return np.sqrt(np.maximum(res, 1e-30))


def get_ang(a, b, c):
    v = a - b
    w = c - b
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    w = w / np.linalg.norm(w, axis=-1, keepdims=True)
    return np.arccos(np.sum(v * w, axis=-1))


def get_dih(a, b, c, d):
    b0 = a - b
    b1 = c - b
    b2 = d - c
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)
    return np.arctan2(y, x)


def xyz_to_c6d(xyz, params=PARAMS):
    batch, nres = xyz.shape[:2]
    N, Ca, C = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]
    Cb = generate_Cbeta(N, Ca, C)

    c6d = np.zeros((batch, nres, nres, 4), dtype=xyz.dtype)
    dist = get_pair_dist(Cb, Cb)
    dist[np.isnan(dist)] = 999.9
    c6d[..., 0] = dist + 999.9 * np.eye(nres, dtype=xyz.dtype)[None, ...]
    b, i, j = np.nonzero(c6d[..., 0] < params["DMAX"])

    c6d[b, i, j, 1] = get_dih(Ca[b, i], Cb[b, i], Cb[b, j], Ca[b, j])
    c6d[b, i, j, 2] = get_dih(N[b, i], Ca[b, i], Cb[b, i], Cb[b, j])
    c6d[b, i, j, 3] = get_ang(Ca[b, i], Cb[b, i], Cb[b, j])

    d0 = c6d[..., 0]
    d0[d0 >= params["DMAX"]] = 999.9

    mask = np.zeros((batch, nres, nres), dtype=xyz.dtype)
    mask[b, i, j] = 1.0
    return c6d, mask


def dist_to_onehot(dist, params=PARAMS):
    dist[np.isnan(dist)] = 999.9
    dstep = (params["DMAX"] - params["DMIN"]) / params["DBINS"]
    dbins = torch_linspace_f32(params["DMIN"] + dstep, params["DMAX"], params["DBINS"])
    db = np.searchsorted(dbins, dist, side="left")   # == torch.bucketize(right=False)
    return np.eye(params["DBINS"] + 1, dtype=F32)[db]


def xyz_to_t2d(xyz_t, params=PARAMS):
    """[B, T, L, >=3, 3] -> [B, T, L, L, 44]"""
    B, T, L = xyz_t.shape[:3]
    with np.errstate(invalid="ignore"):
        c6d, mask = xyz_to_c6d(xyz_t[:, :, :, :3].reshape(B * T, L, 3, 3), params=params)
        c6d = c6d.reshape(B, T, L, L, 4)
        mask = mask.reshape(B, T, L, L, 1)
        dist = dist_to_onehot(c6d[..., 0], params)
        orien = np.concatenate((np.sin(c6d[..., 1:]), np.cos(c6d[..., 1:])), axis=-1) * mask
        present = ~np.isnan(c6d[:, :, :, :, 0])
        t2d = np.concatenate((dist, orien, present[..., None].astype(F32)), axis=-1)
    t2d[np.isnan(t2d)] = 0.0
    return t2d


def get_init_xyz(xyz_t):
    """[B, T, L, 27, 3] -> same, with missing residues placed at the nearest
    present residue's CA (ideal backbone, NaN beyond atom 3)."""
    B, T, L = xyz_t.shape[:3]
    init = np.tile(INIT_CRDS.reshape(1, 1, 1, 27, 3), (B, T, L, 1, 1))
    if np.isnan(xyz_t).all():
        return init

    mask = np.isnan(xyz_t[:, :, :, :3]).any(axis=-1).any(axis=-1)          # [B, T, L]
    present = ~mask[:, :, :, None]
    num = (present * np.nan_to_num(xyz_t[:, :, :, 1, :])).sum(axis=2)
    den = present.sum(axis=2).astype(F32) + F32(1e-4)
    center_CA = num / den
    xyz_t = xyz_t - center_CA.reshape(B, T, 1, 1, 3)

    for i_b in range(B):
        for i_T in range(T):
            if mask[i_b, i_T].all():
                continue
            exist_in_templ = np.nonzero(~mask[i_b, i_T])[0]
            seqmap = np.abs(np.arange(L)[:, None] - exist_in_templ[None, :])
            seqmap = np.argmin(seqmap, axis=-1)
            idx = exist_in_templ[seqmap]
            offset_CA = xyz_t[i_b, i_T, idx, 1, :]
            init[i_b, i_T] += offset_CA.reshape(L, 1, 3)

    return np.where(mask.reshape(B, T, L, 1, 1), init, xyz_t)


def torch_linspace_f32(start, end, steps):
    """`torch.linspace(start, end, steps)` in float32, bit for bit: the step
    is taken in float32, the points in float64, from both ends inward."""
    s32, e32 = F32(start), F32(end)
    if steps == 1:
        return np.array([s32], dtype=F32)
    step = float((e32 - s32) / F32(steps - 1))
    i = np.arange(steps)
    half = steps // 2
    vals = np.where(i < half, float(s32) + step * i, float(e32) - step * (steps - i - 1))
    return vals.astype(F32)
