"""Post-export graph rewrites that make the model correct and fast in ONNX Runtime.

Rewrites 1 and 2 replace standard ONNX ops with ORT's `com.microsoft` contrib
ops, which compute the same thing with faster CPU kernels. Rewrite 3 works
around an ORT CUDA bug with a standard op. All three edit the exported file in
under a minute, so no re-export is needed, and the weights are not touched.

**With rewrites 1 or 2, the result needs ONNX Runtime.** Contrib ops ship in
the standard `onnxruntime` / `onnxruntime-gpu` wheels, but other ONNX runtimes
will not load the file. With only rewrite 3 (`--no-layernorm
--no-fused-matmul`), the file stays standard ONNX.

1. LayerNormalization -> SkipLayerNormalization  (`--no-layernorm` to skip)

   ORT's CPU kernel for the standard op is slow on inputs with many rows --
   the pair (L*L rows) and edge (E rows) features this model normalises ~600
   times per forward. `SkipLayerNormalization` computes `LayerNorm(input +
   skip)` and is 2-5x faster on the same data. There is no skip connection to
   fuse, so each node becomes

       SkipLayerNormalization(x, x, scale, bias, epsilon=4*eps)

   which equals `LayerNormalization(x, scale, bias, epsilon=eps)`: LayerNorm
   is scale-invariant apart from eps, `x + x = 2x` is exact in floating point,
   and `LN(2x; 4*eps) = (2x - 2mu) / sqrt(4*var + 4*eps) = LN(x; eps)`. No
   zero tensor is allocated and no extra pass is made. The kernel only takes
   2-D or 3-D input, so 4-D inputs are reshaped to (1, rows, d) and back;
   Reshape does not copy data in ORT.

2. Transpose -> MatMul  becomes  cheaper Transpose -> FusedMatMul
   (`--no-fused-matmul` to skip)

   ORT's Transpose is single-threaded and copies. A permutation that moves the
   LAST axis into second-to-last place (e.g. (0,1,3,4,2)) copies element by
   element. When such a Transpose feeds only MatMul, its last two output axes
   are swapped back and the MatMul becomes `FusedMatMul(transA/transB=1)`,
   which lets BLAS do that swap without a copy. What remains is a
   permutation that keeps the last axis in place, and that copies whole
   blocks. If what remains is the identity, the Transpose is dropped.
   Bit-identical output.

3. ReduceL2 over length-1 axes -> Abs  (`--no-reducel2-fix` to skip)

   ORT's CUDA ReduceL2 (1.23.2) returns its input unchanged when every
   reduced axis has length 1, sign included, where the answer is `|x|`. The
   SE(3) layers take exactly that norm of their degree-0 features, and on the
   CUDA provider the model's coordinates came out as garbage (~1e5). Such
   nodes become `Abs`, the same value. Current model code no longer emits the
   pattern (`se3/layers.py:_feature_norm`); this fixes files exported before.

Usage:
    python tools/export/onnx_ort_rewrites.py \\
        models/rf_loop_dynamic.onnx models/rf_loop_dynamic_ort.onnx
    python tools/export/verify_onnx_model.py models/rf_loop_dynamic_ort.onnx
"""
import argparse
import collections
import os
import shutil
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

MS = "com.microsoft"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _is_std(node):
    return node.domain in ("", "ai.onnx")


def rewrite_layernorm(model):
    g = model.graph
    init = {i.name: i for i in g.initializer}
    ranks = {v.name: len(v.type.tensor_type.shape.dim)
             for v in list(g.value_info) + list(g.input)
             if v.type.tensor_type.HasField("shape")}

    to3_consts = {}  # d -> initializer name holding [1, -1, d]
    new_nodes = []
    counts = collections.Counter()
    for node in g.node:
        if node.op_type != "LayerNormalization" or not _is_std(node):
            new_nodes.append(node)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        x, scale = node.input[0], node.input[1]
        bias = node.input[2] if len(node.input) > 2 else ""
        extra_outputs = [o for o in node.output[1:] if o]
        # Only the plain case this model produces; anything else is left alone.
        if (attrs.get("axis", -1) != -1 or attrs.get("stash_type", 1) != 1
                or extra_outputs or scale not in init or not bias or bias not in init):
            new_nodes.append(node)
            counts["skipped"] += 1
            continue

        d = int(init[scale].dims[0])
        eps4 = float(np.float32(attrs.get("epsilon", 1e-5)) * np.float32(4))  # exact
        y = node.output[0]
        base = node.name or f"ln_{len(new_nodes)}"
        rank = ranks.get(x)

        skip = helper.make_node(
            "SkipLayerNormalization", [x, x, scale, bias], [y],
            name=f"{base}_skipln", domain=MS, epsilon=eps4)
        # keep the exporter's provenance (torch source location, module path)
        skip.metadata_props.extend(node.metadata_props)

        if rank in (2, 3):
            new_nodes.append(skip)
            counts[f"rank{rank}"] += 1
            continue

        if d not in to3_consts:
            name = f"skipln_to3_d{d}"
            g.initializer.append(numpy_helper.from_array(np.array([1, -1, d], np.int64), name))
            to3_consts[d] = name
        shp, x3, y3 = f"{base}_skipln_shape", f"{base}_skipln_x3", f"{base}_skipln_y3"
        skip.input[:2] = [x3, x3]
        skip.output[0] = y3
        new_nodes += [
            helper.make_node("Shape", [x], [shp], name=f"{base}_skipln_shapeof"),
            helper.make_node("Reshape", [x, to3_consts[d]], [x3], name=f"{base}_skipln_in"),
            skip,
            helper.make_node("Reshape", [y3, shp], [y], name=f"{base}_skipln_out"),
        ]
        counts[f"rank{rank}" if rank else "rank?"] += 1

    del g.node[:]
    g.node.extend(new_nodes)
    return counts


def rewrite_transpose_matmul(model):
    g = model.graph
    graph_outputs = {o.name for o in g.output}
    consumers = collections.defaultdict(list)
    for n in g.node:
        for i, x in enumerate(n.input):
            consumers[x].append((n, i))

    counts = collections.Counter()
    drop = set()
    rename = {}  # output of a dropped Transpose -> its input
    reshaped = set()  # tensors whose shape this rewrite changes
    for t in g.node:
        if t.op_type != "Transpose" or not _is_std(t):
            continue
        perm = [helper.get_attribute_value(a) for a in t.attribute if a.name == "perm"]
        if not perm:
            continue
        perm = list(perm[0])
        n = len(perm)
        users = consumers[t.output[0]]
        if (n < 2 or perm[-2] != n - 1 or t.output[0] in graph_outputs or not users
                or any(u.op_type != "MatMul" or not _is_std(u) for u, _ in users)
                or any(list(u.input).count(t.output[0]) > 1 for u, _ in users)):
            continue
        new_perm = perm[:-2] + [perm[-1], perm[-2]]
        for u, idx in users:
            u.op_type, u.domain = "FusedMatMul", MS
            u.attribute.append(helper.make_attribute("transA" if idx == 0 else "transB", 1))
        reshaped.add(t.output[0])
        if new_perm == list(range(n)):
            drop.add(id(t))
            rename[t.output[0]] = t.input[0]
            counts["transpose_dropped"] += 1
        else:
            del t.attribute[:]
            t.attribute.append(helper.make_attribute("perm", new_perm))
            counts["transpose_cheapened"] += 1
        counts["matmul_fused"] += len(users)

    # The exporter recorded each tensor's shape in value_info, and ORT plans
    # buffer reuse from it. A Transpose whose perm changed now produces a
    # different shape, so its stale entry must go (ORT re-infers it); leaving
    # it fails at run time with "Shape mismatch attempting to re-use buffer".
    stale = [v for v in g.value_info if v.name in reshaped]
    for v in stale:
        g.value_info.remove(v)
    counts["value_info_dropped"] = len(stale)

    if drop:
        kept = [n for n in g.node if id(n) not in drop]
        for n in kept:
            for i, x in enumerate(n.input):
                if x in rename:
                    n.input[i] = rename[x]
        del g.node[:]
        g.node.extend(kept)
    return counts


def rewrite_reducel2_unit_axes(model):
    g = model.graph
    shapes = {v.name: [d.dim_value if d.HasField("dim_value") else None
                       for d in v.type.tensor_type.shape.dim]
              for v in list(g.value_info) + list(g.input)
              if v.type.tensor_type.HasField("shape")}
    consts = {i.name: numpy_helper.to_array(i) for i in g.initializer
              if i.data_location != onnx.TensorProto.EXTERNAL}
    for n in g.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    consts[n.output[0]] = numpy_helper.to_array(a.t)

    counts = collections.Counter()
    for n in g.node:
        if n.op_type != "ReduceL2" or not _is_std(n):
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        shape = shapes.get(n.input[0])
        axes = consts.get(n.input[1]) if len(n.input) > 1 and n.input[1] else None
        if shape is None or axes is None or attrs.get("keepdims", 1) != 1:
            counts["left_alone"] += 1
            continue
        axes = [int(a) % len(shape) for a in np.atleast_1d(axes)]
        if not axes or not all(shape[a] == 1 for a in axes):
            counts["left_alone"] += 1
            continue
        del n.input[1:]
        del n.attribute[:]
        n.op_type = "Abs"
        counts["to_abs"] += 1
    return counts


def _replace_file(tmp, final):
    # os.replace swaps the directory entry atomically. A process that already
    # has the old file open or memory-mapped (ORT maps the weights) keeps
    # reading the old bytes instead of seeing a half-written file.
    os.replace(tmp, final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--no-layernorm", action="store_true")
    ap.add_argument("--no-fused-matmul", action="store_true")
    ap.add_argument("--no-reducel2-fix", action="store_true")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)

    # Weights stay where they are: the graph is loaded WITHOUT its external
    # data, the `.data` file is copied as a whole, and the initializers are
    # re-pointed at the copy. Round-tripping ~1000 tensors through onnx's
    # external-data writer opens the file once per tensor, which took minutes
    # on NFS for a byte-identical result.
    log(f"loading {src} (graph only) ...")
    model = onnx.load(str(src), load_external_data=False)
    before = collections.Counter(n.op_type for n in model.graph.node)
    if not args.no_layernorm:
        log(f"LayerNorm rewrite: {dict(rewrite_layernorm(model))}")
    if not args.no_fused_matmul:
        log(f"Transpose->MatMul rewrite: {dict(rewrite_transpose_matmul(model))}")
    if not args.no_reducel2_fix:
        log(f"ReduceL2 (length-1 axes) -> Abs: {dict(rewrite_reducel2_unit_axes(model))}")
    after = collections.Counter(n.op_type for n in model.graph.node)
    for op in ("LayerNormalization", "SkipLayerNormalization", "Transpose", "MatMul", "FusedMatMul",
               "ReduceL2", "Abs"):
        log(f"  {op:24s} {before[op]:6d} -> {after[op]}")
    if any(not _is_std(n) for n in model.graph.node) and not any(o.domain == MS for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid(MS, 1))

    dst.parent.mkdir(parents=True, exist_ok=True)
    data_name = dst.name + ".data"
    src_locations = set()
    for t in model.graph.initializer:
        if t.data_location == onnx.TensorProto.EXTERNAL:
            for e in t.external_data:
                if e.key == "location":
                    src_locations.add(e.value)
                    e.value = data_name
    if len(src_locations) > 1:
        raise SystemExit(f"expected at most one external data file, found {src_locations}")
    if src_locations:
        src_data = (src.parent / src_locations.pop()).resolve()
        dst_data = (dst.parent / data_name).resolve()
        if src_data != dst_data:  # rewriting in place: the weights are already there
            shutil.copyfile(src_data, str(dst_data) + ".tmp")
            _replace_file(str(dst_data) + ".tmp", dst_data)
    onnx.save_model(model, str(dst) + ".tmp")
    _replace_file(str(dst) + ".tmp", dst)
    for f in sorted(dst.parent.glob(dst.name + "*")):
        log(f"  {f.name}  {f.stat().st_size / 1e6:.0f} MB")
    log("NOW VERIFY IT:")
    log(f"  python tools/export/verify_onnx_model.py {dst}")


if __name__ == "__main__":
    main()
