"""One-off diagnosis: same itq weights, three forward paths, one metric.

  dense-fp16  exactly Step 1's object (weights written in place, model.half())
  path-fp32   the factored BitPath + SDPA student in float32
  path-fp16   the same factored student in fp16

If dense-fp16 reproduces Step 1's 0.1375 and the path variants do not, the
factored wiring is wrong. If all three agree, the difference was numerical.
"""
import importlib.util
import json
import sys

import chess
import torch

sys.path.insert(0, "vendor/little-bit")

from chessformer_lens.adapters.lc0 import _leela_planes, _load_bt4   # noqa: E402

import pathlib

CKPT = "data/bt4/BT4-1024x15x32h-swa-6147500.pb.gz"


def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


q1 = _load("q1bit", "harness/bt4/q1bit.py")
bp = _load("bitpath", "harness/bt4/bitpath.py")
qat = _load("qat", "harness/bt4/qat.py")

rows = [json.loads(l) for l in pathlib.Path("data/bt4/puzzles.jsonl").read_text().splitlines() if l.strip()]
cal = [r for r in rows if r["split"] == "calib"]
cal_planes = torch.stack([_leela_planes(qat.live_board(r)) for r in cal]).cuda().half()

model = _load_bt4(CKPT, "cpu").half().cuda()
_, _, paths = q1.quantise(model, cal_planes, "itq", 0)
print("quantised (dense weights written in place, fp16)")

ev = qat.Eval("test")
print("dense-fp16 ", json.dumps(ev(model)))
with torch.no_grad():
    ref_logits = torch.empty((ev.planes.size(0), 4288), device="cuda")
    for s in range(0, ev.planes.size(0), 64):
        pol, _, _ = model(ev.planes[s:s + 64])
        ref_logits[s:s + 64] = pol.float()

bp.install(model, paths)
model = model.float()
print("path-fp32  ", json.dumps(ev(model)))
with torch.no_grad():
    p32 = torch.empty_like(ref_logits)
    for s in range(0, ev.planes.size(0), 64):
        pol, _, _ = model(ev.planes[s:s + 64].float())
        p32[s:s + 64] = pol.float()

model = model.half()
print("path-fp16  ", json.dumps(ev(model)))
d = (p32 - ref_logits)
print("max|dense-path| fp32", float(d.abs().max()),
      "rel", float(d.abs().max()) / float(ref_logits.abs().max()))
print("corr", float(torch.corrcoef(torch.stack([p32.flatten(), ref_logits.flatten()]))[0, 1]))
