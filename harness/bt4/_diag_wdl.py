"""Diagnostic: how good is the 1.04 bpw student's WDL head, and is it why PUCT stalls?

PUCT strength leans on the value head for every leaf and every backup. If WDL
is noisy, 64 nodes of search mostly propagate noise -- which would explain why
1 -> 64 nodes bought only ~265 Elo while the teacher sits ~1140 above.

Reports, on the 2,437 held-out game positions:
  wdl_argmax_agree   fraction where student argmax(W/D/L) == teacher's
  win_prob_mae       |P(win)_student - P(win)_teacher|
  score_corr         Pearson r of (P(win)-P(loss)) between student and teacher
  value_mse          MSE of the three-way distribution (the trained term)

usage:
  python harness/bt4/_diag_wdl.py --student results/bt4/step2_weights_mix4000.pt
"""
import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "vendor/little-bit")

from chessformer_lens.adapters.lc0 import _load_bt4   # noqa: E402

CKPT = "data/bt4/BT4-1024x15x32h-swa-6147500.pb.gz"
EVAL_PLANES = pathlib.Path("data/bt4/_planes_games_eval.npy")


def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    qat = _load("qat", "harness/bt4/qat.py")
    planes = torch.tensor(np.load(EVAL_PLANES)).cuda().half()
    n = planes.size(0)

    teacher = _load_bt4(CKPT, "cpu").half().cuda().eval()
    student = _load_bt4(CKPT, "cpu").half().cuda()
    student, head = qat.load_factors(student, args.student)
    student = student.half().cuda().eval()

    tv, sv = [], []
    for s in range(0, n, args.batch):
        x = planes[s:s + args.batch]
        with torch.no_grad():
            _, tval, _ = teacher(x)
            _, sval, _ = student(x)
        tv.append(tval.float())
        sv.append(sval.float())
    tv, sv = torch.cat(tv), torch.cat(sv)

    tp, sp = F.softmax(tv, -1), F.softmax(sv, -1)
    t_score = tp[:, 0] - tp[:, 2]
    s_score = sp[:, 0] - sp[:, 2]

    def corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))

    rec = dict(label=pathlib.Path(args.student).stem, n=n,
               wdl_argmax_agree=round(float((tv.argmax(1) == sv.argmax(1)).float().mean()), 4),
               win_prob_mae=round(float((tp[:, 0] - sp[:, 0]).abs().mean()), 4),
               score_corr=round(corr(t_score, s_score), 4),
               value_mse=round(float(F.mse_loss(sp, tp)), 5),
               teacher_score_mean=round(float(t_score.mean()), 4),
               student_score_mean=round(float(s_score.mean()), 4),
               teacher_win_mean=round(float(tp[:, 0].mean()), 4),
               student_win_mean=round(float(sp[:, 0].mean()), 4),
               student_draw_rate=round(float(sp[:, 1].mean()), 4),
               teacher_draw_rate=round(float(tp[:, 1].mean()), 4))
    out = pathlib.Path("results/bt4") / f"step4_wdl_{rec['label']}.json"
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
