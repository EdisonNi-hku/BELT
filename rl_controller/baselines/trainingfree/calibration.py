from __future__ import annotations

import bisect

from rl_controller.iclr.chains import is_correct

CALIB_SALT = "iclr-ladder-cv-2026-08-27"
N_FOLDS = 4


class Isotonic:

    def __init__(self, xs, ys):
        if len(xs) != len(ys):
            raise ValueError("xs and ys must align")
        if not xs:
            raise ValueError("an isotonic fit needs at least one block")
        self.xs = list(xs)
        self.ys = list(ys)

    def predict(self, x: float) -> float:
        i = bisect.bisect_left(self.xs, float(x))
        if i >= len(self.ys):
            return self.ys[-1]
        return self.ys[i]

    def to_dict(self) -> dict:
        return {"xs": self.xs, "ys": self.ys, "n_blocks": len(self.xs)}


def fit_isotonic(pairs) -> Isotonic:
    rows = sorted((float(x), float(y)) for x, y in pairs)
    if not rows:
        raise ValueError("no calibration examples")
    merged: list[list[float]] = []
    for x, y in rows:
        if merged and merged[-1][0] == x:
            merged[-1][1] += y
            merged[-1][2] += 1.0
        else:
            merged.append([x, y, 1.0])
    blocks: list[list[float]] = []
    for x, total, weight in merged:
        blocks.append([x, total, weight])
        while len(blocks) > 1 and (blocks[-2][1] / blocks[-2][2]) > (blocks[-1][1] / blocks[-1][2]):
            x_hi, t2, w2 = blocks.pop()
            _, t1, w1 = blocks.pop()
            blocks.append([x_hi, t1 + t2, w1 + w2])
    return Isotonic([b[0] for b in blocks], [b[1] / b[2] for b in blocks])


def calibration_examples(chains, belief: str = "q"):
    out = []
    for chain in chains:
        for probe, score in zip(chain["probes"], chain["scores"]):
            out.append((float(score[belief]), float(is_correct(probe["forced"], chain["gold"]))))
    return out


def attach_isotonic(val_chains, read_chains, *, belief: str = "q", key: str = "q_iso",
                    salt: str = CALIB_SALT, n_folds: int = N_FOLDS) -> dict:
    qkeys = sorted({c["qkey"] for c in val_chains})
    if not qkeys:
        raise ValueError("no val chains to calibrate on")
    if len(qkeys) < n_folds:
        raise ValueError(
            f"val has {len(qkeys)} question(s) but the cross-fit needs at least {n_folds}. "
            "The fold split is by QUESTION, so a smaller val leaves folds with no training "
            "data. Either widen val or pass --no-isotonic; do NOT drop the cross-fit, "
            "which would make the selection split in-sample.")
    from rl_controller.iclr import learning_curve as lc

    folds = lc.question_folds(qkeys, n_folds=n_folds, salt=f"{salt}:iso")
    per_fold = {}
    for fold in range(n_folds):
        train = [c for c in val_chains if folds[c["qkey"]] != fold]
        examples = calibration_examples(train, belief)
        if not examples:
            raise ValueError(f"fold {fold} has no training examples to calibrate on")
        per_fold[fold] = fit_isotonic(examples)
    for chain in val_chains:
        model = per_fold[folds[chain["qkey"]]]
        for score in chain["scores"]:
            score[key] = model.predict(score[belief])
    full = fit_isotonic(calibration_examples(val_chains, belief))
    for chain in read_chains:
        for score in chain["scores"]:
            score[key] = full.predict(score[belief])
    return {"key": key, "belief": belief, "n_val_questions": len(qkeys),
            "n_folds": n_folds, "salt": f"{salt}:iso",
            "full_fit_blocks": len(full.xs),
            "crossfit": "val rows are out-of-fold; the read split uses the full-val fit"}
