"""
scripts/eval_with_grammar_fix.py

Re-runs the comparison with the grammar crash fixed.
Also adds one-char error analysis to understand the remaining 9.9% gap.
"""
import sys, os, json, re, torch, cv2
import numpy as np
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from backend.scripts.train_plate_recognizer import CRNN, ctc_decode, IMG_H, IMG_W
from backend.scripts.indian_plate_grammar import decode_plate

gt_path = ROOT / "data/plate_real/verified_all.jsonl"
records = [json.loads(l) for l in open(gt_path, encoding="utf-8") if l.strip()]

def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m  = CRNN(n_classes=len(ck["chars"]) + 1)
    m.load_state_dict(ck["model"])
    m.eval()
    return m

def safe_grammar(raw_pred: str) -> str:
    """
    Wrapper around decode_plate that never crashes and never returns None.
    Returns the grammar-corrected plate, or the raw pred if correction fails.
    """
    if not raw_pred:
        return raw_pred
    try:
        result = decode_plate(raw_pred)
        # decode_plate can return None, or a dict with plate=None
        if result is None:
            return raw_pred
        corrected = result.get("plate")
        if corrected is None or len(corrected) < 4:
            return raw_pred
        return corrected
    except Exception:
        return raw_pred

def char_error_rate(pred: str, gt: str) -> float:
    if not gt:
        return 0.0 if not pred else 1.0
    dp = list(range(len(gt) + 1))
    for pc in pred:
        prev, dp[0] = dp[0], dp[0] + 1
        for j, gc in enumerate(gt, 1):
            cur  = dp[j]
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + (pc != gc))
            prev = cur
    return dp[len(gt)] / len(gt)

def classify_error(pred: str, gt: str) -> str:
    if pred == gt:
        return "EXACT"
    if not pred:
        return "NO_READ"
    cer = char_error_rate(pred, gt)
    if len(pred) == len(gt):
        n_diff = sum(a != b for a, b in zip(pred, gt))
        if n_diff == 1:
            return "ONE_CHAR"
        if n_diff == 2:
            return "TWO_CHAR"
    return "WRONG"

def eval_detailed(model, label: str, use_grammar: bool = False):
    outcomes     = Counter()
    confusables  = Counter()
    position_err = Counter()
    cer_total    = 0.0
    total        = 0

    for rec in records:
        img_path = ROOT / "data/plate_real" / rec.get("file", "")
        if not img_path.exists():
            img_path = ROOT / "data/plate_real/images" / rec.get("file", "")
        gt = re.sub(r"[^A-Z0-9]", "", rec.get("text", "").upper())
        if not img_path.exists() or not gt:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img_r = cv2.resize(img, (IMG_W, IMG_H))
        t     = torch.from_numpy(img_r).float().div(127.5).sub(1.0)[None, None]

        with torch.no_grad():
            logits = model(t)

        raw   = ctc_decode(logits)
        pred  = re.sub(r"[^A-Z0-9]", "", raw[0].upper()) if raw else ""
        final = safe_grammar(pred) if use_grammar else pred

        outcome = classify_error(final, gt)
        outcomes[outcome] += 1
        cer_total += char_error_rate(final, gt)
        total     += 1

        # Track which characters are being confused
        if outcome == "ONE_CHAR" and len(final) == len(gt):
            for i, (p, g) in enumerate(zip(final, gt)):
                if p != g:
                    confusables[(g, p)] += 1   # (correct, predicted_wrong)
                    position_err[i]     += 1

    exact   = outcomes["EXACT"]
    one_ch  = outcomes["ONE_CHAR"]
    two_ch  = outcomes["TWO_CHAR"]
    wrong   = outcomes["WRONG"]
    no_read = outcomes["NO_READ"]

    gram_tag = "+Grammar" if use_grammar else "        "
    print(f"\n{label} {gram_tag}")
    print(f"  Exact:          {exact:3d}/{total} = {exact/max(1,total):6.1%}")
    print(f"  One-char error: {one_ch:3d}/{total} = {one_ch/max(1,total):6.1%}  ← recoverable")
    print(f"  Two-char error: {two_ch:3d}/{total} = {two_ch/max(1,total):6.1%}")
    print(f"  Wrong:          {wrong:3d}/{total} = {wrong/max(1,total):6.1%}")
    print(f"  No read:        {no_read:3d}/{total} = {no_read/max(1,total):6.1%}")
    print(f"  CER:            {cer_total/max(1,total):6.2%}")

    if confusables:
        print(f"\n  Top confusable pairs (gt_char → predicted_as):")
        for (correct, wrong_char), cnt in confusables.most_common(10):
            known = {frozenset(p) for p in [('0','O'),('1','I'),('8','B'),
                                             ('5','S'),('6','G'),('2','Z')]}
            flag  = " ← known confusable" \
                    if frozenset([correct, wrong_char]) in known else ""
            print(f"    {correct} → {wrong_char} : {cnt:3d}{flag}")

    if position_err:
        print(f"\n  One-char errors by position:")
        pos_labels = {0:"state1", 1:"state2", 2:"dist1", 3:"dist2",
                      4:"ser1",   5:"ser2",   6:"ser3",  7:"num1",
                      8:"num2",   9:"num3",  10:"num4"}
        for pos, cnt in sorted(position_err.items()):
            bar   = "█" * cnt
            label_pos = pos_labels.get(pos, f"pos{pos}")
            print(f"    [{pos}] {label_pos:<8}: {cnt:2d} {bar}")

    return exact / max(1, total)


print("Loading models...")
m_fine = load_model(ROOT / "models/plate_recognizer/finetuned.pt")

print("\n" + "=" * 60)
print("finetuned.pt — FULL ERROR BREAKDOWN")
print("=" * 60)

acc_raw  = eval_detailed(m_fine, "finetuned.pt", use_grammar=False)
acc_gram = eval_detailed(m_fine, "finetuned.pt", use_grammar=True)

print("\n" + "=" * 60)
print("GRAMMAR CORRECTION IMPACT")
print("=" * 60)
gain = (acc_gram - acc_raw) * 100
print(f"  Raw:          {acc_raw:.1%}")
print(f"  With grammar: {acc_gram:.1%}")
print(f"  Delta:        {gain:+.1f} points")

print("\n" + "=" * 60)
print("END-TO-END PROJECTION  (assuming 91% detection)")
print("=" * 60)
det = 0.91
print(f"  Raw recognition:          {acc_raw:.1%} × {det:.0%} = {acc_raw*det:.1%}")
print(f"  With grammar correction:  {acc_gram:.1%} × {det:.0%} = {acc_gram*det:.1%}")
print(f"  Target:                   70.0%")
print(f"  Already met?              {'YES' if acc_gram*det >= 0.70 else 'NOT YET — gap: ' + f'{(0.70 - acc_gram*det)*100:.1f} points'}")