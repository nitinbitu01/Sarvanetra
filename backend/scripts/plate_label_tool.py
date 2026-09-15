"""backend/scripts/plate_label_tool.py — build a browser labelling tool for real plates.

WHY HAND LABELS ARE UNAVOIDABLE
  Four automated routes to labels were tried and measured, and all four failed:
    synthetic-only recogniser  97.2% synthetic -> read GJ32K4588 as 'G33ZK458'
    synthetic composites       taught the detector to find paste artifacts
    OCR bootstrap              trained a detector on the burned-in caption
    pseudo-labels + voting     ~15% correct when checked against the images

  The last one failed for a reason worth keeping: voting only corrects RANDOM
  errors. EasyOCR's errors are SYSTEMATIC - it misreads the same glyph the same
  way on every frame, so twenty-five frames agree on the same wrong answer.
  'GJ1CO9007' carried agreement 1.00 and was wrong.

  Meanwhile the plates are plainly legible to a human eye in these very crops.
  The information is present; only correct labels are missing.

WHAT MAKES THIS FAST
  Each crop arrives with the OCR guess pre-filled, so the work is CORRECTION,
  not typing - most plates need one or two characters changed. Progress is
  saved to localStorage on every keystroke, so the browser can be closed and
  reopened without losing anything.

  OCR is used here purely for LOCALISATION - finding where the plate sits,
  which it does well - and its reading is treated as a first draft only.

USAGE
  python -m backend.scripts.plate_label_tool --n 100
  then open output/plate_labeller.html in a browser
"""
from __future__ import annotations

import argparse
import base64
import json
from collections import defaultdict
from pathlib import Path

import cv2

REAL = Path("data/plate_real")
OUT = Path("output/plate_labeller.html")

PAGE_HEAD = """<title>Plate Labeller</title>
<style>
 :root{--bg:#12141a;--fg:#e8eaf0;--mut:#8b93a7;--acc:#5eead4;--card:#1b1f28;
       --edge:#2a3040;--warn:#fbbf24}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:15px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}
 header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--edge);
        padding:14px 20px;z-index:9}
 h1{margin:0 0 4px;font-size:17px;letter-spacing:.01em}
 .sub{color:var(--mut);font-size:13px}
 .bar{height:6px;background:var(--edge);border-radius:3px;margin-top:10px;overflow:hidden}
 .fill{height:100%;background:var(--acc);width:0;transition:width .2s}
 main{padding:18px 20px 120px;max-width:1180px;margin:0 auto}
 .card{background:var(--card);border:1px solid var(--edge);border-radius:10px;
       padding:12px;margin-bottom:14px;display:flex;gap:16px;align-items:center;
       flex-wrap:wrap}
 .card img{image-rendering:pixelated;height:104px;border-radius:6px;background:#000;
           flex:0 0 auto}
 .meta{color:var(--mut);font-size:12px;min-width:150px}
 input{background:#0d1016;border:1px solid var(--edge);color:var(--fg);
       border-radius:7px;padding:11px 13px;font:600 19px ui-monospace,Menlo,monospace;
       letter-spacing:.12em;width:250px;text-transform:uppercase}
 input:focus{outline:2px solid var(--acc);border-color:transparent}
 input.done{border-color:var(--acc)}
 input.skip{border-color:var(--warn);opacity:.55}
 .hint{color:var(--mut);font-size:12px;max-width:230px}
 footer{position:fixed;bottom:0;left:0;right:0;background:var(--card);
        border-top:1px solid var(--edge);padding:12px 20px;display:flex;
        gap:12px;align-items:center}
 button{background:var(--acc);color:#06231f;border:0;border-radius:7px;
        padding:10px 18px;font-weight:700;cursor:pointer;font-size:14px}
 button.ghost{background:transparent;color:var(--mut);border:1px solid var(--edge)}
 code{background:#0d1016;padding:1px 6px;border-radius:4px;color:var(--acc)}
</style>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--height", type=int, default=104)
    ap.add_argument("--src", default=str(REAL),
                    help="Directory holding images/ and labels.jsonl.")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    src = Path(args.src)
    out_path = Path(args.out)
    rows = [json.loads(l) for l in (src / "labels.jsonl").open(encoding="utf-8")]
    # One crop per vehicle - labelling ten frames of the same plate adds work
    # without adding information. Sharpest-looking first (highest agreement is
    # a decent proxy) so early cards are the quickest to judge.
    per_track: dict[tuple, dict] = {}
    for r in rows:
        k = (r["camera"], r["track"])
        if k not in per_track or r["agreement"] > per_track[k]["agreement"]:
            per_track[k] = r
    items = sorted(per_track.values(), key=lambda r: -r["agreement"])[:args.n]
    print(f"unique vehicles available : {len(per_track)}")
    print(f"preparing                 : {len(items)}")

    cards = []
    for i, r in enumerate(items):
        p = src / "images" / r["file"]
        img = cv2.imread(str(p))
        if img is None:
            continue
        # Upscale small crops so glyphs are judgeable by eye without zooming.
        if img.shape[0] < args.height:
            f = args.height / img.shape[0]
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            continue
        b64 = base64.b64encode(buf).decode()
        cards.append(
            f'<div class="card">'
            f'<img src="data:image/png;base64,{b64}" alt="plate {i}">'
            f'<div class="meta">#{i+1} &middot; {r["camera"]}<br>'
            f'agree {r["agreement"]:.2f} &middot; {r["votes"]} frames</div>'
            f'<input id="p{i}" data-i="{i}" value="{r["text"]}" '
            f'autocomplete="off" spellcheck="false">'
            f'<div class="hint">Correct it, then press <code>Enter</code>. '
            f'Unreadable? Clear the box and press Enter.</div>'
            f'</div>')

    meta = [{"i": i, "file": r["file"], "camera": r["camera"],
             "ocr": r["text"], "track": r["track"]}
            for i, r in enumerate(items)]

    script = """
<script>
const META = __META__;
// Key is per-batch. With a shared key, opening batch 2 would load batch 1's
// saved answers BY INDEX and attach them to different images - silently
// producing wrong labels, which is the one failure mode this whole exercise
// exists to avoid.
const KEY = '__STOREKEY__';
let store = {};
try { store = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch(e) { store = {}; }

function paint(){
  let n = 0;
  document.querySelectorAll('input[data-i]').forEach(el=>{
    const i = el.dataset.i;
    if (i in store){
      n++;
      el.value = store[i];
      el.classList.toggle('done', store[i] !== '');
      el.classList.toggle('skip', store[i] === '');
    }
  });
  const total = META.length;
  document.getElementById('cnt').textContent = n + ' / ' + total;
  document.getElementById('fill').style.width = (100*n/total) + '%';
}

document.addEventListener('keydown', e=>{
  if (e.key !== 'Enter') return;
  const el = e.target;
  if (!el.matches('input[data-i]')) return;
  const i = el.dataset.i;
  store[i] = el.value.trim().toUpperCase().replace(/[^A-Z0-9]/g,'');
  localStorage.setItem(KEY, JSON.stringify(store));
  paint();
  // Jump to the next unlabelled field so the flow never stalls.
  const all = [...document.querySelectorAll('input[data-i]')];
  const next = all.slice(all.indexOf(el)+1).find(x=>!(x.dataset.i in store));
  (next || all[all.indexOf(el)+1])?.focus();
  e.preventDefault();
});

function download(){
  const out = META.filter(m => String(m.i) in store)
                  .map(m => ({file:m.file, camera:m.camera, track:m.track,
                              ocr:m.ocr, text:store[String(m.i)]}))
                  .filter(m => m.text !== '');
  const blob = new Blob([out.map(o=>JSON.stringify(o)).join('\\n')],
                        {type:'application/x-ndjson'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'plate_labels_verified.jsonl';
  a.click();
  alert(out.length + ' labels exported.\\nSave the file into data/plate_real/');
}

function reset(){
  if (!confirm('Clear all labels entered so far?')) return;
  store = {}; localStorage.removeItem(KEY);
  document.querySelectorAll('input[data-i]').forEach((el,i)=>{
    el.value = META[i].ocr; el.classList.remove('done','skip');
  });
  paint();
}
paint();
document.querySelector('input[data-i]')?.focus();
</script>
"""
    script = script.replace("__META__", json.dumps(meta))
    script = script.replace("__STOREKEY__", f"plate_labels_{out_path.stem}")

    html = (PAGE_HEAD +
            '<header><h1>Plate Labeller</h1>'
            '<div class="sub">The box is pre-filled with the OCR guess &mdash; '
            'correct it and press Enter. Progress saves automatically.</div>'
            '<div class="bar"><div class="fill" id="fill"></div></div></header>'
            '<main>' + "\n".join(cards) + '</main>'
            '<footer><button onclick="download()">Export labels</button>'
            '<button class="ghost" onclick="reset()">Reset</button>'
            '<div class="sub">labelled: <b id="cnt">0 / 0</b></div></footer>'
            + script)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    mb = out_path.stat().st_size / 1_048_576
    print(f"\nlabelling tool -> {out_path.resolve()}  ({mb:.1f} MB)")
    print("\nOpen it in a browser. Correct each plate, press Enter, then click")
    print("'Export labels' and save the file into data/plate_real/.")
    print("Images are embedded, so the file works offline and can be moved.")


if __name__ == "__main__":
    main()
