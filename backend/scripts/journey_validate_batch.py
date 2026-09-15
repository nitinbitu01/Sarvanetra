"""backend/scripts/journey_validate_batch.py — a short batch that can tell an
attentive judgement from a click-through.

WHAT WENT WRONG WITH THE FIRST BATCH
  All 120 candidate links came back marked "different" and 0 marked "unsure".
  That may be correct - cross-camera linking may simply not work on this
  footage. It may equally be what a batch looks like when someone presses D a
  hundred and twenty times.

  Nothing in the data distinguishes those, because the controls were built
  wrong. They were pairs the system scores as unrelated, expected to be marked
  "different" - so a run of D passes every control. A control that only
  detects one failure mode is not a control.

WHAT THIS BATCH ADDS
  POSITIVE controls: two frames of the SAME track, far apart in time, so the
  vehicle is at a different distance and angle. There is no doubt about the
  answer. If these come back "different", the batch was not being judged and
  the earlier 120 verdicts have to be discarded.

  Captions are removed. The first tool showed each side's camera and time, and
  on a positive control both sides would read the same camera - which answers
  the question for the labeller instead of asking it. With no captions the
  only available evidence is the vehicles.

  UNSURE is pushed harder. Many real pairs show one vehicle from the front and
  the other from behind, and no one can rule on those with confidence. A batch
  with zero unsure answers on that material is itself a warning sign.

USAGE
  python -m backend.scripts.journey_validate_batch
"""
from __future__ import annotations

import argparse
import base64
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2

CORPUS = Path("output/plate_corpus")
INDEX = Path("output/journey_index")
VERIFY = Path("data/journey_verify")

PAGE = """<!doctype html><meta charset="utf-8">
<title>Same vehicle? - validation</title>
<style>
:root{--bg:#0f1216;--card:#181d24;--fg:#e6eaf0;--dim:#8b96a4;--rule:#2a323c;
 --ok:#4ade80;--no:#f87171;--maybe:#fbbf24;--accent:#60a5fa}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:15px/1.5 ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;z-index:9;background:#0b0e12;
 border-bottom:1px solid var(--rule);padding:12px 18px;
 display:flex;gap:18px;align-items:center;flex-wrap:wrap}
header b{font-size:17px}
.keys{color:var(--dim);font-size:13px}
kbd{background:#222a34;border:1px solid #39424e;border-radius:4px;
 padding:1px 7px;font-family:ui-monospace,monospace;font-size:12px}
.bar{flex:1;height:7px;background:#222a34;border-radius:4px;overflow:hidden;
 min-width:110px}
.bar i{display:block;height:100%;background:var(--accent);width:0}
.tip{background:#1a2430;border-bottom:1px solid var(--rule);
 padding:10px 18px;color:var(--dim);font-size:13.5px}
.tip b{color:var(--maybe)}
main{padding:16px 18px 120px;max-width:1180px;margin:0 auto}
.pair{background:var(--card);border:1px solid var(--rule);border-radius:12px;
 padding:14px;margin-bottom:16px}
.pair.done{opacity:.45}
.pair.cur{border-color:var(--accent);box-shadow:0 0 0 2px rgba(96,165,250,.2)}
.imgs{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:700px){.imgs{grid-template-columns:1fr}}
.side{background:#000;border-radius:9px;overflow:hidden;border:1px solid #2a323c}
.side img{display:block;width:100%;height:auto}
.row{display:flex;gap:10px;align-items:center;margin-top:12px;flex-wrap:wrap}
button{border:0;border-radius:8px;padding:9px 18px;font-weight:700;
 cursor:pointer;font-size:14px}
.b-same{background:var(--ok);color:#06210f}
.b-diff{background:var(--no);color:#2a0a0a}
.b-uns{background:var(--maybe);color:#2a1e00}
.verdict{margin-left:auto;font-weight:700}
.v-same{color:var(--ok)} .v-diff{color:var(--no)} .v-uns{color:var(--maybe)}
footer{position:fixed;bottom:0;left:0;right:0;background:#0b0e12;
 border-top:1px solid var(--rule);padding:11px 18px;display:flex;gap:12px;
 align-items:center;flex-wrap:wrap}
footer button{background:var(--accent);color:#06172b}
.hint{color:var(--dim);font-size:12.5px}
</style>
<header>
  <b>Same vehicle?</b>
  <span id="count" class="hint"></span>
  <span class="bar"><i id="prog"></i></span>
  <span class="keys"><kbd>S</kbd> same &nbsp; <kbd>D</kbd> different
   &nbsp; <kbd>U</kbd> unsure &nbsp; <kbd>&larr;</kbd> back</span>
</header>
<div class="tip">Only 30 pairs. Some are easy on purpose &mdash; they are
 checking the batch, not you. <b>Where one photo shows the front and the other
 the rear, UNSURE is usually the right answer.</b> A guess is worse than an
 unsure.</div>
<main id="app"></main>
<footer>
  <button onclick="dl()">Export judgements</button>
  <span class="hint">Judge the VEHICLES. No camera or time is shown, on
   purpose.</span>
</footer>
<script>
const DATA = __DATA__;
const KEY = "journey_validate_v1";
let store = {};
try { store = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e){}
let cur = 0;
const app = document.getElementById("app");
DATA.forEach((d, i) => {
  const el = document.createElement("div");
  el.className = "pair"; el.id = "p" + i;
  el.innerHTML = '<div class="imgs">'
    + '<div class="side"><img src="data:image/jpeg;base64,'+d.a+'"></div>'
    + '<div class="side"><img src="data:image/jpeg;base64,'+d.b+'"></div>'
    + '</div><div class="row">'
    + '<button class="b-same" onclick="mark('+i+',\\'same\\')">Same vehicle</button>'
    + '<button class="b-diff" onclick="mark('+i+',\\'diff\\')">Different</button>'
    + '<button class="b-uns" onclick="mark('+i+',\\'unsure\\')">Unsure</button>'
    + '<span class="verdict" id="v'+i+'"></span></div>';
  app.appendChild(el);
});
function mark(i,v){ store[DATA[i].id]=v;
  localStorage.setItem(KEY,JSON.stringify(store)); paint(i); count();
  if(i===cur) focus(Math.min(cur+1,DATA.length-1)); }
function paint(i){ const v=store[DATA[i].id];
  const el=document.getElementById("p"+i), s=document.getElementById("v"+i);
  el.classList.toggle("done",!!v);
  s.className="verdict "+(v==="same"?"v-same":v==="diff"?"v-diff":v?"v-uns":"");
  s.textContent=v?v.toUpperCase():""; }
function focus(i){ document.getElementById("p"+cur)?.classList.remove("cur");
  cur=i; const el=document.getElementById("p"+cur); el.classList.add("cur");
  el.scrollIntoView({block:"center",behavior:"smooth"}); }
function count(){ const n=Object.keys(store).length;
  document.getElementById("count").textContent=n+" of "+DATA.length;
  document.getElementById("prog").style.width=(n/DATA.length*100)+"%"; }
document.addEventListener("keydown",e=>{ const k=e.key.toLowerCase();
  if(k==="s"){mark(cur,"same");e.preventDefault();}
  else if(k==="d"){mark(cur,"diff");e.preventDefault();}
  else if(k==="u"){mark(cur,"unsure");e.preventDefault();}
  else if(e.key==="ArrowLeft"){focus(Math.max(0,cur-1));e.preventDefault();}
  else if(e.key==="ArrowRight"){focus(Math.min(DATA.length-1,cur+1));e.preventDefault();} });
function dl(){ const out=DATA.filter(d=>store[d.id])
    .map(d=>({id:d.id,kind:d.kind,verdict:store[d.id],score:d.score}));
  if(!out.length){alert("Nothing judged yet.");return;}
  const b=new Blob([out.map(o=>JSON.stringify(o)).join("\\n")+"\\n"],
    {type:"application/json"});
  const a=document.createElement("a"); a.href=URL.createObjectURL(b);
  a.download="journey_validation.jsonl"; a.click();
  alert(out.length+" judgements exported.\\nSave into data/journey_verify/"); }
DATA.forEach((_,i)=>paint(i)); count(); focus(0);
</script>
"""


def thumb(path, h=200):
    img = cv2.imread(str(path))
    if img is None:
        return ""
    s = h / img.shape[0]
    img = cv2.resize(img, (max(1, int(img.shape[1] * s)), h),
                     interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode() if ok else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pos", type=int, default=10)
    ap.add_argument("--neg", type=int, default=10)
    ap.add_argument("--real", type=int, default=10)
    ap.add_argument("--out", default="journey_validate.html")
    args = ap.parse_args()

    recs = [json.loads(l) for l in
            (INDEX / "records.jsonl").open(encoding="utf-8")]
    by_key = {f'{r["camera"]}|{r["track"]}': r for r in recs}

    frames = defaultdict(list)
    for line in (CORPUS / "manifest.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        frames[(r["camera"], r["track"])].append(r)
    long_tracks = {k: sorted(v, key=lambda r: r["frame"])
                   for k, v in frames.items() if len(v) >= 8}

    rng = random.Random(21)
    items = []

    # POSITIVE controls: first and last frame of one track. Same vehicle,
    # different distance and angle - and no caption to give it away.
    keys = [k for k in long_tracks if k[0] in
            {"CAM_08", "CAM_09", "CAM_10", "CAM_21", "CAM_11", "CAM_07"}]
    rng.shuffle(keys)
    for k in keys:
        if sum(1 for i in items if i["kind"] == "positive") >= args.pos:
            break
        fs = long_tracks[k]
        a, b = thumb(fs[0]["path"]), thumb(fs[-1]["path"])
        if a and b:
            items.append({"id": f"pos{len(items)}", "kind": "positive",
                          "a": a, "b": b, "score": None})

    # NEGATIVE controls: two different vehicles on the same camera.
    for _ in range(args.neg * 4):
        if sum(1 for i in items if i["kind"] == "negative") >= args.neg:
            break
        k1, k2 = rng.sample(keys, 2)
        if k1[0] != k2[0]:
            continue
        a = thumb(long_tracks[k1][0]["path"])
        b = thumb(long_tracks[k2][-1]["path"])
        if a and b:
            items.append({"id": f"neg{len(items)}", "kind": "negative",
                          "a": a, "b": b, "score": None})

    # REAL candidates: the highest-scoring cross-camera links from the batch
    # that produced 120 "different" verdicts - the ones the answer hinges on.
    prev = []
    vf = VERIFY / "journey_verdicts.jsonl"
    if vf.is_file():
        for line in vf.open(encoding="utf-8"):
            r = json.loads(line)
            if not r.get("control") and r.get("score") is not None:
                prev.append(r)
    prev.sort(key=lambda r: -r["score"])
    for r in prev[:args.real * 3]:
        if sum(1 for i in items if i["kind"] == "real") >= args.real:
            break
        ra, rb = by_key.get(r["a"]), by_key.get(r["b"])
        if not ra or not rb:
            continue
        pa = next((p for p in ra.get("crops", []) if Path(p).is_file()), None)
        pb = next((p for p in rb.get("crops", []) if Path(p).is_file()), None)
        if not pa or not pb:
            continue
        a, b = thumb(pa), thumb(pb)
        if a and b:
            items.append({"id": f"real{len(items)}", "kind": "real",
                          "a": a, "b": b, "score": r["score"]})

    rng.shuffle(items)
    out = Path(args.out)
    out.write_text(PAGE.replace("__DATA__", json.dumps(items)),
                   encoding="utf-8")
    n = defaultdict(int)
    for i in items:
        n[i["kind"]] += 1
    print(f"pairs: {len(items)}  "
          f"({n['positive']} positive controls, {n['negative']} negative "
          f"controls, {n['real']} real candidates)")
    print(f"tool -> {out.resolve()}  ({out.stat().st_size/1e6:.1f} MB)")
    print("\nThe positive controls are the point: if those come back "
          "'different',\nthe earlier 120 verdicts cannot be used.")


if __name__ == "__main__":
    main()
