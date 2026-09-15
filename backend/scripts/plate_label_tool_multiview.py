"""backend/scripts/plate_label_tool_multiview.py — label one plate from several
frames of the same vehicle.

WHY THE SINGLE-VIEW TOOL WAS REPLACED
  A human worked through 672 single-frame crops and read 77 of them - eleven
  percent. Re-rendering the failures with several frames per vehicle made
  roughly four in nine of the real plates readable. The information was in the
  corpus the whole time; the tool was showing one frame of it.

  Motion blur, glare and occlusion fall on different characters in different
  frames. A reader who can see four frames resolves character three from the
  first and character seven from the third. That is how a person reads a plate
  off video, and the tool now allows it.

DISPLAY ENHANCEMENT IS FOR THE EYE ONLY
  The zoom, contrast and sharpen controls affect what is drawn on screen and
  nothing else. The saved crops are untouched, because the recogniser must
  train on the distribution the live pipeline actually produces - enhancing
  the training data would create a model that only works on enhanced input,
  which is the mistake an earlier ablation already measured and rejected.

  For the human the calculus is opposite: anything that makes a character
  legible is worth having, since the output is a string, not an image.

WHAT THE LABELLER SHOULD DO WITH A DOUBTFUL PLATE
  Leave it blank. A guessed label is trained on as truth and its damage is
  invisible in every number afterwards, whereas a skipped plate costs only
  itself. The skip is the safe action and should be the default when unsure.

USAGE
  python -m backend.scripts.plate_label_tool_multiview \
      --src data/plate_label_batch7 --out data/plate_real/label_batch7.html
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

PAGE = """<!doctype html><meta charset="utf-8">
<title>Plate labelling - multi-view</title>
<style>
 :root{--bg:#14161a;--card:#1e2229;--fg:#e8eaed;--dim:#9aa3ad;--ok:#4ade80;
       --accent:#60a5fa}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:15px/1.45 ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif}
 header{position:sticky;top:0;z-index:9;background:#0f1114;
        border-bottom:1px solid #2a2f37;padding:10px 16px;
        display:flex;gap:18px;align-items:center;flex-wrap:wrap}
 header b{font-size:16px}
 .ctl{display:flex;gap:6px;align-items:center;font-size:13px;color:var(--dim)}
 .ctl input[type=range]{width:110px}
 main{padding:14px 16px 120px;max-width:1500px;margin:0 auto}
 .card{background:var(--card);border:1px solid #2a2f37;border-radius:10px;
       padding:12px;margin-bottom:14px}
 .card.done{border-color:var(--ok)}
 .meta{color:var(--dim);font-size:12px;margin-bottom:8px;
       display:flex;gap:14px;flex-wrap:wrap}
 .views{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
 .views img{background:#000;border-radius:6px;border:1px solid #333;
            image-rendering:auto}
 input.plate{margin-top:10px;width:320px;padding:9px 11px;font-size:19px;
      letter-spacing:2px;text-transform:uppercase;background:#0f1114;
      color:var(--fg);border:1px solid #39404a;border-radius:7px;
      font-family:ui-monospace,Consolas,monospace}
 input.plate:focus{outline:none;border-color:var(--accent)}
 input.plate.good{border-color:var(--ok)}
 .hint{color:var(--dim);font-size:12px;margin-left:10px}
 footer{position:fixed;bottom:0;left:0;right:0;background:#0f1114;
        border-top:1px solid #2a2f37;padding:10px 16px;display:flex;gap:12px;
        align-items:center}
 button{background:var(--accent);color:#08131f;border:0;border-radius:7px;
        padding:9px 16px;font-weight:600;cursor:pointer;font-size:14px}
 button.ghost{background:#2a2f37;color:var(--fg)}
 .fmt{font-family:ui-monospace,Consolas,monospace;color:var(--dim);
      font-size:12px}
</style>
<header>
  <b>Plate labelling</b>
  <span id="count" class="hint"></span>
  <span class="fmt">G J 0 1 A B 1 2 3 4 &nbsp;=&nbsp; LL DD LL DDDD</span>
  <span class="ctl">zoom <input type="range" id="zoom" min="1" max="6"
        step="0.25" value="2.5"></span>
  <span class="ctl">contrast <input type="range" id="con" min="100" max="260"
        step="5" value="115"></span>
  <span class="ctl"><label><input type="checkbox" id="sharp"> sharpen</label></span>
</header>
<main id="app"></main>
<footer>
  <button onclick="dl()">Export labels</button>
  <button class="ghost" onclick="clr()">Clear all</button>
  <span class="hint">Enter = next &nbsp;·&nbsp; blank = skip &nbsp;·&nbsp;
    saved automatically</span>
</footer>
<script>
const DATA = __DATA__;
const KEY = "__STOREKEY__";
let store = {};
try { store = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { store = {}; }

const app = document.getElementById("app");
DATA.forEach((d, i) => {
  const c = document.createElement("div");
  c.className = "card";
  c.id = "c" + i;
  const views = d.v.map(v =>
    `<img data-w="${v.w}" src="data:image/jpeg;base64,${v.b}">`).join("");
  c.innerHTML =
    `<div class="meta"><span>${i+1} / ${DATA.length}</span>` +
    `<span>${d.cam}</span><span>track ${d.t}</span>` +
    `<span>${d.v.length} views</span><span>${d.pw}px wide</span></div>` +
    `<div class="views">${views}</div>` +
    `<input class="plate" id="i${i}" maxlength="11" autocomplete="off" ` +
    `spellcheck="false"><span class="hint">can't read it? leave blank</span>`;
  app.appendChild(c);
});

const inputs = [...document.querySelectorAll("input.plate")];
inputs.forEach((el, i) => {
  const k = DATA[i].k;
  if (store[k]) { el.value = store[k]; mark(i); }
  el.addEventListener("input", () => {
    el.value = el.value.toUpperCase().replace(/[^A-Z0-9]/g, "");
    const v = el.value.trim();
    if (v) store[k] = v; else delete store[k];
    localStorage.setItem(KEY, JSON.stringify(store));
    mark(i); count();
  });
  el.addEventListener("keydown", e => {
    if (e.key === "Enter") {
      e.preventDefault();
      const n = inputs[i + 1];
      if (n) { n.focus(); n.scrollIntoView({block:"center",behavior:"smooth"}); }
    }
  });
});

function mark(i){
  const el = inputs[i], c = document.getElementById("c" + i);
  // 9-11 characters is the plausible range for an Indian plate; the marker is
  // a nudge, never a gate - unusual series exist and must remain enterable.
  const ok = el.value.length >= 9 && el.value.length <= 11;
  el.classList.toggle("good", ok);
  c.classList.toggle("done", !!el.value);
}
function count(){
  const n = Object.values(store).filter(v => v && v.trim()).length;
  document.getElementById("count").textContent =
    n + " labelled of " + DATA.length;
}
function applyView(){
  const z = +document.getElementById("zoom").value;
  const con = +document.getElementById("con").value;
  const sh = document.getElementById("sharp").checked;
  document.querySelectorAll(".views img").forEach(im => {
    im.style.height = Math.round(26 * z) + "px";
    im.style.filter = `contrast(${con}%)` + (sh ? " saturate(0)" : "");
    im.style.imageRendering = sh ? "crisp-edges" : "auto";
  });
}
["zoom","con","sharp"].forEach(id =>
  document.getElementById(id).addEventListener("input", applyView));
applyView(); count();

function dl(){
  const out = [];
  DATA.forEach(d => {
    const v = (store[d.k] || "").trim();
    if (v) out.push({file: d.v[0].f, camera: d.cam, track: d.t,
                     ocr: "", text: v});
  });
  if (!out.length) { alert("Nothing labelled yet."); return; }
  const blob = new Blob([out.map(o => JSON.stringify(o)).join("\\n") + "\\n"],
                        {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "__DLNAME__";
  a.click();
  alert(out.length + " labels exported.\\nSave the file into data/plate_real/");
}
function clr(){
  if (!confirm("Clear every label entered so far?")) return;
  store = {}; localStorage.removeItem(KEY);
  inputs.forEach((el, i) => { el.value = ""; mark(i); });
  count();
}
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/plate_label_batch7")
    ap.add_argument("--out", default="data/plate_real/label_batch7.html")
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--dlname", default="plate_labels_batch7.jsonl")
    args = ap.parse_args()

    src = Path(args.src)
    rows = [json.loads(l) for l in (src / "labels.jsonl").open(encoding="utf-8")]
    rows.sort(key=lambda r: -r.get("agreement", 0))
    rows = rows[:args.n]
    print(f"vehicles : {len(rows)}")

    data = []
    for r in rows:
        views = []
        for v in r["views"]:
            p = src / "images" / v["file"]
            if not p.is_file():
                continue
            views.append({"f": v["file"], "w": v["plate_w"],
                          "b": base64.b64encode(p.read_bytes()).decode()})
        if not views:
            continue
        data.append({"k": f'{r["camera"]}_{r["track"]}', "cam": r["camera"],
                     "t": r["track"], "pw": int(r.get("plate_w", 0)),
                     "v": views})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    html = (PAGE.replace("__DATA__", json.dumps(data))
                .replace("__STOREKEY__", f"plate_mv_{out.stem}")
                .replace("__DLNAME__", args.dlname))
    out.write_text(html, encoding="utf-8")
    n_views = sum(len(d["v"]) for d in data)
    print(f"views    : {n_views} ({n_views/max(len(data),1):.1f} per vehicle)")
    print(f"\nlabelling tool -> {out.resolve()}  "
          f"({out.stat().st_size/1e6:.1f} MB)")
    print("\nOpen in a browser. Use zoom/contrast if a plate is marginal.")
    print("Leave doubtful plates BLANK - a guess is worse than a skip.")


if __name__ == "__main__":
    main()
