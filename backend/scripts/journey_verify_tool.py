"""backend/scripts/journey_verify_tool.py — a page for ruling on candidate links.

WHAT IS BEING ASKED, AND WHY IT IS FAST
  Two vehicle crops from two cameras: same vehicle or not. That is a glance,
  not a reading. Labelling a plate means transcribing ten characters from a
  60-pixel smear; this is "is that the same car", which a person answers in
  about three seconds. A hundred and twenty pairs is under ten minutes, and it
  is the only input the journey feature still lacks.

DESIGN CHOICES THAT PROTECT THE ANSWER

  The plate readings are hidden. Shown two identical strings, a labeller
  agrees with the system and the exercise measures nothing. The question is
  whether the VEHICLES match.

  The score is hidden for the same reason. A pair presented as "high
  confidence" biases the judgement toward confirming it.

  Keyboard-first: S for same, D for different, U for unsure. Reaching for a
  mouse doubles the time per pair and the batch stops getting finished.

  UNSURE is a first-class answer with its own key. A forced binary makes
  people guess on hard pairs, and a guessed label is worse than a missing one -
  it will be trained on and scored against as though it were known.

USAGE
  python -m backend.scripts.journey_verify_tool
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SRC = Path("data/journey_verify")

PAGE = """<!doctype html><meta charset="utf-8">
<title>Same vehicle?</title>
<style>
:root{--bg:#0f1216;--card:#181d24;--fg:#e6eaf0;--dim:#8b96a4;--rule:#2a323c;
 --ok:#4ade80;--no:#f87171;--maybe:#fbbf24;--accent:#60a5fa}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:15px/1.5 ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;z-index:9;background:#0b0e12;
 border-bottom:1px solid var(--rule);padding:12px 18px;
 display:flex;gap:20px;align-items:center;flex-wrap:wrap}
header b{font-size:17px}
.keys{color:var(--dim);font-size:13px}
kbd{background:#222a34;border:1px solid #39424e;border-radius:4px;
 padding:1px 7px;font-family:ui-monospace,monospace;font-size:12px}
.bar{flex:1;height:7px;background:#222a34;border-radius:4px;overflow:hidden;
 min-width:120px}
.bar i{display:block;height:100%;background:var(--accent);width:0}
main{padding:16px 18px 130px;max-width:1180px;margin:0 auto}
.pair{background:var(--card);border:1px solid var(--rule);border-radius:12px;
 padding:14px;margin-bottom:16px}
.pair.done{opacity:.5}
.pair.cur{border-color:var(--accent);box-shadow:0 0 0 2px rgba(96,165,250,.2)}
.imgs{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:700px){.imgs{grid-template-columns:1fr}}
.side{background:#000;border-radius:9px;overflow:hidden;
 border:1px solid #2a323c}
.side img{display:block;width:100%;height:auto}
.cap{padding:7px 10px;color:var(--dim);font-size:12.5px;
 font-family:ui-monospace,monospace;background:#11161c}
.row{display:flex;gap:10px;align-items:center;margin-top:12px;flex-wrap:wrap}
button{border:0;border-radius:8px;padding:9px 18px;font-weight:700;
 cursor:pointer;font-size:14px}
.b-same{background:var(--ok);color:#06210f}
.b-diff{background:var(--no);color:#2a0a0a}
.b-uns{background:#2a323c;color:var(--fg)}
.verdict{margin-left:auto;font-weight:700;font-size:14px}
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
<main id="app"></main>
<footer>
  <button onclick="dl()">Export judgements</button>
  <span class="hint">Judge the VEHICLES, not the plates. Unsure is a real
    answer &mdash; use it rather than guessing.</span>
</footer>
<script>
const DATA = __DATA__;
const KEY = "journey_verify_v1";
let store = {};
try { store = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e){}
let cur = 0;

const app = document.getElementById("app");
DATA.forEach((d, i) => {
  const el = document.createElement("div");
  el.className = "pair"; el.id = "p" + i;
  el.innerHTML =
    '<div class="imgs">'
    + '<div class="side"><img src="data:image/jpeg;base64,'+d.a_img+'">'
      + '<div class="cap">'+d.a_cam+' &middot; '+(d.a_ts||'')+'</div></div>'
    + '<div class="side"><img src="data:image/jpeg;base64,'+d.b_img+'">'
      + '<div class="cap">'+d.b_cam+' &middot; '+(d.b_ts||'')+'</div></div>'
    + '</div>'
    + '<div class="row">'
    + '<button class="b-same" onclick="mark('+i+',\\'same\\')">Same vehicle</button>'
    + '<button class="b-diff" onclick="mark('+i+',\\'diff\\')">Different</button>'
    + '<button class="b-uns" onclick="mark('+i+',\\'unsure\\')">Unsure</button>'
    + '<span class="verdict" id="v'+i+'"></span></div>';
  app.appendChild(el);
});

function mark(i, v){
  store[DATA[i].id] = v;
  localStorage.setItem(KEY, JSON.stringify(store));
  paint(i);
  count();
  if (i === cur) focus(Math.min(cur + 1, DATA.length - 1));
}
function paint(i){
  const v = store[DATA[i].id];
  const el = document.getElementById("p" + i);
  const s = document.getElementById("v" + i);
  el.classList.toggle("done", !!v);
  s.className = "verdict " + (v === "same" ? "v-same"
    : v === "diff" ? "v-diff" : v ? "v-uns" : "");
  s.textContent = v === "same" ? "SAME" : v === "diff" ? "DIFFERENT"
    : v === "unsure" ? "UNSURE" : "";
}
function focus(i){
  document.getElementById("p" + cur)?.classList.remove("cur");
  cur = i;
  const el = document.getElementById("p" + cur);
  el.classList.add("cur");
  el.scrollIntoView({block: "center", behavior: "smooth"});
}
function count(){
  const n = Object.keys(store).length;
  document.getElementById("count").textContent = n + " of " + DATA.length;
  document.getElementById("prog").style.width =
    (n / DATA.length * 100) + "%";
}
document.addEventListener("keydown", e => {
  const k = e.key.toLowerCase();
  if (k === "s") { mark(cur, "same"); e.preventDefault(); }
  else if (k === "d") { mark(cur, "diff"); e.preventDefault(); }
  else if (k === "u") { mark(cur, "unsure"); e.preventDefault(); }
  else if (e.key === "ArrowLeft") { focus(Math.max(0, cur - 1)); e.preventDefault(); }
  else if (e.key === "ArrowRight") { focus(Math.min(DATA.length-1, cur+1)); e.preventDefault(); }
});
function dl(){
  const out = DATA.filter(d => store[d.id]).map(d => ({
    id: d.id, verdict: store[d.id], score: d.score, control: d.control,
    a: d.a_key, b: d.b_key, a_cam: d.a_cam, b_cam: d.b_cam
  }));
  if (!out.length){ alert("Nothing judged yet."); return; }
  const blob = new Blob([out.map(o => JSON.stringify(o)).join("\\n")+"\\n"],
                        {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "journey_verdicts.jsonl";
  a.click();
  alert(out.length + " judgements exported.\\nSave into data/journey_verify/");
}
DATA.forEach((_, i) => paint(i));
count(); focus(0);
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default="journey_verify.html")
    args = ap.parse_args()

    items = json.loads((Path(args.src) / "pairs.json").read_text(encoding="utf-8"))
    # The score and the plate never reach the page - see the module docstring.
    safe = [{k: v for k, v in it.items() if k not in ("score",)}
            for it in items]
    for s, it in zip(safe, items):
        s["score"] = it["score"]          # kept in the EXPORT, not displayed
    out = Path(args.out)
    out.write_text(PAGE.replace("__DATA__", json.dumps(safe)),
                   encoding="utf-8")
    print(f"pairs: {len(items)} "
          f"({sum(1 for i in items if i['control'])} controls mixed in)")
    print(f"tool -> {out.resolve()}  ({out.stat().st_size/1e6:.1f} MB)")
    print("\nS = same, D = different, U = unsure. Judge the VEHICLES.")
    print("Export when done and save into data/journey_verify/")


if __name__ == "__main__":
    main()
