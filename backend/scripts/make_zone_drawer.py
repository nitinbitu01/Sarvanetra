"""backend/scripts/make_zone_drawer.py — browser-based zone drawing tool.

WHY A BROWSER AND NOT AN OpenCV WINDOW
  draw_zones.py uses cv2.imshow, which on Windows opens behind other windows,
  ignores focus, and sometimes fails to appear at all. The plate labelling tool
  built earlier in this project worked reliably in Chrome, so zones use the
  same approach: a self-contained HTML file with the camera frame embedded, no
  server and no install.

WHAT THE ZONE TYPES MEAN
  exempt      Bus stop, auto stand, shop frontage, hospital gate. Waiting is
              the purpose of these places. Never alerts, at any duration.
  normal      Footpath, open road. 10 min day / 5 min night.
  sensitive   Parked vehicles, ATM, isolated corner. 3 min day / 90s night.

  Draw exempt zones generously. A missed alert at a bus stop costs nothing.
  Fifty alerts from a bus stop cost the operator's attention, and once that is
  gone the genuine alerts go unread too - which is how analytics end up
  switched off after deployment.

USAGE
  python -m backend.scripts.make_zone_drawer --camera CAM_04 \
      --clip demo/clips/cam_04_0700.mp4 --time 30
  then open output/zone_drawer_CAM_04.html
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "timeout;30000000|stimeout;30000000|rw_timeout;30000000")

import cv2

PAGE = """<title>Zone Drawer</title>
<style>
 :root{--bg:#12141a;--fg:#e8eaf0;--mut:#8b93a7;--acc:#5eead4;--card:#1b1f28;
       --edge:#2a3040;--exempt:#5ac85a;--normal:#fabb3c;--sens:#f5503a}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:15px/1.5 ui-sans-serif,system-ui,Segoe UI,sans-serif}
 header{padding:14px 20px;border-bottom:1px solid var(--edge)}
 h1{margin:0 0 4px;font-size:17px}
 .sub{color:var(--mut);font-size:13px}
 main{display:flex;gap:16px;padding:16px 20px;align-items:flex-start}
 #wrap{position:relative;line-height:0;border:1px solid var(--edge);
       border-radius:8px;overflow:hidden}
 canvas{display:block;cursor:crosshair;max-width:100%}
 aside{width:320px;flex:0 0 auto}
 .card{background:var(--card);border:1px solid var(--edge);border-radius:10px;
       padding:14px;margin-bottom:12px}
 .t{display:flex;gap:8px;margin:10px 0}
 .t button{flex:1;border:0;border-radius:7px;padding:10px 6px;font-weight:700;
           cursor:pointer;font-size:13px;color:#07120f}
 .t .e{background:var(--exempt)} .t .n{background:var(--normal)}
 .t .s{background:var(--sens);color:#fff}
 .t button.off{opacity:.35}
 button.act{background:var(--acc);color:#06231f;border:0;border-radius:7px;
            padding:10px 16px;font-weight:700;cursor:pointer;width:100%;
            margin-top:6px;font-size:14px}
 button.ghost{background:transparent;color:var(--mut);
              border:1px solid var(--edge)}
 ul{list-style:none;padding:0;margin:8px 0 0;font-size:13px}
 li{padding:6px 8px;border-radius:6px;background:#0d1016;margin-bottom:5px;
    display:flex;justify-content:space-between}
 .k{color:var(--mut);font-size:12px;line-height:1.7}
 code{background:#0d1016;padding:1px 6px;border-radius:4px;color:var(--acc)}
</style>
<header>
  <h1>Zone Drawer &mdash; __CAM__</h1>
  <div class="sub">Frame par click karke polygon banao &rarr; type chuno &rarr; Export</div>
</header>
<main>
  <div id="wrap"><canvas id="c"></canvas></div>
  <aside>
    <div class="card">
      <b>1. Polygon banao</b>
      <div class="k">Frame par click karo (min 3 points).<br>
      <code>Right-click</code> = last point hatao</div>
      <div class="t">
        <button class="e" onclick="finish('exempt')">1 EXEMPT</button>
        <button class="n" onclick="finish('normal')">2 NORMAL</button>
        <button class="s" onclick="finish('sensitive')">3 SENSITIVE</button>
      </div>
      <div class="k">
        <b style="color:var(--exempt)">EXEMPT</b> &mdash; bus stop, auto stand,
        dukaan, hospital gate. Kabhi alert nahi.<br>
        <b style="color:var(--normal)">NORMAL</b> &mdash; footpath, sadak.
        10min din / 5min raat.<br>
        <b style="color:var(--sens)">SENSITIVE</b> &mdash; khadi gaadiyan, ATM,
        sunsaan corner. 3min din / 90s raat.
      </div>
    </div>
    <div class="card">
      <b>Zones (<span id="n">0</span>)</b>
      <ul id="list"></ul>
      <button class="act ghost" onclick="undo()">Last zone hatao</button>
    </div>
    <div class="card">
      <button class="act" onclick="save()">Export zones JSON</button>
      <div class="k" style="margin-top:8px">File ko
      <code>config/zones/__CAM__.json</code> mein rakhna</div>
    </div>
  </aside>
</main>
<script>
const CAM="__CAM__", W=__W__, H=__H__, CLIP="__CLIP__";
const DEF={exempt:[null,null], normal:[600,300], sensitive:[180,90]};
const COL={exempt:"#5ac85a", normal:"#fabb3c", sensitive:"#f5503a"};
const img=new Image(); img.src="data:image/jpeg;base64,__IMG__";
const c=document.getElementById("c"), ctx=c.getContext("2d");
let pts=[], zones=[];

img.onload=()=>{ c.width=W; c.height=H; draw(); };

function draw(){
  ctx.drawImage(img,0,0,W,H);
  zones.forEach(z=>{
    ctx.beginPath();
    z.polygon.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));
    ctx.closePath();
    ctx.fillStyle=COL[z.type]+"55"; ctx.fill();
    ctx.strokeStyle=COL[z.type]; ctx.lineWidth=3; ctx.stroke();
    const cx=z.polygon.reduce((a,p)=>a+p[0],0)/z.polygon.length;
    const cy=z.polygon.reduce((a,p)=>a+p[1],0)/z.polygon.length;
    ctx.fillStyle="#fff"; ctx.font="bold 20px sans-serif";
    ctx.textAlign="center"; ctx.fillText(z.type.toUpperCase(),cx,cy);
  });
  if(pts.length){
    ctx.beginPath();
    pts.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));
    ctx.strokeStyle="#fff"; ctx.lineWidth=2; ctx.stroke();
    pts.forEach(p=>{ ctx.beginPath(); ctx.arc(p[0],p[1],6,0,7);
                     ctx.fillStyle="#fff"; ctx.fill(); });
  }
}
function xy(e){
  const r=c.getBoundingClientRect();
  return [Math.round((e.clientX-r.left)*W/r.width),
          Math.round((e.clientY-r.top)*H/r.height)];
}
c.addEventListener("click",e=>{ pts.push(xy(e)); draw(); });
c.addEventListener("contextmenu",e=>{ e.preventDefault(); pts.pop(); draw(); });

function finish(type){
  if(pts.length<3){ alert("Kam se kam 3 points chahiye"); return; }
  const [d,n]=DEF[type];
  zones.push({id:type+"_"+(zones.length+1), type:type, label:type+"_"+(zones.length+1),
              polygon:pts, threshold_day_sec:d, threshold_night_sec:n});
  pts=[]; draw(); refresh();
}
function undo(){ zones.pop(); draw(); refresh(); }
function refresh(){
  document.getElementById("n").textContent=zones.length;
  document.getElementById("list").innerHTML=zones.map(z=>
    `<li><span style="color:${COL[z.type]}">${z.type}</span>`+
    `<span style="color:#8b93a7">${z.polygon.length} pts</span></li>`).join("");
}
document.addEventListener("keydown",e=>{
  if(e.key==="1")finish("exempt"); if(e.key==="2")finish("normal");
  if(e.key==="3")finish("sensitive"); if(e.key==="u")undo();
});
function save(){
  if(!zones.length){ alert("Pehle zones banao"); return; }
  const out={camera_id:CAM, frame_width:W, frame_height:H,
             source_clip:CLIP, zones:zones};
  const b=new Blob([JSON.stringify(out,null,2)],{type:"application/json"});
  const a=document.createElement("a");
  a.href=URL.createObjectURL(b); a.download=CAM+".json"; a.click();
  alert(zones.length+" zones exported.\\nFile ko config/zones/ mein daalo.");
}
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--time", type=float, default=30.0,
                    help="Seconds into the clip. Pick a BUSY moment so the "
                         "bus stops and auto stands are visibly in use.")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.clip}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.time * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame at {args.time}s")

    h, w = frame.shape[:2]
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    b64 = base64.b64encode(buf).decode()

    html = (PAGE.replace("__CAM__", args.camera)
                .replace("__W__", str(w)).replace("__H__", str(h))
                .replace("__CLIP__", args.clip.replace("\\", "/"))
                .replace("__IMG__", b64))

    out = Path(f"output/zone_drawer_{args.camera}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"camera : {args.camera}   frame {w}x{h} at {args.time}s")
    print(f"tool   -> {out.resolve()}  ({out.stat().st_size/1_048_576:.1f} MB)")
    print("\nBrowser mein kholo, zones banao, Export dabao,")
    print(f"phir file ko config/zones/{args.camera}.json mein rakho.")


if __name__ == "__main__":
    main()
