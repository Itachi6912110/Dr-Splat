#
# 3D object localization directly on the Gaussians — no cameras, no rasterization.
# Loads a Dr-Splat checkpoint (chkpnt0.pth), PQ-decodes each Gaussian's registered
# CLIP embedding, scores it against one or more text queries, and reports each
# object's 3D centroid / bounding box. Optionally writes a bird-eye-view png, a
# heatmap-colored point cloud (PLY), and a self-contained interactive HTML viewer.
#
# Usage:
#   python localize_3d.py -m output/teatime_1_pq_openclip_topk45_weight_128 \
#       --pq_index ckpts/pq_index.faiss --threshold 0.6 \
#       --img_label "teddy bear" "sheep" "coffee mug"
#
import os
import time
import base64
from argparse import ArgumentParser

import faiss
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
from plyfile import PlyData, PlyElement

from evaluation.openclip_encoder import OpenCLIPNetwork

SH_C0 = 0.28209479177387814


def load_gaussians(model_path):
    ckpt = os.path.join(model_path, "chkpnt0.pth")
    (model_params, _) = torch.load(ckpt, weights_only=False)
    assert len(model_params) == 13, f"expected feature checkpoint (13-tuple), got {len(model_params)}"
    (_, xyz, f_dc, _, _, _, opacity, language_feature, *_rest) = model_params
    xyz = xyz.detach().cuda()
    rgb = torch.clamp(0.5 + SH_C0 * f_dc.detach().cuda().squeeze(1), 0.0, 1.0)
    opacity = torch.sigmoid(opacity.detach().cuda()).squeeze(-1)
    return xyz, rgb, opacity, language_feature.detach().cuda()


def write_ply(path, xyz, colors):
    data = np.empty(xyz.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                         ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["red"], data["green"], data["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    PlyData([PlyElement.describe(data, "vertex")]).write(path)


def write_bev(path, query, threshold, xyz_np, hit_np, act_np, cluster_np, lo_np, hi_np, c_np):
    scene = xyz_np[~hit_np]
    obj = xyz_np[hit_np][cluster_np]
    fig, ax = plt.subplots(figsize=(8, 8))
    sub = np.random.default_rng(0).choice(scene.shape[0], min(200000, scene.shape[0]), replace=False)
    ax.scatter(scene[sub, 0], scene[sub, 2], s=0.3, c="0.75", linewidths=0, rasterized=True)
    ax.scatter(obj[:, 0], obj[:, 2], s=0.6, c=act_np[hit_np][cluster_np],
               cmap="turbo", vmin=threshold, vmax=1.0, linewidths=0)
    ax.add_patch(Rectangle((lo_np[0], lo_np[2]), hi_np[0] - lo_np[0], hi_np[2] - lo_np[2],
                           fill=False, edgecolor="red", linewidth=1.5))
    ax.plot(c_np[0], c_np[2], "r+", markersize=12, markeredgewidth=2)
    ax.set_aspect("equal")
    # zoom to the object with context (unbounded scenes have background floaters far away)
    half = max(float(hi_np[0] - lo_np[0]), float(hi_np[2] - lo_np[2])) * 1.8
    ax.set_xlim(c_np[0] - half, c_np[0] + half)
    ax.set_ylim(c_np[2] - half, c_np[2] + half)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(f'bird-eye view: "{query}" ({cluster_np.sum()} Gaussians, thr {threshold})')
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_html(path, xyz, colors, rgb_true, activated, query, stats, bbox_lo, bbox_hi):
    pos = xyz.astype(np.float32)
    center = pos.mean(axis=0)
    pos -= center
    radius = float(np.percentile(np.linalg.norm(pos, axis=1), 95))
    b64_pos = base64.b64encode(pos.tobytes()).decode()
    b64_col = base64.b64encode(colors.astype(np.uint8).tobytes()).decode()
    b64_rgb = base64.b64encode(rgb_true.astype(np.uint8).tobytes()).decode()
    b64_act = base64.b64encode(activated.astype(np.uint8).tobytes()).decode()
    obj_center = (stats["centroid"] - center).tolist()
    fmt = lambda v: "[" + ",".join(f"{x:.4f}" for x in v) + "]"
    html = HTML_TEMPLATE
    for key, val in [("__QUERY__", query), ("__NPTS__", str(pos.shape[0])),
                     ("__NACT__", str(stats["count"])), ("__RADIUS__", f"{radius:.4f}"),
                     ("__OBJ__", fmt(obj_center)),
                     ("__BLO__", fmt(bbox_lo - center)), ("__BHI__", fmt(bbox_hi - center)),
                     ("__POS__", b64_pos), ("__COL__", b64_col), ("__RGB__", b64_rgb),
                     ("__ACT__", b64_act)]:
        html = html.replace(key, val)
    with open(path, "w") as f:
        f.write(html)


HTML_TEMPLATE = r"""<meta charset="utf-8"><title>Dr-Splat 3D localization</title>
<style>
  html,body{margin:0;height:100%;overflow:hidden;background:#111;color:#ddd;font:13px system-ui,sans-serif}
  canvas{display:block;width:100vw;height:100vh;cursor:grab}
  #hud{position:fixed;top:10px;left:10px;background:rgba(20,20,25,.85);padding:10px 14px;border-radius:8px;line-height:1.5}
  #hud b{color:#ffb057}
  #hud .dim{color:#888}
  label{user-select:none}
</style>
<div id="hud">
  query: <b>__QUERY__</b><br>
  <span class="dim">__NACT__ / __NPTS__ Gaussians activated</span><br>
  <label><input type="checkbox" id="only"> show activated only</label><br>
  <label><input type="checkbox" id="truecol"> true colors</label><br>
  <label><input type="checkbox" id="bbox" checked> show 3D bounding box</label><br>
  <span class="dim">drag: orbit &nbsp; wheel: zoom &nbsp; shift-drag: pan</span>
</div>
<canvas id="c"></canvas>
<script>
const B=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
const pos=new Float32Array(B("__POS__").buffer), col=B("__COL__"), rgb=B("__RGB__"), act=B("__ACT__");
const N=__NPTS__, R=__RADIUS__, OBJ=__OBJ__, BLO=__BLO__, BHI=__BHI__;
const cv=document.getElementById("c"), gl=cv.getContext("webgl");
const vs=`attribute vec3 p;attribute vec3 c;attribute vec3 c2;attribute float a;
uniform mat4 mvp;uniform float ps;uniform float onlyAct;uniform float trueCol;
varying vec3 vc;varying float va;
void main(){gl_Position=mvp*vec4(p,1.);float w=max(gl_Position.w,.01);gl_PointSize=clamp(ps/w,1.,8.);
vc=mix(c,c2,trueCol);va=a;if(onlyAct>.5&&a<.5)gl_Position=vec4(2e9,2e9,2e9,1.);}`;
const fs=`precision mediump float;varying vec3 vc;varying float va;uniform float isLine;
void main(){if(isLine<.5){vec2 d=gl_PointCoord-vec2(.5);if(dot(d,d)>.25)discard;}gl_FragColor=vec4(vc,1.);}`;
function sh(t,s){const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);
if(!gl.getShaderParameter(o,gl.COMPILE_STATUS))throw gl.getShaderInfoLog(o);return o;}
const pr=gl.createProgram();gl.attachShader(pr,sh(gl.VERTEX_SHADER,vs));gl.attachShader(pr,sh(gl.FRAGMENT_SHADER,fs));
gl.linkProgram(pr);gl.useProgram(pr);
function mkbuf(data){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);
gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);return b;}
function attrib(b,name,size,type,norm){gl.bindBuffer(gl.ARRAY_BUFFER,b);const l=gl.getAttribLocation(pr,name);
gl.enableVertexAttribArray(l);gl.vertexAttribPointer(l,size,type,norm,0,0);}
const bP=mkbuf(pos),bC=mkbuf(col),bC2=mkbuf(rgb),bA=mkbuf(Float32Array.from(act));
// wireframe bbox: 12 edges from the dominant-cluster min/max corners
const E=[[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[1,1,0,0,1,0],[1,1,0,1,0,0],[1,1,0,1,1,1],
[1,0,1,0,0,1],[1,0,1,1,0,0],[1,0,1,1,1,1],[0,1,1,0,0,1],[0,1,1,0,1,0],[0,1,1,1,1,1]];
const boxPos=new Float32Array(E.flat().map((t,i)=>{const ax=i%3;return t?BHI[ax]:BLO[ax];}));
const NBOX=boxPos.length/3;
const boxCol=new Uint8Array(NBOX*3);for(let i=0;i<NBOX;i++){boxCol[i*3]=255;boxCol[i*3+1]=45;boxCol[i*3+2]=45;}
const bBP=mkbuf(boxPos),bBC=mkbuf(boxCol),bBA=mkbuf(new Float32Array(NBOX).fill(1));
const uMVP=gl.getUniformLocation(pr,"mvp"),uPS=gl.getUniformLocation(pr,"ps"),
uOnly=gl.getUniformLocation(pr,"onlyAct"),uTrue=gl.getUniformLocation(pr,"trueCol"),
uLine=gl.getUniformLocation(pr,"isLine");
const BD=Math.hypot(BHI[0]-BLO[0],BHI[1]-BLO[1],BHI[2]-BLO[2]);
let yaw=.6,pitch=.4,dist=Math.max(2.5*BD,.15*R),tx=OBJ[0],ty=OBJ[1],tz=OBJ[2],only=0,truec=0,box=1;
document.getElementById("only").onchange=e=>{only=e.target.checked?1:0;};
document.getElementById("truecol").onchange=e=>{truec=e.target.checked?1:0;};
document.getElementById("bbox").onchange=e=>{box=e.target.checked?1:0;};
let drag=0,px=0,py=0;
cv.onmousedown=e=>{drag=e.shiftKey?2:1;px=e.clientX;py=e.clientY;};
window.onmouseup=()=>drag=0;
window.onmousemove=e=>{if(!drag)return;const dx=e.clientX-px,dy=e.clientY-py;px=e.clientX;py=e.clientY;
if(drag==1){yaw+=dx*.005;pitch=Math.max(-1.55,Math.min(1.55,pitch+dy*.005));}
else{const s=dist*.0012;tx-=s*(dx*Math.cos(yaw)-0);tz-=s*(dx*Math.sin(yaw));ty+=s*dy;}};
cv.onwheel=e=>{e.preventDefault();dist*=Math.pow(1.1,e.deltaY>0?1:-1);dist=Math.max(.05*R,Math.min(20*R,dist));};
function mat(){const cw=cv.clientWidth,ch=cv.clientHeight,ar=cw/ch,f=1.6,zn=.01*R,zf=60*R;
const cp=Math.cos(pitch),sp=Math.sin(pitch),cy=Math.cos(yaw),sy=Math.sin(yaw);
const ex=tx+dist*cp*sy,ey=ty+dist*sp,ez=tz+dist*cp*cy;
let zx=ex-tx,zy=ey-ty,zz=ez-tz;const zl=Math.hypot(zx,zy,zz);zx/=zl;zy/=zl;zz/=zl;
let xx=zz,xy=0,xz=-zx;const xl=Math.hypot(xx,xy,xz)||1;xx/=xl;xz/=xl;
const yx=zy*xz-zz*xy,yy=zz*xx-zx*xz,yz=zx*xy-zy*xx;
const v=[xx,yx,zx,0, xy,yy,zy,0, xz,yz,zz,0, -(xx*ex+xy*ey+xz*ez),-(yx*ex+yy*ey+yz*ez),-(zx*ex+zy*ey+zz*ez),1];
const p=[f/ar,0,0,0, 0,f,0,0, 0,0,(zf+zn)/(zn-zf),-1, 0,0,2*zf*zn/(zn-zf),0];
const m=new Float32Array(16);
for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;for(let k=0;k<4;k++)s+=v[i*4+k]*p[k*4+j];m[i*4+j]=s;}
return m;}
function frame(){const dpr=window.devicePixelRatio||1;
if(cv.width!==cv.clientWidth*dpr||cv.height!==cv.clientHeight*dpr){cv.width=cv.clientWidth*dpr;cv.height=cv.clientHeight*dpr;}
gl.viewport(0,0,cv.width,cv.height);gl.clearColor(.07,.07,.08,1);gl.enable(gl.DEPTH_TEST);
gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
gl.uniformMatrix4fv(uMVP,false,mat());gl.uniform1f(uPS,cv.height*.0035*R);
gl.uniform1f(uOnly,only);gl.uniform1f(uTrue,truec);gl.uniform1f(uLine,0);
attrib(bP,"p",3,gl.FLOAT,false);attrib(bC,"c",3,gl.UNSIGNED_BYTE,true);
attrib(bC2,"c2",3,gl.UNSIGNED_BYTE,true);attrib(bA,"a",1,gl.FLOAT,false);
gl.drawArrays(gl.POINTS,0,N);
if(box){gl.uniform1f(uLine,1);gl.uniform1f(uTrue,0);gl.uniform1f(uOnly,0);
attrib(bBP,"p",3,gl.FLOAT,false);attrib(bBC,"c",3,gl.UNSIGNED_BYTE,true);
attrib(bBC,"c2",3,gl.UNSIGNED_BYTE,true);attrib(bBA,"a",1,gl.FLOAT,false);
gl.drawArrays(gl.LINES,0,NBOX);}
requestAnimationFrame(frame);}
frame();
</script>
"""


if __name__ == "__main__":
    parser = ArgumentParser(description="Text-query 3D localization directly on Gaussians")
    parser.add_argument("-m", "--model_path", type=str, required=True)
    parser.add_argument("--pq_index", type=str, required=True)
    parser.add_argument("--img_label", type=str, nargs="+", required=True,
                        help="one or more text queries; model/index load once, each query is scored in-loop")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--opacity_min", type=float, default=0.1)
    parser.add_argument("--html_points", type=int, default=400000)
    parser.add_argument("--skip_ply", action="store_true")
    parser.add_argument("--skip_html", action="store_true")
    args = parser.parse_args()

    xyz, rgb, opacity, codes = load_gaussians(args.model_path)
    index = faiss.read_index(args.pq_index)
    clip_model = OpenCLIPNetwork("cuda")
    clip_model.set_positives(args.img_label)

    # PQ decode is query-independent — do it once for all queries
    torch.cuda.synchronize()
    t0 = time.time()
    valid = ~torch.all(codes == -1, dim=-1)
    embeds = torch.from_numpy(index.sa_decode(codes[valid].cpu().numpy())).cuda()
    torch.cuda.synchronize()
    print(f"decoded {int(valid.sum())} of {codes.shape[0]} Gaussian embeddings in {time.time() - t0:.2f} s")

    keep = opacity > args.opacity_min
    xyz_np = xyz[keep].cpu().numpy()
    base = (rgb[keep].cpu().numpy() * 0.35 + 0.45) * 255.0  # desaturated scene
    true_rgb = rgb[keep].cpu().numpy() * 255.0
    out_dir = os.path.join(args.model_path, "localize_3d")
    os.makedirs(out_dir, exist_ok=True)

    for qi, query in enumerate(args.img_label):
        # --- the 3D-native query: CLIP scoring against every Gaussian, no rasterization ---
        torch.cuda.synchronize()
        t0 = time.time()
        activation = torch.zeros(codes.shape[0], device="cuda")
        activation[valid] = clip_model.get_activation(embeds, qi).squeeze(-1)
        torch.cuda.synchronize()
        query_time = time.time() - t0

        hit = (activation > args.threshold) & (opacity > args.opacity_min)
        n_hit = int(hit.sum())
        print(f"\nquery '{query}': scored {int(valid.sum())} Gaussians in {query_time*1000:.1f} ms "
              f"(no rasterization), {n_hit} above threshold {args.threshold}")
        if n_hit == 0:
            print("  no Gaussians activated — lower --threshold")
            continue

        # sigma-clip to the dominant cluster so scattered false positives don't inflate the box
        pts = xyz[hit]
        w = (activation[hit] * opacity[hit]).unsqueeze(-1)
        keep_c = torch.ones(pts.shape[0], dtype=torch.bool, device="cuda")
        for _ in range(5):
            centroid = (pts[keep_c] * w[keep_c]).sum(0) / w[keep_c].sum()
            d = (pts - centroid).norm(dim=-1)
            sigma = d[keep_c].square().mean().sqrt()
            keep_c = d < 2.0 * sigma
        lo = pts[keep_c].min(dim=0).values
        hi = pts[keep_c].max(dim=0).values
        print(f"  3D centroid: {centroid.cpu().numpy().round(4).tolist()} "
              f"({int(keep_c.sum())} of {n_hit} in dominant cluster)")
        print(f"  3D bbox: min {lo.cpu().numpy().round(4).tolist()} max {hi.cpu().numpy().round(4).tolist()}")

        # --- visualization exports ---
        act_np = activation[keep].cpu().numpy()
        hit_np = hit[keep].cpu().numpy()
        cluster_np = keep_c.cpu().numpy()
        heat = cm.turbo(np.clip((act_np - args.threshold) / max(1e-6, 1 - args.threshold), 0, 1))[:, :3] * 255.0
        colors = np.where(hit_np[:, None], heat, base).astype(np.uint8)

        tag = query.replace(" ", "_")
        stats = {"count": n_hit, "centroid": centroid.cpu().numpy()}
        bev_path = os.path.join(out_dir, f"{tag}_bev.png")
        write_bev(bev_path, query, args.threshold, xyz_np, hit_np, act_np, cluster_np,
                  lo.cpu().numpy(), hi.cpu().numpy(), centroid.cpu().numpy())
        print(f"  wrote {bev_path}")

        if not args.skip_ply:
            ply_path = os.path.join(out_dir, f"{tag}.ply")
            write_ply(ply_path, xyz_np, colors)
            print(f"  wrote {ply_path} ({xyz_np.shape[0]} points)")

        if not args.skip_html:
            if xyz_np.shape[0] > args.html_points:
                rest = np.where(~hit_np)[0]
                sub = np.random.default_rng(0).choice(rest, max(0, args.html_points - n_hit), replace=False)
                sel = np.concatenate([np.where(hit_np)[0], sub])
            else:
                sel = np.arange(xyz_np.shape[0])
            html_path = os.path.join(out_dir, f"{tag}.html")
            write_html(html_path, xyz_np[sel], colors[sel], true_rgb[sel], hit_np[sel], query, stats,
                       lo.cpu().numpy(), hi.cpu().numpy())
            print(f"  wrote {html_path} ({sel.shape[0]} points)")
