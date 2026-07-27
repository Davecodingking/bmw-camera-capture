"""
把 rgb.mp4 + depth.mkv 合成对齐检查视频。

默认模式(verify)：完全按 vxr VxrMapConsolidator._alignment_verify 的可视化，但【逐帧】输出 mp4：
  每帧四联 [ RGB | 深度(log 归一化 TURBO) | RGB+黄色深度边缘 | 纯深度边缘 ]
  深度按 camera_params 的 encoding 解码(log_16bit / linear_16bit 都支持)；
  每帧用几何像素(排除天空)的 log 深度 p2..p98 归一化再上 TURBO，Canny(40,100) 提边。

其它模式：
  --gray          深度灰度半透明叠 RGB + 白线描深度不连续(相对梯度)
  --blend         深度 TURBO 半透明叠 RGB
  --side-by-side  左 RGB / 右 深度 TURBO 并排

用法：
  python make_alignment.py --dataset-dir <rgb_depth_dataset 目录>
      [--panel-width 640] [--edge-low 40] [--edge-high 100] [--out xxx.mp4]
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import cv2


def raw_reader(path, pix_fmt):
    return subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-an", "-sn",
         "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def read_frame(proc, nbytes):
    if proc.stdout is None:
        return None
    data = proc.stdout.read(nbytes)
    if not data or len(data) != nbytes:
        return None
    return data


def make_depth_decoder(params):
    """按 camera_params 的 'encoding' 返回 decode(u16 数组)->depth_m(米)。支持 log_16bit / linear_16bit。"""
    dmin = float(params["depth_min_m"]); dmax = float(params["depth_max_m"])
    enc = params.get("encoding", "linear_16bit")
    if enc == "log_16bit":
        lo = np.log(dmin + 1.0); hi = np.log(dmax + 1.0)

        def dec(u16):
            return np.exp(u16.astype(np.float64) / 65535.0 * (hi - lo) + lo) - 1.0
    else:
        def dec(u16):
            u = u16.astype(np.float64)
            d = (u - 1.0) / 65534.0 * (dmax - dmin) + dmin
            d[u16 == 0] = 0.0                     # 0 = invalid (linear 约定)
            return d
    return dec, dmin, dmax, enc


def depth_edges(depth_m, valid, rel, kernel, thick):
    """gray 模式用：log 深度的 Sobel 相对梯度 > rel = 深度不连续。"""
    logd = np.zeros_like(depth_m)
    logd[valid] = np.log(np.clip(depth_m[valid], 1e-3, None))
    logd = cv2.GaussianBlur(logd, (3, 3), 0)
    gx = cv2.Sobel(logd, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(logd, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    edges = ((grad > rel) & valid).astype(np.uint8)
    if thick > 0:
        edges = cv2.dilate(edges, kernel, iterations=thick)
    return edges


def _label(im, txt):
    cv2.putText(im, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(im, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return im


def main():
    ap = argparse.ArgumentParser(description="vxr 式逐帧深度对齐检查 -> alignment.mp4")
    ap.add_argument("--dataset-dir", type=Path, default=None)
    ap.add_argument("--rgb", type=Path, default=None)
    ap.add_argument("--depth-mkv", type=Path, default=None)
    ap.add_argument("--camera-params", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--gray", action="store_true", help="旧模式:深度灰度叠RGB+白色相对梯度边")
    ap.add_argument("--blend", action="store_true", help="深度TURBO半透明叠RGB")
    ap.add_argument("--side-by-side", action="store_true", help="并排 左RGB 右深度TURBO")
    ap.add_argument("--alpha", type=float, default=0.5, help="gray/blend 叠加不透明度")
    ap.add_argument("--panel-width", type=int, default=640, help="verify 每个子图宽度")
    ap.add_argument("--edge-low", type=float, default=40.0, help="verify Canny 低阈值(vxr=40)")
    ap.add_argument("--edge-high", type=float, default=100.0, help="verify Canny 高阈值(vxr=100)")
    ap.add_argument("--max-depth", type=float, default=0.0, help="gray/blend 可视上限(米),0=每帧p95")
    ap.add_argument("--edge-rel", type=float, default=0.3, help="gray 模式相对梯度阈值")
    ap.add_argument("--edge-thick", type=int, default=1, help="gray 模式边线加粗")
    ap.add_argument("--edge-color", type=str, default="255,255,255", help="gray 模式边色 BGR")
    args = ap.parse_args()

    d = args.dataset_dir
    rgb_path = args.rgb or (d / "rgb.mp4")
    depth_path = args.depth_mkv or (d / "depth.mkv")
    cam_path = args.camera_params or (d / "camera_params.json")
    mode = ("gray" if args.gray else "blend" if args.blend
            else "sbs" if args.side_by_side else "verify")
    default_out = {"gray": "alignment_gray.mp4", "blend": "alignment_blend.mp4",
                   "sbs": "alignment_sbs.mp4", "verify": "alignment.mp4"}[mode]
    out_path = args.out or (d / default_out)

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found in PATH")
    for p in (rgb_path, depth_path, cam_path):
        if not Path(p).exists():
            raise SystemExit(f"missing input: {p}")

    edge_color = tuple(int(x) for x in args.edge_color.split(","))
    params = json.load(open(cam_path, "r", encoding="utf-8"))
    W = int(params["width"]); H = int(params["height"]); fps = params.get("fps", 25)
    total = int(params.get("num_frames", 0))
    decode, dmin, dmax, enc = make_depth_decoder(params)
    kernel = np.ones((3, 3), np.uint8)

    if mode == "verify":
        pw = args.panel_width - (args.panel_width % 2)
        ph = int(round(H * pw / W)); ph -= ph % 2
        out_w, out_h = pw * 4, ph
    elif mode == "sbs":
        out_w, out_h = W * 2, H
    else:
        out_w, out_h = W, H

    rp = raw_reader(rgb_path, "bgr24")
    dp = raw_reader(depth_path, "gray16le")
    enc_p = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{out_w}x{out_h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out_path)],
        stdin=subprocess.PIPE,
    )
    print(f"mode={mode} encoding={enc} out={out_path} ({out_w}x{out_h})")

    n = 0
    while True:
        rb = read_frame(rp, W * H * 3)
        db = read_frame(dp, W * H * 2)
        if rb is None or db is None:
            break
        rgb = np.frombuffer(rb, np.uint8).reshape(H, W, 3).copy()
        u16 = np.frombuffer(db, "<u2").reshape(H, W)
        depth_m = decode(u16)
        valid = (depth_m > 0) & (depth_m < dmax * 0.5)     # 排除天空(=dmax)/无效

        if mode == "verify":
            dlog = np.log(np.maximum(depth_m, 0.0) + 1.0)
            if valid.any():
                p2 = float(np.percentile(dlog[valid], 2))
                p98 = float(np.percentile(dlog[valid], 98))
            else:
                p2, p98 = 0.0, 1.0
            dn = np.clip((dlog - p2) / max(p98 - p2, 1e-6) * 255.0, 0, 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(dn, cv2.COLORMAP_TURBO)
            edges = cv2.Canny(dn, int(args.edge_low), int(args.edge_high))
            overlay = rgb.copy(); overlay[edges > 0] = (0, 255, 255)   # 黄
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            tiles = [(rgb, f"F{n} RGB"), (depth_color, f"F{n} Depth(log)"),
                     (overlay, f"F{n} RGB+DepthEdges"), (edges_bgr, f"F{n} DepthEdges")]
            small = [_label(cv2.resize(im, (pw, ph)), txt) for im, txt in tiles]
            frame = np.hstack(small)
        else:
            vmax = args.max_depth if args.max_depth > 0 else (
                float(np.percentile(depth_m[valid], 95)) if valid.any() else dmax)
            vmax = max(vmax, dmin + 1e-3)
            norm = np.clip((depth_m - dmin) / (vmax - dmin), 0.0, 1.0)
            if mode == "sbs":
                dc = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                dc[~valid] = 0
                frame = np.concatenate([rgb, dc], axis=1)
            elif mode == "blend":
                dc = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                dc[~valid] = 0
                a = args.alpha; frame = rgb.copy()
                frame[valid] = (rgb[valid].astype(np.float32) * (1 - a) +
                                dc[valid].astype(np.float32) * a).astype(np.uint8)
            else:  # gray
                gray3 = cv2.cvtColor((norm * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                a = args.alpha; frame = rgb.copy()
                frame[valid] = (rgb[valid].astype(np.float32) * (1 - a) +
                                gray3[valid].astype(np.float32) * a).astype(np.uint8)
                e = depth_edges(depth_m, valid, args.edge_rel, kernel, args.edge_thick)
                frame[e > 0] = edge_color

        enc_p.stdin.write(frame.tobytes())
        n += 1
        if n % 30 == 0:
            pct = f" ({100 * n // total}%)" if total else ""
            print((f"  {n}/{total}{pct}" if total else f"  {n} frames"), flush=True)

    if enc_p.stdin:
        enc_p.stdin.close()
    enc_p.wait()
    for pr in (rp, dp):
        try:
            if pr.stdout:
                pr.stdout.close()
            pr.wait()
        except Exception:
            pass
    if n == 0:
        raise SystemExit("no frames written (rgb/depth 为空或不匹配)")
    print(f"wrote {out_path} ({n} frames)")


if __name__ == "__main__":
    main()
