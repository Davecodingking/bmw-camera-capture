"""
从 rgb.mp4 + depth.mkv + camera_params.json 反投影成世界坐标点云 PLY。
默认输出彩色点云(xyz+rgb); 加 --gaussian 输出 3D Gaussian Splatting PLY(可在 superspl.at/editor 打开)。
默认把选到的帧合并成一个 PLY; 加 --per-frame 则每帧单独一个 PLY(不合并)。

  python dataset_to_ply.py --dataset-dir <dir>
      [--stride 4] [--frame-step 10] [--start-frame 0] [--max-frames 0]
      [--min-depth 0.2] [--max-depth 200] [--voxel 0] [--per-frame]
      [--gaussian] [--scale-factor 0.6] [--out xxx.ply]

原理(与 conver_bmw 一致):
  深度按 camera_params 的 encoding 解码(log_16bit / linear_16bit)。
  ray_x=(u+0.5-cx)/fx, ray_y=(v+0.5-cy)/fy
  X_world = pos + depth * (look + right*ray_x*SX + up*ray_y*SY)   # SX=+1, SY=-1
  丢弃 depth<min 或 >max(天空/远景)。

  --stride 像素下采样  --frame-step 每N帧取1  --start-frame 起始帧  --max-frames 取N帧
  --per-frame 每帧单独输出(文件名 <out_stem>_<帧号>.ply)   --voxel >0 体素(米)去重
  --gaussian 输出3DGS(每点=高斯splat)   --scale-factor gaussian splat尺度(默认0.6)
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

import conver_bmw   # 复用其 save_3dgs_ply (3D Gaussian Splatting 输出)

SX, SY, OFF = 1.0, -1.0, 0.5   # SCREEN_X_RIGHT_SIGN, SCREEN_Y_UP_SIGN, PIXEL_CENTER_OFFSET


def reader(path, fmt):
    return subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", fmt, "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def make_decoder(params):
    dmin = float(params["depth_min_m"]); dmax = float(params["depth_max_m"])
    enc = params.get("encoding", "linear_16bit")
    if enc == "log_16bit":
        lo = np.log(dmin + 1.0); hi = np.log(dmax + 1.0)
        return lambda u: np.exp(u.astype(np.float64) / 65535.0 * (hi - lo) + lo) - 1.0

    def lin(u):
        d = (u.astype(np.float64) - 1.0) / 65534.0 * (dmax - dmin) + dmin
        d[u == 0] = 0.0
        return d
    return lin


def write_rgb_ply(P, C, out_path):
    rec = np.empty(len(P), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"] = P[:, 0]; rec["y"] = P[:, 1]; rec["z"] = P[:, 2]
    rec["red"] = C[:, 0]; rec["green"] = C[:, 1]; rec["blue"] = C[:, 2]
    with open(out_path, "wb") as fp:
        fp.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                  "end_header\n").encode())
        fp.write(rec.tobytes())


def write_cloud(P, C, D, out_path, gaussian, stride, scale_factor, intr):
    if gaussian:
        mb = conver_bmw.save_3dgs_ply(P, C, D, (intr["fx"], intr["fy"]), out_path, stride, scale_factor)
        print(f"wrote {out_path} ({len(P):,} splats, {mb:.1f} MB) [3DGS]")
    else:
        write_rgb_ply(P, C, out_path)
        print(f"wrote {out_path} ({len(P):,} points, {out_path.stat().st_size/1e6:.1f} MB)")


def voxel_downsample(P, C, D, voxel):
    keys = np.floor(P / voxel).astype(np.int64)
    _, uidx = np.unique(keys, axis=0, return_index=True)
    return P[uidx], C[uidx], D[uidx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--stride", type=int, default=4, help="像素下采样(每N像素取1)")
    ap.add_argument("--frame-step", type=int, default=10, help="每N帧取1")
    ap.add_argument("--start-frame", type=int, default=0, help="起始帧号(dataset 帧序)")
    ap.add_argument("--min-depth", type=float, default=0.2)
    ap.add_argument("--max-depth", type=float, default=200.0, help="超过此深度(米)丢弃(天空/远景)")
    ap.add_argument("--voxel", type=float, default=0.0, help=">0 按体素(米)下采样去重")
    ap.add_argument("--max-frames", type=int, default=0, help=">0 只处理前N个(采样后)帧")
    ap.add_argument("--per-frame", action="store_true", help="每帧单独输出一个PLY(不合并)")
    ap.add_argument("--gaussian", action="store_true", help="输出 3D Gaussian Splatting PLY(可在 superspl.at 打开)")
    ap.add_argument("--scale-factor", type=float, default=0.6, help="gaussian splat 尺度系数")
    a = ap.parse_args()

    d = a.dataset_dir
    params = json.load(open(d / "camera_params.json", "r", encoding="utf-8"))
    W = int(params["width"]); H = int(params["height"])
    frames = params["frames"]
    dec = make_decoder(params)
    # Screen-axis signs: BMW/UE datasets write -1/+1; GTA and old datasets default
    # to +1/-1. Read from camera_params.json so reprojection matches the capture.
    sx = float(params.get("screen_x_sign", SX))
    sy = float(params.get("screen_y_sign", SY))
    out = a.out or (d / ("point_cloud_3dgs.ply" if a.gaussian else "point_cloud.ply"))

    rp = reader(d / "rgb.mp4", "bgr24")
    dp = reader(d / "depth.mkv", "gray16le")
    rgb_n = W * H * 3; dep_n = W * H * 2

    us = np.arange(0, W, a.stride); vs = np.arange(0, H, a.stride)
    uu, vv = np.meshgrid(us, vs)
    uu = uu.ravel().astype(np.float64); vv = vv.ravel().astype(np.float64)

    all_pts = []; all_col = []; all_dep = []
    idx = 0; used = 0
    while True:
        rb = rp.stdout.read(rgb_n); db = dp.stdout.read(dep_n)
        if len(rb) < rgb_n or len(db) < dep_n:
            break
        if idx < a.start_frame or (idx - a.start_frame) % a.frame_step != 0 or idx >= len(frames):
            idx += 1
            continue
        f = frames[idx]
        intr = f["intrinsics"]; fx = intr["fx"]; fy = intr["fy"]; cx = intr["cx"]; cy = intr["cy"]
        rgb = np.frombuffer(rb, np.uint8).reshape(H, W, 3)
        u16 = np.frombuffer(db, "<u2").reshape(H, W)
        depth = dec(u16[np.ix_(vs, us)]).ravel()
        col = rgb[np.ix_(vs, us)].reshape(-1, 3)
        m = (depth > a.min_depth) & (depth < a.max_depth) & np.isfinite(depth)
        if m.any():
            dd = depth[m]
            rx = (uu[m] + OFF - cx) / fx
            ry = (vv[m] + OFF - cy) / fy
            pos = np.array([f["camera_position_m"][k] for k in "xyz"])
            right = np.array(f["right"]); up = np.array(f["up"]); look = np.array(f["look"])
            dirs = (look[None, :]
                    + right[None, :] * (rx * sx)[:, None]
                    + up[None, :] * (ry * sy)[:, None])
            pts = pos[None, :] + dd[:, None] * dirs
            P = pts.astype(np.float32)
            C = col[m][:, [2, 1, 0]].astype(np.uint8)          # BGR -> RGB
            Dd = dd.astype(np.float32)                         # 相机 z 距离(给 gaussian scale 用)
            if a.per_frame:
                if a.voxel > 0:
                    P, C, Dd = voxel_downsample(P, C, Dd, a.voxel)
                fpath = out.parent / f"{out.stem}_{idx:06d}{out.suffix}"
                write_cloud(P, C, Dd, fpath, a.gaussian, a.stride, a.scale_factor, intr)
            else:
                all_pts.append(P); all_col.append(C); all_dep.append(Dd)
            used += 1
            if not a.per_frame and used % 50 == 0:
                print(f"  {used} frames, {sum(len(p) for p in all_pts)} pts")
            if a.max_frames and used >= a.max_frames:
                break
        idx += 1

    for pr in (rp, dp):
        try:
            pr.stdout.close(); pr.wait()
        except Exception:
            pass

    if a.per_frame:
        print(f"done: {used} per-frame PLY(s)")
        return

    if not all_pts:
        raise SystemExit("no points produced")
    P = np.concatenate(all_pts); C = np.concatenate(all_col); D = np.concatenate(all_dep)
    print(f"used {used} frames -> {len(P):,} points (before voxel)")
    if a.voxel > 0:
        P, C, D = voxel_downsample(P, C, D, a.voxel)
        print(f"after voxel {a.voxel}m: {len(P):,} points")
    intr = frames[0]["intrinsics"]
    write_cloud(P, C, D, out, a.gaussian, a.stride, a.scale_factor, intr)


if __name__ == "__main__":
    main()
