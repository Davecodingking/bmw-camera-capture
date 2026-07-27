"""
Direct BMWCameraCapture .bin -> 3D Gaussian Splatting PLY, numpy-only.

Same math/format as dataset_to_ply.py --gaussian and conver_bmw.save_3dgs_ply,
but reads the raw color/depth .bin straight from disk (no rgb.mp4/depth.mkv
round-trip), so it needs neither ffmpeg nor OpenCV. Handy for a quick look at a
couple of frames as splats (open the .ply in https://superspl.at/editor).

  python bin_to_splat.py --csv <BMWCameraCapture/bmw_camera_pos.csv>
      [--out-dir splat_out] [--start-index 0] [--count 2]
      [--stride 4] [--min-depth 0.2] [--max-depth 200] [--scale-factor 0.6]
      [--combined]

By default writes one splat PLY per frame (2 consecutive frames). --combined
also writes a single merged PLY of all selected frames.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

# --- reprojection constants ---
INITIAL_OPACITY_LOGIT = 5.0
SH_C0 = 0.28209479177387814
NEGATE_CSV_FWD_FOR_LOOK = True
# Screen-axis signs for Black Myth / UE captures. NOTE: these are the OPPOSITE of
# the GTA/RAGE tuning (conver_bmw's defaults are +1/-1). Using +1/-1 here makes
# multi-frame splats of a static scene NOT overlap (verified: 45% vs 87% overlap
# for a pure-rotation pair). This file is BMW-only, so we hardcode the BMW signs.
SCREEN_X_RIGHT_SIGN = -1.0
SCREEN_Y_UP_SIGN = 1.0
PIXEL_CENTER_OFFSET = 0.5


def normalize(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def fget(row, key):
    return float(row[key])


# --- decoders (mirror conver_bmw.decode_direct_*_bytes) ---
def decode_depth(raw_bytes, width, height, row_pitch, tight_row_pitch, near_m):
    if tight_row_pitch <= 0:
        tight_row_pitch = row_pitch
    bpp = tight_row_pitch // width
    expected = row_pitch * height
    rows = np.frombuffer(raw_bytes[:expected], np.uint8).reshape(height, row_pitch)
    tight = rows[:, :width * bpp].copy()
    if bpp == 2:
        raw = tight.view("<u2").reshape(height, width).astype(np.float64) / 65535.0
    elif bpp == 8:
        raw = tight.reshape(height, width, 8)[:, :, :4].copy().view("<f4").reshape(height, width).astype(np.float64)
    elif bpp == 4:
        raw = tight.view("<f4").reshape(height, width).astype(np.float64)
    else:
        raise ValueError(f"unsupported depth bpp {bpp}")
    raw = np.clip(raw, 1e-9, 1.0)
    return near_m / raw   # reversed-Z -> meters


def decode_color(raw_bytes, width, height, row_pitch, tight_row_pitch, format_id):
    if tight_row_pitch <= 0:
        tight_row_pitch = row_pitch
    bpp = tight_row_pitch // width
    expected = row_pitch * height
    rows = np.frombuffer(raw_bytes[:expected], np.uint8).reshape(height, row_pitch)
    tight = rows[:, :width * bpp].copy()
    if format_id in (23, 24, 25) or (format_id == 0 and bpp == 4):
        packed = tight.view("<u4").reshape(height, width)
        r = (packed & 0x3FF).astype(np.float32) / 1023.0
        g = ((packed >> 10) & 0x3FF).astype(np.float32) / 1023.0
        b = ((packed >> 20) & 0x3FF).astype(np.float32) / 1023.0
        return np.clip(np.stack([b, g, r], axis=2) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if format_id in (27, 28, 29, 30) and bpp == 4:
        return tight.reshape(height, width, 4)[:, :, [2, 1, 0]].copy()
    if format_id in (87, 90, 91, 88, 92, 93) and bpp == 4:
        return tight.reshape(height, width, 4)[:, :, :3].copy()
    raise ValueError(f"unsupported color format={format_id} bpp={bpp}")


def load_bin_and_meta(path, fallback):
    meta = dict(fallback)
    jp = path.with_suffix(".json")
    if jp.exists():
        j = json.load(open(jp, "r", encoding="utf-8"))
        for k in ("width", "height", "row_pitch", "tight_row_pitch", "format"):
            if k in j:
                meta[k] = int(j[k])
    return path.read_bytes(), meta


def save_3dgs_ply(positions, colors_rgb, depths, fx, fy, path, stride, scale_factor):
    n = positions.shape[0]
    favg = (fx + fy) / 2.0
    rgb_norm = colors_rgb.astype(np.float32) / 255.0
    f_dc = (rgb_norm - 0.5) / SH_C0
    pixel_world_size = np.maximum(depths / favg * stride * scale_factor, 1e-8)
    log_scale = np.log(pixel_world_size).astype(np.float32)
    scales = np.stack([log_scale, log_scale, log_scale], axis=1)
    opacity = np.full((n, 1), INITIAL_OPACITY_LOGIT, dtype=np.float32)
    rotations = np.zeros((n, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    normals = np.zeros((n, 3), dtype=np.float32)
    data = np.concatenate([positions, normals, f_dc, opacity, scales, rotations], axis=1).astype(np.float32)
    header = [
        "ply", "format binary_little_endian 1.0", f"element vertex {n}",
        "property float x", "property float y", "property float z",
        "property float nx", "property float ny", "property float nz",
        "property float f_dc_0", "property float f_dc_1", "property float f_dc_2",
        "property float opacity",
        "property float scale_0", "property float scale_1", "property float scale_2",
        "property float rot_0", "property float rot_1", "property float rot_2", "property float rot_3",
        "end_header",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fp:
        fp.write(("\n".join(header) + "\n").encode("ascii"))
        data.tofile(fp)
    return path.stat().st_size / 1e6


def unproject(row, base_dir, stride, min_d, max_d):
    color_file = row["color_file"]
    depth_file = row["depth_file"]
    if not color_file or not depth_file:
        return None

    cbytes, cmeta = load_bin_and_meta(base_dir / color_file, {
        "width": int(row["color_width"]), "height": int(row["color_height"]),
        "row_pitch": int(row["color_row_pitch"]), "tight_row_pitch": int(row["color_tight_row_pitch"]),
        "format": int(row["color_format"])})
    dbytes, dmeta = load_bin_and_meta(base_dir / depth_file, {
        "width": int(row["depth_width"]), "height": int(row["depth_height"]),
        "row_pitch": int(row["depth_row_pitch"]), "tight_row_pitch": int(row["depth_tight_row_pitch"])})

    near_m = fget(row, "near_cm") * 0.01
    rgb_bgr = decode_color(cbytes, cmeta["width"], cmeta["height"], cmeta["row_pitch"], cmeta["tight_row_pitch"], cmeta["format"])
    depth_m = decode_depth(dbytes, dmeta["width"], dmeta["height"], dmeta["row_pitch"], dmeta["tight_row_pitch"], near_m)
    if rgb_bgr.shape[:2] != depth_m.shape:
        raise ValueError(f"rgb {rgb_bgr.shape[:2]} != depth {depth_m.shape}")

    h, w = depth_m.shape
    fx = (w / 2.0) / math.tan(math.radians(fget(row, "hfov_deg")) / 2.0)
    fy = (h / 2.0) / math.tan(math.radians(fget(row, "vfov_deg")) / 2.0)
    cx, cy = w / 2.0, h / 2.0

    vs = np.arange(0, h, stride)
    us = np.arange(0, w, stride)
    uu, vv = np.meshgrid(us, vs)
    d = depth_m[vv, uu]
    bgr = rgb_bgr[vv, uu]
    mask = (d > min_d) & (d < max_d) & np.isfinite(d)
    d = d[mask].astype(np.float64)
    bgr = bgr[mask]
    uu = uu[mask].astype(np.float64)
    vv = vv[mask].astype(np.float64)

    ray_x = (uu + PIXEL_CENTER_OFFSET - cx) / fx
    ray_y = (vv + PIXEL_CENTER_OFFSET - cy) / fy

    pos = np.array([fget(row, "pos_x_m"), fget(row, "pos_y_m"), fget(row, "pos_z_m")])
    right = normalize(np.array([fget(row, "right_x"), fget(row, "right_y"), fget(row, "right_z")]))
    up = normalize(np.array([fget(row, "up_x"), fget(row, "up_y"), fget(row, "up_z")]))
    fwd = np.array([fget(row, "fwd_x"), fget(row, "fwd_y"), fget(row, "fwd_z")])
    look = normalize(-fwd if NEGATE_CSV_FWD_FOR_LOOK else fwd)

    dirs = (look[None, :]
            + right[None, :] * (ray_x * SCREEN_X_RIGHT_SIGN)[:, None]
            + up[None, :] * (ray_y * SCREEN_Y_UP_SIGN)[:, None])
    pts = pos[None, :] + d[:, None] * dirs
    P = pts.astype(np.float32)
    C = bgr[:, [2, 1, 0]].astype(np.uint8)   # BGR -> RGB
    D = d.astype(np.float32)
    return P, C, D, fx, fy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--count", type=int, default=2)
    ap.add_argument("--indices", type=str, default=None,
                    help="comma-separated CSV row indices (e.g. 2,5); overrides --start-index/--count")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--min-depth", type=float, default=0.2)
    ap.add_argument("--max-depth", type=float, default=200.0)
    ap.add_argument("--scale-factor", type=float, default=0.6)
    ap.add_argument("--combined", action="store_true")
    a = ap.parse_args()

    base_dir = a.csv.parent
    out_dir = a.out_dir or (base_dir / "splats")
    with open(a.csv, newline="") as fp:
        rows = list(csv.DictReader(fp))
    if a.indices:
        idxs = [int(x) for x in a.indices.split(",") if x.strip() != ""]
    else:
        idxs = list(range(a.start_index, a.start_index + a.count))
    pairs = [(ix, rows[ix]) for ix in idxs if 0 <= ix < len(rows)]
    if not pairs:
        raise SystemExit(f"no valid rows in {idxs} (csv has {len(rows)} rows)")

    merged_P, merged_C, merged_D = [], [], []
    fx = fy = None
    for idx, row in pairs:
        res = unproject(row, base_dir, a.stride, a.min_depth, a.max_depth)
        if res is None:
            print(f"frame idx {idx} (frame {row['frame']}): missing color/depth file, skipped")
            continue
        P, C, D, fx, fy = res
        out = out_dir / f"splat_idx{idx:03d}_frame{int(row['frame']):08d}.ply"
        mb = save_3dgs_ply(P, C, D, fx, fy, out, a.stride, a.scale_factor)
        print(f"wrote {out} ({len(P):,} splats, {mb:.1f} MB)")
        merged_P.append(P); merged_C.append(C); merged_D.append(D)

    if a.combined and merged_P:
        P = np.concatenate(merged_P); C = np.concatenate(merged_C); D = np.concatenate(merged_D)
        label = "-".join(str(ix) for ix, _ in pairs)
        out = out_dir / f"splat_combined_idx{label}.ply"
        mb = save_3dgs_ply(P, C, D, fx, fy, out, a.stride, a.scale_factor)
        print(f"wrote {out} ({len(P):,} splats, {mb:.1f} MB) [combined]")


if __name__ == "__main__":
    main()
