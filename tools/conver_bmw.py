"""
Black Myth: Wukong camera/depth/color captures -> Inria 3D Gaussian Splat PLY.

Preferred inputs:
  - bmw_camera_pos.csv from BMWCameraCapture.addon64
  - add-on GPU readback files:
      bmw_camera_frame_XXXXXXXX.depth.bin/.json
      bmw_camera_frame_XXXXXXXX.color.bin/.json

Fallback input:
  - ReShade screenshots containing QuadViewRaw output:
      top-left     = color
      top-right    = raw depth high byte
      bottom-left  = raw depth middle byte
      bottom-right = raw depth low byte

Depth model:
  UE reversed-Z infinite projection stores raw ~= near / z_camera.
  With near_cm from CSV, z_camera_m = near_m / raw.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = DEFAULT_ROOT / "bmw_camera_pos.csv"
DEFAULT_SCREEN_DIR = DEFAULT_ROOT / "screen"
DEFAULT_OUT_DIR = DEFAULT_ROOT / "ply"
DEFAULT_CANDIDATE_CSV_NAME = "bmw_camera_candidates.csv"

STRIDE = 1
MIN_DEPTH_M = 0.1
MAX_DEPTH_M = 100.0
INITIAL_OPACITY_LOGIT = 5.0
SCALE_FACTOR = 0.6
SH_C0 = 0.28209479177387814

# The UE ViewToWorld row named "fwd" in the CSV is opposite to the visible
# camera look direction for these captures.
NEGATE_CSV_FWD_FOR_LOOK = True

# GTA/RAGE captures use the opposite screen-basis convention from the BMW/UE
# captures this script was copied from. Cross-frame depth reprojection on the
# GTA sample frames showed the stable mapping is:
#   screen +X -> +right
#   screen +Y -> -up
SCREEN_X_RIGHT_SIGN = 1.0
SCREEN_Y_UP_SIGN = -1.0

# Pixel indices address the top-left corner of each pixel. Use the pixel center
# for projection math to avoid a small, consistent sub-pixel reprojection bias.
PIXEL_CENTER_OFFSET = 0.5

# Wrong-camera gate: if the CBV-sniffed forward and the IGCS engine-camera forward
# differ by more than this many degrees, the sniff locked onto the wrong camera that
# frame -> drop it. Good frames agree to <1 deg; wrong-camera frames diverge 15+ deg,
# so there is a wide safe margin.
WRONG_CAMERA_MAX_FWD_DEG = 3.0

# When BMWCameraCapture writes bmw_camera_candidates.csv, a single frame can
# contain many UE ViewUniform-shaped CBVs. The stable main camera is the pose
# that appears most often in that frame, not necessarily the first CBV that the
# add-on matched for bmw_camera_pos.csv.
DOMINANT_POSE_ROUND_DECIMALS = 4
DOMINANT_ROOT_PREFERENCE = ("3", "12", "8", "4")

CSV_COLUMNS = [
    "frame",
    "root_param",
    "draw_count",
    "resource",
    "cb_offset_hex",
    "pos_x_cm",
    "pos_y_cm",
    "pos_z_cm",
    "pos_x_m",
    "pos_y_m",
    "pos_z_m",
    "right_x",
    "right_y",
    "right_z",
    "up_x",
    "up_y",
    "up_z",
    "fwd_x",
    "fwd_y",
    "fwd_z",
    "hfov_deg",
    "vfov_deg",
    "near_cm",
    "projection_offset_hex",
    "view_origin_offset_hex",
    "view_to_world_offset_hex",
    "screenshot_postfix",
]

BASE_CSV_COLUMN_COUNT = len(CSV_COLUMNS)

CSV_COLUMNS += [f"proj_m{row}{col}" for row in range(4) for col in range(4)]
CSV_COLUMNS += [f"view_m3{col}" for col in range(4)]
CSV_COLUMNS += [
    "depth_file",
    "depth_width",
    "depth_height",
    "depth_format",
    "depth_row_pitch",
    "depth_tight_row_pitch",
    "depth_byte_size",
    "depth_samples",
]
CSV_COLUMNS += [
    "color_file",
    "color_width",
    "color_height",
    "color_format",
    "color_row_pitch",
    "color_tight_row_pitch",
    "color_byte_size",
    "color_samples",
]
CSV_COLUMNS += [
    "camera_frame",
    "depth_frame",
    "color_frame",
    "strict_sync_ok",
]

# Exact column order written by GTACameraCapture.addon64 (write_csv_header).
# Used to parse a GTA CSV that has no header line (the addon only writes the
# header when the file is new/empty, so an appended or cleared file can lack it).
GTA_CSV_COLUMNS = [
    "frame", "root_param", "draw_count", "resource", "cb_offset_hex",
    "pos_x", "pos_y", "pos_z", "pos_x_0p01", "pos_y_0p01", "pos_z_0p01",
    "right_x", "right_y", "right_z", "up_x", "up_y", "up_z",
    "fwd_x", "fwd_y", "fwd_z",
    "hfov_deg", "vfov_deg", "projection_depth_param",
    "projection_offset_hex", "view_origin_offset_hex", "view_to_world_offset_hex",
    "screenshot_postfix",
]
GTA_CSV_COLUMNS += [f"proj_m{r}{c}" for r in range(4) for c in range(4)]
GTA_CSV_COLUMNS += [f"view_m3{c}" for c in range(4)]
GTA_CSV_COLUMNS += [f"world_to_view_m{r}{c}" for r in range(4) for c in range(4)]
GTA_CSV_COLUMNS += [f"world_to_clip_m{r}{c}" for r in range(4) for c in range(4)]
GTA_CSV_COLUMNS += [f"view_to_world_m{r}{c}" for r in range(4) for c in range(4)]
GTA_CSV_COLUMNS += [
    "depth_file", "depth_width", "depth_height", "depth_format",
    "depth_row_pitch", "depth_tight_row_pitch", "depth_byte_size", "depth_samples",
    "color_file", "color_width", "color_height", "color_format",
    "color_row_pitch", "color_tight_row_pitch", "color_byte_size", "color_samples",
    "camera_frame", "depth_frame", "color_frame", "strict_sync_ok",
    "segment", "seg_start",
]

CANDIDATE_CSV_COLUMNS = [
    "frame",
    "candidate_index",
    "selected_camera_sample",
    "root_param",
    "draw_count",
    "pipeline_layout",
    "resource",
    "cb_offset_hex",
    "pos_x_cm",
    "pos_y_cm",
    "pos_z_cm",
    "pos_x_m",
    "pos_y_m",
    "pos_z_m",
    "right_x",
    "right_y",
    "right_z",
    "up_x",
    "up_y",
    "up_z",
    "fwd_x",
    "fwd_y",
    "fwd_z",
    "hfov_deg",
    "vfov_deg",
    "near_cm",
    "projection_offset_hex",
    "view_origin_offset_hex",
    "view_to_world_offset_hex",
]
CANDIDATE_CSV_COLUMNS += [f"proj_m{row}{col}" for row in range(4) for col in range(4)]
CANDIDATE_CSV_COLUMNS += [f"view_m3{col}" for col in range(4)]
CANDIDATE_CSV_COLUMNS += [
    "depth_resource",
    "depth_width",
    "depth_height",
    "depth_format",
    "depth_samples",
]


def parse_float(row, name):
    return float(row[name].strip())


def parse_int(row, name):
    return int(row[name].strip())


def parse_optional_int(row, name, default=0):
    value = row.get(name, "").strip()
    if not value:
        return default
    return int(value, 0)


def load_camera_csv(path):
    # Inspect the first non-empty line: a header (starts with "frame") or data.
    header = None
    first_data_len = 0
    with open(path, "r", encoding="utf-8-sig", newline="") as fp:
        for cells in csv.reader(fp):
            if not cells or not any(c.strip() for c in cells):
                continue
            if cells[0].strip() == "frame":
                header = [c.strip() for c in cells]
            else:
                first_data_len = len(cells)
            break

    # Headerless GTA CSV: the addon only writes the header when the file is
    # new/empty, so an appended or cleared file can have none. Detect by the exact
    # GTA column count and supply the known column names.
    if header is None and first_data_len == len(GTA_CSV_COLUMNS):
        header = list(GTA_CSV_COLUMNS)
    elif header is None and first_data_len == len(GTA_CSV_COLUMNS) - 2:
        # older headerless CSV written before segment/seg_start columns were added
        header = list(GTA_CSV_COLUMNS[:-2])
    elif header is None and first_data_len not in (len(CSV_COLUMNS), BASE_CSV_COLUMN_COUNT):
        # 不认识的列数：以前会硬按 BMW 的 CSV_COLUMNS 去 zip，depth_file(GTA 第 96 列)
        # 会被读成投影矩阵里的一个浮点数，于是每帧都报 "missing rgb/depth readback"，
        # 而位姿列还“看着正常”——排查起来极其误导。现在直接失败。
        raise SystemExit(
            f"{path}: 无法识别的 CSV 格式。\n"
            f"  首行不是表头(frame,...)，数据行有 {first_data_len} 列；"
            f"GTA 应为 {len(GTA_CSV_COLUMNS)} 或 {len(GTA_CSV_COLUMNS) - 2} 列，"
            f"BMW 应为 {len(CSV_COLUMNS)} 列。\n"
            f"  常见原因：采集插件版本与本工具不一致 / CSV 被截断或人工合并过 / 末行写了一半。\n"
            f"  （强行解析会让 depth_file 列错位，表现为每帧都 missing rgb/depth readback）"
        )

    # GTA CSV uses pos_x/pos_y/pos_z (meters) and projection_depth_param (meters).
    # BMW CSV uses pos_x_cm/m and near_cm (cm; *0.01 -> meters).
    is_gta = header is not None and "projection_depth_param" in header

    if header is not None and "depth_file" not in header:
        print(f"warning: {path} 的表头共 {len(header)} 列但没有 depth_file 列；"
              f"除非走 QuadView 截图流程，否则导出会 100% 报 missing rgb/depth readback",
              file=sys.stderr)

    rows = []
    short_rows = 0
    with open(path, "r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.reader(fp)
        for cells in reader:
            if not cells or not any(cell.strip() for cell in cells):
                continue
            if cells[0].strip() == "frame":
                continue
            if header is not None and len(cells) >= len(header):
                rows.append(dict(zip(header, cells)))
            elif header is None and len(cells) >= BASE_CSV_COLUMN_COUNT:
                rows.append(dict(zip(CSV_COLUMNS, cells)))
            else:
                short_rows += 1

    frames = []
    sync_dropped = 0
    parse_dropped = 0
    igcs_used = 0
    wrong_cam_dropped = 0
    for row in rows:
        try:
            strict_sync = row.get("strict_sync_ok", "").strip().lower()
            if strict_sync and strict_sync not in ("1", "true", "yes"):
                sync_dropped += 1
                continue

            if is_gta:
                pos = np.array([
                    parse_float(row, "pos_x"),
                    parse_float(row, "pos_y"),
                    parse_float(row, "pos_z"),
                ], dtype=np.float64)
                near_m = parse_float(row, "projection_depth_param")
            else:
                pos = np.array([
                    parse_float(row, "pos_x_m"),
                    parse_float(row, "pos_y_m"),
                    parse_float(row, "pos_z_m"),
                ], dtype=np.float64)
                near_m = parse_float(row, "near_cm") * 0.01
            right = np.array([
                parse_float(row, "right_x"),
                parse_float(row, "right_y"),
                parse_float(row, "right_z"),
            ], dtype=np.float64)
            up = np.array([
                parse_float(row, "up_x"),
                parse_float(row, "up_y"),
                parse_float(row, "up_z"),
            ], dtype=np.float64)
            fwd = np.array([
                parse_float(row, "fwd_x"),
                parse_float(row, "fwd_y"),
                parse_float(row, "fwd_z"),
            ], dtype=np.float64)

            look = -fwd if NEGATE_CSV_FWD_FOR_LOOK else fwd

            frames.append({
                "frame": parse_int(row, "frame"),
                "root_param": parse_optional_int(row, "root_param"),
                "draw_count": parse_optional_int(row, "draw_count"),
                "resource": row.get("resource", "").strip(),
                "cb_offset_hex": row.get("cb_offset_hex", "").strip(),
                "pos": pos,
                "right": normalize(right),
                "up": normalize(up),
                "look": normalize(look),
                "hfov_deg": parse_float(row, "hfov_deg"),
                "vfov_deg": parse_float(row, "vfov_deg"),
                "near_m": near_m,
                "screenshot_postfix": row.get("screenshot_postfix", "").strip(),
                "depth_file": row.get("depth_file", "").strip(),
                "depth_width": parse_optional_int(row, "depth_width"),
                "depth_height": parse_optional_int(row, "depth_height"),
                "depth_format": parse_optional_int(row, "depth_format"),
                "depth_row_pitch": parse_optional_int(row, "depth_row_pitch"),
                "depth_tight_row_pitch": parse_optional_int(row, "depth_tight_row_pitch"),
                "depth_byte_size": parse_optional_int(row, "depth_byte_size"),
                "depth_samples": parse_optional_int(row, "depth_samples"),
                "color_file": row.get("color_file", "").strip(),
                "color_width": parse_optional_int(row, "color_width"),
                "color_height": parse_optional_int(row, "color_height"),
                "color_format": parse_optional_int(row, "color_format"),
                "color_row_pitch": parse_optional_int(row, "color_row_pitch"),
                "color_tight_row_pitch": parse_optional_int(row, "color_tight_row_pitch"),
                "color_byte_size": parse_optional_int(row, "color_byte_size"),
                "color_samples": parse_optional_int(row, "color_samples"),
                "segment": parse_optional_int(row, "segment"),
                "seg_start": parse_optional_int(row, "seg_start"),
                # Screen-axis signs used by unproject_frame. BMW (UE) needs the
                # opposite of the GTA/RAGE tuning, otherwise a static scene's frames
                # do not overlap when reprojected to world space.
                "screen_x_sign": SCREEN_X_RIGHT_SIGN if is_gta else -1.0,
                "screen_y_sign": SCREEN_Y_UP_SIGN if is_gta else 1.0,
                "camera_source": "csv",
            })

            # Use the MATRIX camera (recorded projection + view_to_world) for BMW/UE
            # when those columns are present. This is the game-agnostic ground truth:
            # verified correct for rotation AND translation on both Black Myth: Wukong
            # and Clair Obscur: Expedition 33. It supersedes both (a) the old right/up/
            # look + screen-sign method (which only aligns pure-rotation; it is a mirror
            # image of the correct result so it fails under translation) and (b) the IGCS
            # engine camera (which is per-game unreliable: UUU's reported FOV / forward
            # convention did not match Expedition 33's actual render, giving fanned-out
            # misalignment even at green light). The recorded matrices always match what
            # the game rendered. We keep the existing sign formula and just bake the
            # matrix-equivalent camera into it:
            #   matrix:  World = -pos_m + d*(fwd + rx*R - ry*U)   [pos_m is UE PreViewTranslation = -origin]
            #   formula: World =  pos   + d*(look + sx*rx*R + sy*ry*U)
            #   => pos=-pos_m, look=+fwd, R,U unchanged, sx=+1, sy=-1.
            # hfov_deg/vfov_deg already come from the projection matrix (addon computed
            # them as 2*atan(1/p00) / 2*atan(1/p11)), so intrinsics need no change.
            if (not is_gta
                    and row.get("view_to_world_m00", "").strip() != ""
                    and row.get("proj_m00", "").strip() != ""):
                f = frames[-1]
                f["pos"] = -pos                       # -PreViewTranslation = camera origin
                f["look"] = normalize(fwd)            # +forward (view_to_world row 2), NOT negated
                f["screen_x_sign"] = 1.0
                f["screen_y_sign"] = -1.0
                f["camera_source"] = "matrix"
                igcs_used += 1

                # WRONG-CAMERA GATE (this is what IGCS is FOR). The CBV sniff can lock
                # onto a different camera on some frames — one at the SAME origin but a
                # different orientation (e.g. a cubemap/reflection/shadow pass), so a
                # position check misses it. The IGCS engine camera (UUU) gives the true
                # main-camera orientation, so compare the sniffed forward against the
                # IGCS forward: they agree to <1 deg on good frames and diverge 15-27 deg
                # on wrong-camera frames (verified on Wukong: 5/18 frames were wrong).
                # Such a frame's sniffed matrices are for the wrong camera, so the matrix
                # reconstruction of it is garbage and there is no correct projection to
                # fall back to -> DROP it. Only possible when IGCS is present; without it
                # we cannot tell, so we keep the frame (degrades to trusting the sniff).
                if row.get("igcs_ok", "").strip() == "1" and row.get("igcs_fwd_x", "").strip() != "":
                    gf = np.array([parse_float(row, "igcs_fwd_x"),
                                   parse_float(row, "igcs_fwd_y"),
                                   parse_float(row, "igcs_fwd_z")], dtype=np.float64)
                    sf = normalize(fwd)
                    gfn = normalize(gf)
                    ang = math.degrees(math.acos(max(-1.0, min(1.0, float(sf @ gfn)))))
                    if ang > WRONG_CAMERA_MAX_FWD_DEG:
                        frames.pop()
                        igcs_used -= 1
                        wrong_cam_dropped += 1
        except (KeyError, ValueError):
            parse_dropped += 1
            continue

    # 诊断信息一律走 stderr：setup_and_run.sh 会用 stdout 取 --list-segments 的结果。
    if short_rows:
        print(f"warning: {path}: {short_rows} 行列数少于表头"
              f"({len(header) if header is not None else '?'} 列)，已丢弃"
              f" —— 可能是采集中途插件升级或写入被截断", file=sys.stderr)
    if sync_dropped:
        print(f"warning: {path}: {sync_dropped} 行 strict_sync_ok 不为真，已丢弃",
              file=sys.stderr)
    if parse_dropped:
        print(f"warning: {path}: {parse_dropped} 行字段解析失败，已丢弃", file=sys.stderr)
    if igcs_used:
        print(f"camera: {igcs_used}/{len(frames)} 帧使用矩阵法重建(投影+view_to_world矩阵，"
              f"旋转/平移通用)，其余回退符号法", file=sys.stderr)
    if wrong_cam_dropped:
        print(f"camera: 丢弃 {wrong_cam_dropped} 帧抓错相机(嗅探朝向与 IGCS 主相机差 "
              f">{WRONG_CAMERA_MAX_FWD_DEG}°)", file=sys.stderr)

    return frames


def candidate_pose_key(row):
    names = [
        "pos_x_m", "pos_y_m", "pos_z_m",
        "right_x", "right_y", "right_z",
        "up_x", "up_y", "up_z",
        "fwd_x", "fwd_y", "fwd_z",
        "proj_m00", "proj_m11", "proj_m20", "proj_m21",
    ]
    return tuple(round(float(row[name]), DOMINANT_POSE_ROUND_DECIMALS) for name in names)


def load_candidate_rows(path):
    if not path or not path.exists():
        return {}

    by_frame = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.reader(fp)
        header = None
        for cells in reader:
            if not cells or not any(cell.strip() for cell in cells):
                continue

            if cells[0].strip() == "frame":
                header = [cell.strip() for cell in cells]
                continue

            if header is not None:
                row = dict(zip(header, cells))
            elif len(cells) >= len(CANDIDATE_CSV_COLUMNS):
                row = dict(zip(CANDIDATE_CSV_COLUMNS, cells))
            else:
                continue

            try:
                frame = int(row["frame"])
                candidate_pose_key(row)
            except (KeyError, ValueError):
                continue
            by_frame.setdefault(frame, []).append(row)
    return by_frame


def choose_dominant_candidate(rows, root_preference=DOMINANT_ROOT_PREFERENCE):
    groups = {}
    for row in rows:
        groups.setdefault(candidate_pose_key(row), []).append(row)
    if not groups:
        return None, 0, 0

    dominant = max(groups.values(), key=len)

    for preferred_root in root_preference:
        for row in dominant:
            if row.get("root_param") == preferred_root:
                return row, len(dominant), len(rows)

    for row in dominant:
        if row.get("selected_camera_sample") == "1":
            return row, len(dominant), len(rows)

    return dominant[0], len(dominant), len(rows)


def apply_candidate_camera_to_frame(frame, row, cluster_size, candidate_count):
    pos = np.array([
        parse_float(row, "pos_x_m"),
        parse_float(row, "pos_y_m"),
        parse_float(row, "pos_z_m"),
    ], dtype=np.float64)
    right = np.array([
        parse_float(row, "right_x"),
        parse_float(row, "right_y"),
        parse_float(row, "right_z"),
    ], dtype=np.float64)
    up = np.array([
        parse_float(row, "up_x"),
        parse_float(row, "up_y"),
        parse_float(row, "up_z"),
    ], dtype=np.float64)
    fwd = np.array([
        parse_float(row, "fwd_x"),
        parse_float(row, "fwd_y"),
        parse_float(row, "fwd_z"),
    ], dtype=np.float64)

    frame["root_param"] = parse_optional_int(row, "root_param")
    frame["draw_count"] = parse_optional_int(row, "draw_count")
    frame["resource"] = row.get("resource", "").strip()
    frame["cb_offset_hex"] = row.get("cb_offset_hex", "").strip()
    frame["candidate_pipeline_layout"] = row.get("pipeline_layout", "").strip()
    frame["candidate_cluster_size"] = cluster_size
    frame["candidate_count"] = candidate_count
    frame["candidate_index"] = parse_optional_int(row, "candidate_index", -1)
    frame["pos"] = pos
    frame["right"] = normalize(right)
    frame["up"] = normalize(up)
    frame["look"] = normalize(-fwd if NEGATE_CSV_FWD_FOR_LOOK else fwd)
    frame["hfov_deg"] = parse_float(row, "hfov_deg")
    frame["vfov_deg"] = parse_float(row, "vfov_deg")
    frame["near_m"] = parse_float(row, "near_cm") * 0.01


def apply_dominant_candidates(frames, candidate_csv, root_preference=DOMINANT_ROOT_PREFERENCE):
    by_frame = load_candidate_rows(candidate_csv)
    if not by_frame:
        return {
            "enabled": False,
            "candidate_csv": candidate_csv,
            "applied": 0,
            "missing": len(frames),
            "candidate_frames": 0,
        }

    applied = 0
    missing = 0
    for frame in frames:
        rows = by_frame.get(frame["frame"])
        if not rows:
            missing += 1
            continue
        candidate, cluster_size, candidate_count = choose_dominant_candidate(rows, root_preference)
        if candidate is None:
            missing += 1
            continue
        apply_candidate_camera_to_frame(frame, candidate, cluster_size, candidate_count)
        applied += 1

    return {
        "enabled": applied > 0,
        "candidate_csv": candidate_csv,
        "applied": applied,
        "missing": missing,
        "candidate_frames": len(by_frame),
    }


def normalize(v):
    n = np.linalg.norm(v)
    if n <= 1e-12:
        return v
    return v / n


def _frame_rotation_matrix(frame):
    # 用 right/up/look 三个基向量组成旋转矩阵（每行一个基向量）
    return np.array([frame["right"], frame["up"], frame["look"]], dtype=np.float64)


def _rotation_angle_deg(Ra, Rb):
    # 两个朝向之间的整体旋转角（含 yaw/pitch/roll）
    m = Ra @ Rb.T
    cos = (np.trace(m) - 1.0) / 2.0
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos)))


def dedupe_frames(frames, pos_thresh_m=0.03, rot_thresh_deg=3.0, window=30):
    """滑动窗口去重：若某帧在前 window 帧内，存在一个已保留帧与它
    位置 < pos_thresh_m 且 旋转 < rot_thresh_deg，则判为重复帧并丢弃。
    返回 (保留帧列表, 丢弃数量)。保持原有顺序。"""
    kept = []      # [(orig_idx, frame, R)]
    result = []
    dropped = 0
    for i, f in enumerate(frames):
        Ri = _frame_rotation_matrix(f)
        pi = f["pos"]
        is_dup = False
        for j, fj, Rj in reversed(kept):
            if i - j > window:          # 超出窗口（reversed 从近到远，可提前停止）
                break
            if (np.linalg.norm(pi - fj["pos"]) < pos_thresh_m
                    and _rotation_angle_deg(Ri, Rj) < rot_thresh_deg):
                is_dup = True
                break
        if is_dup:
            dropped += 1
        else:
            kept.append((i, f, Ri))
            result.append(f)
    return result, dropped


def index_screens(screen_dir):
    by_postfix = {}
    for path in sorted(screen_dir.glob("*.png")):
        stem = path.stem
        marker = "bmw_camera_frame_"
        idx = stem.find(marker)
        if idx >= 0:
            by_postfix[stem[idx:]] = path
    return by_postfix


def decode_quadview_raw(bgr_full):
    h_full, w_full = bgr_full.shape[:2]
    hh, hw = h_full // 2, w_full // 2

    rgb_bgr = bgr_full[:hh, :hw]
    hi = cv2.cvtColor(bgr_full[:hh, hw:], cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    mid = cv2.cvtColor(bgr_full[hh:, :hw], cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    lo = cv2.cvtColor(bgr_full[hh:, hw:], cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

    raw = hi + mid / 255.0 + lo / 65025.0
    return rgb_bgr, raw


def split_rgb_for_direct_depth(bgr_full, depth_shape):
    depth_h, depth_w = depth_shape
    image_h, image_w = bgr_full.shape[:2]

    if image_h == depth_h and image_w == depth_w:
        return bgr_full
    if image_h >= depth_h * 2 and image_w >= depth_w * 2:
        return bgr_full[:depth_h, :depth_w]

    raise ValueError(
        f"screenshot resolution {image_w}x{image_h} does not match direct depth "
        f"resolution {depth_w}x{depth_h} or QuadView {depth_w * 2}x{depth_h * 2}"
    )


def decode_direct_depth_bytes(raw_bytes, width, height, row_pitch, tight_row_pitch):
    if width <= 0 or height <= 0 or row_pitch <= 0:
        raise ValueError("invalid direct depth metadata")

    if tight_row_pitch <= 0:
        tight_row_pitch = row_pitch

    bytes_per_pixel = tight_row_pitch // width
    if bytes_per_pixel not in (2, 4, 8):
        raise ValueError(f"unsupported direct depth pixel size: {bytes_per_pixel} bytes")

    expected = row_pitch * height
    if len(raw_bytes) < expected:
        raise ValueError(f"depth file is too small: got {len(raw_bytes)}, expected {expected}")

    rows = np.frombuffer(raw_bytes[:expected], dtype=np.uint8).reshape(height, row_pitch)
    tight = rows[:, :width * bytes_per_pixel].copy()

    if bytes_per_pixel == 2:
        return tight.view("<u2").reshape(height, width).astype(np.float64) / 65535.0

    if bytes_per_pixel == 8:
        # D32_FLOAT_S8X24 stores the 32-bit float depth in the first four bytes
        # and stencil/padding in the remaining four bytes.
        depth_bytes = tight.reshape(height, width, 8)[:, :, :4].copy()
        return depth_bytes.view("<f4").reshape(height, width).astype(np.float64)

    as_float = tight.view("<f4").reshape(height, width).astype(np.float64)
    finite = np.isfinite(as_float)
    if finite.mean() > 0.95:
        finite_values = as_float[finite]
        if finite_values.size and np.nanpercentile(finite_values, 99.0) <= 1.5 and np.nanpercentile(finite_values, 1.0) >= -0.01:
            return as_float

    as_uint = tight.view("<u4").reshape(height, width)
    return (as_uint & 0x00FFFFFF).astype(np.float64) / 16777215.0


def load_direct_depth(frame, depth_dir):
    depth_file = frame.get("depth_file", "")
    if not depth_file:
        return None

    depth_path = depth_dir / depth_file
    if not depth_path.exists():
        return None

    meta = {}
    json_path = depth_path.with_suffix(".json")
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as fp:
            meta = json.load(fp)

    width = int(meta.get("width", frame.get("depth_width", 0)))
    height = int(meta.get("height", frame.get("depth_height", 0)))
    row_pitch = int(meta.get("row_pitch", frame.get("depth_row_pitch", 0)))
    tight_row_pitch = int(meta.get("tight_row_pitch", frame.get("depth_tight_row_pitch", 0)))

    raw_bytes = depth_path.read_bytes()
    raw = decode_direct_depth_bytes(raw_bytes, width, height, row_pitch, tight_row_pitch)
    return raw_reversed_z_to_meters(raw, frame["near_m"])


def extract_pitched_rows(raw_bytes, width, height, row_pitch, tight_row_pitch, label):
    if width <= 0 or height <= 0 or row_pitch <= 0:
        raise ValueError(f"invalid direct {label} metadata")

    if tight_row_pitch <= 0:
        tight_row_pitch = row_pitch

    if tight_row_pitch % width != 0:
        raise ValueError(f"direct {label} tight row pitch is not divisible by width")

    bytes_per_pixel = tight_row_pitch // width
    expected = row_pitch * height
    if len(raw_bytes) < expected:
        raise ValueError(f"{label} file is too small: got {len(raw_bytes)}, expected {expected}")

    rows = np.frombuffer(raw_bytes[:expected], dtype=np.uint8).reshape(height, row_pitch)
    return rows[:, :width * bytes_per_pixel].copy(), bytes_per_pixel


def decode_direct_color_bytes(raw_bytes, width, height, row_pitch, tight_row_pitch, format_id):
    tight, bytes_per_pixel = extract_pitched_rows(
        raw_bytes, width, height, row_pitch, tight_row_pitch, "color"
    )

    # ReShade uses API format enum values that match the underlying DXGI format
    # values for D3D12. Black Myth's back buffer has appeared as
    # DXGI_FORMAT_R10G10B10A2_UNORM (24) in the ReShade log.
    r10g10b10a2_formats = {23, 24, 25}
    r8g8b8a8_formats = {27, 28, 29, 30}
    b8g8r8a8_formats = {87, 90, 91}
    b8g8r8x8_formats = {88, 92, 93}
    r16g16b16a16_float_formats = {10}
    r32g32b32a32_float_formats = {2}

    if format_id in r10g10b10a2_formats or (format_id == 0 and bytes_per_pixel == 4):
        packed = tight.view("<u4").reshape(height, width)
        r = (packed & 0x3FF).astype(np.float32) / 1023.0
        g = ((packed >> 10) & 0x3FF).astype(np.float32) / 1023.0
        b = ((packed >> 20) & 0x3FF).astype(np.float32) / 1023.0
        return np.clip(np.stack([b, g, r], axis=2) * 255.0 + 0.5, 0, 255).astype(np.uint8)

    if format_id in r8g8b8a8_formats and bytes_per_pixel == 4:
        rgba = tight.reshape(height, width, 4)
        return rgba[:, :, [2, 1, 0]].copy()

    if (format_id in b8g8r8a8_formats or format_id in b8g8r8x8_formats) and bytes_per_pixel == 4:
        bgra = tight.reshape(height, width, 4)
        return bgra[:, :, :3].copy()

    if format_id in r16g16b16a16_float_formats and bytes_per_pixel == 8:
        rgba = tight.view("<f2").reshape(height, width, 4).astype(np.float32)
        bgr = np.clip(rgba[:, :, [2, 1, 0]], 0.0, 1.0)
        return (bgr * 255.0 + 0.5).astype(np.uint8)

    if format_id in r32g32b32a32_float_formats and bytes_per_pixel == 16:
        rgba = tight.view("<f4").reshape(height, width, 4).astype(np.float32)
        bgr = np.clip(rgba[:, :, [2, 1, 0]], 0.0, 1.0)
        return (bgr * 255.0 + 0.5).astype(np.uint8)

    raise ValueError(
        f"unsupported direct color format={format_id}, bytes_per_pixel={bytes_per_pixel}"
    )


def load_direct_color(frame, color_dir):
    color_file = frame.get("color_file", "")
    if not color_file:
        return None

    color_path = color_dir / color_file
    if not color_path.exists():
        return None

    meta = {}
    json_path = color_path.with_suffix(".json")
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as fp:
            meta = json.load(fp)

    width = int(meta.get("width", frame.get("color_width", 0)))
    height = int(meta.get("height", frame.get("color_height", 0)))
    row_pitch = int(meta.get("row_pitch", frame.get("color_row_pitch", 0)))
    tight_row_pitch = int(meta.get("tight_row_pitch", frame.get("color_tight_row_pitch", 0)))
    format_id = int(meta.get("format", frame.get("color_format", 0)))

    raw_bytes = color_path.read_bytes()
    return decode_direct_color_bytes(raw_bytes, width, height, row_pitch, tight_row_pitch, format_id)


def raw_reversed_z_to_meters(raw, near_m):
    raw = np.clip(raw, 1e-9, 1.0)
    return near_m / raw


def unproject_frame(frame, rgb_bgr, depth_m, stride, min_depth_m, max_depth_m):
    h, w = depth_m.shape

    hfov_rad = math.radians(frame["hfov_deg"])
    vfov_rad = math.radians(frame["vfov_deg"])
    fx = (w / 2.0) / math.tan(hfov_rad / 2.0)
    fy = (h / 2.0) / math.tan(vfov_rad / 2.0)
    cx = w / 2.0
    cy = h / 2.0

    vs = np.arange(0, h, stride)
    us = np.arange(0, w, stride)
    uu, vv = np.meshgrid(us, vs)

    d = depth_m[vv, uu]
    bgr = rgb_bgr[vv, uu]

    mask = (d > min_depth_m) & (d < max_depth_m) & np.isfinite(d)
    d = d[mask].astype(np.float64)
    bgr = bgr[mask]
    uu = uu[mask].astype(np.float64)
    vv = vv[mask].astype(np.float64)

    ray_x = (uu + PIXEL_CENTER_OFFSET - cx) / fx
    ray_y = (vv + PIXEL_CENTER_OFFSET - cy) / fy

    sx = frame.get("screen_x_sign", SCREEN_X_RIGHT_SIGN)
    sy = frame.get("screen_y_sign", SCREEN_Y_UP_SIGN)
    pos = frame["pos"][np.newaxis, :]
    right = (frame["right"] * sx)[np.newaxis, :]
    up = (frame["up"] * sy)[np.newaxis, :]
    look = frame["look"][np.newaxis, :]

    d_col = d[:, np.newaxis]
    rx_col = ray_x[:, np.newaxis]
    ry_col = ray_y[:, np.newaxis]

    points = pos + d_col * (look + right * rx_col + up * ry_col)
    colors_rgb = bgr[:, [2, 1, 0]]

    return (
        points.astype(np.float32),
        colors_rgb.astype(np.uint8),
        d.astype(np.float32),
        (float(fx), float(fy)),
    )


def save_3dgs_ply(positions, colors_rgb, depths, fx_fy, path, stride, scale_factor):
    n = positions.shape[0]
    fx, fy = fx_fy
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

    data = np.concatenate([
        positions, normals, f_dc, opacity, scales, rotations
    ], axis=1).astype(np.float32)

    header_lines = [
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
        fp.write(("\n".join(header_lines) + "\n").encode("ascii"))
        data.tofile(fp)

    return os.path.getsize(path) / 1e6


def save_rgb_pointcloud_ply(positions, colors_rgb, path):
    n = positions.shape[0]
    vertex_dtype = np.dtype([
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ])

    data = np.empty(n, dtype=vertex_dtype)
    data["x"] = positions[:, 0].astype(np.float32)
    data["y"] = positions[:, 1].astype(np.float32)
    data["z"] = positions[:, 2].astype(np.float32)
    data["red"] = colors_rgb[:, 0]
    data["green"] = colors_rgb[:, 1]
    data["blue"] = colors_rgb[:, 2]

    header_lines = [
        "ply", "format binary_little_endian 1.0", f"element vertex {n}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        "end_header",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fp:
        fp.write(("\n".join(header_lines) + "\n").encode("ascii"))
        data.tofile(fp)

    return os.path.getsize(path) / 1e6


def compute_origin(frames, mode):
    if mode == "none":
        return np.zeros(3, dtype=np.float64)
    if mode == "frame0_cam":
        return frames[0]["pos"]
    if mode == "each_cam":
        return None
    raise ValueError(f"unsupported origin mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description="Convert Black Myth camera/depth/color captures to PLY.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--screen-dir", type=Path, default=DEFAULT_SCREEN_DIR)
    parser.add_argument("--depth-dir", type=Path, default=None, help="Directory containing *.depth.bin files; defaults to --screen-dir.")
    parser.add_argument("--color-dir", type=Path, default=None, help="Directory containing *.color.bin files; defaults to --depth-dir.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=None,
        help="Path to BMWCameraCapture bmw_camera_candidates.csv; defaults to a file with that name next to --csv.",
    )
    parser.add_argument(
        "--camera-selection",
        choices=["auto", "csv", "dominant"],
        default="auto",
        help="Use camera rows as-is, or replace each frame with the dominant pose from bmw_camera_candidates.csv.",
    )
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--min-depth", type=float, default=MIN_DEPTH_M)
    parser.add_argument("--max-depth", type=float, default=MAX_DEPTH_M)
    parser.add_argument("--scale-factor", type=float, default=SCALE_FACTOR)
    parser.add_argument("--origin-mode", choices=["frame0_cam", "each_cam", "none"], default="frame0_cam")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frame count for testing; 0 means all.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--no-rgb-ply", action="store_true", help="Do not write CloudCompare-friendly *_rgb.ply files.")
    parser.add_argument("--only-rgb-ply", action="store_true", help="Only write CloudCompare-friendly *_rgb.ply files.")
    args = parser.parse_args()
    depth_dir = args.depth_dir if args.depth_dir is not None else args.screen_dir
    color_dir = args.color_dir if args.color_dir is not None else depth_dir

    frames = load_camera_csv(args.csv)
    candidate_csv = args.candidate_csv if args.candidate_csv is not None else args.csv.parent / DEFAULT_CANDIDATE_CSV_NAME
    candidate_info = {
        "enabled": False,
        "candidate_csv": candidate_csv,
        "applied": 0,
        "missing": len(frames),
        "candidate_frames": 0,
    }
    if args.camera_selection in ("auto", "dominant"):
        if candidate_csv.exists():
            candidate_info = apply_dominant_candidates(frames, candidate_csv)
        elif args.camera_selection == "dominant":
            raise SystemExit(f"candidate CSV not found: {candidate_csv}")

    screens = index_screens(args.screen_dir)

    matched = []
    for frame in frames:
        screenshot = screens.get(frame["screenshot_postfix"])
        if screenshot is None and not frame.get("color_file"):
            print(
                f"skip frame {frame['frame']}: missing screenshot postfix "
                f"{frame['screenshot_postfix']!r} and no direct color_file"
            )
            continue
        frame = dict(frame)
        frame["screenshot"] = screenshot
        matched.append(frame)

    if args.start_index:
        matched = matched[args.start_index:]
    if args.max_frames > 0:
        matched = matched[:args.max_frames]

    if not matched:
        raise SystemExit("no usable CSV rows matched screenshot files or direct color readbacks")

    origin = compute_origin(matched, args.origin_mode)
    print(f"matched frames: {len(matched)}")
    if args.camera_selection == "csv":
        print("camera selection: csv rows")
    elif candidate_info["enabled"]:
        print(
            "camera selection: dominant candidates "
            f"({candidate_info['applied']}/{len(frames)} rows, "
            f"candidate frames={candidate_info['candidate_frames']}, "
            f"source={candidate_info['candidate_csv']})"
        )
        if candidate_info["missing"]:
            print(f"camera selection warning: {candidate_info['missing']} CSV rows had no candidate replacement")
    else:
        print("camera selection: csv rows (no candidate CSV found)")
    print(f"origin mode: {args.origin_mode}")
    if origin is not None:
        print(f"origin: {origin}")

    for idx, frame in enumerate(matched):
        print(f"\n=== {idx:04d} frame {frame['frame']} ===")
        if frame["screenshot"] is not None:
            print(f"  screenshot: {frame['screenshot'].name}")
        else:
            print("  screenshot: none")
        print(f"  camera pos: {frame['pos']}")
        print(f"  look: {frame['look']}")
        if frame.get("candidate_cluster_size"):
            print(
                "  camera candidate: "
                f"root={frame.get('root_param')} draw={frame.get('draw_count')} "
                f"cluster={frame.get('candidate_cluster_size')}/{frame.get('candidate_count')} "
                f"layout={frame.get('candidate_pipeline_layout')}"
            )

        depth_m = load_direct_depth(frame, depth_dir)
        rgb_bgr = load_direct_color(frame, color_dir)
        color_source = "direct add-on color" if rgb_bgr is not None else ""

        bgr = None
        if frame["screenshot"] is not None and (depth_m is None or rgb_bgr is None):
            bgr = cv2.imread(str(frame["screenshot"]), cv2.IMREAD_COLOR)
            if bgr is None:
                print("  skip: failed to read image")
                continue

        if depth_m is not None:
            depth_source = "direct add-on depth"
        else:
            if bgr is None:
                print("  skip: no direct depth and no QuadView screenshot fallback")
                continue
            rgb_bgr, raw = decode_quadview_raw(bgr)
            depth_m = raw_reversed_z_to_meters(raw, frame["near_m"])
            depth_source = "QuadViewRaw PNG"
            color_source = "QuadViewRaw PNG"

        if rgb_bgr is None:
            if bgr is None:
                print("  skip: no direct color and no screenshot fallback")
                continue
            rgb_bgr = split_rgb_for_direct_depth(bgr, depth_m.shape)
            color_source = "PNG"

        if rgb_bgr.shape[:2] != depth_m.shape:
            print(
                f"  skip: color resolution {rgb_bgr.shape[1]}x{rgb_bgr.shape[0]} "
                f"does not match depth {depth_m.shape[1]}x{depth_m.shape[0]}"
            )
            continue

        finite_depth = depth_m[np.isfinite(depth_m)]
        print(
            f"  rgb/depth resolution: {rgb_bgr.shape[1]}x{rgb_bgr.shape[0]}, "
            f"depth={depth_source}, color={color_source}, "
            f"depth p50={np.percentile(finite_depth, 50):.2f}m, "
            f"p95={np.percentile(finite_depth, 95):.2f}m"
        )

        positions, colors, depths, fxfy = unproject_frame(
            frame, rgb_bgr, depth_m, args.stride, args.min_depth, args.max_depth
        )

        if len(positions) == 0:
            print("  skip: no valid points after depth filtering")
            continue

        if args.origin_mode == "each_cam":
            frame_origin = frame["pos"]
        else:
            frame_origin = origin

        positions = positions - frame_origin.astype(np.float32)

        out_path = args.out_dir / f"bmw_frame_{frame['frame']:08d}.ply"
        size_mb = None
        if not args.only_rgb_ply:
            size_mb = save_3dgs_ply(
                positions, colors, depths, fxfy, out_path, args.stride, args.scale_factor
            )
        rgb_out_path = args.out_dir / f"bmw_frame_{frame['frame']:08d}_rgb.ply"
        rgb_size_mb = None
        if not args.no_rgb_ply or args.only_rgb_ply:
            rgb_size_mb = save_rgb_pointcloud_ply(positions, colors, rgb_out_path)

        print(f"  points: {len(positions):,}")
        if size_mb is not None:
            print(f"  saved 3DGS: {out_path} ({size_mb:.1f} MB)")
        if rgb_size_mb is not None:
            print(f"  saved RGB : {rgb_out_path} ({rgb_size_mb:.1f} MB)")

    print("\nDone. Open the PLY files in SuperSplat or a 3DGS-compatible viewer.")


if __name__ == "__main__":
    main()
