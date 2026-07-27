"""
Pack BMWCameraCapture RGB/depth/camera rows into a compact dataset:

  rgb.mp4
  depth.mkv
  camera_params.json

Depth is stored as lossless FFV1 gray16le video. The 16-bit values use a
linear meter range and reserve 0 for invalid depth:

  depth_m = (value - 1) / 65534 * (depth_max_m - depth_min_m) + depth_min_m

for value > 0.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice as _islice
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import conver_bmw as bmw  # noqa: E402


DEFAULT_CAPTURE_DIR = SCRIPT_DIR / "scripts"
DEFAULT_CSV = DEFAULT_CAPTURE_DIR / "bmw_camera_pos.csv"
DEFAULT_OUT_DIR = SCRIPT_DIR / "rgb_depth_dataset"
DEFAULT_FPS = 25
# Log-compression range (METERS), vxr-style: fixed, NOT scene-adaptive, NO range clip.
# depth_m is already meters (conver_bmw.raw_reversed_z_to_meters = near_m / raw), so NO /100.
# Real GTA surfaces ~0.15..few-hundred m; reversed-Z sky ~= near/1e-9 ~= 1.5e8 m -> clamps to 65535.
DEFAULT_DEPTH_MIN_M = 0.0
DEFAULT_DEPTH_MAX_M = 1.0e6

# 退出码约定，setup_and_run.sh 据此决定「重试 / 跳过 / 删原始」：
#   0 = 成功
#   1 = 真失败（要重试，别删原始）
#   3 = 这一段本来就没内容（F8 残段，帧数太少，跳过即可）
#   4 = 有一批帧，但它们的 depth/color .bin 整段都不在磁盘上（采集/拷贝不完整）。
#       和 1 分开：这段数据是缺的、重试也没用，脚本会自动判 skipped——但只有在
#       这天【还有别的段成功产出】时才删原始，整天全缺 bin 时仍保留（疑似没拷进来）。
EXIT_NO_USABLE_FRAMES = 3
EXIT_ALL_BINS_MISSING = 4
DEFAULT_MIN_SEGMENT_FRAMES = 5

# 缺 .bin 的帧最多打印多少行，之后只计数（避免 10w 行 skip 刷屏，也看不出重点）
MAX_SKIP_PRINTS = 20


def encode_h264(png_dir, pattern, out_path, fps, crf):
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-nostdin",
        "-framerate",
        str(fps),
        "-i",
        str(png_dir / pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-preset",
        "fast",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def encode_ffv1_gray16(png_dir, pattern, out_path, fps):
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-nostdin",
        "-framerate",
        str(fps),
        "-i",
        str(png_dir / pattern),
        "-c:v",
        "ffv1",
        "-pix_fmt",
        "gray16le",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def compress_depth_log_16bit(depth_m, depth_min_m, depth_max_m):
    """Log-compress depth (m) -> uint16 over [min, max], vxr-style.

    NO range clip on the depth itself (operate on the raw depth); only the normalized code is
    clamped to [0,1] to avoid uint16 overflow. Non-finite pixels are treated as far/sky.
    Near surfaces get sub-mm precision, far proportionally coarser (relative precision ~const).
      code = (log(depth+1) - log(min+1)) / (log(max+1) - log(min+1)) * 65535
    """
    if depth_max_m <= depth_min_m:
        raise ValueError("depth_max_m must be greater than depth_min_m")
    lo = math.log(depth_min_m + 1.0)
    hi = math.log(depth_max_m + 1.0)
    d = np.where(np.isfinite(depth_m), depth_m, depth_max_m)
    norm = (np.log(np.maximum(d, 0.0) + 1.0) - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, 1.0)
    return (norm * 65535.0).astype(np.uint16)


def frame_intrinsics(frame, width, height):
    fx = (width / 2.0) / math.tan(math.radians(frame["hfov_deg"]) / 2.0)
    fy = (height / 2.0) / math.tan(math.radians(frame["vfov_deg"]) / 2.0)
    return {
        "fx": fx,
        "fy": fy,
        "cx": width / 2.0,
        "cy": height / 2.0,
        "hfov_deg": frame["hfov_deg"],
        "vfov_deg": frame["vfov_deg"],
        "near_m": frame["near_m"],
    }


def vec3(v):
    return [float(v[0]), float(v[1]), float(v[2])]


def segment_of(frame):
    seg = frame.get("segment")
    return 0 if seg is None else seg


def load_frames(args):
    frames = bmw.load_camera_csv(args.csv)
    if not frames:
        raise SystemExit(f"no camera frames loaded from {args.csv}")
    candidate_csv = (
        args.candidate_csv
        if args.candidate_csv is not None
        else args.csv.parent / bmw.DEFAULT_CANDIDATE_CSV_NAME
    )

    candidate_info = {
        "enabled": False,
        "candidate_csv": str(candidate_csv),
        "applied": 0,
        "missing": len(frames),
        "candidate_frames": 0,
    }
    if args.camera_selection in ("auto", "dominant"):
        if candidate_csv.exists():
            candidate_info = bmw.apply_dominant_candidates(frames, candidate_csv)
        elif args.camera_selection == "dominant":
            raise SystemExit(f"candidate CSV not found: {candidate_csv}")

    # 段过滤必须在去重【之前】：dedup 是跨全部段做的，如果玩家在同一位置 F8 停了又重开，
    # 后一段开头的帧会被前一段的相近位姿判成重复而删光，导致 "no frames for segment N"。
    if args.segment >= 0:
        frames = [f for f in frames if segment_of(f) == args.segment]
        print(f"segment {args.segment}: {len(frames)} frames (去重前)")

    if args.start_index:
        frames = frames[args.start_index:]
    if args.max_frames > 0:
        frames = frames[:args.max_frames]

    if getattr(args, "dedup", False):
        before = len(frames)
        frames, dropped = bmw.dedupe_frames(
            frames, args.dedup_pos, args.dedup_rot, args.dedup_window)
        print(f"dedup: {before} -> {len(frames)} frames (dropped {dropped}; "
              f"pos<{args.dedup_pos}m rot<{args.dedup_rot}deg window={args.dedup_window})")

    if getattr(args, "keep_frames", 0) > 0 and len(frames) > args.keep_frames:
        print(f"keep-frames: {len(frames)} -> {args.keep_frames} (take first {args.keep_frames} after dedup)")
        frames = frames[:args.keep_frames]

    return frames, candidate_info


def write_camera_params(path, frames, width, height, fps, depth_min_m, depth_max_m, candidate_info):
    out_frames = []
    for idx, frame in enumerate(frames):
        info = frame_intrinsics(frame, width, height)
        out_frames.append({
            "frame_id": idx,
            "source_frame": int(frame["frame"]),
            "timestamp": idx / fps,
            "camera_position_m": {
                "x": float(frame["pos"][0]),
                "y": float(frame["pos"][1]),
                "z": float(frame["pos"][2]),
            },
            "right": vec3(frame["right"]),
            "up": vec3(frame["up"]),
            "look": vec3(frame["look"]),
            "intrinsics": info,
            "root_param": int(frame.get("root_param", 0)),
            "draw_count": int(frame.get("draw_count", 0)),
        })

    # Screen-axis signs for reprojection (dataset_to_ply reads these). BMW/UE needs
    # -1/+1; GTA/RAGE needs +1/-1. conver_bmw tags each frame in load_camera_csv;
    # fall back to GTA defaults for old data lacking the field.
    screen_x_sign = float(frames[0].get("screen_x_sign", 1.0)) if frames else 1.0
    screen_y_sign = float(frames[0].get("screen_y_sign", -1.0)) if frames else -1.0

    payload = {
        "encoding": "log_16bit",
        "rgb": "rgb.mp4",
        "depth_mkv": "depth.mkv",
        "width": int(width),
        "height": int(height),
        "fps": fps,
        "num_frames": len(out_frames),
        "screen_x_sign": screen_x_sign,
        "screen_y_sign": screen_y_sign,
        "depth_min_m": depth_min_m,
        "depth_max_m": depth_max_m,
        "sky_value": 65535,
        "depth_decode": "depth_m = exp(value / 65535 * (log(depth_max_m + 1) - log(depth_min_m + 1)) + log(depth_min_m + 1)) - 1",
        "camera_selection": {
            "mode": "dominant" if candidate_info.get("enabled") else "csv",
            "applied": int(candidate_info.get("applied", 0)),
            "missing": int(candidate_info.get("missing", 0)),
            "candidate_frames": int(candidate_info.get("candidate_frames", 0)),
            "candidate_csv": str(candidate_info.get("candidate_csv", "")),
        },
        "frames": out_frames,
    }

    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Export BMWCameraCapture RGB/depth readbacks to rgb.mp4, depth.mkv and camera_params.json."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--depth-dir", type=Path, default=None, help="Directory used as base for CSV depth_file paths.")
    parser.add_argument("--color-dir", type=Path, default=None, help="Directory used as base for CSV color_file paths.")
    parser.add_argument("--candidate-csv", type=Path, default=None)
    parser.add_argument("--camera-selection", choices=["auto", "csv", "dominant"], default="auto")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--depth-min", type=float, default=DEFAULT_DEPTH_MIN_M)
    parser.add_argument("--depth-max", type=float, default=DEFAULT_DEPTH_MAX_M)
    parser.add_argument("--rgb-crf", type=int, default=18)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=0,
                        help="并行加载线程数；0 = 自动取 CPU 核心数（默认）")
    parser.add_argument("--dedup", action="store_true",
                        help="去重：滑动窗口内位置与旋转都很接近的帧视为重复，丢弃")
    parser.add_argument("--dedup-pos", type=float, default=0.03,
                        help="去重位置阈值(米)，默认 0.03=3cm")
    parser.add_argument("--dedup-rot", type=float, default=3.0,
                        help="去重旋转阈值(度)，默认 3")
    parser.add_argument("--dedup-window", type=int, default=30,
                        help="去重滑动窗口帧数，默认 30")
    parser.add_argument("--keep-frames", type=int, default=0,
                        help="去重后只保留最早的前 N 帧（0=全部保留）")
    parser.add_argument("--segment", type=int, default=-1,
                        help="只导出 CSV segment 列等于该值的帧(某段F8录制); -1=全部。旧无segment列的CSV视作段0")
    parser.add_argument("--min-segment-frames", type=int, default=DEFAULT_MIN_SEGMENT_FRAMES,
                        help=f"帧数不超过该值就判为 F8 残段，以退出码 {EXIT_NO_USABLE_FRAMES} 跳过"
                             f"（不算失败）；0=不跳过。默认 {DEFAULT_MIN_SEGMENT_FRAMES}")
    parser.add_argument("--list-segments", action="store_true",
                        help="只打印 CSV 里有哪些 segment(空格分隔到 stdout) 然后退出，供批处理脚本探段用")
    args = parser.parse_args()

    if args.list_segments:
        counts = {}
        for frame in bmw.load_camera_csv(args.csv) or []:
            seg = segment_of(frame)
            counts[seg] = counts.get(seg, 0) + 1
        if not counts:
            raise SystemExit(f"no camera frames loaded from {args.csv}")
        for seg in sorted(counts):
            print(f"  segment {seg}: {counts[seg]} frames", file=sys.stderr)
        print(" ".join(str(seg) for seg in sorted(counts)))
        return

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found in PATH")

    depth_dir = args.depth_dir if args.depth_dir is not None else args.csv.parent
    color_dir = args.color_dir if args.color_dir is not None else depth_dir

    frames, candidate_info = load_frames(args)

    # 空段 / 残段 -> 退出码 3，调用方标记 skipped 继续，而不是把整天判失败后无限重试。
    if not frames:
        print(f"segment {args.segment}: 没有帧，跳过该段")
        sys.exit(EXIT_NO_USABLE_FRAMES)
    if 0 < args.min_segment_frames and len(frames) <= args.min_segment_frames:
        print(f"只有 {len(frames)} 帧 (<= --min-segment-frames {args.min_segment_frames})，"
              f"判为 F8 残段/无效采集，跳过")
        sys.exit(EXIT_NO_USABLE_FRAMES)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = args.out_dir / "rgb.mp4"
    depth_path = args.out_dir / "depth.mkv"
    camera_params_path = args.out_dir / "camera_params.json"

    existing = [p for p in (rgb_path, depth_path, camera_params_path) if p.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(p) for p in existing)
        raise SystemExit(f"output exists; pass --overwrite to replace: {names}")

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 4)
    print(f"frame loading threads: {workers}")

    def _load_frame(frame):
        return (
            bmw.load_direct_color(frame, color_dir),
            bmw.load_direct_depth(frame, depth_dir),
        )

    written_frames = []
    width = height = None
    skipped = {"missing": 0, "mismatch": 0}

    def note_skip(kind, message):
        skipped[kind] += 1
        total_skipped = skipped["missing"] + skipped["mismatch"]
        if total_skipped <= MAX_SKIP_PRINTS:
            print(message)
        elif total_skipped == MAX_SKIP_PRINTS + 1:
            print(f"... 跳过的帧超过 {MAX_SKIP_PRINTS} 个，后面不再逐帧打印（结尾给总数）")

    # 流式编码：逐帧通过管道喂给 ffmpeg，不落临时 PNG（10w+ 帧也内存/磁盘恒定）。
    # 内存由“有界 prefetch 窗口”+“ffmpeg stdin 背压”双重限制，不随帧数增长。
    rgb_proc = None
    depth_proc = None
    prefetch = max(4, workers * 2)   # 内存中最多保留的已解码帧数（有界，防 OOM）
    total = len(frames)
    prog_every = max(50, total // 200)   # 约打印 200 行进度
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            frame_iter = iter(frames)
            pending = deque()
            for frame in _islice(frame_iter, prefetch):
                pending.append((frame, pool.submit(_load_frame, frame)))

            while pending:
                frame, future = pending.popleft()
                try:
                    nxt = next(frame_iter)
                    pending.append((nxt, pool.submit(_load_frame, nxt)))
                except StopIteration:
                    pass

                rgb_bgr, depth_m = future.result()
                if rgb_bgr is None or depth_m is None:
                    which = "rgb" if rgb_bgr is None else ""
                    which += ("+" if which and depth_m is None else "") + ("depth" if depth_m is None else "")
                    note_skip("missing",
                              f"skip frame {frame['frame']}: missing {which} readback "
                              f"({frame.get('color_file', '?')} / {frame.get('depth_file', '?')})")
                    continue
                if rgb_bgr.shape[:2] != depth_m.shape:
                    note_skip(
                        "mismatch",
                        f"skip frame {frame['frame']}: rgb {rgb_bgr.shape[1]}x{rgb_bgr.shape[0]} "
                        f"does not match depth {depth_m.shape[1]}x{depth_m.shape[0]}"
                    )
                    continue

                if width is None:
                    height, width = depth_m.shape
                    rgb_proc = subprocess.Popen(
                        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                         "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(args.fps),
                         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                         "-crf", str(args.rgb_crf), "-preset", "fast", str(rgb_path)],
                        stdin=subprocess.PIPE)
                    depth_proc = subprocess.Popen(
                        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                         "-pix_fmt", "gray16le", "-s", f"{width}x{height}", "-r", str(args.fps),
                         "-i", "-", "-c:v", "ffv1", str(depth_path)],
                        stdin=subprocess.PIPE)
                    print(f"streaming encode start ({width}x{height}) -> rgb.mp4 + depth.mkv")
                elif depth_m.shape != (height, width):
                    note_skip("mismatch", f"skip frame {frame['frame']}: resolution changed")
                    continue

                rgb_proc.stdin.write(np.ascontiguousarray(rgb_bgr, dtype=np.uint8).tobytes())
                depth_u16 = compress_depth_log_16bit(depth_m, args.depth_min, args.depth_max)
                depth_proc.stdin.write(np.ascontiguousarray(depth_u16, dtype="<u2").tobytes())
                written_frames.append(frame)

                n_done = len(written_frames)
                if n_done == 1 or n_done % prog_every == 0:
                    el = time.time() - t0
                    rate = n_done / el if el > 0 else 0.0
                    eta = (total - n_done) / rate / 60.0 if rate > 0 else 0.0
                    print(f"  encoded {n_done}/{total} ({100.0*n_done/max(total,1):.0f}%)  "
                          f"{rate:.0f} fps  eta {eta:.1f} min", flush=True)

        if not written_frames:
            detail = (f"输入 {total} 帧: 缺 .bin {skipped['missing']} 帧, "
                      f"尺寸不匹配 {skipped['mismatch']} 帧")
            if skipped["missing"] > 0 and skipped["mismatch"] == 0:
                # 整段每一帧的 .bin 都不在磁盘上：这段采集/拷贝不完整，不是深度质量问题，
                # 重试也没用。用退出码 4 让批处理脚本自动判为缺失并跳过（收尾时统一删）。
                print(f"!! 本段 {total} 帧的 depth/color .bin 一个都不在磁盘上 —— "
                      f"这段采集/拷贝不完整（不是深度没采到），判为缺失。({detail})")
                sys.exit(EXIT_ALL_BINS_MISSING)
            raise SystemExit(
                f"no frames with both rgb and depth were written ({detail})\n"
                f"  依次检查: 1) {depth_dir} 下 depth/ color/ 是否跟 CSV 一起完整拷贝过来了\n"
                f"            2) CSV 的 depth_file/color_file 列是不是绝对路径(目录改名后会失效)\n"
                f"            3) 采集端是否真的没写出深度(MSAA=Off / 磁盘满)")
        print(f"encoded {len(written_frames)} frames (streaming, log_16bit); "
              f"skipped {skipped['missing']} missing + {skipped['mismatch']} mismatched")
    finally:
        for pr in (rgb_proc, depth_proc):
            if pr is not None and pr.stdin is not None:
                try:
                    pr.stdin.close()
                except Exception:
                    pass
        for pr in (rgb_proc, depth_proc):
            if pr is not None:
                pr.wait()

    write_camera_params(
        camera_params_path,
        written_frames,
        width,
        height,
        args.fps,
        args.depth_min,
        args.depth_max,
        candidate_info,
    )

    print(f"wrote {rgb_path}")
    print(f"wrote {depth_path}")
    print(f"wrote {camera_params_path}")


if __name__ == "__main__":
    main()
