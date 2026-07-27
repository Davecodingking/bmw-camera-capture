#!/usr/bin/env bash
# ------------------------------------------------------------------
# 渲染机一键脚本：配置环境 + 批量处理 + 断点续传 + 处理完删原始数据
#
# 渲染机每天把当天的 GTACameraCapture 放进 DATASET 目录、按日期命名，例如：
#     DATASET/
#       GTACameraCapture_0702/   (内含 gta_camera_pos.csv + color/ + depth/)
#       GTACameraCapture_0703/
#       ...
# 本脚本扫描 DATASET 下所有 GTACameraCapture_* ，只处理【还没处理完】的，
# 每个处理成 <该目录>/rgb_depth_dataset/ ：
#     rgb.mp4            彩色 (H.264)
#     depth.mkv          16bit 深度，LOG 压缩，FFV1 无损（decode 见 camera_params.json）
#     camera_params.json 内外参 + 每帧位姿（right/up/look + camera_position_m）
#     alignment.mp4      vxr 式逐帧四联：RGB | 深度(log-TURBO) | RGB+深度边缘 | 纯边缘
#
# 关键特性：
#   - 流式编码：export 逐帧管道喂 ffmpeg，不落临时 PNG，10w+ 帧也内存/磁盘恒定
#   - 断点续传：DATASET/process_state.json 记录到【段】粒度，中断重跑只补没做完的段。
#       某一段失败不会连累已成功的段（不会重编码），也不会 break 掉后面的段。
#       只有 1~2 帧的 F8 残段会被标记 skipped 跳过，不算失败（否则整天永远重试）。
#   - 处理成功后：删除该天原始 .bin(color/ depth/) 省空间（DELETE_RAW 控制）
#       前提是断点记录写入成功；写失败则保留原始数据。
#
# 用法（在 Git Bash 里）：
#   bash setup_and_run.sh  "D:/DATASET"
#   （参数 = DATASET 根目录；不传则用下面默认值）
#
# 需要：Windows 10/11 自带的 winget，以及网络（首次装 Python/ffmpeg）。
# ------------------------------------------------------------------
set -u

# Python 在 Git Bash(mintty) 里默认按系统 ANSI 代码页(cp936)输出，而 mintty 是 UTF-8，
# 于是中文提示全变成 ▒▒ 乱码。强制 Python 用 UTF-8 收发，乱码即消失。
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# ===== 可改：DATASET 根目录（渲染机放各天数据的地方）=====
DATASET_DEFAULT="E:/DATASET"
DATASET="${1:-$DATASET_DEFAULT}"
# ===== 可改：去重参数（滑窗内位置<3cm 且 旋转<3度 视为重复帧丢弃）=====
DEDUP=1            # 1=开启去重, 0=关闭
DEDUP_POS=0.03     # 位置阈值(米)
DEDUP_ROT=3        # 旋转阈值(度)
DEDUP_WINDOW=30    # 滑动窗口帧数
# ===== 可改：处理成功后是否删除原始采集(.bin) 省空间 =====
DELETE_RAW=1       # 1=处理成功后删除该天原始采集目录省空间; 0=保留
# ===== 可改：是否按 segment(每次F8录制段) 拆分 =====
SPLIT_BY_SEG=1     # 1=按CSV的segment列, 每段各出一套 rgb_depth_dataset_<日期>_seg<id>/; 0=整个采集目录出一套
# ===== 可改：帧数不超过这个值的段判为 F8 残段，标记 skipped 跳过（不算失败）=====
MIN_SEG_FRAMES=5   # 0=不跳过（一帧的残段也当正常段处理，会导致整天失败）
# ==========================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="$DATASET/process_state.json"   # 断点续传记录（哪天已处理完成）
echo "脚本目录     : $DIR"
echo "DATASET 目录 : $DATASET"
echo "断点记录     : $STATE"
echo "删原始数据   : DELETE_RAW=$DELETE_RAW"
[ -d "$DATASET" ] || { echo "!! 找不到 DATASET 目录 $DATASET，请把正确的根目录作为参数传入"; exit 1; }

# ---- 1) Python（优先 3.12/3.13/3.11 有 opencv wheel 的版本，其次系统 python）----
PY=""
for cand in \
  "$LOCALAPPDATA/Programs/Python/Python312/python.exe" \
  "$LOCALAPPDATA/Programs/Python/Python313/python.exe" \
  "$LOCALAPPDATA/Programs/Python/Python311/python.exe" \
  "$LOCALAPPDATA/Programs/Python/Python310/python.exe"; do
  [ -f "$cand" ] && PY="$cand" && break
done
[ -n "$PY" ] || PY="$(command -v python 2>/dev/null || command -v python3 2>/dev/null || true)"
if [ -z "${PY:-}" ] && command -v winget >/dev/null 2>&1; then
  echo "[env 1/3] 未发现 Python，winget 安装 3.12 ..."
  winget install -e --id Python.Python.3.12 --scope user --silent \
    --accept-source-agreements --accept-package-agreements || true
  PY="$LOCALAPPDATA/Programs/Python/Python312/python.exe"
fi
[ -n "${PY:-}" ] || { echo "!! 未找到 Python。请装 Python 3.12: https://www.python.org/downloads/ (勾 Add to PATH) 再重试"; exit 1; }
pyver="$("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo '?')"
echo "[env 1/3] Python: $PY ($pyver)"

# ---- 2) ffmpeg（winget -> 已解压的 C:\ffmpeg -> curl 自动下载静态包；无 winget 也能用）----
if ! command -v ffmpeg >/dev/null 2>&1; then
  FF="$(find /c/ffmpeg "$DIR/ffmpeg" -iname ffmpeg.exe 2>/dev/null | head -1)"
  if [ -z "${FF:-}" ] && command -v winget >/dev/null 2>&1; then
    echo "[env 2/3] winget 安装 ffmpeg ..."
    winget install -e --id Gyan.FFmpeg --silent --accept-source-agreements --accept-package-agreements || true
    FF="$(find "$LOCALAPPDATA/Microsoft/WinGet/Packages" -iname ffmpeg.exe 2>/dev/null | head -1)"
  fi
  if [ -z "${FF:-}" ]; then
    echo "[env 2/3] 无 winget，自动下载静态 ffmpeg 到 C:\\ffmpeg ..."
    curl -fL -o "$DIR/ffdl.zip" 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' \
      && powershell -NoProfile -Command "Expand-Archive -Path '$(cygpath -w "$DIR/ffdl.zip")' -DestinationPath 'C:\\ffmpeg' -Force" \
      && rm -f "$DIR/ffdl.zip" || echo "   下载/解压失败(检查网络)"
    FF="$(find /c/ffmpeg -iname ffmpeg.exe 2>/dev/null | head -1)"
  fi
  [ -n "${FF:-}" ] && export PATH="$(dirname "$FF"):$PATH"
fi
command -v ffmpeg >/dev/null 2>&1 || { echo "!! ffmpeg 不可用。手动下载 https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip 解压到 C:\\ffmpeg 再重试"; exit 1; }
echo "     ffmpeg: $(command -v ffmpeg)"

# ---- 3) Python 依赖 ----
echo "[env 3/3] 安装 numpy / opencv-python ..."
"$PY" -m pip install --quiet --user --upgrade pip >/dev/null 2>&1 || true
if ! "$PY" -m pip install --quiet --user numpy opencv-python; then
  echo "!! numpy/opencv 安装失败。Python $pyver 可能太新(还没有 opencv 预编译 wheel)。"
  echo "   请装 Python 3.12 后重试: https://www.python.org/downloads/ (勾 Add to PATH)"
  exit 1
fi

# ---- 4) 扫描 DATASET，断点续传逐天处理 ----
DEDUP_ARGS=""
[ "$DEDUP" = "1" ] && DEDUP_ARGS="--dedup --dedup-pos $DEDUP_POS --dedup-rot $DEDUP_ROT --dedup-window $DEDUP_WINDOW"

EXPORT="$DIR/export_bmw_rgb_depth_dataset.py"
ALIGN="$DIR/make_alignment.py"
ST="$DIR/state_tool.py"          # 断点记录读写（段级 + 原子写）

shopt -s nullglob
caps=("$DATASET"/GTACameraCapture_*)
echo
echo "扫描到 ${#caps[@]} 个数据集（GTACameraCapture_*）"
done_cnt=0; skip_cnt=0; fail_cnt=0; miss_cnt=0; idx_cap=0; total_cap=${#caps[@]}
for cap in "${caps[@]}"; do
  [ -d "$cap" ] || continue
  idx_cap=$((idx_cap+1))
  tag="$(basename "$cap")"
  CSV="$cap/gta_camera_pos.csv"
  if [ ! -f "$CSV" ]; then
    echo "[$tag] 没有 gta_camera_pos.csv，跳过"; continue
  fi
  # 断点检查：整天已完成就跳过（exit 0=已完成）
  if "$PY" "$ST" is-day-done "$STATE" "$tag"; then
    echo "[$tag] 已处理完成，跳过"; skip_cnt=$((skip_cnt+1)); continue
  fi

  echo "[$tag] (数据集 $idx_cap/$total_cap) 开始处理 ..."
  date_tag="${tag#GTACameraCapture_}"                 # GTACameraCapture_0702 -> 0702
  ok_all=1; outs=""; day_status="done"
  any_real_ok=0        # 本次运行有没有段真正产出数据（exit 0）
  missing_segs=""      # 整段缺 bin 的段（exit 4）：延后到收尾再决定 skip/保留
  if [ "$SPLIT_BY_SEG" = "1" ]; then
    # 探测 CSV 里有哪些 segment(每次F8录制段); 旧无segment列 -> 段0(退回整套)
    # CSV 格式不对时 --list-segments 会非零退出并打印原因，这天直接判失败，不去猜。
    if ! segs="$("$PY" "$EXPORT" --csv "$CSV" --list-segments)"; then
      echo "[$tag] !! CSV 无法解析（原因见上），本天跳过"; fail_cnt=$((fail_cnt+1)); continue
    fi
    [ -z "$segs" ] && segs=0
    echo "  [$tag] 按段拆分, segment: $segs"
    for s in $segs; do
      OUT="$DATASET/rgb_depth_dataset_${date_tag}/seg${s}"
      # 段级断点：这一段做过了(done 或 skipped)就直接跳过，不重复编码
      if "$PY" "$ST" is-seg-done "$STATE" "$tag" "$s"; then
        echo "  [$tag] 段 $s 之前已完成/已跳过"; outs="$outs seg${s}"; continue
      fi
      echo "  [$tag] 段 $s -> $OUT"
      "$PY" "$EXPORT" --csv "$CSV" --depth-dir "$cap" --color-dir "$cap" \
            --out-dir "$OUT" --overwrite --segment "$s" \
            --min-segment-frames "$MIN_SEG_FRAMES" $DEDUP_ARGS
      rc=$?
      case "$rc" in
        0)
          # 段成功 -> 立刻记账。哪怕后面的段全挂，这一段也不会再重做。
          if "$PY" "$ALIGN" --dataset-dir "$OUT" \
             && "$PY" "$ST" mark-seg "$STATE" "$tag" "$s" done "$OUT"; then
            outs="$outs seg${s}"; any_real_ok=1
          else
            echo "  [$tag] !! 段 $s 对齐视频或记账失败，下次重试该段"; ok_all=0
          fi
          ;;
        3)
          # F8 残段/空段：不是错误，标记 skipped，别让它把整天拖住
          echo "  [$tag] 段 $s 无可用帧(残段)，标记跳过"
          "$PY" "$ST" mark-seg "$STATE" "$tag" "$s" skipped "$OUT" || ok_all=0
          ;;
        4)
          # 整段 depth/color .bin 都不在：数据缺失、重试无用。先记下，收尾再定 skip/保留，
          # 因为要区分「这天还有别的段成功(可安全跳过并删)」和「整天全缺(疑似没拷进来)」。
          echo "  [$tag] 段 $s 整段缺 .bin，暂记为缺失（收尾时统一处理）"
          missing_segs="$missing_segs $s"
          ;;
        *)
          echo "  [$tag] !! 段 $s 处理失败 (exit $rc)，继续处理后面的段"; ok_all=0
          ;;
      esac
    done
  else
    OUT="$DATASET/rgb_depth_dataset_${date_tag}"        # 整个采集目录一套，输出到 DATASET 根
    "$PY" "$EXPORT" --csv "$CSV" --depth-dir "$cap" --color-dir "$cap" \
          --out-dir "$OUT" --overwrite --min-segment-frames "$MIN_SEG_FRAMES" $DEDUP_ARGS
    rc=$?
    case "$rc" in
      0) if "$PY" "$ALIGN" --dataset-dir "$OUT"; then outs="$OUT"; any_real_ok=1; else ok_all=0; fi ;;
      3) echo "  [$tag] 整个采集无可用帧(残段)，标记跳过，不删原始"
         day_status="skipped"; DELETE_THIS_RAW=0 ;;
      4) echo "  [$tag] !! 整个采集的 depth/color .bin 都不在（疑似没拷进来），保留原始不删"
         ok_all=0 ;;   # 单套模式没有"其他段"可依靠，一律判失败保留原始
      *) ok_all=0 ;;
    esac
  fi

  # 收尾前处理「整段缺 bin」的段：只有这天确实有别的段产出(本次成功 + 以前已 done)
  # 才把它们正式记为 skipped 并允许删原始；整天全缺则保留、判失败、下次重试。
  if [ -n "$missing_segs" ]; then
    real_prior="$("$PY" "$ST" day-real-count "$STATE" "$tag" 2>/dev/null || echo 0)"
    has_real=0
    { [ "$any_real_ok" = "1" ] || [ "${real_prior:-0}" -gt 0 ]; } && has_real=1
    if [ "$has_real" = "1" ]; then
      for s in $missing_segs; do
        if "$PY" "$ST" mark-seg "$STATE" "$tag" "$s" skipped ""; then
          echo "  [$tag] 段 $s 缺 .bin，判为缺失并跳过（这天其它段有产出，随其一起收尾）"
          miss_cnt=$((miss_cnt+1))
        else
          ok_all=0
        fi
      done
    else
      echo "[$tag] !! 整天有效输出为 0，所有段都缺 .bin —— 极可能 depth/ color/ 整个没拷进来或还在传输中。"
      echo "         保留原始、不记账、不删除，下次运行会重试（确认数据永久丢失可手动 mark-seg skipped）。"
      ok_all=0
    fi
  fi

  if [ "$ok_all" = "1" ]; then
    # 所有段都 done/skipped -> 写整天记录。写失败(盘满/只读/记录损坏)就【不删原始】。
    if "$PY" "$ST" mark-day "$STATE" "$tag" "$outs" "$day_status"; then
      done_cnt=$((done_cnt+1))
      if [ "$DELETE_RAW" = "1" ] && [ "${DELETE_THIS_RAW:-1}" = "1" ]; then
        if rm -rf "$cap"; then
          echo "  [$tag] 已删除原始目录 $tag（产物已在 DATASET 根）"
        else
          echo "  [$tag] !! 删除原始目录失败（可能被占用），请手动清理 $cap"
        fi
      fi
    else
      echo "[$tag] !! 断点记录写入失败，保留原始数据，下次运行重试"
      fail_cnt=$((fail_cnt+1))
    fi
  else
    echo "[$tag] !! 有段没做完（成功的段已记账、原始未删，下次只补没做完的段）。"
    echo "         常见原因：depth/ 为空（深度没采到）或 depth/color 没跟 CSV 一起拷过来"
    fail_cnt=$((fail_cnt+1))
  fi
  unset DELETE_THIS_RAW
done

echo
echo "==== 全部完成 ===="
echo "本次新处理: $done_cnt   跳过(已完成): $skip_cnt   失败: $fail_cnt   缺bin判跳过的段: $miss_cnt"
echo "断点记录  : $STATE"
echo "            想重跑某天：删掉里面对应的那一条；想重跑某段：删该天 segs 里那一条"
echo "各天产物在: <DATASET>/rgb_depth_dataset_<日期>/   (SPLIT_BY_SEG=1 时段各一套子目录: .../seg<id>/)"
echo "  rgb.mp4 / depth.mkv(log 16bit 无损) / camera_params.json / alignment.mp4(四联对齐检查)"
