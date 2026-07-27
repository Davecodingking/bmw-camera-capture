GTA 数据处理工具（放到渲染机运行）
======================================

【用途】
把渲染机每天产出的 GTACameraCapture 数据（.bin + csv）批量转成：
  rgb.mp4 + depth.mkv + camera_params.json + alignment.mp4
带去重、断点续传、处理完删原始数据；10w+ 帧也不爆内存/磁盘。

【准备：目录结构】
渲染机把每天的数据放进一个 DATASET 目录，子目录按日期命名：
  DATASET/
    GTACameraCapture_0702/   (内含 gta_camera_pos.csv + color/ + depth/)
    GTACameraCapture_0703/
    ...

【运行】（在 Git Bash 里）
  bash setup_and_run.sh  "D:/DATASET"
  （参数 = DATASET 根目录；不传则用脚本顶部默认 D:/DATASET）

它会：
  1) 首次自动装 Python3.12 / ffmpeg / numpy / opencv（需要联网 + winget）
  2) 扫描 DATASET 下所有 GTACameraCapture_*，跳过已处理，处理没做过的
  3) 每天 -> DATASET/rgb_depth_dataset_<日期>/   （输出到 DATASET 根，集中、带日期）
       rgb.mp4            彩色 (H.264)
       depth.mkv          16bit 深度，LOG 压缩，FFV1 无损
       camera_params.json 内外参 + 每帧位姿(right/up/look + camera_position_m)
       alignment.mp4      对齐检查四联: RGB | 深度(log-TURBO) | RGB+深度边缘 | 纯边缘
  4) 每一【段】成功后就立刻写断点记录 DATASET/process_state.json（段级）；
     整天所有段都完成后，才删除整个 GTACameraCapture_<日期>/（原始 .bin）

处理完 DATASET 目录长这样：
  DATASET/
    process_state.json           断点记录（哪天已完成）
    rgb_depth_dataset_0702/      当天产物（原始 GTACameraCapture_0702/ 已删）
    rgb_depth_dataset_0703/
    ...

【关键特性】
  - 流式编码：export 逐帧管道喂 ffmpeg，不落临时 PNG，内存/磁盘不随帧数增长
  - 断点续传（段级）：process_state.json 记到每个 seg。某段失败【不会】连累已成功的段，
           也不会中断后面的段；重跑只补没做完的段，不会重新编码已完成的段。
           想重跑某天：删掉 process_state.json 里对应那一条；只重跑某段：删该天 segs 里那一条
  - 残段容错：帧数 <= MIN_SEG_FRAMES(默认5) 的段（F8 刚按下就停）标记 skipped 跳过，
           不算失败——否则这一段会让整天永远重试、永远重做前面已经成功的段
  - 缺 bin 容错：某段的 depth/color .bin 整段都不在磁盘上（采集/拷贝不完整），自动判为
           缺失并 skipped，不用手动处理。但有安全阀：只有这天【还有别的段成功产出】时才
           跳过并随整天一起删原始；若整天每个段都缺 bin（多半是 depth/color 整个没拷进来
           或还在传），则保留原始、判失败、下次重试，绝不静默删除。收尾统计会单列
           “缺bin判跳过的段”数量。
  - 删原始：脚本顶部 DELETE_RAW=1（默认删整个当天原始目录省空间；改 0 保留）
           只有「所有段都完成」且「断点记录写入成功」才删；写记录失败一律保留原始
           失败的段不记账、不删原始，下次自动重试
  - 去重：DEDUP=1（位置<3cm 且 旋转<3度 视为重复帧丢弃；阈值在脚本顶部可改）
  - 进度：每天打印“数据集 i/N”，编码打印 已完成/总数 + % + fps + 预计剩余时间

【深度编码】
depth.mkv = log 压缩 16bit（范围 [0,1e6]m）+ FFV1 无损。
decode 公式在 camera_params.json 的 "depth_decode" 字段：
  depth_m = exp(value/65535*(log(max+1)-log(min+1)) + log(min+1)) - 1
单位=米；天空/极远 -> 65535。相机位置 camera_position_m 也是米。

【可选：出点云 PLY】（不在一键流程里，需单独跑某天的产物）
  # 3D Gaussian Splatting（可在 https://superspl.at/editor 打开）：
  python dataset_to_ply.py --dataset-dir D:/DATASET/rgb_depth_dataset_0702 --gaussian
  # 普通彩色点云：
  python dataset_to_ply.py --dataset-dir D:/DATASET/rgb_depth_dataset_0702
  常用参数：--stride 像素下采样  --frame-step 每N帧  --per-frame 每帧单独一个PLY
            --start-frame 起始帧  --voxel 体素去重(米)  --scale-factor splat大小

【文件清单】
  setup_and_run.sh                   一键批量脚本（入口）
  export_bmw_rgb_depth_dataset.py    .bin -> rgb.mp4 + depth.mkv + camera_params.json（流式）
  make_alignment.py                  -> alignment.mp4（对齐检查四联）
  conver_bmw.py                      深度解码/反投影/点云 库（被上面依赖，勿删）
  state_tool.py                      断点记录读写（段级、原子写，被入口脚本调用，勿删）
  dataset_to_ply.py                  数据集 -> 点云/3DGS PLY（可选）

【常见问题】
  - 某段一直失败、每帧都 "missing rgb/depth readback"：按报错里列的三步查
      1) depth/ color/ 有没有跟 gta_camera_pos.csv 一起完整拷过来（人工搬目录最容易漏）
      2) CSV 的 depth_file/color_file 是不是绝对路径——目录改过名(比如变成 xxx(2))就会失效
      3) 采集端是不是真没写出深度（MSAA=Off、磁盘满）
  - 报 "无法识别的 CSV 格式"：CSV 列数不对（GTA 应为 117 列，旧版 115 列）。
      多半是采集插件版本和本工具不一致，或 CSV 被截断/人工合并过。
      注意：插件只在文件新建时写表头，所以清空重写过的 CSV 可能没有表头行。
  - 想强制重处理某天/某段：删 process_state.json 里对应那一条（原始已删则需重新渲染）
  - 内存吃紧：脚本会用 CPU 核数并行加载，如需降内存可给 export 传 --workers 4
