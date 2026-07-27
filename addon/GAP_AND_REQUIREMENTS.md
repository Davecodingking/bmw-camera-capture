# BMWCameraCapture 功能差距与需求文档

> 记录 `BMWCameraCapture.cpp`(UE / 黑神话悟空,D3D12)相对 `GTACameraCapture.cpp`
> (GTA V / RAGE,D3D11+D3D12)的功能差距,以及新的需求。

## 0. 两个插件是干什么的

两者是**同一类 ReShade 插件**,目的完全一样:游戏运行时,每隔 N 帧抽取一次
「相机位姿 + 投影矩阵 + 该帧 RGB 图 + 该帧深度图」,对齐后写成数据集:

```
<输出目录>/
  ├─ *_camera_pos.csv          # 每帧一行:相机参数 + 指向对应的 bin 文件
  ├─ color/*.color.bin(+.json) # 该帧 RGB
  └─ depth/*.depth.bin(+.json) # 该帧深度
```

产出的**数据种类相同**。BMW 是更早、更薄的版本;GTA 是经过实战打磨的成熟版本。
不存在「GTA 能抽某种 BMW 抽不了的 pass」这种类别差异,差别都在下面这几点的
**覆盖面 / 稳健性 / 可用性**上。

---

## 1. 功能差距(BMW 需要补齐到 GTA)


| # | 差距               | GTA 的做法                                                                                    | BMW 现状                                                         |
| - | ------------------ | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| 1 | **API 覆盖**       | D3D11 用原生 staging 纹理回读 + D3D12 拷贝,两条路都支持                                       | **只支持 D3D12**;D3D11 游戏根本导不出 RGB/深度                   |
| 2 | **深度与画面对齐** | 单独追踪「全分辨率主深度」,即使相机绑定在半分辨率 pass 上,导出的深度也和 RGB 同分辨率、同视角 | 假设「当前 DSV 就是主深度」,引擎不这么干时会抓到错位或缺失的深度 |
| 3 | **相机信息丰富度** | CSV 额外导出完整的`world_to_view`、`world_to_clip`、`view_to_world` 三个 4×4 矩阵            | 只有投影矩阵 + view 第 3 行                                      |
| 4 | **录制分段**       | CSV 有`segment`/`seg_start` 两列,每次按 F8 = 新的一段,下游能切分不同 take                     | 无,所有帧混在一起分不清                                          |
| 5 | **录制反馈**       | F8 有开始/停止音效 + 画面左上角红色 REC 点                                                    | 无任何提示,不知道在不在录                                        |

> 支撑上面这些的底层实现(后台异步写盘线程、buffer shadow 追踪、RAGE 环形缓冲
> 处理等)属于实现手段,不是独立的用户功能,补齐上述 5 点时会顺带引入。

---

## 2. 新需求(超出「对齐 GTA」之外)

**BMW 的相机 CSV 要比 GTA 多一列:每帧的 `timestamp`(帧时间戳)。**

- 这是连 GTA 都没有的额外列。
- 位置:加到 `bmw_camera_pos.csv` 的每一行里。
- 目的:下游能按真实时间对齐/回放,而不只是按帧序号。

所以 BMW 的最终目标 = **补齐第 1 节的 5 个差距 + 新增第 2 节的时间戳列**。

---

## 3. 构建说明(重要,避免踩坑)

- 这是一个独立的 CMake 工程,header-only 插件:只 `#include <reshade.hpp>`,
  **不链接任何 ReShade 的 .lib**,纯 C++17 + Windows SDK 的 D3D11/D3D12 头。
- 产出两个文件:`build/Release/BMWCameraCapture.addon64`、`GTACameraCapture.addon64`
  (`.addon64` 就是改了后缀的 DLL),拷到游戏 exe 旁边由 ReShade 加载。
- **不需要为它专门装 VS2019。** `-G "Visual Studio 16 2019"` 只是生成器选择,
  代码本身不绑定任何 VS 版本(README 自己的示例用的是 VS2022)。VS2026 能编。
  - ⚠️ 坑:仓库里 `build/` 文件夹的 `CMakeCache.txt` 是从同事机器同步过来的,
    里面**写死了 VS2019 路径**。在自己的 VS2026 机器上想重新编译,要先
    **删掉整个 `build/` 目录再重新 `cmake -S . -B build -G "<你的生成器>"`**,
    否则 CMake 会去找不存在的 VS2019 而报错——这多半就是「必须用 VS2019」误解的来源。
- 真正必须版本对齐的是 `RESHADE_INCLUDE_DIR` 指向的头文件,要和游戏里加载的
  ReShade DLL 版本一致(当前是 6.7.3 / API 18)。

---

## 4. 完整运行流程(黑神话悟空)

游戏目录:`D:\game\Black.Myth.Wukong.Digital.Deluxe.Edition-InsaneRamZes\b1\Binaries\Win64\`
(此目录里 ReShade 的 `dxgi.dll`、`BMWCameraCapture.addon64`、输出目录均已就位)

1. **(前置,已完成)** ReShade 已装:`dxgi.dll` + `ReShade.ini`;`BMWCameraCapture.addon64`
   已拷到游戏 exe 旁。
2. 启动游戏 `b1-Win64-Shipping.exe`,ReShade 会自动加载 `dxgi.dll` 和插件。
   打开 ReShade 覆盖层,在 Add-ons 页确认 `BMWCameraCapture` 已加载。
3. 启动 `UUU\IGCSClient.exe`(UUU = Universal UE5 Unlocker,负责自由相机 + 暂停/
   定格游戏)。点 **Select** 选中 `b1-Win64-Shipping.exe` 进程,点 **Inject dll**。
   - 暂停:小键盘 `Numpad 0`(硬暂停)或 `Page Down`(slomo 暂停,推荐,不抖动)。
4. 在游戏里按 **F8** 开始/停止采集(BMW 默认关,按一次开始)。
5. 采集结果写到:`...\Win64\BMWCameraCapture\bmw_camera_pos.csv` + `color/` + `depth/`。
   调试信息看 `bmw_camera_capture_debug.log`。

> 同事描述的顺序(配置 ReShade → 加入黑神话 → 开游戏 → 再启动 UUU → 选进程注入)
> 与 UUU 的官方 Readme 一致,是对的。ReShade 和 UUU 是两个独立的注入,ReShade 随
> 游戏启动自动加载,UUU 在游戏跑起来后手动注入。

---

## 5. 进度(2026-07-27)

| 项 | 状态 |
|---|---|
| gap 1 API 覆盖(D3D11) | ❌ 未做(黑神话是 D3D12,暂不需要) |
| gap 2 深度/颜色对齐 | ✅ 已做并游戏内验证(Step 1) |
| gap 3 完整矩阵导出 | ✅ 已做:CSV 增加 `world_to_view_m*`/`world_to_clip_m*`/`view_to_world_m*`(Step 2,已编译未实测) |
| gap 4 录制分段 | ✅ 已做:CSV 增加 `segment`/`seg_start`(Step 2) |
| gap 5 录制反馈 | ✅ 已做:彩色指示灯(Step 1)+ F8 升/降调音效(Step 2) |
| 新增 frame 时间戳 | ✅ 已做:两列 `timestamp_s`(单调秒/从开录起)+ `timestamp_unix`(墙钟)(Step 2) |
| jitter | ⏸ 暂缓(用户定) |

**Step 1**(已实测):深度对齐 + 放宽严格同步(`kRequireEnabledReadbacksForCsv=false`)+
朝向过滤开关默认关(`kFilterLikelyMainView=false`)+ 预览图(`kSavePreviewImages`)+ 录制指示灯。

**Step 2**(已编译,待用户部署实测):gap 3/4/5 音效 + 双时间戳列。
⚠️ **表头变了**:用新插件采集前请把旧的 `bmw_camera_pos.csv`(及输出目录)改名/清空,或换新的
输出目录,否则新行的列数与旧表头不符会错位。

**下游符号修正**(已随 Step 2 固化):重建工具对黑神话(UE)必须用 `SX=-1, SY=+1`(与 GTA 相反),
否则多帧 splat 不重叠。已改 `bin_to_splat.py`(本机已验证)、`conver_bmw.py`/`dataset_to_ply.py`/
`export_bmw_rgb_depth_dataset.py`(共用文件,用 BMW/GTA 判别分支,本机无 cv2/ffmpeg 未实跑)。
