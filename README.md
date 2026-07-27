# bmw-camera-capture

Tooling to extract an aligned **camera pose + RGB + depth** dataset from
Unreal-Engine games (primarily **Black Myth: Wukong**, D3D12) via a ReShade
add-on, and to turn the captured frames into point clouds / 3D Gaussian Splats.

## Layout

```
addon/                     ReShade add-on (C++, D3D12) that does the in-game capture
  BMWCameraCapture.cpp        Black Myth / UE capture add-on
  GTACameraCapture.cpp        GTA V / RAGE capture add-on (D3D11 + D3D12)
  CMakeLists.txt              builds both into *.addon64
  README.md                   build & run notes
  GAP_AND_REQUIREMENTS.md     feature status: BMW vs GTA gaps + requirements
tools/                     Python post-processing (numpy; some need OpenCV/ffmpeg)
  bin_to_splat.py             raw capture .bin -> 3D Gaussian Splat .ply (numpy only)
  conver_bmw.py               decode/reproject library (RGB + reversed-Z depth)
  export_bmw_rgb_depth_dataset.py   pack capture -> rgb.mp4 + depth.mkv + camera_params.json
  dataset_to_ply.py           packed dataset -> point cloud / 3DGS .ply
  make_alignment.py, state_tool.py, setup_and_run.sh   batch/verify helpers
```

## Build the add-on

Needs ReShade **6.7.3** SDK headers (API version 18 — must match the ReShade DLL
loaded by the game). Any modern MSVC works (VS2022/VS2026); VS2019 is not required.

```
cmake -S addon -B addon/build -G "Visual Studio 18 2026" -A x64 \
      -DRESHADE_INCLUDE_DIR=<path to reshade-6.7.3/include>
cmake --build addon/build --config Release
```

Output: `addon/build/Release/BMWCameraCapture.addon64`. Copy it next to the game
exe (alongside ReShade's `dxgi.dll`); ReShade auto-loads any `*.addon64` there.

## Capture (Black Myth: Wukong)

1. Launch the game with ReShade installed; press `Home` to confirm the add-on loaded.
2. **Turn off super-resolution / render at native (100%)** — the capture aligns
   color and depth only when the internal render resolution equals the output.
3. Use UUU (Universal UE5 Unlocker) for a free camera + pause if desired.
4. Press `F8` to start/stop capture (green corner marker = writing; ascending beep = started).
5. Output lands in `BMWCameraCapture/` next to the game exe: `bmw_camera_pos.csv`
   + `color/` + `depth/` (each frame has `.bin`, `.json`, and a viewable `.bmp`).

## Reconstruct

Quick splat straight from the raw `.bin` (numpy only, no ffmpeg/OpenCV):

```
python tools/bin_to_splat.py --csv <...>/BMWCameraCapture/bmw_camera_pos.csv \
       --out-dir splat_out --indices 0,2,5 --combined
```

Open the resulting `.ply` at https://superspl.at/editor.

> Note: for UE captures the reprojection screen-axis signs are `SX=-1, SY=+1`
> (opposite of the GTA/RAGE defaults); this is handled automatically per-dataset.
