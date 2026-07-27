# BMWCameraCapture ReShade Add-on

This add-on is a first-pass probe for Black Myth: Wukong D3D12 camera data.

It watches D3D12 constant-buffer root descriptors, looks for the UE reversed-Z projection matrix found in the RenderDoc capture at relative offset `0x200`, then reads:

- `ViewOrigin` at relative offset `0x480`
- `ViewToWorld` basis rows at relative offset `0x100`
- near plane from the projection matrix

When it finds a match, it writes one row per frame to:

```text
b1\Binaries\Win64\bmw_camera_pos.csv
```

It also writes a small debug log next to the add-on:

```text
b1\Binaries\Win64\bmw_camera_capture_debug.log
```

## Build

Use ReShade headers that match the ReShade DLL currently loaded by the game. If your log says ReShade `6.7.3` supports API version `18`, compile this add-on against the matching ReShade `6.7.3` SDK/source headers, not the latest `main` branch headers.

Example in a Visual Studio Developer PowerShell:

```powershell
cd C:\path\to\pack_gta\addons\bmw_camera_capture
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DRESHADE_INCLUDE_DIR=C:\path\to\reshade-6.7.3\include
cmake --build build --config Release
```

The output is:

```text
build\Release\BMWCameraCapture.addon64
```

Copy it to the same directory as the game executable and ReShade `dxgi.dll`, for example:

```text
...\b1\Binaries\Win64\BMWCameraCapture.addon64
```

## Run

1. Start the game.
2. Open the ReShade overlay.
3. In the Add-ons page, confirm `BMWCameraCapture` is loaded.
4. Move the camera for a few seconds.
5. Close the game or tab out and check `bmw_camera_pos.csv`.

Expected columns include position in centimeters and meters:

```text
pos_x_cm,pos_y_cm,pos_z_cm,pos_x_m,pos_y_m,pos_z_m
```

For the earlier RenderDoc sample, the expected first matched position was roughly:

```text
60098.89, -21310.31, 1929.70 cm
600.99, -213.10, 19.30 m
```

## If It Does Not Output Rows

Check `bmw_camera_capture_debug.log` first.

- If it says the add-on loaded but never matched a CBV, the offsets may differ in the live frame or the game build.
- If it says direct mapping failed, the add-on will still try cached mapped upload pointers. If the CSV is still empty, the next step is to add a D3D12 readback copy path.
- If ReShade refuses to load the add-on with an API version error, rebuild with the exact ReShade headers for the installed runtime.
