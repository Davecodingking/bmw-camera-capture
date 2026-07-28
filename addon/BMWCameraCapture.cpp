#include <reshade.hpp>

#include <Windows.h>
#include <d3d12.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// --- IGCS camera-tools connector (engine-authoritative camera from UUU) ---------
// UUU (an IGCS camera tool) hunts loaded modules for an add-on exporting
// connectFromCameraTools()/getDataFromCameraToolsBuffer(); it then writes the
// CameraToolsData struct into that buffer every frame. By exporting these two
// functions we receive UUU's TRUE main-camera pose (no CBV sniffing / guessing).
// Struct layout must match FransBouma/IgcsConnector src/CameraToolsData.h exactly.
struct IgcsVec3 { float x, y, z; };
struct IgcsVec4 { float x, y, z, w; };
struct IgcsCameraToolsData
{
	uint8_t cameraEnabled;
	uint8_t cameraMovementLocked;
	uint8_t reserved1;
	uint8_t reserved2;
	float fov;
	IgcsVec3 coordinates;
	IgcsVec4 lookQuaternion;
	IgcsVec3 rotationMatrixUpVector;
	IgcsVec3 rotationMatrixRightVector;
	IgcsVec3 rotationMatrixForwardVector;
	float pitch;
	float yaw;
	float roll;
};
static LPBYTE g_igcs_camera_buffer = nullptr;   // 8KB buffer UUU writes into
static std::atomic_bool g_igcs_connected = false;

extern "C" __declspec(dllexport) bool connectFromCameraTools()
{
	if (g_igcs_camera_buffer == nullptr)
		g_igcs_camera_buffer = static_cast<LPBYTE>(calloc(8 * 1024, 1));
	g_igcs_connected = (g_igcs_camera_buffer != nullptr);
	return g_igcs_connected;
}

extern "C" __declspec(dllexport) LPBYTE getDataFromCameraToolsBuffer()
{
	return g_igcs_camera_buffer;
}

namespace
{
namespace api = reshade::api;

constexpr uint64_t kViewToWorldOffset = 0x100;
constexpr uint64_t kProjectionOffset = 0x200;
constexpr uint64_t kViewOriginOffset = 0x480;
constexpr uint64_t kReadSize = 0x520;
constexpr uint32_t kMinDrawVertices = 1000;
constexpr uint64_t kCaptureEveryNFrames = 30;
constexpr bool kSaveScreenshotOnCapture = false;
// Keep GPU readback disabled by default. The raw copy path is experimental and
// can trigger DXGI_ERROR_DEVICE_HUNG if a resource state assumption is wrong.
// Test one path at a time: first camera-only, then color-only, then depth-only.
constexpr bool kSaveDepthReadbackOnCapture = true;
constexpr bool kSaveColorReadbackOnCapture = true;
constexpr bool kCaptureStartsEnabled = false;
constexpr uint32_t kToggleCaptureKey = VK_F8;
// Strict mode avoids writing a CSV row unless every enabled readback path was
// queued for the exact same add-on frame as the camera sample. Default OFF so a
// row is still written (with strict_sync_ok=0) when e.g. depth could not be
// aligned this frame; matches GTACameraCapture behavior.
constexpr bool kRequireEnabledReadbacksForCsv = false;
// Do not repeatedly map/read camera CBVs after the first main-view match in a
// frame. Re-reading every matching draw can destabilize D3D12 in this title.
constexpr bool kUseLastMatchedCameraSampleInFrame = false;
// Orientation-based main-view filter (up.z / forward.z limits). Default OFF: we
// record manually with a free camera, so steep down/up shots must not be dropped.
// Flip to true only if auto-capture needs to reject non-main-view CBVs by angle.
constexpr bool kFilterLikelyMainView = false;
// Diagnostic path: dump every unique CBV that passes the camera-shape test on
// sampled capture frames. This is intentionally separate from the main camera
// CSV, so bad candidates can be inspected without changing the conversion path.
// Default OFF: only needed when first calibrating the camera CBV for a new game.
// When on, every draw on a capture frame re-scans ALL CBVs and re-maps camera
// CBVs under g_mutex — the code above warns this can destabilize D3D12 in this
// title, and it is a likely contributor to mid-capture crashes. GTA keeps it off.
constexpr bool kLogCameraCandidates = false;
// CALIBRATION for a NEW game: dump the raw bytes of every constant buffer bound
// on a captured frame to BMWCameraCapture/cbv_dumps/, so the camera-CBV offsets
// (kProjectionOffset/kViewOriginOffset/kViewToWorldOffset) can be found offline
// by scanning for the UE reversed-Z projection matrix. Candidate logging can't do
// this because it re-parses with the OLD (game-specific) offsets and never matches
// a different layout. Turn OFF for normal capture. Each unique (resource,offset)
// is dumped once, capped at kDumpMaxFiles.
constexpr bool kDumpRawCbvs = false;
constexpr uint32_t kDumpMaxFiles = 80;
constexpr uint64_t kDumpBytes = 0x1000;  // 4 KB per CBV: covers UE's view uniform buffer at any offset
constexpr uint32_t kMaxCameraCandidateRowsPerFrame = 512;
constexpr uint32_t kMaxCameraCandidateDrawScansPerFrame = 1024;
constexpr float kMinMainViewUpZ = 0.7f;
constexpr float kMaxMainViewAbsForwardZ = 0.35f;
constexpr uint32_t kTextureReadbackPitchAlignment = 256;
constexpr uint64_t kReadbackDelayFrames = 8;
constexpr bool kWaitIdleBeforeReadbackMap = false;
// Minimum size for a depth target to be accepted as the full-resolution main
// depth (used by make_readable_depth_target). Rejects shadow maps / small passes.
constexpr uint32_t kMinMainTargetWidth = 1000;
constexpr uint32_t kMinMainTargetHeight = 600;
// Also write a directly-viewable .bmp next to each color/depth .bin for quick
// eyeball verification (color decoded to 8-bit BGR; depth linearized to grayscale).
constexpr bool kSavePreviewImages = true;
// Draw a small colored square in the top-left WHILE recording (D3D12 only):
// green = recording and a CSV row was written last sampled frame,
// yellow = recording but last sampled frame produced no row. Drawn after the
// color readback is queued, so it never appears in the captured dataset.
constexpr bool kRecIndicator = true;
constexpr int kRecMarkerSizePx = 8;
// Play a short beep when capture toggles (ascending = start, descending = stop)
// so F8 gives immediate audible confirmation without watching the log.
constexpr bool kAudioCueOnToggle = true;

struct TrackedCBV
{
	api::resource buffer = {};
	uint64_t offset = 0;
	uint64_t size = UINT64_MAX;
	uint64_t pipeline_layout = 0;
	uint32_t root_param = 0;
};

struct CommandListState
{
	std::vector<TrackedCBV> cbvs;
	api::resource_view current_dsv = {};
};

struct MappedBuffer
{
	uint64_t offset = 0;
	uint64_t size = UINT64_MAX;
	const uint8_t *data = nullptr;
};

struct CameraSample
{
	bool valid = false;
	uint64_t frame = 0;
	uint32_t root_param = 0;
	uint32_t draw_count = 0;
	uint64_t pipeline_layout = 0;
	uint64_t resource = 0;
	uint64_t cb_offset = 0;
	float pos_cm[3] = {};
	float right[3] = {};
	float up[3] = {};
	float fwd[3] = {};
	float hfov_deg = 0.0f;
	float vfov_deg = 0.0f;
	float near_cm = 0.0f;
	float projection[16] = {};
	float view_to_world_row3[4] = {};
	float view_to_world[16] = {};
	float world_to_view[16] = {};
	float world_to_clip[16] = {};
	double timestamp_s = 0.0;      // monotonic seconds since capture (F8) start
	double timestamp_unix = 0.0;   // absolute wall-clock seconds since Unix epoch
	api::resource depth_resource = {};
	uint32_t depth_width = 0;
	uint32_t depth_height = 0;
	uint16_t depth_samples = 0;
	api::format depth_format = api::format::unknown;
	uint32_t depth_row_pitch = 0;
	uint64_t depth_byte_size = 0;
	// Diagnostics: whether the camera pass's OWN depth matched the color target
	// (=1 trustworthy; =0 we fell back to the tracked global depth), and the
	// camera-pass depth dimensions before any fallback.
	bool cam_depth_ok = false;
	uint32_t cam_pass_depth_width = 0;
	uint32_t cam_pass_depth_height = 0;
};

struct DepthReadback
{
	bool valid = false;
	uint64_t frame = 0;
	api::device *device = nullptr;
	api::resource buffer = {};
	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t row_pitch = 0;
	uint32_t tight_row_pitch = 0;
	uint64_t byte_size = 0;
	api::format format = api::format::unknown;
	uint16_t samples = 0;
	float near_cm = 0.0f;
};

struct ColorReadback
{
	bool valid = false;
	uint64_t frame = 0;
	api::device *device = nullptr;
	api::resource buffer = {};
	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t row_pitch = 0;
	uint32_t tight_row_pitch = 0;
	uint64_t byte_size = 0;
	api::format format = api::format::unknown;
	uint16_t samples = 0;
};

// The most-recent full-resolution main depth target seen this frame. Used when the
// camera pass binds a lower-res depth than the final color target (see the depth
// choice in on_reshade_begin_effects).
struct DepthTarget
{
	bool valid = false;
	uint64_t frame = 0;
	api::resource resource = {};
	uint32_t width = 0;
	uint32_t height = 0;
	uint16_t samples = 0;
	api::format format = api::format::unknown;
	uint32_t row_pitch = 0;
	uint32_t tight_row_pitch = 0;
	uint64_t byte_size = 0;
};

HMODULE g_module = nullptr;
std::filesystem::path g_game_dir;
std::filesystem::path g_output_dir;
std::filesystem::path g_depth_dir;
std::filesystem::path g_color_dir;
std::filesystem::path g_csv_path;
std::filesystem::path g_candidate_csv_path;
std::filesystem::path g_debug_path;
std::filesystem::path g_cbv_dump_dir;

std::mutex g_mutex;
std::unordered_set<std::string> g_dumped_cbv_keys;   // (resource,offset) already dumped
std::atomic<uint32_t> g_dumped_cbv_count = 0;
std::unordered_map<api::command_list *, CommandListState> g_cmd_states;
std::unordered_map<uint64_t, MappedBuffer> g_mapped_buffers;
CameraSample g_latest_sample;
DepthTarget g_latest_readable_depth;
DepthReadback g_pending_depth;
ColorReadback g_pending_color;
std::vector<DepthReadback> g_delayed_depths;
std::vector<ColorReadback> g_delayed_colors;
std::vector<CameraSample> g_frame_candidates;
std::unordered_set<std::string> g_frame_candidate_keys;
api::command_queue *g_graphics_queue = nullptr;
std::atomic<uint64_t> g_frame_index = 0;
std::atomic_bool g_capture_enabled = kCaptureStartsEnabled;
// Whether to also write viewable .bmp previews next to each .bin. Defaults to
// kSavePreviewImages, but an artist can override it at runtime by putting a
// "BMWCameraCapture.ini" next to the add-on with a line "save_bmp=0" (bin only)
// or "save_bmp=1" (bin + bmp). Read once in init_paths().
std::atomic_bool g_save_preview_images{kSavePreviewImages};
std::atomic_bool g_logged_map_failure = false;
std::atomic_bool g_logged_depth_failure = false;
std::atomic_bool g_logged_color_failure = false;
std::atomic_bool g_logged_depth_unsupported_format = false;
std::atomic_bool g_logged_depth_readback_info = false;
std::atomic_bool g_logged_color_readback_info = false;
std::atomic_bool g_logged_strict_sync_skip = false;
std::atomic_bool g_logged_depth_size_mismatch = false;
std::atomic_bool g_logged_candidate_truncation = false;
std::atomic_bool g_logged_depth_target_found = false;
std::atomic_bool g_logged_depth_target_miss = false;
std::atomic_bool g_logged_depth_apply_miss = false;
std::atomic_bool g_logged_depth_choice = false;
std::atomic_bool g_logged_preview_unsupported = false;
std::atomic_bool g_logged_igcs_connected = false;
// REC indicator state: true when the last sampled frame produced no CSV row.
std::atomic_bool g_rec_skipped = false;
// Recording-segment tracking: bump the session id on each F8 start; the first CSV
// row written in a segment gets seg_start=1.
std::atomic<uint32_t> g_capture_session = 0;
std::atomic_bool g_segment_start_pending = false;
// High-resolution timestamps. QPC frequency is fixed at load; the QPC epoch is
// re-latched on each F8 start so timestamp_s is "seconds since this capture began".
LARGE_INTEGER g_qpc_freq = {};
std::atomic<int64_t> g_capture_qpc_start = 0;
bool g_logged_first_match = false;
uint64_t g_frame_candidates_frame = UINT64_MAX;
uint32_t g_frame_candidate_draw_scans = 0;

uint64_t resource_key(api::resource resource)
{
	return resource.handle;
}

bool is_finite(float value)
{
	return std::isfinite(value) != 0;
}

float length3(const float v[3])
{
	return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

float dot3(const float a[3], const float b[3])
{
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

float radians_to_degrees(float radians)
{
	return radians * 57.29577951308232f;
}

// Row-major 4x4 multiply: out = a * b. Ported from GTACameraCapture.
void multiply_row_major_4x4(const float a[16], const float b[16], float out[16])
{
	float tmp[16] = {};
	for (int row = 0; row < 4; ++row)
		for (int col = 0; col < 4; ++col)
		{
			float v = 0.0f;
			for (int k = 0; k < 4; ++k)
				v += a[row * 4 + k] * b[k * 4 + col];
			tmp[row * 4 + col] = v;
		}
	std::memcpy(out, tmp, sizeof(tmp));
}

// Inverse of a row-vector affine transform whose 3x3 rotation part is (near-)
// orthonormal and translation is in row 3 (the UE ViewToWorld layout BMW reads):
//   M = [ R (3x3) 0 ; t (row3) 1 ]  with world = view * M.
// Inverse = [ R^T 0 ; -t*R^T 1 ]. Good enough here (basis passed validate_basis).
void invert_affine_row_major(const float m[16], float out[16])
{
	// R^T
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			out[r * 4 + c] = m[c * 4 + r];
	out[3] = 0.0f; out[7] = 0.0f; out[11] = 0.0f; out[15] = 1.0f;
	// -t * R^T  (t = row 3 of m)
	const float tx = m[12], ty = m[13], tz = m[14];
	for (int c = 0; c < 3; ++c)
		out[12 + c] = -(tx * out[0 * 4 + c] + ty * out[1 * 4 + c] + tz * out[2 * 4 + c]);
}

// Audible F8 confirmation. Beep() is synchronous, so run it detached to avoid
// hitching the present thread. Ascending = started, descending = stopped.
void play_toggle_cue(bool started)
{
	if (!kAudioCueOnToggle)
		return;
	std::thread([started] {
		if (started) { Beep(784, 90); Beep(1175, 140); }   // G5 -> D6
		else         { Beep(1175, 90); Beep(784, 140); }   // D6 -> G5
	}).detach();
}

double qpc_seconds_since(int64_t start_ticks)
{
	LARGE_INTEGER now = {};
	QueryPerformanceCounter(&now);
	if (g_qpc_freq.QuadPart == 0)
		return 0.0;
	return static_cast<double>(now.QuadPart - start_ticks) / static_cast<double>(g_qpc_freq.QuadPart);
}

double wall_clock_unix_seconds()
{
	FILETIME ft = {};
	GetSystemTimePreciseAsFileTime(&ft);
	ULARGE_INTEGER u = {};
	u.LowPart = ft.dwLowDateTime;
	u.HighPart = ft.dwHighDateTime;
	// FILETIME is 100ns ticks since 1601-01-01; 116444736000000000 = ticks to 1970.
	return (static_cast<double>(u.QuadPart) - 116444736000000000.0) / 1.0e7;
}

bool should_capture_frame(uint64_t frame)
{
	return kCaptureEveryNFrames <= 1 || (frame % kCaptureEveryNFrames) == 0;
}

uint32_t align_to(uint32_t value, uint32_t alignment)
{
	return (value + alignment - 1) / alignment * alignment;
}

bool supports_direct_depth_buffer_copy(api::format format)
{
	switch (format)
	{
	case api::format::d16_unorm:
	case api::format::d16_unorm_s8_uint:
	case api::format::d24_unorm_x8_uint:
	case api::format::d24_unorm_s8_uint:
	case api::format::d32_float:
	case api::format::d32_float_s8_uint:
		return true;
	default:
		return false;
	}
}

bool get_d3d12_copyable_footprint(api::resource resource, uint32_t subresource, uint32_t &row_pitch, uint32_t &tight_row_pitch, uint64_t &byte_size)
{
	if (resource.handle == 0)
		return false;

	ID3D12Resource *native_resource = reinterpret_cast<ID3D12Resource *>(resource.handle);
	ID3D12Device *native_device = nullptr;
	if (FAILED(native_resource->GetDevice(IID_PPV_ARGS(&native_device))) || native_device == nullptr)
		return false;

	const D3D12_RESOURCE_DESC desc = native_resource->GetDesc();
	D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = {};
	UINT num_rows = 0;
	UINT64 row_size = 0;
	UINT64 total_bytes = 0;
	native_device->GetCopyableFootprints(&desc, subresource, 1, 0, &footprint, &num_rows, &row_size, &total_bytes);
	native_device->Release();

	if (footprint.Footprint.RowPitch == 0 || row_size == 0 || total_bytes == 0)
		return false;

	row_pitch = footprint.Footprint.RowPitch;
	tight_row_pitch = row_size > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(row_size);
	byte_size = total_bytes;
	return true;
}

std::string screenshot_postfix(uint64_t frame)
{
	std::ostringstream ss;
	ss << "bmw_camera_frame_" << std::setw(8) << std::setfill('0') << frame;
	return ss.str();
}

std::string depth_postfix(uint64_t frame)
{
	return screenshot_postfix(frame) + ".depth";
}

std::string color_postfix(uint64_t frame)
{
	return screenshot_postfix(frame) + ".color";
}

std::string depth_relative_bin_path(const std::string &postfix)
{
	return "depth/" + postfix + ".bin";
}

std::string color_relative_bin_path(const std::string &postfix)
{
	return "color/" + postfix + ".bin";
}

void log_line(const std::string &message, reshade::log::level level = reshade::log::level::info)
{
	reshade::log::message(level, message.c_str());

	if (!g_debug_path.empty())
	{
		std::ofstream fp(g_debug_path, std::ios::app);
		if (fp)
			fp << message << '\n';
	}
}

std::string narrow_path(const std::filesystem::path &path)
{
	return path.string();
}

void write_csv_header(std::ofstream &fp)
{
	fp << "frame,root_param,draw_count,resource,cb_offset_hex,"
	      "pos_x_cm,pos_y_cm,pos_z_cm,pos_x_m,pos_y_m,pos_z_m,"
	      "right_x,right_y,right_z,up_x,up_y,up_z,fwd_x,fwd_y,fwd_z,"
	      "hfov_deg,vfov_deg,near_cm,projection_offset_hex,view_origin_offset_hex,view_to_world_offset_hex,"
	      "screenshot_postfix";

	for (int row = 0; row < 4; ++row)
	{
		for (int col = 0; col < 4; ++col)
			fp << ",proj_m" << row << col;
	}

	for (int col = 0; col < 4; ++col)
		fp << ",view_m3" << col;

	for (int row = 0; row < 4; ++row)
		for (int col = 0; col < 4; ++col)
			fp << ",world_to_view_m" << row << col;
	for (int row = 0; row < 4; ++row)
		for (int col = 0; col < 4; ++col)
			fp << ",world_to_clip_m" << row << col;
	for (int row = 0; row < 4; ++row)
		for (int col = 0; col < 4; ++col)
			fp << ",view_to_world_m" << row << col;

	fp << ",depth_file,depth_width,depth_height,depth_format,depth_row_pitch,depth_tight_row_pitch,depth_byte_size,depth_samples";
	fp << ",color_file,color_width,color_height,color_format,color_row_pitch,color_tight_row_pitch,color_byte_size,color_samples";
	fp << ",camera_frame,depth_frame,color_frame,strict_sync_ok,segment,seg_start,timestamp_s,timestamp_unix";
	fp << ",cam_depth_ok,cam_pass_depth_w,cam_pass_depth_h";
	// IGCS = engine-authoritative camera pushed by UUU (see connector above).
	fp << ",igcs_ok,igcs_cam_enabled,igcs_fov,igcs_pos_x,igcs_pos_y,igcs_pos_z";
	fp << ",igcs_qx,igcs_qy,igcs_qz,igcs_qw";
	fp << ",igcs_up_x,igcs_up_y,igcs_up_z,igcs_right_x,igcs_right_y,igcs_right_z,igcs_fwd_x,igcs_fwd_y,igcs_fwd_z";

	fp << '\n';
}

void write_candidate_csv_header(std::ofstream &fp)
{
	fp << "frame,candidate_index,selected_camera_sample,root_param,draw_count,pipeline_layout,resource,cb_offset_hex,"
	      "pos_x_cm,pos_y_cm,pos_z_cm,pos_x_m,pos_y_m,pos_z_m,"
	      "right_x,right_y,right_z,up_x,up_y,up_z,fwd_x,fwd_y,fwd_z,"
	      "hfov_deg,vfov_deg,near_cm,projection_offset_hex,view_origin_offset_hex,view_to_world_offset_hex";

	for (int row = 0; row < 4; ++row)
	{
		for (int col = 0; col < 4; ++col)
			fp << ",proj_m" << row << col;
	}

	for (int col = 0; col < 4; ++col)
		fp << ",view_m3" << col;

	fp << ",depth_resource,depth_width,depth_height,depth_format,depth_samples";
	fp << '\n';
}

void init_paths()
{
	wchar_t module_path[MAX_PATH] = {};
	GetModuleFileNameW(g_module, module_path, MAX_PATH);

	g_game_dir = std::filesystem::path(module_path).parent_path();
	g_output_dir = g_game_dir / L"BMWCameraCapture";
	g_depth_dir = g_output_dir / L"depth";
	g_color_dir = g_output_dir / L"color";
	g_csv_path = g_output_dir / L"bmw_camera_pos.csv";
	g_candidate_csv_path = g_output_dir / L"bmw_camera_candidates.csv";
	g_debug_path = g_output_dir / L"bmw_camera_capture_debug.log";
	g_cbv_dump_dir = g_output_dir / L"cbv_dumps";

	// Artist toggle: BMWCameraCapture.ini next to the add-on / game exe. A line
	// "save_bmp=0" -> write only the .bin (no viewable .bmp); "save_bmp=1" -> both.
	// No file / no such line -> keep the built-in default (kSavePreviewImages).
	{
		std::ifstream cfg(g_game_dir / L"BMWCameraCapture.ini");
		std::string line;
		while (cfg && std::getline(cfg, line))
		{
			line.erase(std::remove_if(line.begin(), line.end(),
				[](unsigned char c) { return c == ' ' || c == '\t' || c == '\r'; }), line.end());
			if (line.rfind("save_bmp=", 0) == 0)
			{
				const std::string v = line.substr(9);
				g_save_preview_images.store(!(v == "0" || v == "false" || v == "off"), std::memory_order_relaxed);
			}
		}
	}

	std::error_code ec;
	std::filesystem::create_directories(g_depth_dir, ec);
	std::filesystem::create_directories(g_color_dir, ec);

	log_line(g_save_preview_images.load(std::memory_order_relaxed)
		? "BMWCameraCapture: preview .bmp = ON (writing .bin + .bmp). Set save_bmp=0 in BMWCameraCapture.ini for .bin only."
		: "BMWCameraCapture: preview .bmp = OFF (writing .bin only, per BMWCameraCapture.ini save_bmp=0).");

	const bool need_header = !std::filesystem::exists(g_csv_path) || std::filesystem::file_size(g_csv_path) == 0;
	std::ofstream fp(g_csv_path, std::ios::app);
	if (fp && need_header)
		write_csv_header(fp);
	else if (!need_header)
	{
		std::ifstream existing(g_csv_path);
		std::string header;
		std::getline(existing, header);
		if (header.find("proj_m00") == std::string::npos)
			log_line("BMWCameraCapture: existing CSV header has no projection matrix columns. Rename/delete old bmw_camera_pos.csv before a new capture if you want a clean header.", reshade::log::level::warning);
		else if (header.find("depth_file") == std::string::npos)
			log_line("BMWCameraCapture: existing CSV header has no depth readback columns. Rename/delete old bmw_camera_pos.csv before a new capture if you want a clean header.", reshade::log::level::warning);
		else if (header.find("color_file") == std::string::npos)
			log_line("BMWCameraCapture: existing CSV header has no color readback columns. Rename/delete old bmw_camera_pos.csv before a new capture if you want a clean header.", reshade::log::level::warning);
		else if (header.find("strict_sync_ok") == std::string::npos)
			log_line("BMWCameraCapture: existing CSV header has no strict sync columns. Rename/delete old bmw_camera_pos.csv before a new capture if you want sync validation columns.", reshade::log::level::warning);
	}

	if (kLogCameraCandidates)
	{
		const bool need_candidate_header = !std::filesystem::exists(g_candidate_csv_path) || std::filesystem::file_size(g_candidate_csv_path) == 0;
		std::ofstream candidates(g_candidate_csv_path, std::ios::app);
		if (candidates && need_candidate_header)
			write_candidate_csv_header(candidates);
	}
}

bool validate_basis(const float right[3], const float up[3], const float fwd[3])
{
	const float rl = length3(right);
	const float ul = length3(up);
	const float fl = length3(fwd);

	if (std::abs(rl - 1.0f) > 0.15f || std::abs(ul - 1.0f) > 0.15f || std::abs(fl - 1.0f) > 0.15f)
		return false;

	if (std::abs(dot3(right, up)) > 0.2f || std::abs(dot3(right, fwd)) > 0.2f || std::abs(dot3(up, fwd)) > 0.2f)
		return false;

	return true;
}

bool parse_camera_from_bytes(const uint8_t *base, CameraSample &sample)
{
	const float *proj = reinterpret_cast<const float *>(base + kProjectionOffset);

	for (int i = 0; i < 16; ++i)
	{
		if (!is_finite(proj[i]))
			return false;
		sample.projection[i] = proj[i];
	}

	const float p00 = proj[0];
	const float p11 = proj[5];
	const float p22 = proj[10];
	const float p23 = proj[11];
	const float p32 = proj[14];
	const float p33 = proj[15];

	if (p00 < 0.1f || p00 > 10.0f || p11 < 0.1f || p11 > 10.0f)
		return false;

	// UE reversed-Z infinite projection in this capture:
	// [x 0 0 0; 0 y 0 0; 0 0 0 1; 0 0 Near 0]
	if (std::abs(p22) > 0.02f || std::abs(p23 - 1.0f) > 0.02f || std::abs(p33) > 0.02f)
		return false;
	if (p32 < 1.0f || p32 > 1000.0f)
		return false;

	const float *origin = reinterpret_cast<const float *>(base + kViewOriginOffset);
	for (int i = 0; i < 3; ++i)
	{
		if (!is_finite(origin[i]) || std::abs(origin[i]) > 100000000.0f)
			return false;
		sample.pos_cm[i] = origin[i];
	}

	const float *view_to_world = reinterpret_cast<const float *>(base + kViewToWorldOffset);
	for (int i = 0; i < 16; ++i)
	{
		if (!is_finite(view_to_world[i]))
			return false;
		sample.view_to_world[i] = view_to_world[i];
	}
	for (int i = 0; i < 3; ++i)
	{
		sample.right[i] = view_to_world[i];
		sample.up[i] = view_to_world[4 + i];
		sample.fwd[i] = view_to_world[8 + i];
	}
	for (int i = 0; i < 4; ++i)
		sample.view_to_world_row3[i] = view_to_world[12 + i];

	if (!validate_basis(sample.right, sample.up, sample.fwd))
		return false;

	if (kFilterLikelyMainView)
	{
		if (sample.up[2] < kMinMainViewUpZ)
			return false;
		if (std::abs(sample.fwd[2]) > kMaxMainViewAbsForwardZ)
			return false;
	}

	// Derive the extra matrices GTA also exports (UE gives us ViewToWorld + Projection
	// directly; WorldToView is its inverse, WorldToClip = WorldToView * Projection).
	invert_affine_row_major(sample.view_to_world, sample.world_to_view);
	multiply_row_major_4x4(sample.world_to_view, sample.projection, sample.world_to_clip);

	sample.hfov_deg = radians_to_degrees(2.0f * std::atan(1.0f / p00));
	sample.vfov_deg = radians_to_degrees(2.0f * std::atan(1.0f / p11));
	sample.near_cm = p32;
	return true;
}

bool try_read_from_mapped_shadow(const TrackedCBV &cbv, CameraSample &sample)
{
	const auto it = g_mapped_buffers.find(resource_key(cbv.buffer));
	if (it == g_mapped_buffers.end() || it->second.data == nullptr)
		return false;

	const MappedBuffer &mapped = it->second;
	if (cbv.offset < mapped.offset)
		return false;

	const uint64_t relative = cbv.offset - mapped.offset;
	if (mapped.size != UINT64_MAX && relative + kReadSize > mapped.size)
		return false;

	return parse_camera_from_bytes(mapped.data + relative, sample);
}

bool try_read_from_device(api::device *device, const TrackedCBV &cbv, CameraSample &sample)
{
	api::resource_desc desc = device->get_resource_desc(cbv.buffer);
	if (desc.type != api::resource_type::buffer)
		return false;

	if (desc.buffer.size != 0 && cbv.offset + kReadSize > desc.buffer.size)
		return false;

	void *mapped = nullptr;
	if (!device->map_buffer_region(cbv.buffer, cbv.offset, kReadSize, api::map_access::read_only, &mapped) || mapped == nullptr)
	{
		if (!g_logged_map_failure.exchange(true))
			log_line("BMWCameraCapture: direct map failed at least once; trying cached mapped upload pointers.", reshade::log::level::warning);
		return false;
	}

	const bool ok = parse_camera_from_bytes(static_cast<const uint8_t *>(mapped), sample);
	device->unmap_buffer_region(cbv.buffer);
	return ok;
}

// CALIBRATION: dump the raw bytes of one constant buffer to cbv_dumps/, once per
// unique (resource,offset). Used to find a new game's camera-CBV offsets offline.
void dump_cbv_raw(api::device *device, const TrackedCBV &cbv)
{
	if (g_dumped_cbv_count.load(std::memory_order_relaxed) >= kDumpMaxFiles)
		return;

	const std::string key = std::to_string(resource_key(cbv.buffer)) + "_" + std::to_string(cbv.offset);
	{
		std::lock_guard<std::mutex> lock(g_mutex);
		if (g_dumped_cbv_keys.count(key))
			return;
	}

	api::resource_desc desc = device->get_resource_desc(cbv.buffer);
	if (desc.type != api::resource_type::buffer)
		return;

	uint64_t avail = (desc.buffer.size != 0 && desc.buffer.size > cbv.offset)
		? (desc.buffer.size - cbv.offset) : kDumpBytes;
	const uint64_t n = avail < kDumpBytes ? avail : kDumpBytes;
	if (n < 0x100)   // too small to hold a UE view uniform buffer
		return;

	void *mapped = nullptr;
	if (!device->map_buffer_region(cbv.buffer, cbv.offset, n, api::map_access::read_only, &mapped) || mapped == nullptr)
		return;
	std::vector<uint8_t> bytes(static_cast<const uint8_t *>(mapped), static_cast<const uint8_t *>(mapped) + n);
	device->unmap_buffer_region(cbv.buffer);

	std::error_code ec;
	std::filesystem::create_directories(g_cbv_dump_dir, ec);
	const std::filesystem::path out = g_cbv_dump_dir / (std::string("cbv_") + key + ".bin");
	std::ofstream fp(out, std::ios::binary);
	if (!fp)
		return;
	fp.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
	fp.close();

	std::lock_guard<std::mutex> lock(g_mutex);
	if (g_dumped_cbv_keys.insert(key).second)
		g_dumped_cbv_count.fetch_add(1, std::memory_order_relaxed);
}

void publish_sample(const CameraSample &sample)
{
	g_latest_sample = sample;

	if (!g_logged_first_match)
	{
		g_logged_first_match = true;

		std::ostringstream ss;
		ss << "BMWCameraCapture: matched UE camera CBV. CSV: " << narrow_path(g_csv_path);
		log_line(ss.str());
	}
}

std::string candidate_key(uint64_t pipeline_layout, uint32_t root_param, uint64_t resource, uint64_t cb_offset)
{
	std::ostringstream ss;
	ss << std::hex << pipeline_layout << '|'
	   << std::dec << root_param << '|'
	   << std::hex << resource << '|'
	   << cb_offset;
	return ss.str();
}

std::string candidate_key(const CameraSample &sample)
{
	return candidate_key(sample.pipeline_layout, sample.root_param, sample.resource, sample.cb_offset);
}

void reset_candidate_frame_locked(uint64_t frame)
{
	if (g_frame_candidates_frame == frame)
		return;

	g_frame_candidates_frame = frame;
	g_frame_candidates.clear();
	g_frame_candidate_keys.clear();
	g_frame_candidate_draw_scans = 0;
}

bool candidate_draw_scan_budget_available_locked(uint64_t frame)
{
	if (!kLogCameraCandidates)
		return false;

	reset_candidate_frame_locked(frame);
	if (g_frame_candidate_draw_scans >= kMaxCameraCandidateDrawScansPerFrame)
		return false;

	++g_frame_candidate_draw_scans;
	return true;
}

void record_candidate_locked(const CameraSample &sample)
{
	if (!kLogCameraCandidates)
		return;

	reset_candidate_frame_locked(sample.frame);

	if (g_frame_candidates.size() >= kMaxCameraCandidateRowsPerFrame)
	{
		if (!g_logged_candidate_truncation.exchange(true))
			log_line("BMWCameraCapture: candidate CBV logging hit the per-frame row cap; increase kMaxCameraCandidateRowsPerFrame only for short diagnostic captures.", reshade::log::level::warning);
		return;
	}

	const std::string key = candidate_key(sample);
	if (g_frame_candidate_keys.find(key) == g_frame_candidate_keys.end())
		return;

	g_frame_candidates.push_back(sample);
}

bool should_skip_candidate_cbv_locked(uint64_t frame, const TrackedCBV &cbv)
{
	if (!kLogCameraCandidates)
		return false;

	reset_candidate_frame_locked(frame);

	if (g_frame_candidates.size() >= kMaxCameraCandidateRowsPerFrame)
	{
		if (!g_logged_candidate_truncation.exchange(true))
			log_line("BMWCameraCapture: candidate CBV logging hit the per-frame row cap; increase kMaxCameraCandidateRowsPerFrame only for short diagnostic captures.", reshade::log::level::warning);
		return true;
	}

	const std::string key = candidate_key(cbv.pipeline_layout, cbv.root_param, resource_key(cbv.buffer), cbv.offset);
	if (g_frame_candidate_keys.find(key) != g_frame_candidate_keys.end())
		return true;

	g_frame_candidate_keys.insert(key);
	return false;
}

bool same_camera_identity(const CameraSample &a, const CameraSample &b)
{
	return a.valid && b.valid &&
		a.frame == b.frame &&
		a.root_param == b.root_param &&
		a.pipeline_layout == b.pipeline_layout &&
		a.resource == b.resource &&
		a.cb_offset == b.cb_offset;
}

void write_candidate_rows(const std::vector<CameraSample> &candidates, const CameraSample &selected)
{
	if (!kLogCameraCandidates || candidates.empty() || g_candidate_csv_path.empty())
		return;

	std::ofstream fp(g_candidate_csv_path, std::ios::app);
	if (!fp)
		return;

	fp << std::fixed << std::setprecision(6);

	for (size_t i = 0; i < candidates.size(); ++i)
	{
		const CameraSample &sample = candidates[i];
		fp << sample.frame << ','
		   << i << ','
		   << (same_camera_identity(sample, selected) ? 1 : 0) << ','
		   << sample.root_param << ','
		   << sample.draw_count << ','
		   << "0x" << std::hex << sample.pipeline_layout << std::dec << ','
		   << "0x" << std::hex << sample.resource << std::dec << ','
		   << "0x" << std::hex << sample.cb_offset << std::dec << ','
		   << sample.pos_cm[0] << ',' << sample.pos_cm[1] << ',' << sample.pos_cm[2] << ','
		   << sample.pos_cm[0] * 0.01f << ',' << sample.pos_cm[1] * 0.01f << ',' << sample.pos_cm[2] * 0.01f << ','
		   << sample.right[0] << ',' << sample.right[1] << ',' << sample.right[2] << ','
		   << sample.up[0] << ',' << sample.up[1] << ',' << sample.up[2] << ','
		   << sample.fwd[0] << ',' << sample.fwd[1] << ',' << sample.fwd[2] << ','
		   << sample.hfov_deg << ',' << sample.vfov_deg << ',' << sample.near_cm << ','
		   << "0x" << std::hex << kProjectionOffset << std::dec << ','
		   << "0x" << std::hex << kViewOriginOffset << std::dec << ','
		   << "0x" << std::hex << kViewToWorldOffset << std::dec;

		for (int j = 0; j < 16; ++j)
			fp << ',' << sample.projection[j];

		for (int j = 0; j < 4; ++j)
			fp << ',' << sample.view_to_world_row3[j];

		fp << ','
		   << "0x" << std::hex << resource_key(sample.depth_resource) << std::dec << ','
		   << sample.depth_width << ','
		   << sample.depth_height << ','
		   << static_cast<uint32_t>(sample.depth_format) << ','
		   << sample.depth_samples << '\n';
	}
}

bool attach_depth_from_dsv(api::device *device, api::resource_view dsv, CameraSample &sample)
{
	if (device == nullptr || dsv.handle == 0)
		return false;

	const api::resource depth = device->get_resource_from_view(dsv);
	if (depth.handle == 0)
		return false;

	const api::resource_desc desc = device->get_resource_desc(depth);
	if (desc.type != api::resource_type::texture_2d)
		return false;

	if (desc.texture.width == 0 || desc.texture.height == 0)
		return false;

	const api::resource_view_desc view_desc = device->get_resource_view_desc(dsv);
	const api::format depth_format = view_desc.format != api::format::unknown ? view_desc.format : desc.texture.format;

	if (desc.texture.samples > 1)
	{
		if (!g_logged_depth_failure.exchange(true))
			log_line("BMWCameraCapture: depth readback currently skips MSAA depth resources.", reshade::log::level::warning);
		return false;
	}

	const uint32_t tight_row_pitch = api::format_row_pitch(depth_format, desc.texture.width);
	if (tight_row_pitch == 0)
		return false;

	const uint32_t row_pitch = align_to(tight_row_pitch, kTextureReadbackPitchAlignment);

	sample.depth_resource = depth;
	sample.depth_width = desc.texture.width;
	sample.depth_height = desc.texture.height;
	sample.depth_samples = desc.texture.samples;
	sample.depth_format = depth_format;
	sample.depth_row_pitch = row_pitch;
	sample.depth_byte_size = static_cast<uint64_t>(row_pitch) * desc.texture.height;

	return true;
}

// Records the given DSV as the "full-resolution main depth" candidate for this
// frame, if it is large enough and non-MSAA. Ported from GTACameraCapture.
bool make_readable_depth_target(api::device *device, api::resource_view dsv, uint64_t frame, DepthTarget &target)
{
	target = DepthTarget{};
	if (device == nullptr || dsv.handle == 0)
		return false;

	const api::resource depth = device->get_resource_from_view(dsv);
	if (depth.handle == 0)
		return false;

	const api::resource_desc desc = device->get_resource_desc(depth);
	if (desc.type != api::resource_type::texture_2d)
		return false;
	if (desc.texture.width < kMinMainTargetWidth || desc.texture.height < kMinMainTargetHeight)
		return false;
	if (desc.texture.samples > 1)
	{
		if (!g_logged_depth_target_miss.exchange(true))
		{
			std::ostringstream ss;
			ss << "BMWCameraCapture: make_readable_depth_target: skipped MSAA depth. samples="
			   << desc.texture.samples << " size=" << desc.texture.width << 'x' << desc.texture.height;
			log_line(ss.str(), reshade::log::level::warning);
		}
		return false;
	}

	const api::resource_view_desc view_desc = device->get_resource_view_desc(dsv);
	const api::format depth_format = view_desc.format != api::format::unknown ? view_desc.format : desc.texture.format;
	const uint32_t tight_row_pitch = api::format_row_pitch(depth_format, desc.texture.width);
	if (tight_row_pitch == 0)
		return false;

	const uint32_t row_pitch = align_to(tight_row_pitch, kTextureReadbackPitchAlignment);

	target.valid = true;
	target.frame = frame;
	target.resource = depth;
	target.width = desc.texture.width;
	target.height = desc.texture.height;
	target.samples = desc.texture.samples;
	target.format = depth_format;
	target.row_pitch = row_pitch;
	target.tight_row_pitch = tight_row_pitch;
	target.byte_size = static_cast<uint64_t>(row_pitch) * desc.texture.height;

	if (!g_logged_depth_target_found.exchange(true))
	{
		std::ostringstream ss;
		ss << "BMWCameraCapture: depth target found. size=" << desc.texture.width << 'x' << desc.texture.height
		   << " depth_fmt=" << static_cast<uint32_t>(depth_format)
		   << " row_pitch=" << row_pitch << " frame=" << frame;
		log_line(ss.str());
	}
	return true;
}

// Copies a tracked full-resolution depth target into the sample. No size check
// here; the final size validation vs the color target is done later in
// depth_size_matches_color_target. Ported from GTACameraCapture.
void apply_readable_depth_target(CameraSample &sample, const DepthTarget &target)
{
	if (!target.valid || target.frame != sample.frame)
	{
		if (!g_logged_depth_apply_miss.exchange(true))
		{
			std::ostringstream ss;
			ss << "BMWCameraCapture: apply_readable_depth_target: skipped."
			   << " target.valid=" << target.valid
			   << " target.frame=" << target.frame
			   << " sample.frame=" << sample.frame;
			log_line(ss.str(), reshade::log::level::warning);
		}
		return;
	}

	sample.depth_resource = target.resource;
	sample.depth_width = target.width;
	sample.depth_height = target.height;
	sample.depth_samples = target.samples;
	sample.depth_format = target.format;
	sample.depth_row_pitch = target.row_pitch;
	sample.depth_byte_size = target.byte_size;
}

void destroy_depth_readback(DepthReadback &readback)
{
	if (readback.valid && readback.device != nullptr && readback.buffer.handle != 0)
		readback.device->destroy_resource(readback.buffer);
	readback = DepthReadback{};
}

// Minimal dependency-free 24-bit BMP writer. `bgr_topdown` is width*height*3 bytes,
// B,G,R per pixel, top row first. BMP is stored bottom-up with 4-byte row padding.
void write_bmp_bgr(const std::filesystem::path &path, uint32_t width, uint32_t height, const std::vector<uint8_t> &bgr_topdown)
{
	if (width == 0 || height == 0 || bgr_topdown.size() < static_cast<size_t>(width) * height * 3)
		return;

	const uint32_t row_stride = (width * 3 + 3) & ~3u;
	const uint32_t image_size = row_stride * height;
	const uint32_t file_size = 54 + image_size;

	std::ofstream f(path, std::ios::binary);
	if (!f)
		return;

	auto put16 = [&](uint16_t v) { f.put(static_cast<char>(v & 0xFF)); f.put(static_cast<char>((v >> 8) & 0xFF)); };
	auto put32 = [&](uint32_t v) {
		f.put(static_cast<char>(v & 0xFF)); f.put(static_cast<char>((v >> 8) & 0xFF));
		f.put(static_cast<char>((v >> 16) & 0xFF)); f.put(static_cast<char>((v >> 24) & 0xFF));
	};

	f.put('B'); f.put('M');
	put32(file_size); put32(0); put32(54);
	put32(40); put32(width); put32(height); put16(1); put16(24);
	put32(0); put32(image_size); put32(0); put32(0); put32(0); put32(0);

	std::vector<uint8_t> row(row_stride, 0);
	for (int y = static_cast<int>(height) - 1; y >= 0; --y)
	{
		std::memcpy(row.data(), bgr_topdown.data() + static_cast<size_t>(y) * width * 3, static_cast<size_t>(width) * 3);
		f.write(reinterpret_cast<const char *>(row.data()), row_stride);
	}
}

// Decode a mapped color readback into top-down 8-bit BGR. Mirrors the channel
// handling in gta_process_tools/conver_bmw.py. Returns false for unsupported formats.
bool decode_color_to_bgr(const uint8_t *data, uint32_t width, uint32_t height, uint32_t row_pitch, api::format format, std::vector<uint8_t> &out_bgr)
{
	if (data == nullptr || width == 0 || height == 0)
		return false;

	out_bgr.assign(static_cast<size_t>(width) * height * 3, 0);

	for (uint32_t y = 0; y < height; ++y)
	{
		const uint8_t *src = data + static_cast<size_t>(y) * row_pitch;
		uint8_t *dst = out_bgr.data() + static_cast<size_t>(y) * width * 3;
		for (uint32_t x = 0; x < width; ++x)
		{
			uint8_t b = 0, g = 0, r = 0;
			switch (format)
			{
			case api::format::r10g10b10a2_unorm:
			{
				uint32_t v = 0;
				std::memcpy(&v, src + static_cast<size_t>(x) * 4, 4);
				r = static_cast<uint8_t>((v & 0x3FF) * 255u / 1023u);
				g = static_cast<uint8_t>(((v >> 10) & 0x3FF) * 255u / 1023u);
				b = static_cast<uint8_t>(((v >> 20) & 0x3FF) * 255u / 1023u);
				break;
			}
			case api::format::r8g8b8a8_unorm:
			case api::format::r8g8b8a8_unorm_srgb:
			{
				const uint8_t *p = src + static_cast<size_t>(x) * 4;
				r = p[0]; g = p[1]; b = p[2];
				break;
			}
			case api::format::b8g8r8a8_unorm:
			{
				const uint8_t *p = src + static_cast<size_t>(x) * 4;
				b = p[0]; g = p[1]; r = p[2];
				break;
			}
			default:
				return false;
			}
			dst[static_cast<size_t>(x) * 3 + 0] = b;
			dst[static_cast<size_t>(x) * 3 + 1] = g;
			dst[static_cast<size_t>(x) * 3 + 2] = r;
		}
	}
	return true;
}

// Decode a mapped R32-float reversed-Z depth readback into a top-down grayscale
// BGR preview: linearize z = near / raw (like conver_bmw.py raw_reversed_z_to_meters),
// then normalize in log space (near = bright, far/sky = dark). Preview only.
bool decode_depth_to_gray(const uint8_t *data, uint32_t width, uint32_t height, uint32_t row_pitch, uint32_t tight_row_pitch, float near_cm, std::vector<uint8_t> &out_bgr)
{
	if (data == nullptr || width == 0 || height == 0)
		return false;
	const uint32_t bpp = tight_row_pitch / width;
	if (bpp != 4)   // only the R32-float depth plane is previewed
		return false;

	const float near_m = near_cm * 0.01f;
	std::vector<float> lz(static_cast<size_t>(width) * height, 0.0f);
	std::vector<uint8_t> sky(static_cast<size_t>(width) * height, 1);
	float lmin = 0.0f, lmax = 0.0f;
	bool have = false;

	for (uint32_t y = 0; y < height; ++y)
	{
		const uint8_t *src = data + static_cast<size_t>(y) * row_pitch;
		for (uint32_t x = 0; x < width; ++x)
		{
			float raw = 0.0f;
			std::memcpy(&raw, src + static_cast<size_t>(x) * 4, 4);
			if (!is_finite(raw) || raw <= 1e-6f)
				continue;   // sky / far

			const float z = near_m / raw;
			const float l = std::log(z + 1.0f);
			const size_t idx = static_cast<size_t>(y) * width + x;
			lz[idx] = l;
			sky[idx] = 0;
			if (!have) { lmin = lmax = l; have = true; }
			else { lmin = std::min(lmin, l); lmax = std::max(lmax, l); }
		}
	}

	out_bgr.assign(static_cast<size_t>(width) * height * 3, 0);
	const float range = (lmax > lmin) ? (lmax - lmin) : 1.0f;
	for (size_t i = 0; i < lz.size(); ++i)
	{
		uint8_t gray = 0;
		if (!sky[i])
		{
			float n = (lz[i] - lmin) / range;
			n = n < 0.0f ? 0.0f : (n > 1.0f ? 1.0f : n);
			gray = static_cast<uint8_t>((1.0f - n) * 255.0f);   // near bright, far dark
		}
		out_bgr[i * 3 + 0] = gray;
		out_bgr[i * 3 + 1] = gray;
		out_bgr[i * 3 + 2] = gray;
	}
	return true;
}

bool write_depth_readback(DepthReadback &readback, const std::string &postfix)
{
	if (!readback.valid || readback.device == nullptr || readback.buffer.handle == 0)
		return false;

	if (kWaitIdleBeforeReadbackMap && g_graphics_queue != nullptr)
		g_graphics_queue->wait_idle();

	void *mapped = nullptr;
	if (!readback.device->map_buffer_region(readback.buffer, 0, readback.byte_size, api::map_access::read_only, &mapped) || mapped == nullptr)
	{
		log_line("BMWCameraCapture: failed to map depth readback buffer.", reshade::log::level::warning);
		return false;
	}

	const std::filesystem::path bin_path = g_depth_dir / (postfix + ".bin");
	std::ofstream bin(bin_path, std::ios::binary);
	if (bin)
		bin.write(static_cast<const char *>(mapped), static_cast<std::streamsize>(readback.byte_size));

	if (g_save_preview_images.load(std::memory_order_relaxed))
	{
		std::vector<uint8_t> bgr;
		if (decode_depth_to_gray(static_cast<const uint8_t *>(mapped), readback.width, readback.height, readback.row_pitch, readback.tight_row_pitch, readback.near_cm, bgr))
			write_bmp_bgr(g_depth_dir / (postfix + ".bmp"), readback.width, readback.height, bgr);
	}

	readback.device->unmap_buffer_region(readback.buffer);

	const std::filesystem::path json_path = g_depth_dir / (postfix + ".json");
	std::ofstream json(json_path);
	if (json)
	{
		json << "{\n"
		     << "  \"frame\": " << readback.frame << ",\n"
		     << "  \"file\": \"" << postfix << ".bin\",\n"
		     << "  \"width\": " << readback.width << ",\n"
		     << "  \"height\": " << readback.height << ",\n"
		     << "  \"format\": " << static_cast<uint32_t>(readback.format) << ",\n"
		     << "  \"row_pitch\": " << readback.row_pitch << ",\n"
		     << "  \"tight_row_pitch\": " << readback.tight_row_pitch << ",\n"
		     << "  \"byte_size\": " << readback.byte_size << ",\n"
		     << "  \"samples\": " << readback.samples << ",\n"
		     << "  \"near_cm\": " << std::fixed << std::setprecision(6) << readback.near_cm << "\n"
		     << "}\n";
	}

	return static_cast<bool>(bin);
}

void destroy_color_readback(ColorReadback &readback)
{
	if (readback.valid && readback.device != nullptr && readback.buffer.handle != 0)
		readback.device->destroy_resource(readback.buffer);
	readback = ColorReadback{};
}

bool write_color_readback(ColorReadback &readback, const std::string &postfix)
{
	if (!readback.valid || readback.device == nullptr || readback.buffer.handle == 0)
		return false;

	if (kWaitIdleBeforeReadbackMap && g_graphics_queue != nullptr)
		g_graphics_queue->wait_idle();

	void *mapped = nullptr;
	if (!readback.device->map_buffer_region(readback.buffer, 0, readback.byte_size, api::map_access::read_only, &mapped) || mapped == nullptr)
	{
		log_line("BMWCameraCapture: failed to map color readback buffer.", reshade::log::level::warning);
		return false;
	}

	const std::filesystem::path bin_path = g_color_dir / (postfix + ".bin");
	std::ofstream bin(bin_path, std::ios::binary);
	if (bin)
		bin.write(static_cast<const char *>(mapped), static_cast<std::streamsize>(readback.byte_size));

	if (g_save_preview_images.load(std::memory_order_relaxed))
	{
		std::vector<uint8_t> bgr;
		if (decode_color_to_bgr(static_cast<const uint8_t *>(mapped), readback.width, readback.height, readback.row_pitch, readback.format, bgr))
			write_bmp_bgr(g_color_dir / (postfix + ".bmp"), readback.width, readback.height, bgr);
		else if (!g_logged_preview_unsupported.exchange(true))
		{
			std::ostringstream ss;
			ss << "BMWCameraCapture: preview image skipped for unsupported color format="
			   << static_cast<uint32_t>(readback.format) << " (.bin still written).";
			log_line(ss.str(), reshade::log::level::warning);
		}
	}

	readback.device->unmap_buffer_region(readback.buffer);

	const std::filesystem::path json_path = g_color_dir / (postfix + ".json");
	std::ofstream json(json_path);
	if (json)
	{
		json << "{\n"
		     << "  \"frame\": " << readback.frame << ",\n"
		     << "  \"file\": \"" << postfix << ".bin\",\n"
		     << "  \"width\": " << readback.width << ",\n"
		     << "  \"height\": " << readback.height << ",\n"
		     << "  \"format\": " << static_cast<uint32_t>(readback.format) << ",\n"
		     << "  \"row_pitch\": " << readback.row_pitch << ",\n"
		     << "  \"tight_row_pitch\": " << readback.tight_row_pitch << ",\n"
		     << "  \"byte_size\": " << readback.byte_size << ",\n"
		     << "  \"samples\": " << readback.samples << "\n"
		     << "}\n";
	}

	return static_cast<bool>(bin);
}

void queue_delayed_readbacks(DepthReadback &depth, ColorReadback &color)
{
	if (!depth.valid && !color.valid)
		return;

	std::lock_guard<std::mutex> lock(g_mutex);
	if (depth.valid)
	{
		g_delayed_depths.push_back(depth);
		depth = DepthReadback{};
	}
	if (color.valid)
	{
		g_delayed_colors.push_back(color);
		color = ColorReadback{};
	}
}

void flush_ready_readbacks(uint64_t frame, bool force)
{
	std::vector<DepthReadback> ready_depths;
	std::vector<ColorReadback> ready_colors;

	{
		std::lock_guard<std::mutex> lock(g_mutex);

		const auto depth_it = std::remove_if(
			g_delayed_depths.begin(),
			g_delayed_depths.end(),
			[&](const DepthReadback &readback) {
				if (force || readback.frame + kReadbackDelayFrames <= frame)
				{
					ready_depths.push_back(readback);
					return true;
				}
				return false;
			});
		g_delayed_depths.erase(depth_it, g_delayed_depths.end());

		const auto color_it = std::remove_if(
			g_delayed_colors.begin(),
			g_delayed_colors.end(),
			[&](const ColorReadback &readback) {
				if (force || readback.frame + kReadbackDelayFrames <= frame)
				{
					ready_colors.push_back(readback);
					return true;
				}
				return false;
			});
		g_delayed_colors.erase(color_it, g_delayed_colors.end());
	}

	for (DepthReadback &readback : ready_depths)
	{
		write_depth_readback(readback, depth_postfix(readback.frame));
		destroy_depth_readback(readback);
	}

	for (ColorReadback &readback : ready_colors)
	{
		write_color_readback(readback, color_postfix(readback.frame));
		destroy_color_readback(readback);
	}
}

bool enqueue_depth_readback(api::command_list *cmd_list, const CameraSample &sample)
{
	if (!kSaveDepthReadbackOnCapture || cmd_list == nullptr || sample.depth_resource.handle == 0)
		return false;
	if (sample.depth_width == 0 || sample.depth_height == 0 || sample.depth_row_pitch == 0 || sample.depth_byte_size == 0)
		return false;

	api::device *device = cmd_list->get_device();
	if (device == nullptr)
		return false;

	uint32_t row_pitch = sample.depth_row_pitch;
	uint32_t tight_row_pitch = api::format_row_pitch(sample.depth_format, sample.depth_width);
	uint64_t byte_size = sample.depth_byte_size;
	if (tight_row_pitch == 0)
		return false;

	if (!supports_direct_depth_buffer_copy(sample.depth_format))
	{
		if (!g_logged_depth_unsupported_format.exchange(true))
		{
			std::ostringstream ss;
			ss << "BMWCameraCapture: skipping direct depth readback for format="
			   << static_cast<uint32_t>(sample.depth_format)
			   << ". Stencil/planar depth formats need a shader/backup-texture path; direct texture-to-buffer copy can invalidate the D3D12 command list.";
			log_line(ss.str(), reshade::log::level::warning);
		}
		return false;
	}

	if (device->get_api() == api::device_api::d3d12)
	{
		uint32_t native_row_pitch = 0;
		uint32_t native_tight_row_pitch = 0;
		uint64_t native_byte_size = 0;
		if (get_d3d12_copyable_footprint(sample.depth_resource, 0, native_row_pitch, native_tight_row_pitch, native_byte_size))
		{
			row_pitch = native_row_pitch;
			tight_row_pitch = native_tight_row_pitch;
			byte_size = native_byte_size;
		}
	}

	const uint32_t bytes_per_pixel = sample.depth_width != 0 ? std::max(1u, tight_row_pitch / sample.depth_width) : 1u;
	const uint32_t row_length_pixels = row_pitch / bytes_per_pixel;

	api::resource readback_buffer = {};
	const api::resource_desc buffer_desc(byte_size, api::memory_heap::gpu_to_cpu, api::resource_usage::copy_dest);
	if (!device->create_resource(buffer_desc, nullptr, api::resource_usage::copy_dest, &readback_buffer))
	{
		if (!g_logged_depth_failure.exchange(true))
			log_line("BMWCameraCapture: failed to create depth readback buffer.", reshade::log::level::warning);
		return false;
	}

	if (!g_logged_depth_readback_info.exchange(true))
	{
		std::ostringstream ss;
		ss << "BMWCameraCapture: depth readback copy queued. size="
		   << sample.depth_width << 'x' << sample.depth_height
		   << " format=" << static_cast<uint32_t>(sample.depth_format)
		   << " row_pitch=" << row_pitch
		   << " tight_row_pitch=" << tight_row_pitch
		   << " bytes=" << byte_size
		   << " resource=0x" << std::hex << sample.depth_resource.handle << std::dec;
		log_line(ss.str());
	}

	cmd_list->barrier(sample.depth_resource, api::resource_usage::depth_stencil_write, api::resource_usage::copy_source);
	if (device->get_api() == api::device_api::d3d12)
		cmd_list->copy_texture_region(sample.depth_resource, 0, nullptr, readback_buffer, 0, nullptr);
	else
		cmd_list->copy_texture_to_buffer(sample.depth_resource, 0, nullptr, readback_buffer, 0, row_length_pixels, sample.depth_height);
	cmd_list->barrier(sample.depth_resource, api::resource_usage::copy_source, api::resource_usage::depth_stencil_write);

	DepthReadback readback = {};
	readback.valid = true;
	readback.frame = sample.frame;
	readback.device = device;
	readback.buffer = readback_buffer;
	readback.width = sample.depth_width;
	readback.height = sample.depth_height;
	readback.row_pitch = row_pitch;
	readback.tight_row_pitch = tight_row_pitch;
	readback.byte_size = byte_size;
	readback.format = sample.depth_format;
	readback.samples = sample.depth_samples;
	readback.near_cm = sample.near_cm;

	std::lock_guard<std::mutex> lock(g_mutex);
	destroy_depth_readback(g_pending_depth);
	g_pending_depth = readback;
	return true;
}

bool enqueue_color_readback(api::command_list *cmd_list, api::resource_view rtv, uint64_t frame)
{
	if (!kSaveColorReadbackOnCapture || cmd_list == nullptr || rtv.handle == 0)
		return false;

	api::device *device = cmd_list->get_device();
	if (device == nullptr)
		return false;

	const api::resource color = device->get_resource_from_view(rtv);
	if (color.handle == 0)
		return false;

	const api::resource_desc desc = device->get_resource_desc(color);
	if (desc.type != api::resource_type::texture_2d)
		return false;
	if (desc.texture.width == 0 || desc.texture.height == 0)
		return false;

	const api::resource_view_desc view_desc = device->get_resource_view_desc(rtv);
	const api::format color_format = view_desc.format != api::format::unknown ? view_desc.format : desc.texture.format;

	if (desc.texture.samples > 1)
	{
		if (!g_logged_color_failure.exchange(true))
			log_line("BMWCameraCapture: color readback currently skips MSAA render targets.", reshade::log::level::warning);
		return false;
	}

	const uint32_t tight_row_pitch = api::format_row_pitch(color_format, desc.texture.width);
	if (tight_row_pitch == 0)
		return false;

	const uint32_t row_pitch = align_to(tight_row_pitch, kTextureReadbackPitchAlignment);
	const uint64_t byte_size = static_cast<uint64_t>(row_pitch) * desc.texture.height;

	api::resource readback_buffer = {};
	const api::resource_desc buffer_desc(byte_size, api::memory_heap::gpu_to_cpu, api::resource_usage::copy_dest);
	if (!device->create_resource(buffer_desc, nullptr, api::resource_usage::copy_dest, &readback_buffer))
	{
		if (!g_logged_color_failure.exchange(true))
			log_line("BMWCameraCapture: failed to create color readback buffer.", reshade::log::level::warning);
		return false;
	}

	const uint32_t bytes_per_pixel = desc.texture.width != 0 ? std::max(1u, tight_row_pitch / desc.texture.width) : 1u;
	const uint32_t row_length_pixels = row_pitch / bytes_per_pixel;

	if (!g_logged_color_readback_info.exchange(true))
	{
		std::ostringstream ss;
		ss << "BMWCameraCapture: color readback copy queued. size="
		   << desc.texture.width << 'x' << desc.texture.height
		   << " format=" << static_cast<uint32_t>(color_format)
		   << " row_pitch=" << row_pitch
		   << " tight_row_pitch=" << tight_row_pitch
		   << " bytes=" << byte_size
		   << " resource=0x" << std::hex << color.handle << std::dec;
		log_line(ss.str());
	}

	cmd_list->barrier(color, api::resource_usage::render_target, api::resource_usage::copy_source);
	cmd_list->copy_texture_to_buffer(color, 0, nullptr, readback_buffer, 0, row_length_pixels, desc.texture.height);
	cmd_list->barrier(color, api::resource_usage::copy_source, api::resource_usage::render_target);

	ColorReadback readback = {};
	readback.valid = true;
	readback.frame = frame;
	readback.device = device;
	readback.buffer = readback_buffer;
	readback.width = desc.texture.width;
	readback.height = desc.texture.height;
	readback.row_pitch = row_pitch;
	readback.tight_row_pitch = tight_row_pitch;
	readback.byte_size = byte_size;
	readback.format = color_format;
	readback.samples = desc.texture.samples;

	std::lock_guard<std::mutex> lock(g_mutex);
	destroy_color_readback(g_pending_color);
	g_pending_color = readback;
	return true;
}

bool depth_size_matches_color_target(api::command_list *cmd_list, api::resource_view rtv, const CameraSample &sample)
{
	if (cmd_list == nullptr || rtv.handle == 0 || sample.depth_resource.handle == 0)
		return true;

	api::device *device = cmd_list->get_device();
	if (device == nullptr)
		return true;

	const api::resource color = device->get_resource_from_view(rtv);
	if (color.handle == 0)
		return true;

	const api::resource_desc desc = device->get_resource_desc(color);
	if (desc.type != api::resource_type::texture_2d || desc.texture.width == 0 || desc.texture.height == 0)
		return true;

	if (sample.depth_width == desc.texture.width && sample.depth_height == desc.texture.height)
		return true;

	if (!g_logged_depth_size_mismatch.exchange(true))
	{
		std::ostringstream ss;
		ss << "BMWCameraCapture: skipping depth readback because DSV size "
		   << sample.depth_width << 'x' << sample.depth_height
		   << " does not match color target size "
		   << desc.texture.width << 'x' << desc.texture.height
		   << ". This is likely a non-main-view depth pass.";
		log_line(ss.str(), reshade::log::level::warning);
	}

	return false;
}

void scan_cbvs(api::command_list *cmd_list, uint32_t draw_count)
{
	if (!g_capture_enabled.load(std::memory_order_relaxed))
		return;

	const uint64_t frame = g_frame_index.load(std::memory_order_relaxed);
	if (!should_capture_frame(frame))
		return;

	api::device *device = cmd_list->get_device();
	if (device == nullptr || device->get_api() != api::device_api::d3d12)
		return;

	std::vector<TrackedCBV> cbvs;
	api::resource_view dsv = {};
	bool log_candidates_this_draw = false;

	{
		std::lock_guard<std::mutex> lock(g_mutex);
		log_candidates_this_draw = candidate_draw_scan_budget_available_locked(frame);
		if (!log_candidates_this_draw && !kUseLastMatchedCameraSampleInFrame && g_latest_sample.valid && g_latest_sample.frame == frame)
			return;

		const auto state_it = g_cmd_states.find(cmd_list);
		if (state_it == g_cmd_states.end())
			return;

		cbvs = state_it->second.cbvs;
		dsv = state_it->second.current_dsv;
	}

	for (const TrackedCBV &cbv : cbvs)
	{
		if (cbv.buffer.handle == 0)
			continue;
		if (cbv.size != UINT64_MAX && cbv.size < kReadSize)
			continue;
		if (kDumpRawCbvs)
			dump_cbv_raw(device, cbv);
		if (log_candidates_this_draw)
		{
			std::lock_guard<std::mutex> lock(g_mutex);
			if (should_skip_candidate_cbv_locked(frame, cbv))
				continue;
		}

		CameraSample sample = {};
		sample.frame = frame;
		sample.root_param = cbv.root_param;
		sample.draw_count = draw_count;
		sample.pipeline_layout = cbv.pipeline_layout;
		sample.resource = resource_key(cbv.buffer);
		sample.cb_offset = cbv.offset;

		bool matched = false;
		if (log_candidates_this_draw)
		{
			std::lock_guard<std::mutex> lock(g_mutex);
			matched = try_read_from_mapped_shadow(cbv, sample);
		}

		if (!matched)
			matched = try_read_from_device(device, cbv, sample);
		if (!matched)
		{
			std::lock_guard<std::mutex> lock(g_mutex);
			matched = try_read_from_mapped_shadow(cbv, sample);
		}

		if (matched)
		{
			sample.valid = true;
			attach_depth_from_dsv(device, dsv, sample);

			std::lock_guard<std::mutex> lock(g_mutex);
			if (log_candidates_this_draw)
				record_candidate_locked(sample);

			if (!kUseLastMatchedCameraSampleInFrame && g_latest_sample.valid && g_latest_sample.frame == frame)
			{
				if (log_candidates_this_draw)
					continue;
				return;
			}

			publish_sample(sample);
			if (!log_candidates_this_draw)
				return;
		}
	}
}

void on_init_command_list(api::command_list *cmd_list)
{
	std::lock_guard<std::mutex> lock(g_mutex);
	g_cmd_states.emplace(cmd_list, CommandListState{});
}

void on_destroy_command_list(api::command_list *cmd_list)
{
	std::lock_guard<std::mutex> lock(g_mutex);
	g_cmd_states.erase(cmd_list);
}

void on_init_command_queue(api::command_queue *queue)
{
	if (queue != nullptr && (queue->get_type() & api::command_queue_type::graphics) == api::command_queue_type::graphics)
	{
		std::lock_guard<std::mutex> lock(g_mutex);
		g_graphics_queue = queue;
	}
}

void on_destroy_command_queue(api::command_queue *queue)
{
	std::lock_guard<std::mutex> lock(g_mutex);
	if (g_graphics_queue == queue)
		g_graphics_queue = nullptr;
}

void on_push_descriptors(
	api::command_list *cmd_list,
	api::shader_stage,
	api::pipeline_layout layout,
	uint32_t layout_param,
	const api::descriptor_table_update &update)
{
	if (update.type != api::descriptor_type::constant_buffer || update.descriptors == nullptr || update.count == 0)
		return;

	const api::buffer_range *ranges = static_cast<const api::buffer_range *>(update.descriptors);

	std::lock_guard<std::mutex> lock(g_mutex);
	CommandListState &state = g_cmd_states[cmd_list];

	state.cbvs.erase(
		std::remove_if(state.cbvs.begin(), state.cbvs.end(),
			[layout_param](const TrackedCBV &cbv) { return cbv.root_param == layout_param; }),
		state.cbvs.end());

	for (uint32_t i = 0; i < update.count; ++i)
	{
		TrackedCBV cbv = {};
		cbv.buffer = ranges[i].buffer;
		cbv.offset = ranges[i].offset;
		cbv.size = ranges[i].size;
		cbv.pipeline_layout = layout.handle;
		cbv.root_param = layout_param;
		state.cbvs.push_back(cbv);
	}
}

void on_begin_render_pass(
	api::command_list *cmd_list,
	uint32_t,
	const api::render_pass_render_target_desc *,
	const api::render_pass_depth_stencil_desc *ds)
{
	// Track the latest full-resolution main depth only while recording a sampled
	// frame; the device queries are skipped on every other render pass.
	DepthTarget readable_depth = {};
	const uint64_t frame = g_frame_index.load(std::memory_order_relaxed);
	if (g_capture_enabled.load(std::memory_order_relaxed) && should_capture_frame(frame) && ds != nullptr)
	{
		api::device *device = cmd_list != nullptr ? cmd_list->get_device() : nullptr;
		make_readable_depth_target(device, ds->view, frame, readable_depth);
	}

	std::lock_guard<std::mutex> lock(g_mutex);
	CommandListState &state = g_cmd_states[cmd_list];
	state.current_dsv = ds != nullptr ? ds->view : api::resource_view{};
	if (readable_depth.valid)
		g_latest_readable_depth = readable_depth;
}

void on_bind_render_targets_and_depth_stencil(
	api::command_list *cmd_list,
	uint32_t,
	const api::resource_view *,
	api::resource_view dsv)
{
	DepthTarget readable_depth = {};
	const uint64_t frame = g_frame_index.load(std::memory_order_relaxed);
	if (g_capture_enabled.load(std::memory_order_relaxed) && should_capture_frame(frame))
	{
		api::device *device = cmd_list != nullptr ? cmd_list->get_device() : nullptr;
		make_readable_depth_target(device, dsv, frame, readable_depth);
	}

	std::lock_guard<std::mutex> lock(g_mutex);
	CommandListState &state = g_cmd_states[cmd_list];
	state.current_dsv = dsv;
	if (readable_depth.valid)
		g_latest_readable_depth = readable_depth;
}

bool on_draw(api::command_list *cmd_list, uint32_t vertex_count, uint32_t instance_count, uint32_t, uint32_t)
{
	if (vertex_count * std::max(instance_count, 1u) >= kMinDrawVertices)
		scan_cbvs(cmd_list, vertex_count);
	return false;
}

bool on_draw_indexed(api::command_list *cmd_list, uint32_t index_count, uint32_t instance_count, uint32_t, int32_t, uint32_t)
{
	if (index_count * std::max(instance_count, 1u) >= kMinDrawVertices)
		scan_cbvs(cmd_list, index_count);
	return false;
}

void on_map_buffer_region(api::device *, api::resource resource, uint64_t offset, uint64_t size, api::map_access, void **data)
{
	if (resource.handle == 0 || data == nullptr || *data == nullptr)
		return;

	std::lock_guard<std::mutex> lock(g_mutex);
	g_mapped_buffers[resource_key(resource)] = MappedBuffer{ offset, size, static_cast<const uint8_t *>(*data) };
}

void on_unmap_buffer_region(api::device *, api::resource resource)
{
	std::lock_guard<std::mutex> lock(g_mutex);
	g_mapped_buffers.erase(resource_key(resource));
}

void on_reshade_begin_effects(api::effect_runtime *, api::command_list *cmd_list, api::resource_view rtv, api::resource_view)
{
	if (!g_capture_enabled.load(std::memory_order_relaxed))
		return;

	const uint64_t frame = g_frame_index.load(std::memory_order_relaxed);
	if (!should_capture_frame(frame))
		return;

	CameraSample sample = {};
	bool need_depth = false;
	bool need_color = false;
	{
		std::lock_guard<std::mutex> lock(g_mutex);
		if (!g_latest_sample.valid || g_latest_sample.frame != frame)
			return;
		need_depth = kSaveDepthReadbackOnCapture && !(g_pending_depth.valid && g_pending_depth.frame == frame);
		need_color = kSaveColorReadbackOnCapture && !(g_pending_color.valid && g_pending_color.frame == frame);
		if (!need_depth && !need_color)
			return;
		sample = g_latest_sample;

		// Determine the final color-target (backbuffer) size from the RTV, then
		// choose the depth to read back: prefer the camera pass's own depth if it
		// already matches the color size, otherwise fall back to the tracked
		// full-resolution main depth (g_latest_readable_depth).
		uint32_t color_w = 0, color_h = 0;
		if (cmd_list != nullptr && rtv.handle != 0)
		{
			api::device *dev = cmd_list->get_device();
			if (dev != nullptr)
			{
				const api::resource backbuf = dev->get_resource_from_view(rtv);
				if (backbuf.handle != 0)
				{
					const api::resource_desc bd = dev->get_resource_desc(backbuf);
					if (bd.type == api::resource_type::texture_2d && bd.texture.width != 0)
					{
						color_w = bd.texture.width;
						color_h = bd.texture.height;
					}
				}
			}
		}

		const bool cam_depth_ok = sample.depth_resource.handle != 0 &&
			color_w != 0 && sample.depth_width == color_w && sample.depth_height == color_h;

		// Record diagnostics per frame (the cam-pass depth dims, captured BEFORE any
		// fallback overwrites sample.depth_*).
		sample.cam_depth_ok = cam_depth_ok;
		sample.cam_pass_depth_width = sample.depth_width;
		sample.cam_pass_depth_height = sample.depth_height;

		if (!g_logged_depth_choice.exchange(true))
		{
			std::ostringstream ss;
			ss << "BMWCameraCapture: DEPTH-CHOICE color=" << color_w << 'x' << color_h
			   << " cam_pass_depth=" << sample.depth_width << 'x' << sample.depth_height
			   << " global_readable_depth=" << g_latest_readable_depth.width << 'x' << g_latest_readable_depth.height
			   << " cam_depth_ok=" << (cam_depth_ok ? 1 : 0);
			log_line(ss.str());
		}

		if (!cam_depth_ok)
			apply_readable_depth_target(sample, g_latest_readable_depth);

		g_latest_sample = sample;
	}

	if (need_depth && !depth_size_matches_color_target(cmd_list, rtv, sample))
	{
		if (kRequireEnabledReadbacksForCsv)
			return;
		need_depth = false;
	}

	if (need_depth)
		enqueue_depth_readback(cmd_list, sample);
	if (need_color)
		enqueue_color_readback(cmd_list, rtv, frame);
}

// Draw a small colored square in the top-left of the final image WHILE recording.
// D3D12 only. Runs in reshade_finish_effects (after the color readback was queued
// in begin_effects), so it never appears in the captured dataset.
void draw_rec_indicator(api::command_list *cmd_list, api::resource_view rtv)
{
	if (!kRecIndicator)
		return;
	if (!g_capture_enabled.load(std::memory_order_relaxed))
		return;
	if (cmd_list == nullptr || rtv.handle == 0)
		return;

	api::device *device = cmd_list->get_device();
	if (device == nullptr || device->get_api() != api::device_api::d3d12)
		return;

	auto *native_cmd = reinterpret_cast<ID3D12GraphicsCommandList *>(cmd_list->get_native());
	if (native_cmd == nullptr)
		return;

	// For D3D12, a resource_view handle is a D3D12_CPU_DESCRIPTOR_HANDLE.
	D3D12_CPU_DESCRIPTOR_HANDLE handle = {};
	handle.ptr = static_cast<SIZE_T>(rtv.handle);

	const int m = 12;                   // margin from the top-left corner (px)
	const int s = kRecMarkerSizePx;     // square side (px)
	const D3D12_RECT box = { m, m, m + s, m + s };

	const FLOAT green[4]  = { 0.10f, 0.85f, 0.10f, 1.0f };
	const FLOAT yellow[4] = { 0.95f, 0.85f, 0.05f, 1.0f };
	const bool skipped = g_rec_skipped.load(std::memory_order_relaxed);
	native_cmd->ClearRenderTargetView(handle, skipped ? yellow : green, 1, &box);
}

void on_reshade_finish_effects(api::effect_runtime *, api::command_list *cmd_list, api::resource_view rtv, api::resource_view)
{
	draw_rec_indicator(cmd_list, rtv);
}

void on_reshade_present(api::effect_runtime *runtime)
{
	CameraSample sample = {};
	DepthReadback depth = {};
	ColorReadback color = {};
	std::vector<CameraSample> candidates;
	const uint64_t frame = g_frame_index.load(std::memory_order_relaxed);
	bool wrote_row = false;   // for the REC indicator: did this sampled frame emit a CSV row?

	if (g_igcs_connected.load(std::memory_order_relaxed) && !g_logged_igcs_connected.exchange(true))
		log_line("BMWCameraCapture: IGCS camera tool (UUU) connected -- engine-authoritative camera available.");

	if (runtime != nullptr && runtime->is_key_pressed(kToggleCaptureKey))
	{
		const bool enabled = !g_capture_enabled.load(std::memory_order_relaxed);
		g_capture_enabled.store(enabled, std::memory_order_relaxed);

		if (enabled)
		{
			// New recording segment: bump session id, mark first-row seg_start, and
			// re-latch the monotonic timestamp epoch to "now".
			g_capture_session.fetch_add(1, std::memory_order_relaxed);
			g_segment_start_pending.store(true, std::memory_order_relaxed);
			LARGE_INTEGER now = {};
			QueryPerformanceCounter(&now);
			g_capture_qpc_start.store(now.QuadPart, std::memory_order_relaxed);
		}

		{
			std::lock_guard<std::mutex> lock(g_mutex);
			g_latest_sample.valid = false;
			g_latest_readable_depth = DepthTarget{};   // drop stale depth across toggles
		}
		g_rec_skipped.store(false, std::memory_order_relaxed);   // reset indicator to green

		log_line(enabled ? "BMWCameraCapture: capture started." : "BMWCameraCapture: capture stopped.");
		play_toggle_cue(enabled);
	}

	if (g_capture_enabled.load(std::memory_order_relaxed) && should_capture_frame(frame))
	{
		std::lock_guard<std::mutex> lock(g_mutex);
		if (g_latest_sample.valid && g_latest_sample.frame == frame)
		{
			sample = g_latest_sample;
			g_latest_sample.valid = false;
		}
		if (kLogCameraCandidates && g_frame_candidates_frame == frame)
		{
			candidates = g_frame_candidates;
			g_frame_candidates.clear();
			g_frame_candidate_keys.clear();
			g_frame_candidate_draw_scans = 0;
		}
		if (g_pending_depth.valid && g_pending_depth.frame == frame)
		{
			depth = g_pending_depth;
			g_pending_depth = DepthReadback{};
		}
		if (g_pending_color.valid && g_pending_color.frame == frame)
		{
			color = g_pending_color;
			g_pending_color = ColorReadback{};
		}
	}

	if (kLogCameraCandidates && !candidates.empty())
		write_candidate_rows(candidates, sample);

	if (sample.valid)
	{
		const std::string shot_postfix = screenshot_postfix(sample.frame);
		const std::string depth_name = depth.valid ? depth_postfix(sample.frame) : "";
		const std::string color_name = color.valid ? color_postfix(sample.frame) : "";
		const bool depth_queued = depth.valid;
		const bool color_queued = color.valid;
		const uint64_t depth_frame = depth_queued ? depth.frame : UINT64_MAX;
		const uint64_t color_frame = color_queued ? color.frame : UINT64_MAX;
		const bool strict_sync_ok =
			sample.frame == frame &&
			(!kSaveDepthReadbackOnCapture || (depth_queued && depth.frame == sample.frame)) &&
			(!kSaveColorReadbackOnCapture || (color_queued && color.frame == sample.frame));

		if (kRequireEnabledReadbacksForCsv && !strict_sync_ok)
		{
			if (!g_logged_strict_sync_skip.exchange(true))
				log_line("BMWCameraCapture: skipped at least one CSV row because camera/depth/color were not all queued for the same frame.", reshade::log::level::warning);
		}
		else
		{
			std::ofstream fp(g_csv_path, std::ios::app);
			if (fp)
			{
				fp << std::fixed << std::setprecision(6)
				   << sample.frame << ','
				   << sample.root_param << ','
				   << sample.draw_count << ','
				   << "0x" << std::hex << sample.resource << std::dec << ','
				   << "0x" << std::hex << sample.cb_offset << std::dec << ','
				   << sample.pos_cm[0] << ',' << sample.pos_cm[1] << ',' << sample.pos_cm[2] << ','
				   << sample.pos_cm[0] * 0.01f << ',' << sample.pos_cm[1] * 0.01f << ',' << sample.pos_cm[2] * 0.01f << ','
				   << sample.right[0] << ',' << sample.right[1] << ',' << sample.right[2] << ','
				   << sample.up[0] << ',' << sample.up[1] << ',' << sample.up[2] << ','
				   << sample.fwd[0] << ',' << sample.fwd[1] << ',' << sample.fwd[2] << ','
				   << sample.hfov_deg << ',' << sample.vfov_deg << ',' << sample.near_cm << ','
				   << "0x" << std::hex << kProjectionOffset << std::dec << ','
				   << "0x" << std::hex << kViewOriginOffset << std::dec << ','
				   << "0x" << std::hex << kViewToWorldOffset << std::dec << ','
				   << shot_postfix;

				for (int i = 0; i < 16; ++i)
					fp << ',' << sample.projection[i];

				for (int i = 0; i < 4; ++i)
					fp << ',' << sample.view_to_world_row3[i];

				for (int i = 0; i < 16; ++i)
					fp << ',' << sample.world_to_view[i];
				for (int i = 0; i < 16; ++i)
					fp << ',' << sample.world_to_clip[i];
				for (int i = 0; i < 16; ++i)
					fp << ',' << sample.view_to_world[i];

				fp << ',';
				if (depth_queued)
					fp << depth_relative_bin_path(depth_name);
				fp << ','
				   << sample.depth_width << ','
				   << sample.depth_height << ','
				   << static_cast<uint32_t>(sample.depth_format) << ','
				   << (depth_queued ? depth.row_pitch : sample.depth_row_pitch) << ','
				   << (depth_queued ? depth.tight_row_pitch : api::format_row_pitch(sample.depth_format, sample.depth_width)) << ','
				   << (depth_queued ? depth.byte_size : sample.depth_byte_size) << ','
				   << sample.depth_samples;

				fp << ',';
				if (color_queued)
					fp << color_relative_bin_path(color_name);
				fp << ','
				   << color.width << ','
				   << color.height << ','
				   << static_cast<uint32_t>(color.format) << ','
				   << color.row_pitch << ','
				   << color.tight_row_pitch << ','
				   << color.byte_size << ','
				   << color.samples << ','
				   << sample.frame << ',';

				if (depth_frame != UINT64_MAX)
					fp << depth_frame;
				fp << ',';
				if (color_frame != UINT64_MAX)
					fp << color_frame;
				fp << ',' << (strict_sync_ok ? 1 : 0)
				   << ',' << g_capture_session.load(std::memory_order_relaxed)
				   << ',' << (g_segment_start_pending.exchange(false, std::memory_order_relaxed) ? 1 : 0)
				   << ',' << qpc_seconds_since(g_capture_qpc_start.load(std::memory_order_relaxed))
				   << ',' << wall_clock_unix_seconds()
				   << ',' << (sample.cam_depth_ok ? 1 : 0)
				   << ',' << sample.cam_pass_depth_width
				   << ',' << sample.cam_pass_depth_height;

				// Engine-authoritative camera from UUU via the IGCS connector buffer
				// (zeroed if UUU hasn't connected / isn't running).
				IgcsCameraToolsData ig{};
				const bool ig_ok = g_igcs_camera_buffer != nullptr;
				if (ig_ok)
					ig = *reinterpret_cast<const IgcsCameraToolsData *>(g_igcs_camera_buffer);
				fp << ',' << (ig_ok ? 1 : 0) << ',' << static_cast<int>(ig.cameraEnabled) << ',' << ig.fov
				   << ',' << ig.coordinates.x << ',' << ig.coordinates.y << ',' << ig.coordinates.z
				   << ',' << ig.lookQuaternion.x << ',' << ig.lookQuaternion.y << ',' << ig.lookQuaternion.z << ',' << ig.lookQuaternion.w
				   << ',' << ig.rotationMatrixUpVector.x << ',' << ig.rotationMatrixUpVector.y << ',' << ig.rotationMatrixUpVector.z
				   << ',' << ig.rotationMatrixRightVector.x << ',' << ig.rotationMatrixRightVector.y << ',' << ig.rotationMatrixRightVector.z
				   << ',' << ig.rotationMatrixForwardVector.x << ',' << ig.rotationMatrixForwardVector.y << ',' << ig.rotationMatrixForwardVector.z
				   << '\n';
				wrote_row = true;
			}

			if (kSaveScreenshotOnCapture && runtime != nullptr)
				runtime->save_screenshot(shot_postfix.c_str());

			queue_delayed_readbacks(depth, color);
		}
	}

	if (depth.valid)
		destroy_depth_readback(depth);
	if (color.valid)
		destroy_color_readback(color);

	// Update the REC indicator state on sampled frames: green if a row was written,
	// yellow otherwise (matched-but-skipped, or nothing matched this frame).
	if (g_capture_enabled.load(std::memory_order_relaxed) && should_capture_frame(frame))
		g_rec_skipped.store(!wrote_row, std::memory_order_relaxed);

	flush_ready_readbacks(frame, false);
	g_frame_index.fetch_add(1, std::memory_order_relaxed);
}

void register_events()
{
	reshade::register_event<reshade::addon_event::init_command_list>(&on_init_command_list);
	reshade::register_event<reshade::addon_event::destroy_command_list>(&on_destroy_command_list);
	reshade::register_event<reshade::addon_event::init_command_queue>(&on_init_command_queue);
	reshade::register_event<reshade::addon_event::destroy_command_queue>(&on_destroy_command_queue);
	reshade::register_event<reshade::addon_event::begin_render_pass>(&on_begin_render_pass);
	reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(&on_bind_render_targets_and_depth_stencil);
	reshade::register_event<reshade::addon_event::push_descriptors>(&on_push_descriptors);
	reshade::register_event<reshade::addon_event::draw>(&on_draw);
	reshade::register_event<reshade::addon_event::draw_indexed>(&on_draw_indexed);
	reshade::register_event<reshade::addon_event::map_buffer_region>(&on_map_buffer_region);
	reshade::register_event<reshade::addon_event::unmap_buffer_region>(&on_unmap_buffer_region);
	reshade::register_event<reshade::addon_event::reshade_begin_effects>(&on_reshade_begin_effects);
	reshade::register_event<reshade::addon_event::reshade_finish_effects>(&on_reshade_finish_effects);
	reshade::register_event<reshade::addon_event::reshade_present>(&on_reshade_present);
}

void unregister_events()
{
	reshade::unregister_event<reshade::addon_event::reshade_present>(&on_reshade_present);
	reshade::unregister_event<reshade::addon_event::reshade_finish_effects>(&on_reshade_finish_effects);
	reshade::unregister_event<reshade::addon_event::reshade_begin_effects>(&on_reshade_begin_effects);
	reshade::unregister_event<reshade::addon_event::unmap_buffer_region>(&on_unmap_buffer_region);
	reshade::unregister_event<reshade::addon_event::map_buffer_region>(&on_map_buffer_region);
	reshade::unregister_event<reshade::addon_event::draw_indexed>(&on_draw_indexed);
	reshade::unregister_event<reshade::addon_event::draw>(&on_draw);
	reshade::unregister_event<reshade::addon_event::push_descriptors>(&on_push_descriptors);
	reshade::unregister_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(&on_bind_render_targets_and_depth_stencil);
	reshade::unregister_event<reshade::addon_event::begin_render_pass>(&on_begin_render_pass);
	reshade::unregister_event<reshade::addon_event::destroy_command_queue>(&on_destroy_command_queue);
	reshade::unregister_event<reshade::addon_event::init_command_queue>(&on_init_command_queue);
	reshade::unregister_event<reshade::addon_event::destroy_command_list>(&on_destroy_command_list);
	reshade::unregister_event<reshade::addon_event::init_command_list>(&on_init_command_list);

	flush_ready_readbacks(UINT64_MAX, true);

	std::lock_guard<std::mutex> lock(g_mutex);
	destroy_depth_readback(g_pending_depth);
	destroy_color_readback(g_pending_color);
	for (DepthReadback &readback : g_delayed_depths)
		destroy_depth_readback(readback);
	g_delayed_depths.clear();
	for (ColorReadback &readback : g_delayed_colors)
		destroy_color_readback(readback);
	g_delayed_colors.clear();
	g_frame_candidates.clear();
	g_frame_candidate_keys.clear();
	g_frame_candidate_draw_scans = 0;
	g_frame_candidates_frame = UINT64_MAX;
}
}

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID)
{
	switch (fdwReason)
	{
	case DLL_PROCESS_ATTACH:
		g_module = hinstDLL;
		DisableThreadLibraryCalls(hinstDLL);
		if (!reshade::register_addon(hinstDLL))
			return FALSE;
		QueryPerformanceFrequency(&g_qpc_freq);
		{
			LARGE_INTEGER now = {};
			QueryPerformanceCounter(&now);
			g_capture_qpc_start.store(now.QuadPart, std::memory_order_relaxed);
		}
		init_paths();
		register_events();
		log_line("BMWCameraCapture: loaded.");
		break;
	case DLL_PROCESS_DETACH:
		unregister_events();
		log_line("BMWCameraCapture: unloaded.");
		reshade::unregister_addon(hinstDLL);
		break;
	}

	return TRUE;
}
