#include <reshade.hpp>

#include <Windows.h>
#include <d3d11.h>
#include <d3d11_1.h>
#include <d3d12.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
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

namespace
{
namespace api = reshade::api;

constexpr const char *kBuildTag = "gta_d3d11_camera_rgb_depth_staging_v20_segment_marker";

// GTA V / RAGE "rage_matrices" constant buffer layout observed in RenderDoc:
//   0x040: world_to_view
//   0x080: world_to_clip
//   0x0C0: view_to_world
// The projection matrix is derived as view_to_world * world_to_clip.
constexpr uint64_t kWorldToViewOffset = 0x40;
constexpr uint64_t kWorldToClipOffset = 0x80;
constexpr uint64_t kViewToWorldOffset = 0xC0;
constexpr uint64_t kProjectionOffset = kWorldToClipOffset;
constexpr uint64_t kViewOriginOffset = kViewToWorldOffset + 12 * sizeof(float);
constexpr uint64_t kReadSize = 0x100;
constexpr uint32_t kMinDrawVertices = 3;
constexpr uint64_t kCaptureEveryNFrames = 30;
constexpr bool kSaveScreenshotOnCapture = false;
// GTA V normally runs through D3D11. ReShade's D3D11 texture-to-buffer readback
// is not implemented, so D3D11 RGB/depth uses native staging textures instead.
constexpr bool kSaveDepthReadbackOnCapture = true;
constexpr bool kSaveColorReadbackOnCapture = true;
constexpr bool kCaptureStartsEnabled = false;
constexpr uint32_t kToggleCaptureKey = VK_F8;
// Play a short audio cue when capture toggles (ascending = start, descending =
// stop) so you get immediate confirmation on F8 without watching the log.
constexpr bool kAudioCueOnToggle = true;
// Draw a red on-screen REC marker WHILE recording. Driven by the same
// g_capture_enabled flag as the capture itself, so it is shown if and only if
// recording is active. Drawn after the color readback, so it never appears in
// the captured dataset.
constexpr bool kRecIndicator = true;
constexpr int kRecMarkerSizePx = 5;   // REC square side in screen pixels (try 5-10)
// Keep CSV rows even if a readback path fails; strict_sync_ok marks whether
// camera/depth/color were queued on the same add-on frame.
constexpr bool kRequireEnabledReadbacksForCsv = false;
// GTA has multiple pass families. Keep the latest valid main-target match in
// the frame so late scene/post passes can override earlier candidates.
constexpr bool kUseLastMatchedCameraSampleInFrame = true;
// Diagnostic path: dump every unique CBV that passes the camera-shape test on
// sampled capture frames. This is intentionally separate from the main camera
// CSV, so bad candidates can be inspected without changing the conversion path.
// PERF: keep this OFF during normal capture. When on, every draw on a capture
// frame re-scans ALL CBVs and grabs g_mutex several times per CBV (millions of
// locked matrix parses per session) -- the dominant in-game stutter source.
// Flip back to true only when re-validating the camera CBV offset for a new game.
constexpr bool kLogCameraCandidates = false;
constexpr uint32_t kMaxCameraCandidateRowsPerFrame = 64;
constexpr uint32_t kMaxDrawScansPerCaptureFrame = 24576;   // was 4096; raised so late main-scene passes (dense views) still get scanned
constexpr float kExpectedAspect = 16.0f / 9.0f;
constexpr float kMaxAspectError = 0.05f;
constexpr float kMinGtaPositionLength = 1000.0f;
constexpr float kMaxGtaAbsPosition = 100000.0f;
constexpr uint32_t kMinMainTargetWidth = 1000;
constexpr uint32_t kMinMainTargetHeight = 600;
// Separate, lower threshold for accepting a matched camera's view target in
// has_main_target(). GTA binds the camera CBV in a fraction-resolution pass, and
// at 1080p that pass is only ~960x540 (below kMinMainTarget*). 640x360 covers
// 1080p and up while still filtering tiny reflection/UI passes. (make_readable_
// depth_target keeps the higher kMinMainTarget* so it won't pick shadow maps.)
constexpr uint32_t kMinCameraViewWidth = 640;
constexpr uint32_t kMinCameraViewHeight = 360;
constexpr uint64_t kMaxShadowBufferBytes = 64ull * 1024ull;
constexpr uint32_t kTextureReadbackPitchAlignment = 256;
constexpr uint64_t kReadbackDelayFrames = 8;
constexpr bool kWaitIdleBeforeReadbackMap = false;

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
	api::resource_view current_rtv0 = {};
	api::resource_view current_dsv = {};
};

struct MappedBuffer
{
	uint64_t offset = 0;
	uint64_t size = UINT64_MAX;
	const uint8_t *data = nullptr;
	api::map_access access = api::map_access::read_only;
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
	float world_to_view[16] = {};
	float world_to_clip[16] = {};
	float view_to_world[16] = {};
	float view_to_world_row3[4] = {};
	api::resource color_resource = {};
	uint32_t color_width = 0;
	uint32_t color_height = 0;
	uint16_t color_samples = 0;
	api::format color_format = api::format::unknown;
	api::resource depth_resource = {};
	uint32_t depth_width = 0;
	uint32_t depth_height = 0;
	uint16_t depth_samples = 0;
	api::format depth_format = api::format::unknown;
	uint32_t depth_row_pitch = 0;
	uint64_t depth_byte_size = 0;
};

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

struct DepthReadback
{
	bool valid = false;
	bool d3d11_staging = false;
	uint64_t frame = 0;
	api::device *device = nullptr;
	api::resource buffer = {};
	ID3D11Texture2D *d3d11_texture = nullptr;
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
	bool d3d11_staging = false;
	uint64_t frame = 0;
	api::device *device = nullptr;
	api::resource buffer = {};
	ID3D11Texture2D *d3d11_texture = nullptr;
	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t row_pitch = 0;
	uint32_t tight_row_pitch = 0;
	uint64_t byte_size = 0;
	api::format format = api::format::unknown;
	uint16_t samples = 0;
};

// A self-contained disk-write job handed to the background writer thread. It
// holds only CPU memory + paths (the GPU staging texture has already been mapped
// and released on the present thread), so the writer never touches D3D objects.
struct WriteJob
{
	std::filesystem::path bin_path;
	std::vector<uint8_t> bin_bytes;
	std::filesystem::path json_path;
	std::string json_text; // empty -> skip the .json sidecar
	const char *label = "";
};

HMODULE g_module = nullptr;
std::filesystem::path g_game_dir;
std::filesystem::path g_output_dir;
std::filesystem::path g_depth_dir;
std::filesystem::path g_color_dir;
std::filesystem::path g_csv_path;
std::filesystem::path g_candidate_csv_path;
std::filesystem::path g_debug_path;

std::mutex g_mutex;
std::unordered_map<api::command_list *, CommandListState> g_cmd_states;
std::unordered_map<uint64_t, MappedBuffer> g_mapped_buffers;
std::unordered_map<uint64_t, std::vector<uint8_t>> g_buffer_shadows;
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
std::atomic_bool g_logged_map_failure = false;
std::atomic_bool g_logged_depth_failure = false;
std::atomic_bool g_logged_color_failure = false;
std::atomic_bool g_logged_depth_unsupported_format = false;
std::atomic_bool g_logged_depth_readback_info = false;
std::atomic_bool g_logged_color_readback_info = false;
std::atomic_bool g_logged_readback_unsupported_api = false;
std::atomic_bool g_logged_strict_sync_skip = false;
std::atomic_bool g_logged_depth_size_mismatch = false;
std::atomic_bool g_logged_depth_target_found = false;
std::atomic_bool g_logged_depth_target_miss = false;
std::atomic_bool g_logged_depth_apply_miss = false;
std::atomic_bool g_logged_depth_resource_zero = false;
std::atomic_bool g_logged_parse_diag = false;
std::atomic_bool g_logged_no_main_target = false;
std::atomic_bool g_logged_candidate_truncation = false;
std::atomic_bool g_logged_present_seen = false;
std::atomic_bool g_logged_depth_choice = false;
std::atomic<uint32_t> g_capture_session = 0;       // ++ on each F8 start; written as CSV 'segment'
std::atomic_bool g_segment_start_pending = false;  // set on F8 start; first CSV row -> seg_start=1
std::atomic_bool g_toggle_key_down = false;
std::atomic<uint64_t> g_scan_calls = 0;
std::atomic<uint64_t> g_cbv_checks = 0;
std::atomic<uint64_t> g_shadow_updates = 0;
std::atomic<uint64_t> g_shadow_bytes = 0;
std::atomic<uint64_t> g_shadow_skips = 0;
bool g_logged_first_match = false;
uint64_t g_frame_candidates_frame = UINT64_MAX;
uint32_t g_frame_candidate_draw_scans = 0;

// Background disk-writer: the present thread maps the GPU readback and hands the
// raw bytes here; this thread does the (slow) ~44MB/file fwrite off the render
// path. Uses its own mutex so it never contends with the per-draw g_mutex.
std::mutex g_writer_mutex;
std::condition_variable g_writer_cv;
std::deque<WriteJob> g_writer_queue;
std::thread g_writer_thread;
bool g_writer_started = false;  // a worker thread exists (guarded by g_writer_mutex)
bool g_writer_shutdown = false; // teardown: enqueue writes inline, no new thread
// Back-pressure cap: bounds queued memory (~44MB/job). When full, the present
// thread blocks until the writer drains -- only if the disk truly can't keep up.
constexpr size_t kMaxWriterQueueJobs = 8;

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

bool should_capture_frame(uint64_t frame)
{
	return kCaptureEveryNFrames <= 1 || (frame % kCaptureEveryNFrames) == 0;
}

// True only when we are actively recording AND this frame is one of the sampled
// capture frames. Per-draw / per-bind / per-map hooks gate on this so that the
// 14-of-15 non-capture frames (and all idle time when not recording) do almost
// no work and take no locks shared with the render thread.
bool capture_active_now()
{
	return g_capture_enabled.load(std::memory_order_relaxed) &&
	       should_capture_frame(g_frame_index.load(std::memory_order_relaxed));
}

// Audible F8 confirmation. Beep() is synchronous, so run it on a short detached
// thread to avoid hitching the present thread. Ascending tones = recording
// started, descending = stopped.
void play_toggle_cue(bool started)
{
	if (!kAudioCueOnToggle)
		return;
	std::thread([started] {
		if (started) { Beep(784, 90); Beep(1175, 140); }   // G5 -> D6
		else         { Beep(1175, 90); Beep(784, 140); }   // D6 -> G5
	}).detach();
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
	case api::format::d24_unorm_x8_uint:
	case api::format::d32_float:
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
	ss << "gta_camera_frame_" << std::setw(8) << std::setfill('0') << frame;
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
	      "pos_x,pos_y,pos_z,pos_x_0p01,pos_y_0p01,pos_z_0p01,"
	      "right_x,right_y,right_z,up_x,up_y,up_z,fwd_x,fwd_y,fwd_z,"
	      "hfov_deg,vfov_deg,projection_depth_param,projection_offset_hex,view_origin_offset_hex,view_to_world_offset_hex,"
	      "screenshot_postfix";

	for (int row = 0; row < 4; ++row)
	{
		for (int col = 0; col < 4; ++col)
			fp << ",proj_m" << row << col;
	}

	for (int col = 0; col < 4; ++col)
		fp << ",view_m3" << col;

	for (int row = 0; row < 4; ++row)
	{
		for (int col = 0; col < 4; ++col)
			fp << ",world_to_view_m" << row << col;
	}
	for (int row = 0; row < 4; ++row)
	{
		for (int col = 0; col < 4; ++col)
			fp << ",world_to_clip_m" << row << col;
	}
	for (int row = 0; row < 4; ++row)
	{
		for (int col = 0; col < 4; ++col)
			fp << ",view_to_world_m" << row << col;
	}

	fp << ",depth_file,depth_width,depth_height,depth_format,depth_row_pitch,depth_tight_row_pitch,depth_byte_size,depth_samples";
	fp << ",color_file,color_width,color_height,color_format,color_row_pitch,color_tight_row_pitch,color_byte_size,color_samples";
	fp << ",camera_frame,depth_frame,color_frame,strict_sync_ok,segment,seg_start";

	fp << '\n';
}

void write_candidate_csv_header(std::ofstream &fp)
{
	fp << "frame,candidate_index,selected_camera_sample,root_param,draw_count,pipeline_layout,resource,cb_offset_hex,"
	      "pos_x,pos_y,pos_z,pos_x_0p01,pos_y_0p01,pos_z_0p01,"
	      "right_x,right_y,right_z,up_x,up_y,up_z,fwd_x,fwd_y,fwd_z,"
	      "hfov_deg,vfov_deg,projection_depth_param,projection_offset_hex,view_origin_offset_hex,view_to_world_offset_hex";

	for (int row = 0; row < 4; ++row)
	{
		for (int col = 0; col < 4; ++col)
			fp << ",proj_m" << row << col;
	}

	for (int col = 0; col < 4; ++col)
		fp << ",view_m3" << col;

	for (int row = 0; row < 4; ++row)
	{
		for (int col = 0; col < 4; ++col)
			fp << ",world_to_view_m" << row << col;
	}
	for (int row = 0; row < 4; ++row)
	{
		for (int col = 0; col < 4; ++col)
			fp << ",world_to_clip_m" << row << col;
	}
	for (int row = 0; row < 4; ++row)
	{
		for (int col = 0; col < 4; ++col)
			fp << ",view_to_world_m" << row << col;
	}

	fp << ",depth_resource,depth_width,depth_height,depth_format,depth_samples";
	fp << '\n';
}

void init_paths()
{
	wchar_t module_path[MAX_PATH] = {};
	GetModuleFileNameW(g_module, module_path, MAX_PATH);

	g_game_dir = std::filesystem::path(module_path).parent_path();
	g_output_dir = g_game_dir / L"GTACameraCapture";
	g_depth_dir = g_output_dir / L"depth";
	g_color_dir = g_output_dir / L"color";
	g_csv_path = g_output_dir / L"gta_camera_pos.csv";
	g_candidate_csv_path = g_output_dir / L"gta_camera_candidates.csv";
	g_debug_path = g_output_dir / L"gta_camera_capture_debug.log";

	std::error_code ec;
	std::filesystem::create_directories(g_depth_dir, ec);
	std::filesystem::create_directories(g_color_dir, ec);

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
			log_line("GTACameraCapture: existing CSV header has no projection matrix columns. Rename/delete old gta_camera_pos.csv before a new capture if you want a clean header.", reshade::log::level::warning);
		else if (header.find("depth_file") == std::string::npos)
			log_line("GTACameraCapture: existing CSV header has no depth readback columns. Rename/delete old gta_camera_pos.csv before a new capture if you want a clean header.", reshade::log::level::warning);
		else if (header.find("color_file") == std::string::npos)
			log_line("GTACameraCapture: existing CSV header has no color readback columns. Rename/delete old gta_camera_pos.csv before a new capture if you want a clean header.", reshade::log::level::warning);
		else if (header.find("strict_sync_ok") == std::string::npos)
			log_line("GTACameraCapture: existing CSV header has no strict sync columns. Rename/delete old gta_camera_pos.csv before a new capture if you want sync validation columns.", reshade::log::level::warning);
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

void multiply_row_major_4x4(const float a[16], const float b[16], float out[16])
{
	float tmp[16] = {};
	for (int row = 0; row < 4; ++row)
	{
		for (int col = 0; col < 4; ++col)
		{
			float v = 0.0f;
			for (int k = 0; k < 4; ++k)
				v += a[row * 4 + k] * b[k * 4 + col];
			tmp[row * 4 + col] = v;
		}
	}
	std::memcpy(out, tmp, sizeof(tmp));
}

float row_major_identity_error(const float m[16])
{
	float err = 0.0f;
	for (int row = 0; row < 4; ++row)
	{
		for (int col = 0; col < 4; ++col)
		{
			const float expected = row == col ? 1.0f : 0.0f;
			err = std::max(err, std::abs(m[row * 4 + col] - expected));
		}
	}
	return err;
}

bool read_matrix(const uint8_t *base, uint64_t offset, float out[16])
{
	const float *src = reinterpret_cast<const float *>(base + offset);
	for (int i = 0; i < 16; ++i)
	{
		if (!is_finite(src[i]))
			return false;
		out[i] = src[i];
	}
	return true;
}

// Returns 0 on success, or a non-zero reason code for the first failing check.
int parse_camera_from_bytes_r(const uint8_t *base, CameraSample &sample)
{
	if (!read_matrix(base, kWorldToViewOffset, sample.world_to_view))
		return 1; // NaN/Inf in world-to-view
	if (!read_matrix(base, kWorldToClipOffset, sample.world_to_clip))
		return 2; // NaN/Inf in world-to-clip
	if (!read_matrix(base, kViewToWorldOffset, sample.view_to_world))
		return 3; // NaN/Inf in view-to-world

	float view_identity[16] = {};
	multiply_row_major_4x4(sample.world_to_view, sample.view_to_world, view_identity);
	const float id_err = row_major_identity_error(view_identity);
	if (id_err > 0.03f)
		return 4; // world_to_view * view_to_world != identity

	multiply_row_major_4x4(sample.view_to_world, sample.world_to_clip, sample.projection);

	const float p00 = sample.projection[0];
	const float p11 = sample.projection[5];
	const float p23 = sample.projection[11];
	const float p32 = sample.projection[14];

	if (p00 < 0.1f || p00 > 10.0f || p11 < 0.1f || p11 > 10.0f)
		return 5; // p00/p11 out of [0.1,10]
	if (std::abs(sample.projection[1]) > 0.05f || std::abs(sample.projection[4]) > 0.05f)
		return 6; // off-diagonal projection
	if (std::abs(p23 + 1.0f) > 0.05f)
		return 7; // p23 != -1
	if (p32 <= 0.0f || p32 > 10.0f)
		return 8; // near <= 0 or > 10

	const float aspect = p11 / p00;
	if (std::abs(aspect - kExpectedAspect) > kMaxAspectError)
		return 9; // aspect != 16/9

	for (int i = 0; i < 3; ++i)
		sample.pos_cm[i] = sample.view_to_world[12 + i];

	const float pos_len = length3(sample.pos_cm);
	if (pos_len < kMinGtaPositionLength)
		return 10; // position too close to origin
	for (int i = 0; i < 3; ++i)
	{
		if (!is_finite(sample.pos_cm[i]) || std::abs(sample.pos_cm[i]) > kMaxGtaAbsPosition)
			return 11; // position out of bounds
	}

	for (int i = 0; i < 3; ++i)
	{
		sample.right[i] = sample.view_to_world[i];
		sample.up[i] = sample.view_to_world[4 + i];
		sample.fwd[i] = sample.view_to_world[8 + i];
	}
	for (int i = 0; i < 4; ++i)
		sample.view_to_world_row3[i] = sample.view_to_world[12 + i];

	if (!validate_basis(sample.right, sample.up, sample.fwd))
		return 12; // non-orthogonal basis vectors

	sample.hfov_deg = radians_to_degrees(2.0f * std::atan(1.0f / p00));
	sample.vfov_deg = radians_to_degrees(2.0f * std::atan(1.0f / p11));
	sample.near_cm = p32;
	return 0;
}

bool parse_camera_from_bytes(const uint8_t *base, CameraSample &sample)
{
	const int reason = parse_camera_from_bytes_r(base, sample);
	if (reason != 0 && !g_logged_parse_diag.exchange(true))
	{
		// One-shot: log the first parse failure with key values (helps re-validate
		// the camera CBV offsets if a game update ever changes them).
		std::ostringstream ss;
		ss << "GTACameraCapture: parse_camera first failure reason=" << reason
		   << " p00=" << sample.projection[0] << " p11=" << sample.projection[5]
		   << " p23=" << sample.projection[11] << " p32=" << sample.projection[14]
		   << " pos=(" << sample.pos_cm[0] << "," << sample.pos_cm[1] << "," << sample.pos_cm[2] << ")";
		log_line(ss.str(), reshade::log::level::warning);
	}
	return reason == 0;
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

bool try_read_from_buffer_shadow(const TrackedCBV &cbv, CameraSample &sample)
{
	const auto it = g_buffer_shadows.find(resource_key(cbv.buffer));
	if (it == g_buffer_shadows.end())
		return false;

	const std::vector<uint8_t> &data = it->second;
	if (cbv.offset + kReadSize > data.size())
		return false;

	return parse_camera_from_bytes(data.data() + cbv.offset, sample);
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
			log_line("GTACameraCapture: direct map failed at least once; trying cached mapped upload pointers.", reshade::log::level::warning);
		return false;
	}

	const bool ok = parse_camera_from_bytes(static_cast<const uint8_t *>(mapped), sample);
	device->unmap_buffer_region(cbv.buffer);
	return ok;
}

void publish_sample(const CameraSample &sample)
{
	g_latest_sample = sample;

	if (!g_logged_first_match)
	{
		g_logged_first_match = true;

		std::ostringstream ss;
		ss << "GTACameraCapture: matched GTA camera CBV. CSV: " << narrow_path(g_csv_path);
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

bool draw_scan_budget_available_locked(uint64_t frame)
{
	reset_candidate_frame_locked(frame);
	if (g_frame_candidate_draw_scans >= kMaxDrawScansPerCaptureFrame)
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
			log_line("GTACameraCapture: candidate CBV logging hit the per-frame row cap; increase kMaxCameraCandidateRowsPerFrame only for short diagnostic captures.", reshade::log::level::warning);
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
			log_line("GTACameraCapture: candidate CBV logging hit the per-frame row cap; increase kMaxCameraCandidateRowsPerFrame only for short diagnostic captures.", reshade::log::level::warning);
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

		for (int j = 0; j < 16; ++j)
			fp << ',' << sample.world_to_view[j];
		for (int j = 0; j < 16; ++j)
			fp << ',' << sample.world_to_clip[j];
		for (int j = 0; j < 16; ++j)
			fp << ',' << sample.view_to_world[j];

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
			log_line("GTACameraCapture: depth readback currently skips MSAA depth resources.", reshade::log::level::warning);
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

bool attach_color_from_rtv(api::device *device, api::resource_view rtv, CameraSample &sample)
{
	if (device == nullptr || rtv.handle == 0)
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

	sample.color_resource = color;
	sample.color_width = desc.texture.width;
	sample.color_height = desc.texture.height;
	sample.color_samples = desc.texture.samples;
	sample.color_format = view_desc.format != api::format::unknown ? view_desc.format : desc.texture.format;
	return true;
}

bool has_main_target(const CameraSample &sample)
{
	if (sample.color_width >= kMinCameraViewWidth && sample.color_height >= kMinCameraViewHeight)
		return true;
	if (sample.depth_width >= kMinCameraViewWidth && sample.depth_height >= kMinCameraViewHeight)
		return true;
	return false;
}

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
			ss << "GTACameraCapture: make_readable_depth_target: skipped MSAA depth. samples="
			   << desc.texture.samples << " size=" << desc.texture.width << 'x' << desc.texture.height;
			log_line(ss.str(), reshade::log::level::warning);
		}
		return false;
	}

	const api::resource_view_desc view_desc = device->get_resource_view_desc(dsv);
	const api::format depth_format = view_desc.format != api::format::unknown ? view_desc.format : desc.texture.format;
	const uint32_t tight_row_pitch = api::format_row_pitch(depth_format, desc.texture.width);
	if (tight_row_pitch == 0)
	{
		if (!g_logged_depth_target_miss.exchange(true))
		{
			std::ostringstream ss;
			ss << "GTACameraCapture: make_readable_depth_target: format_row_pitch=0 for depth."
			   << " view_fmt=" << static_cast<uint32_t>(view_desc.format)
			   << " tex_fmt=" << static_cast<uint32_t>(desc.texture.format)
			   << " depth_fmt=" << static_cast<uint32_t>(depth_format)
			   << " size=" << desc.texture.width << 'x' << desc.texture.height;
			log_line(ss.str(), reshade::log::level::warning);
		}
		return false;
	}

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
		ss << "GTACameraCapture: depth target found. size=" << desc.texture.width << 'x' << desc.texture.height
		   << " depth_fmt=" << static_cast<uint32_t>(depth_format)
		   << " view_fmt=" << static_cast<uint32_t>(view_desc.format)
		   << " tex_fmt=" << static_cast<uint32_t>(desc.texture.format)
		   << " row_pitch=" << row_pitch << " frame=" << frame;
		log_line(ss.str());
	}
	return true;
}

void apply_readable_depth_target(CameraSample &sample, const DepthTarget &target)
{
	if (!target.valid || target.frame != sample.frame)
	{
		if (!g_logged_depth_apply_miss.exchange(true))
		{
			std::ostringstream ss;
			ss << "GTACameraCapture: apply_readable_depth_target: skipped."
			   << " target.valid=" << target.valid
			   << " target.frame=" << target.frame
			   << " sample.frame=" << sample.frame;
			log_line(ss.str(), reshade::log::level::warning);
		}
		return;
	}
	// Note: no color/depth size check here.
	// GTA V uses the camera CBV in a half-resolution pass (e.g. 1280x720) while the
	// depth buffer is always full-resolution (e.g. 2560x1440).  The size validation
	// against the actual backbuffer RTV is done later in depth_size_matches_color_target.

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
	if (readback.d3d11_texture != nullptr)
		readback.d3d11_texture->Release();
	readback = DepthReadback{};
}

bool write_binary_file_checked(const std::filesystem::path &path, const void *data, uint64_t byte_size, const char *label)
{
	std::error_code ec;
	std::filesystem::create_directories(path.parent_path(), ec);
	if (ec)
	{
		std::ostringstream ss;
		ss << "GTACameraCapture: failed to create " << label << " output directory: "
		   << path.parent_path().string() << " error=" << ec.message();
		log_line(ss.str(), reshade::log::level::warning);
		return false;
	}

	errno = 0;
	std::ofstream bin(path, std::ios::binary | std::ios::trunc);
	if (!bin)
	{
		std::ostringstream ss;
		ss << "GTACameraCapture: failed to open " << label << " bin for write: "
		   << path.string();
		if (errno != 0)
			ss << " errno=" << errno << " (" << std::strerror(errno) << ')';
		log_line(ss.str(), reshade::log::level::warning);
		return false;
	}

	bin.write(static_cast<const char *>(data), static_cast<std::streamsize>(byte_size));
	if (!bin)
	{
		std::ostringstream ss;
		ss << "GTACameraCapture: failed to write " << label << " bin: "
		   << path.string() << " expected_bytes=" << byte_size;
		log_line(ss.str(), reshade::log::level::warning);
		return false;
	}

	bin.close();
	if (!bin)
	{
		std::ostringstream ss;
		ss << "GTACameraCapture: failed to close " << label << " bin after write: "
		   << path.string() << " expected_bytes=" << byte_size;
		log_line(ss.str(), reshade::log::level::warning);
		return false;
	}

	const uint64_t actual_size = std::filesystem::file_size(path, ec);
	if (ec || actual_size != byte_size)
	{
		std::ostringstream ss;
		ss << "GTACameraCapture: " << label << " bin size mismatch: "
		   << path.string() << " expected_bytes=" << byte_size;
		if (ec)
			ss << " file_size_error=" << ec.message();
		else
			ss << " actual_bytes=" << actual_size;
		log_line(ss.str(), reshade::log::level::warning);
		return false;
	}

	return true;
}

// Actually writes a job's bytes + JSON sidecar to disk. Called on the background
// writer thread (and inline during teardown). Holds no addon locks.
void do_write_job(const WriteJob &job)
{
	if (!write_binary_file_checked(job.bin_path, job.bin_bytes.data(), job.bin_bytes.size(), job.label))
		return;
	if (!job.json_text.empty())
	{
		std::ofstream json(job.json_path);
		if (json)
			json << job.json_text;
	}
}

void writer_thread_main()
{
	for (;;)
	{
		WriteJob job;
		{
			std::unique_lock<std::mutex> lock(g_writer_mutex);
			g_writer_cv.wait(lock, [] { return g_writer_shutdown || !g_writer_queue.empty(); });
			// Drain remaining jobs even after shutdown is requested, then exit.
			if (g_writer_queue.empty())
				return;
			job = std::move(g_writer_queue.front());
			g_writer_queue.pop_front();
		}
		g_writer_cv.notify_all(); // wake a producer blocked on the back-pressure cap
		do_write_job(job);
	}
}

// Hand a finished readback to the background writer. Falls back to a synchronous
// inline write during teardown (so we never spawn a thread from DllMain) or if
// the worker has already been stopped.
void enqueue_write_job(WriteJob &&job)
{
	std::unique_lock<std::mutex> lock(g_writer_mutex);

	if (g_writer_shutdown)
	{
		lock.unlock();
		do_write_job(job);
		return;
	}

	if (!g_writer_started)
	{
		// Lazy start. This only ever runs on the present thread (first capture
		// flush), never under the DllMain loader lock.
		g_writer_started = true;
		g_writer_thread = std::thread(writer_thread_main);
	}

	// Back-pressure: bound queued memory. Blocks the present thread only when the
	// disk cannot keep up with the capture rate.
	g_writer_cv.wait(lock, [] { return g_writer_queue.size() < kMaxWriterQueueJobs || g_writer_shutdown; });
	if (g_writer_shutdown)
	{
		lock.unlock();
		do_write_job(job);
		return;
	}

	g_writer_queue.push_back(std::move(job));
	lock.unlock();
	g_writer_cv.notify_all();
}

// Stop the writer and wait for the queue to fully drain. Called once from
// unregister_events. After this returns, g_writer_shutdown makes any further
// enqueue_write_job calls write synchronously inline.
void writer_stop_and_join()
{
	std::thread t;
	{
		std::lock_guard<std::mutex> lock(g_writer_mutex);
		g_writer_shutdown = true;
		t = std::move(g_writer_thread);
		g_writer_started = false;
	}
	g_writer_cv.notify_all();
	if (t.joinable())
		t.join();
}

// Maps a D3D11 staging texture and copies its tightly-packed rows into `out`.
// This is the part that must stay on the present thread (D3D11 immediate context
// is single-threaded); the subsequent file write is handed to the writer thread.
bool read_d3d11_staging_to_bytes(ID3D11Texture2D *texture, uint32_t height, uint32_t tight_row_pitch, uint64_t byte_size, std::vector<uint8_t> &out)
{
	if (texture == nullptr || height == 0 || tight_row_pitch == 0 || byte_size == 0)
		return false;

	ID3D11Device *native_device = nullptr;
	texture->GetDevice(&native_device);
	if (native_device == nullptr)
		return false;

	ID3D11DeviceContext *context = nullptr;
	native_device->GetImmediateContext(&context);
	native_device->Release();
	if (context == nullptr)
		return false;

	D3D11_MAPPED_SUBRESOURCE mapped = {};
	const HRESULT hr = context->Map(texture, 0, D3D11_MAP_READ, 0, &mapped);
	if (FAILED(hr))
	{
		context->Release();
		std::ostringstream ss;
		ss << "GTACameraCapture: failed to map D3D11 staging texture. hr=0x"
		   << std::hex << static_cast<unsigned long>(hr) << std::dec;
		log_line(ss.str(), reshade::log::level::warning);
		return false;
	}

	bool ok = false;
	if (mapped.pData != nullptr && mapped.RowPitch >= tight_row_pitch)
	{
		out.resize(static_cast<size_t>(byte_size));
		const uint8_t *src = static_cast<const uint8_t *>(mapped.pData);
		uint8_t *dst = out.data();
		for (uint32_t row = 0; row < height; ++row)
			std::memcpy(dst + static_cast<size_t>(row) * tight_row_pitch, src + static_cast<size_t>(row) * mapped.RowPitch, tight_row_pitch);
		ok = true;
	}
	else
	{
		std::ostringstream ss;
		ss << "GTACameraCapture: invalid D3D11 staging map. mapped_row_pitch="
		   << mapped.RowPitch << " tight_row_pitch=" << tight_row_pitch;
		log_line(ss.str(), reshade::log::level::warning);
	}

	context->Unmap(texture, 0);
	context->Release();
	return ok;
}

static DXGI_FORMAT depth_to_typeless_dxgi_fmt(DXGI_FORMAT fmt)
{
	switch (fmt)
	{
	case DXGI_FORMAT_D16_UNORM:            return DXGI_FORMAT_R16_TYPELESS;
	case DXGI_FORMAT_D24_UNORM_S8_UINT:    return DXGI_FORMAT_R24G8_TYPELESS;
	case DXGI_FORMAT_D32_FLOAT:            return DXGI_FORMAT_R32_TYPELESS;
	case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return DXGI_FORMAT_R32G8X24_TYPELESS;
	default:                               return fmt;
	}
}

bool create_d3d11_staging_copy(api::resource source, ID3D11Texture2D **out_texture, D3D11_TEXTURE2D_DESC *out_desc, const char *label)
{
	if (out_texture == nullptr)
		return false;
	*out_texture = nullptr;

	if (source.handle == 0)
		return false;

	ID3D11Resource *source_resource = reinterpret_cast<ID3D11Resource *>(source.handle);
	ID3D11Texture2D *source_texture = nullptr;
	HRESULT hr = source_resource->QueryInterface(IID_PPV_ARGS(&source_texture));
	if (FAILED(hr) || source_texture == nullptr)
	{
		std::ostringstream ss;
		ss << "GTACameraCapture: " << label << " source is not an ID3D11Texture2D. hr=0x"
		   << std::hex << static_cast<unsigned long>(hr) << std::dec;
		log_line(ss.str(), reshade::log::level::warning);
		return false;
	}

	D3D11_TEXTURE2D_DESC desc = {};
	source_texture->GetDesc(&desc);
	if (desc.Width == 0 || desc.Height == 0 || desc.MipLevels == 0 || desc.ArraySize == 0)
	{
		source_texture->Release();
		return false;
	}
	if (desc.SampleDesc.Count > 1)
	{
		source_texture->Release();
		std::ostringstream ss;
		ss << "GTACameraCapture: skipping D3D11 " << label << " staging copy for MSAA texture samples="
		   << desc.SampleDesc.Count;
		log_line(ss.str(), reshade::log::level::warning);
		return false;
	}

	ID3D11Device *native_device = nullptr;
	source_texture->GetDevice(&native_device);
	if (native_device == nullptr)
	{
		source_texture->Release();
		return false;
	}

	D3D11_TEXTURE2D_DESC staging_desc = desc;
	staging_desc.Usage = D3D11_USAGE_STAGING;
	staging_desc.BindFlags = 0;
	staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	staging_desc.MiscFlags = 0;
	// D3D11 rejects staging textures with depth-stencil formats; convert to the
	// typeless equivalent (same bit layout, same cast-set as the source format).
	staging_desc.Format = depth_to_typeless_dxgi_fmt(staging_desc.Format);

	ID3D11Texture2D *staging = nullptr;
	hr = native_device->CreateTexture2D(&staging_desc, nullptr, &staging);
	if (FAILED(hr) || staging == nullptr)
	{
		std::ostringstream ss;
		ss << "GTACameraCapture: failed to create D3D11 " << label << " staging texture. format="
		   << static_cast<uint32_t>(desc.Format) << " size=" << desc.Width << 'x' << desc.Height
		   << " hr=0x" << std::hex << static_cast<unsigned long>(hr) << std::dec;
		log_line(ss.str(), reshade::log::level::warning);
		native_device->Release();
		source_texture->Release();
		return false;
	}

	ID3D11DeviceContext *context = nullptr;
	native_device->GetImmediateContext(&context);
	if (context == nullptr)
	{
		staging->Release();
		native_device->Release();
		source_texture->Release();
		return false;
	}

	context->CopyResource(staging, source_texture);
	context->Release();
	native_device->Release();
	source_texture->Release();

	*out_texture = staging;
	if (out_desc != nullptr)
		*out_desc = desc;
	return true;
}

// Reads a depth staging texture. D32_FLOAT_S8X24 is 8 bytes/pixel but only the
// first 4 (the R32 float depth) matter — the stencil+padding 4 bytes are useless
// and would double the file. This keeps only the R32 channel (halves depth I/O;
// the decoded values are identical, since the reader already used just those 4
// bytes). Other depth formats are copied tight as-is. out_row_pitch/out_byte_size
// report the packed result so the JSON sidecar matches the .bin.
bool read_d3d11_staging_depth(ID3D11Texture2D *texture, uint32_t width, uint32_t height,
                              uint32_t src_tight_row_pitch,
                              std::vector<uint8_t> &out, uint32_t &out_row_pitch, uint64_t &out_byte_size)
{
	if (texture == nullptr || width == 0 || height == 0 || src_tight_row_pitch == 0)
		return false;
	const uint32_t src_bpp = src_tight_row_pitch / width;

	ID3D11Device *native_device = nullptr;
	texture->GetDevice(&native_device);
	if (native_device == nullptr)
		return false;
	ID3D11DeviceContext *ctx = nullptr;
	native_device->GetImmediateContext(&ctx);
	native_device->Release();
	if (ctx == nullptr)
		return false;

	D3D11_MAPPED_SUBRESOURCE mapped = {};
	const HRESULT hr = ctx->Map(texture, 0, D3D11_MAP_READ, 0, &mapped);
	if (FAILED(hr))
	{
		ctx->Release();
		std::ostringstream ss;
		ss << "GTACameraCapture: failed to map D3D11 depth staging. hr=0x"
		   << std::hex << static_cast<unsigned long>(hr) << std::dec;
		log_line(ss.str(), reshade::log::level::warning);
		return false;
	}

	bool ok = false;
	if (mapped.pData != nullptr && mapped.RowPitch >= src_tight_row_pitch)
	{
		const uint8_t *src = static_cast<const uint8_t *>(mapped.pData);
		const uint32_t out_bpp = (src_bpp == 8) ? 4u : src_bpp;  // strip stencil+pad from R32G8X24
		out_row_pitch = width * out_bpp;
		out_byte_size = static_cast<uint64_t>(out_row_pitch) * height;
		out.resize(static_cast<size_t>(out_byte_size));
		uint8_t *dst = out.data();
		for (uint32_t y = 0; y < height; ++y)
		{
			const uint8_t *srow = src + static_cast<size_t>(y) * mapped.RowPitch;
			uint8_t *drow = dst + static_cast<size_t>(y) * out_row_pitch;
			if (src_bpp == 8)
			{
				for (uint32_t x = 0; x < width; ++x)
					std::memcpy(drow + static_cast<size_t>(x) * 4, srow + static_cast<size_t>(x) * 8, 4);
			}
			else
			{
				std::memcpy(drow, srow, out_row_pitch);
			}
		}
		ok = true;
	}
	else
	{
		std::ostringstream ss;
		ss << "GTACameraCapture: invalid D3D11 depth staging map. mapped_row_pitch="
		   << mapped.RowPitch << " tight_row_pitch=" << src_tight_row_pitch;
		log_line(ss.str(), reshade::log::level::warning);
	}

	ctx->Unmap(texture, 0);
	ctx->Release();
	return ok;
}

bool write_depth_readback(DepthReadback &readback, const std::string &postfix)
{
	if (!readback.valid)
		return false;

	if (kWaitIdleBeforeReadbackMap && g_graphics_queue != nullptr)
		g_graphics_queue->wait_idle();

	// Map + copy the bytes out on the present thread (D3D11 immediate context is
	// single-threaded), then hand the file write to the background writer.
	// Depth keeps only the R32 float channel (see read_d3d11_staging_depth), so the
	// on-disk pitch/size can be smaller than the source staging texture.
	std::vector<uint8_t> bytes;
	uint32_t out_row_pitch = readback.tight_row_pitch;
	uint64_t out_byte_size = readback.byte_size;
	if (readback.d3d11_staging)
	{
		if (!read_d3d11_staging_depth(readback.d3d11_texture, readback.width, readback.height,
		                              readback.tight_row_pitch, bytes, out_row_pitch, out_byte_size))
			return false;
	}
	else
	{
		if (readback.device == nullptr || readback.buffer.handle == 0)
			return false;

		void *mapped = nullptr;
		if (!readback.device->map_buffer_region(readback.buffer, 0, readback.byte_size, api::map_access::read_only, &mapped) || mapped == nullptr)
		{
			log_line("GTACameraCapture: failed to map depth readback buffer.", reshade::log::level::warning);
			return false;
		}

		const uint8_t *src = static_cast<const uint8_t *>(mapped);
		bytes.assign(src, src + readback.byte_size);
		readback.device->unmap_buffer_region(readback.buffer);
	}

	std::ostringstream json;
	json << "{\n"
	     << "  \"frame\": " << readback.frame << ",\n"
	     << "  \"file\": \"" << postfix << ".bin\",\n"
	     << "  \"width\": " << readback.width << ",\n"
	     << "  \"height\": " << readback.height << ",\n"
	     << "  \"format\": " << static_cast<uint32_t>(readback.format) << ",\n"
	     << "  \"row_pitch\": " << out_row_pitch << ",\n"
	     << "  \"tight_row_pitch\": " << out_row_pitch << ",\n"
	     << "  \"byte_size\": " << out_byte_size << ",\n"
	     << "  \"samples\": " << readback.samples << ",\n"
	     << "  \"near_cm\": " << std::fixed << std::setprecision(6) << readback.near_cm << "\n"
	     << "}\n";

	WriteJob job;
	job.bin_path = g_depth_dir / (postfix + ".bin");
	job.bin_bytes = std::move(bytes);
	job.json_path = g_depth_dir / (postfix + ".json");
	job.json_text = json.str();
	job.label = "depth";
	enqueue_write_job(std::move(job));
	return true;
}

void destroy_color_readback(ColorReadback &readback)
{
	if (readback.valid && readback.device != nullptr && readback.buffer.handle != 0)
		readback.device->destroy_resource(readback.buffer);
	if (readback.d3d11_texture != nullptr)
		readback.d3d11_texture->Release();
	readback = ColorReadback{};
}

bool write_color_readback(ColorReadback &readback, const std::string &postfix)
{
	if (!readback.valid)
		return false;

	if (kWaitIdleBeforeReadbackMap && g_graphics_queue != nullptr)
		g_graphics_queue->wait_idle();

	std::vector<uint8_t> bytes;
	if (readback.d3d11_staging)
	{
		if (!read_d3d11_staging_to_bytes(readback.d3d11_texture, readback.height, readback.tight_row_pitch, readback.byte_size, bytes))
			return false;
	}
	else
	{
		if (readback.device == nullptr || readback.buffer.handle == 0)
			return false;

		void *mapped = nullptr;
		if (!readback.device->map_buffer_region(readback.buffer, 0, readback.byte_size, api::map_access::read_only, &mapped) || mapped == nullptr)
		{
			log_line("GTACameraCapture: failed to map color readback buffer.", reshade::log::level::warning);
			return false;
		}

		const uint8_t *src = static_cast<const uint8_t *>(mapped);
		bytes.assign(src, src + readback.byte_size);
		readback.device->unmap_buffer_region(readback.buffer);
	}

	std::ostringstream json;
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

	WriteJob job;
	job.bin_path = g_color_dir / (postfix + ".bin");
	job.bin_bytes = std::move(bytes);
	job.json_path = g_color_dir / (postfix + ".json");
	job.json_text = json.str();
	job.label = "color";
	enqueue_write_job(std::move(job));
	return true;
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

	if (device->get_api() == api::device_api::d3d11)
	{
		ID3D11Texture2D *staging = nullptr;
		D3D11_TEXTURE2D_DESC native_desc = {};
		if (!create_d3d11_staging_copy(sample.depth_resource, &staging, &native_desc, "depth"))
			return false;

		row_pitch = tight_row_pitch;
		byte_size = static_cast<uint64_t>(tight_row_pitch) * sample.depth_height;

		if (!g_logged_depth_readback_info.exchange(true))
		{
			std::ostringstream ss;
			ss << "GTACameraCapture: D3D11 depth staging copy queued. size="
			   << sample.depth_width << 'x' << sample.depth_height
			   << " reshade_format=" << static_cast<uint32_t>(sample.depth_format)
			   << " dxgi_format=" << static_cast<uint32_t>(native_desc.Format)
			   << " row_pitch=" << row_pitch
			   << " bytes=" << byte_size
			   << " resource=0x" << std::hex << sample.depth_resource.handle << std::dec;
			log_line(ss.str());
		}

		DepthReadback readback = {};
		readback.valid = true;
		readback.d3d11_staging = true;
		readback.frame = sample.frame;
		readback.device = device;
		readback.d3d11_texture = staging;
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
	if (device->get_api() != api::device_api::d3d12)
	{
		if (!g_logged_readback_unsupported_api.exchange(true))
			log_line("GTACameraCapture: RGB/depth readback is only implemented for D3D11 staging textures and D3D12 texture copies.", reshade::log::level::warning);
		return false;
	}

	if (!supports_direct_depth_buffer_copy(sample.depth_format))
	{
		if (!g_logged_depth_unsupported_format.exchange(true))
		{
			std::ostringstream ss;
			ss << "GTACameraCapture: skipping direct depth readback for format="
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
			log_line("GTACameraCapture: failed to create depth readback buffer.", reshade::log::level::warning);
		return false;
	}

	if (!g_logged_depth_readback_info.exchange(true))
	{
		std::ostringstream ss;
		ss << "GTACameraCapture: depth readback copy queued. size="
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
			log_line("GTACameraCapture: color readback currently skips MSAA render targets.", reshade::log::level::warning);
		return false;
	}

	const uint32_t tight_row_pitch = api::format_row_pitch(color_format, desc.texture.width);
	if (tight_row_pitch == 0)
		return false;

	if (device->get_api() == api::device_api::d3d11)
	{
		ID3D11Texture2D *staging = nullptr;
		D3D11_TEXTURE2D_DESC native_desc = {};
		if (!create_d3d11_staging_copy(color, &staging, &native_desc, "color"))
			return false;

		const uint32_t row_pitch = tight_row_pitch;
		const uint64_t byte_size = static_cast<uint64_t>(row_pitch) * desc.texture.height;

		if (!g_logged_color_readback_info.exchange(true))
		{
			std::ostringstream ss;
			ss << "GTACameraCapture: D3D11 color staging copy queued. size="
			   << desc.texture.width << 'x' << desc.texture.height
			   << " reshade_format=" << static_cast<uint32_t>(color_format)
			   << " dxgi_format=" << static_cast<uint32_t>(native_desc.Format)
			   << " row_pitch=" << row_pitch
			   << " bytes=" << byte_size
			   << " resource=0x" << std::hex << color.handle << std::dec;
			log_line(ss.str());
		}

		ColorReadback readback = {};
		readback.valid = true;
		readback.d3d11_staging = true;
		readback.frame = frame;
		readback.device = device;
		readback.d3d11_texture = staging;
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
	if (device->get_api() != api::device_api::d3d12)
	{
		if (!g_logged_readback_unsupported_api.exchange(true))
			log_line("GTACameraCapture: RGB/depth readback is only implemented for D3D11 staging textures and D3D12 texture copies.", reshade::log::level::warning);
		return false;
	}

	const uint32_t row_pitch = align_to(tight_row_pitch, kTextureReadbackPitchAlignment);
	const uint64_t byte_size = static_cast<uint64_t>(row_pitch) * desc.texture.height;

	api::resource readback_buffer = {};
	const api::resource_desc buffer_desc(byte_size, api::memory_heap::gpu_to_cpu, api::resource_usage::copy_dest);
	if (!device->create_resource(buffer_desc, nullptr, api::resource_usage::copy_dest, &readback_buffer))
	{
		if (!g_logged_color_failure.exchange(true))
			log_line("GTACameraCapture: failed to create color readback buffer.", reshade::log::level::warning);
		return false;
	}

	const uint32_t bytes_per_pixel = desc.texture.width != 0 ? std::max(1u, tight_row_pitch / desc.texture.width) : 1u;
	const uint32_t row_length_pixels = row_pitch / bytes_per_pixel;

	if (!g_logged_color_readback_info.exchange(true))
	{
		std::ostringstream ss;
		ss << "GTACameraCapture: color readback copy queued. size="
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
		ss << "GTACameraCapture: skipping depth readback because DSV size "
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

	g_scan_calls.fetch_add(1, std::memory_order_relaxed);

	api::device *device = cmd_list->get_device();
	if (device == nullptr)
		return;

	std::vector<TrackedCBV> cbvs;
	api::resource_view rtv0 = {};
	api::resource_view dsv = {};
	bool log_candidates_this_draw = false;

	{
		std::lock_guard<std::mutex> lock(g_mutex);
		if (!draw_scan_budget_available_locked(frame))
			return;
		log_candidates_this_draw = kLogCameraCandidates;
		if (!log_candidates_this_draw && !kUseLastMatchedCameraSampleInFrame && g_latest_sample.valid && g_latest_sample.frame == frame)
			return;

		const auto state_it = g_cmd_states.find(cmd_list);
		if (state_it == g_cmd_states.end())
			return;

		cbvs = state_it->second.cbvs;
		rtv0 = state_it->second.current_rtv0;
		dsv = state_it->second.current_dsv;
	}

	for (const TrackedCBV &cbv : cbvs)
	{
		g_cbv_checks.fetch_add(1, std::memory_order_relaxed);

		if (cbv.buffer.handle == 0)
			continue;
		if (cbv.size != UINT64_MAX && cbv.size < kReadSize)
			continue;
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
			if (!matched)
				matched = try_read_from_buffer_shadow(cbv, sample);
		}

		if (!matched && device->get_api() == api::device_api::d3d12)
			matched = try_read_from_device(device, cbv, sample);
		if (!matched)
		{
			std::lock_guard<std::mutex> lock(g_mutex);
			matched = try_read_from_mapped_shadow(cbv, sample);
			if (!matched)
				matched = try_read_from_buffer_shadow(cbv, sample);
		}

		if (matched)
		{
			sample.valid = true;
			attach_color_from_rtv(device, rtv0, sample);
			attach_depth_from_dsv(device, dsv, sample);
			if (!has_main_target(sample))
			{
				if (!g_logged_no_main_target.exchange(true))
				{
					std::ostringstream ss;
					ss << "GTACameraCapture: parse matched but has_main_target=false."
					   << " color=" << sample.color_width << 'x' << sample.color_height
					   << " depth=" << sample.depth_width << 'x' << sample.depth_height
					   << " rtv0=" << rtv0.handle << " dsv=" << dsv.handle;
					log_line(ss.str(), reshade::log::level::warning);
				}
				continue;
			}

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
	// CBV tracking is only consumed by scan_cbvs while recording. Skip the locked
	// vector rebuild entirely when capture is off.
	if (!g_capture_enabled.load(std::memory_order_relaxed))
		return;
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
	uint32_t count,
	const api::render_pass_render_target_desc *rts,
	const api::render_pass_depth_stencil_desc *ds)
{
	// current_rtv0/current_dsv and g_latest_readable_depth are only consumed on
	// capture frames; skip the device queries + lock on every other render pass.
	if (!capture_active_now())
		return;

	DepthTarget readable_depth = {};
	const uint64_t frame = g_frame_index.load(std::memory_order_relaxed);
	api::device *device = cmd_list != nullptr ? cmd_list->get_device() : nullptr;
	if (ds != nullptr)
		make_readable_depth_target(device, ds->view, frame, readable_depth);

	std::lock_guard<std::mutex> lock(g_mutex);
	CommandListState &state = g_cmd_states[cmd_list];
	state.current_rtv0 = count > 0 && rts != nullptr ? rts[0].view : api::resource_view{};
	state.current_dsv = ds != nullptr ? ds->view : api::resource_view{};
	if (readable_depth.valid)
		g_latest_readable_depth = readable_depth;
}

void on_bind_render_targets_and_depth_stencil(
	api::command_list *cmd_list,
	uint32_t count,
	const api::resource_view *rtvs,
	api::resource_view dsv)
{
	if (!capture_active_now())
		return;

	DepthTarget readable_depth = {};
	const uint64_t frame = g_frame_index.load(std::memory_order_relaxed);
	api::device *device = cmd_list != nullptr ? cmd_list->get_device() : nullptr;
	make_readable_depth_target(device, dsv, frame, readable_depth);

	std::lock_guard<std::mutex> lock(g_mutex);
	CommandListState &state = g_cmd_states[cmd_list];
	state.current_rtv0 = count > 0 && rtvs != nullptr ? rtvs[0] : api::resource_view{};
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

bool on_dispatch(api::command_list *cmd_list, uint32_t group_count_x, uint32_t group_count_y, uint32_t group_count_z)
{
	const uint64_t group_count =
		static_cast<uint64_t>(std::max(group_count_x, 1u)) *
		static_cast<uint64_t>(std::max(group_count_y, 1u)) *
		static_cast<uint64_t>(std::max(group_count_z, 1u));
	scan_cbvs(cmd_list, group_count > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(group_count));
	return false;
}

bool copy_buffer_shadow_locked(api::device *device, const void *data, api::resource resource, uint64_t offset, uint64_t size)
{
	if (device == nullptr || data == nullptr || resource.handle == 0)
		return false;
	if (!should_capture_frame(g_frame_index.load(std::memory_order_relaxed)))
		return false;

	const api::resource_desc desc = device->get_resource_desc(resource);
	if (desc.type != api::resource_type::buffer || desc.buffer.size == 0)
		return false;
	if (offset >= desc.buffer.size)
		return false;

	uint64_t copy_size = size;
	if (copy_size == UINT64_MAX || offset + copy_size > desc.buffer.size)
		copy_size = desc.buffer.size - offset;
	if (copy_size == 0 || copy_size > kMaxShadowBufferBytes)
	{
		g_shadow_skips.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	if (offset + copy_size > kMaxShadowBufferBytes)
	{
		g_shadow_skips.fetch_add(1, std::memory_order_relaxed);
		return false;
	}

	std::vector<uint8_t> &shadow = g_buffer_shadows[resource_key(resource)];
	const size_t required_size = static_cast<size_t>(offset + copy_size);
	if (shadow.size() < required_size)
		shadow.resize(required_size);
	std::memcpy(shadow.data() + static_cast<size_t>(offset), data, static_cast<size_t>(copy_size));
	g_shadow_updates.fetch_add(1, std::memory_order_relaxed);
	g_shadow_bytes.fetch_add(copy_size, std::memory_order_relaxed);
	return true;
}

void on_map_buffer_region(api::device *, api::resource resource, uint64_t offset, uint64_t size, api::map_access access, void **data)
{
	// Gate on capture_enabled ONLY, never should_capture_frame. GTA's RAGE engine
	// keeps its dynamic constant buffer persistently mapped (ring buffer): it is
	// mapped on some frame and stays mapped across many frames, so the camera CBV
	// is read via this live pointer (no unmap -> no shadow ever gets built). If we
	// only tracked maps on capture frames, the pointer would be missed unless a
	// (re)map happened to land on a 1-in-15 capture frame -> sparse/dropped matches.
	if (!g_capture_enabled.load(std::memory_order_relaxed))
		return;
	if (resource.handle == 0 || data == nullptr || *data == nullptr)
		return;

	std::lock_guard<std::mutex> lock(g_mutex);
	g_mapped_buffers[resource_key(resource)] = MappedBuffer{ offset, size, static_cast<const uint8_t *>(*data), access };
}

void on_unmap_buffer_region(api::device *device, api::resource resource)
{
	// Must mirror on_map's gate so the live-pointer table stays consistent.
	if (!g_capture_enabled.load(std::memory_order_relaxed))
		return;

	std::lock_guard<std::mutex> lock(g_mutex);
	const auto it = g_mapped_buffers.find(resource_key(resource));
	if (it != g_mapped_buffers.end())
	{
		const MappedBuffer &mapped = it->second;
		if (mapped.data != nullptr && mapped.access != api::map_access::read_only)
			copy_buffer_shadow_locked(device, mapped.data, resource, mapped.offset, mapped.size);
		g_mapped_buffers.erase(it);
	}
}

bool on_update_buffer_region(api::device *device, const void *data, api::resource dest, uint64_t dest_offset, uint64_t size)
{
	if (!g_capture_enabled.load(std::memory_order_relaxed))
		return false;

	std::lock_guard<std::mutex> lock(g_mutex);
	copy_buffer_shadow_locked(device, data, dest, dest_offset, size);
	return false;
}

bool on_update_buffer_region_command(api::command_list *cmd_list, const void *data, api::resource dest, uint64_t dest_offset, uint64_t size)
{
	if (!g_capture_enabled.load(std::memory_order_relaxed))
		return false;

	api::device *device = cmd_list != nullptr ? cmd_list->get_device() : nullptr;
	std::lock_guard<std::mutex> lock(g_mutex);
	copy_buffer_shadow_locked(device, data, dest, dest_offset, size);
	return false;
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

		// Update sample color dims from the actual backbuffer RTV (the draw-call
		// RTV recorded by scan_cbvs may be a half-resolution intermediate pass).
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
						sample.color_width  = bd.texture.width;
						sample.color_height = bd.texture.height;
					}
				}
			}
		}

		// DIAGNOSTIC (one-shot): what depth does the camera pass carry vs the global one?
		if (!g_logged_depth_choice.exchange(true))
		{
			std::ostringstream ss;
			ss << "GTACameraCapture: DEPTH-CHOICE camera_pass_depth=" << sample.depth_width << 'x' << sample.depth_height
			   << " color=" << sample.color_width << 'x' << sample.color_height
			   << " global_readable_depth=" << g_latest_readable_depth.width << 'x' << g_latest_readable_depth.height
			   << " cam_depth_handle=" << sample.depth_resource.handle;
			log_line(ss.str());
		}
		// Prefer the camera pass's OWN depth (same view as the camera). Only fall back to the
		// global most-recent large depth when the camera-pass depth is missing or its size
		// does not match the (backbuffer-corrected) color size.
		const bool cam_depth_ok = sample.depth_resource.handle != 0
			&& sample.depth_width == sample.color_width && sample.depth_height == sample.color_height;
		if (!cam_depth_ok)
			apply_readable_depth_target(sample, g_latest_readable_depth);
		g_latest_sample = sample;

		if (need_depth && sample.depth_resource.handle == 0)
		{
			if (!g_logged_depth_resource_zero.exchange(true))
			{
				std::ostringstream ss;
				ss << "GTACameraCapture: depth resource is null after apply_readable_depth_target."
				   << " g_latest_readable_depth.valid=" << g_latest_readable_depth.valid
				   << " g_latest_readable_depth.frame=" << g_latest_readable_depth.frame
				   << " sample.frame=" << sample.frame
				   << " color=" << sample.color_width << 'x' << sample.color_height
				   << " depth=" << g_latest_readable_depth.width << 'x' << g_latest_readable_depth.height
				   << " depth_fmt=" << static_cast<uint32_t>(g_latest_readable_depth.format);
				log_line(ss.str(), reshade::log::level::warning);
			}
		}
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

// Draw a red REC square in the top-left of the final image, ONLY while recording.
// Gated on the exact g_capture_enabled flag, so the marker is present iff capture
// is active. Runs in reshade_finish_effects (after the color readback was queued
// in begin_effects), so it is never part of the captured frames.
void draw_rec_indicator(api::command_list *cmd_list, api::resource_view rtv)
{
	if (!kRecIndicator)
		return;
	if (!g_capture_enabled.load(std::memory_order_relaxed))
		return;
	if (cmd_list == nullptr || rtv.handle == 0)
		return;

	api::device *device = cmd_list->get_device();
	if (device == nullptr || device->get_api() != api::device_api::d3d11)
		return;

	const api::resource res = device->get_resource_from_view(rtv);
	if (res.handle == 0)
		return;
	const api::resource_desc desc = device->get_resource_desc(res);
	if (desc.type != api::resource_type::texture_2d || desc.texture.height == 0)
		return;

	ID3D11RenderTargetView *nrtv = reinterpret_cast<ID3D11RenderTargetView *>(rtv.handle);
	ID3D11Device *d3d = nullptr;
	nrtv->GetDevice(&d3d);
	if (d3d == nullptr)
		return;
	ID3D11DeviceContext *ctx0 = nullptr;
	d3d->GetImmediateContext(&ctx0);
	d3d->Release();
	if (ctx0 == nullptr)
		return;

	ID3D11DeviceContext1 *ctx1 = nullptr;
	if (SUCCEEDED(ctx0->QueryInterface(IID_PPV_ARGS(&ctx1))) && ctx1 != nullptr)
	{
		const int m = 12;                   // margin from the top-left corner (px)
		const int s = kRecMarkerSizePx;     // square side (px)
		const D3D11_RECT box = { m, m, m + s, m + s };
		const FLOAT red[4] = { 0.96f, 0.06f, 0.06f, 1.0f }; // bright red
		ctx1->ClearView(nrtv, red, &box, 1);
		ctx1->Release();
	}
	ctx0->Release();
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

	if (!g_logged_present_seen.exchange(true))
		log_line("GTACameraCapture: present hook active.");

	const bool toggle_key_down = runtime != nullptr && runtime->is_key_pressed(kToggleCaptureKey);
	const bool toggle_key_was_down = g_toggle_key_down.exchange(toggle_key_down, std::memory_order_relaxed);
	if (toggle_key_down && !toggle_key_was_down)
	{
		const bool enabled = !g_capture_enabled.load(std::memory_order_relaxed);
		g_capture_enabled.store(enabled, std::memory_order_relaxed);
		if (enabled)
		{
			// New recording segment on each F8 start: bump session id; the first CSV
			// row written this segment gets seg_start=1.
			g_capture_session.fetch_add(1, std::memory_order_relaxed);
			g_segment_start_pending.store(true, std::memory_order_relaxed);
		}

		{
			std::lock_guard<std::mutex> lock(g_mutex);
			g_latest_sample.valid = false;
			g_latest_readable_depth = DepthTarget{};
			g_buffer_shadows.clear();
			g_mapped_buffers.clear();   // drop stale live-pointers across F8 toggles
			g_frame_candidates.clear();
			g_frame_candidate_keys.clear();
			g_frame_candidate_draw_scans = 0;
			g_frame_candidates_frame = UINT64_MAX;
		}
		g_scan_calls.store(0, std::memory_order_relaxed);
		g_cbv_checks.store(0, std::memory_order_relaxed);
		g_shadow_updates.store(0, std::memory_order_relaxed);
		g_shadow_bytes.store(0, std::memory_order_relaxed);
		g_shadow_skips.store(0, std::memory_order_relaxed);

		log_line(enabled ? "GTACameraCapture: capture started." : "GTACameraCapture: capture stopped.");
		play_toggle_cue(enabled);
	}

	if (g_capture_enabled.load(std::memory_order_relaxed) && !g_logged_first_match && frame != 0 && (frame % 300) == 0)
	{
		std::ostringstream ss;
		ss << "GTACameraCapture: waiting for GTA camera match. "
		   << "scan_calls=" << g_scan_calls.load(std::memory_order_relaxed)
		   << " cbv_checks=" << g_cbv_checks.load(std::memory_order_relaxed)
		   << " shadow_updates=" << g_shadow_updates.load(std::memory_order_relaxed)
		   << " shadow_bytes=" << g_shadow_bytes.load(std::memory_order_relaxed)
		   << " shadow_skips=" << g_shadow_skips.load(std::memory_order_relaxed)
		   << " tracked_shadows=";
		{
			std::lock_guard<std::mutex> lock(g_mutex);
			ss << g_buffer_shadows.size();
		}
		log_line(ss.str(), reshade::log::level::warning);
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
				log_line("GTACameraCapture: skipped at least one CSV row because camera/depth/color were not all queued for the same frame.", reshade::log::level::warning);
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
				   << '\n';
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

	flush_ready_readbacks(frame, false);
	g_frame_index.fetch_add(1, std::memory_order_relaxed);
}

void register_events()
{
	// Allow the background writer to start fresh if the addon is reloaded in-process.
	{
		std::lock_guard<std::mutex> lock(g_writer_mutex);
		g_writer_shutdown = false;
	}

	reshade::register_event<reshade::addon_event::init_command_list>(&on_init_command_list);
	reshade::register_event<reshade::addon_event::destroy_command_list>(&on_destroy_command_list);
	reshade::register_event<reshade::addon_event::init_command_queue>(&on_init_command_queue);
	reshade::register_event<reshade::addon_event::destroy_command_queue>(&on_destroy_command_queue);
	reshade::register_event<reshade::addon_event::begin_render_pass>(&on_begin_render_pass);
	reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(&on_bind_render_targets_and_depth_stencil);
	reshade::register_event<reshade::addon_event::push_descriptors>(&on_push_descriptors);
	reshade::register_event<reshade::addon_event::draw>(&on_draw);
	reshade::register_event<reshade::addon_event::draw_indexed>(&on_draw_indexed);
	reshade::register_event<reshade::addon_event::dispatch>(&on_dispatch);
	reshade::register_event<reshade::addon_event::map_buffer_region>(&on_map_buffer_region);
	reshade::register_event<reshade::addon_event::unmap_buffer_region>(&on_unmap_buffer_region);
	reshade::register_event<reshade::addon_event::update_buffer_region>(&on_update_buffer_region);
	reshade::register_event<reshade::addon_event::update_buffer_region_command>(&on_update_buffer_region_command);
	reshade::register_event<reshade::addon_event::reshade_begin_effects>(&on_reshade_begin_effects);
	reshade::register_event<reshade::addon_event::reshade_finish_effects>(&on_reshade_finish_effects);
	reshade::register_event<reshade::addon_event::reshade_present>(&on_reshade_present);
}

void unregister_events()
{
	reshade::unregister_event<reshade::addon_event::reshade_present>(&on_reshade_present);
	reshade::unregister_event<reshade::addon_event::reshade_finish_effects>(&on_reshade_finish_effects);
	reshade::unregister_event<reshade::addon_event::reshade_begin_effects>(&on_reshade_begin_effects);
	reshade::unregister_event<reshade::addon_event::update_buffer_region_command>(&on_update_buffer_region_command);
	reshade::unregister_event<reshade::addon_event::update_buffer_region>(&on_update_buffer_region);
	reshade::unregister_event<reshade::addon_event::unmap_buffer_region>(&on_unmap_buffer_region);
	reshade::unregister_event<reshade::addon_event::map_buffer_region>(&on_map_buffer_region);
	reshade::unregister_event<reshade::addon_event::dispatch>(&on_dispatch);
	reshade::unregister_event<reshade::addon_event::draw_indexed>(&on_draw_indexed);
	reshade::unregister_event<reshade::addon_event::draw>(&on_draw);
	reshade::unregister_event<reshade::addon_event::push_descriptors>(&on_push_descriptors);
	reshade::unregister_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(&on_bind_render_targets_and_depth_stencil);
	reshade::unregister_event<reshade::addon_event::begin_render_pass>(&on_begin_render_pass);
	reshade::unregister_event<reshade::addon_event::destroy_command_queue>(&on_destroy_command_queue);
	reshade::unregister_event<reshade::addon_event::init_command_queue>(&on_init_command_queue);
	reshade::unregister_event<reshade::addon_event::destroy_command_list>(&on_destroy_command_list);
	reshade::unregister_event<reshade::addon_event::init_command_list>(&on_init_command_list);

	// Stop + join the background writer first (drains its runtime queue). After
	// this, g_writer_shutdown makes the teardown flush below write synchronously
	// inline, so we never spawn a thread from under the DllMain loader lock.
	writer_stop_and_join();

	flush_ready_readbacks(UINT64_MAX, true);

	std::lock_guard<std::mutex> lock(g_mutex);
	destroy_depth_readback(g_pending_depth);
	destroy_color_readback(g_pending_color);
	g_buffer_shadows.clear();
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
		init_paths();
		register_events();
		log_line("GTACameraCapture: loaded.");
		{
			std::ostringstream ss;
			ss << "GTACameraCapture: build " << kBuildTag;
			log_line(ss.str());
		}
		if (g_capture_enabled.load(std::memory_order_relaxed))
			log_line("GTACameraCapture: capture starts enabled.");
		else
			log_line("GTACameraCapture: capture idle. Press F8 once to start.");
		break;
	case DLL_PROCESS_DETACH:
		unregister_events();
		log_line("GTACameraCapture: unloaded.");
		reshade::unregister_addon(hinstDLL);
		break;
	}

	return TRUE;
}
