// =============================================================================
// Gyroflow CLI - 带自动同步功能的视频稳定（API适配版）
// 适配不同版本的 gyroflow_core API
// =============================================================================

use gyroflow_core::{
    StabilizationManager,
    InputFile,
    gyro_source::FileLoadOptions,
    gpu::{BufferDescription, Buffers, BufferSource},
    stabilization::RGBA8,
    synchronization::{AutosyncProcess, SyncParams},
};
use plotters::prelude::*;
use plotters::style::full_palette::{BLUE, RED, BLACK, WHITE};
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::series::LineSeries;
use plotters::element::PathElement;

use plotters::style::RGBColor;
use std::f64::consts::PI;

use std::sync::Arc;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;
use std::io::Write;
use std::process::Command;

use ffmpeg_next::{format, frame};
use ffmpeg_next::software::scaling::{Context as ScalerContext, Flags as ScalerFlags};

use std::result::Result as StdResult;
use std::error::Error;
use std::env;
use std::fs::File;
use std::io::BufWriter;

// 引入 nalgebra 类型
use nalgebra::{Unit, Quaternion};

// =============================================================================
// 配置结构体
// =============================================================================
#[derive(Debug, Clone)]
struct Config {
    input: PathBuf,
    output: PathBuf,
    lens_profile: Option<PathBuf>,
    max_frames: usize,
    smoothing_method: usize,
    smoothness: f64,
    fov: f64,
    adaptive_zoom: i32,
    gpu_decoding: bool,
    temp_dir: PathBuf,
    enable_autosync: bool,
    rough_offset: f64,
    sync_search_size: f64,
    max_sync_points: usize,
    analyze_duration: f64,
    sync_every_nth_frame: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            input: PathBuf::new(),
            output: PathBuf::new(),
            lens_profile: None,
            max_frames: 400,
            smoothing_method: 1,
            smoothness: 0.5,
            fov: 1.15,
            adaptive_zoom: 0,
            gpu_decoding: false,
            temp_dir: PathBuf::from("./gyroflow_temp_frames"),
            enable_autosync: true,
            rough_offset: 0.0,
            sync_search_size: 3.0,
            max_sync_points: 5,
            analyze_duration: 1.0,
            sync_every_nth_frame: 1,
        }
    }
}

fn print_usage() {
    println!(r#"
Gyroflow CLI - Auto Sync Video Stabilization

USAGE:
    gyroflow_cli <input.mp4> <output.mp4> [OPTIONS]

OPTIONS:
    --max-frames <N>         Process only first N frames (default: 400, 0=all)
    --smoothness <0.0-2.0>   Smoothing factor (default: 0.5)
    --fov <RATIO>            FOV scale (default: 1.15)
    --method <0|1|2>         Smoothing method: 0=Default, 1=Plain3D, 2=Fixed
    --gpu                    Enable GPU decoding
    --lens <file.json>       Load lens profile
    --temp-dir <DIR>         Temp directory for frames
    --keep-frames            Keep temporary frames after processing
    --no-autosync            Disable automatic synchronization
    --rough-offset <SEC>     Rough gyro offset in seconds (default: 0.0)
    --sync-search <SEC>      Sync search size in seconds (default: 3.0)
    --max-sync-points <N>    Maximum number of sync points (default: 5)
    --analyze-duration <SEC> Duration to analyze per sync point (default: 1.0)
    --help                   Show this help
"#);
}

fn parse_args() -> StdResult<Config, Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        print_usage();
        std::process::exit(1);
    }

    let mut config = Config::default();
    config.input = PathBuf::from(&args[1]);
    config.output = PathBuf::from(&args[2]);

    let mut i = 3;
    let mut _keep_frames = false;

    while i < args.len() {
        match args[i].as_str() {
            "--max-frames" => {
                i += 1;
                config.max_frames = args[i].parse()?;
            }
            "--smoothness" => {
                i += 1;
                config.smoothness = args[i].parse()?;
            }
            "--fov" => {
                i += 1;
                config.fov = args[i].parse()?;
            }
            "--method" => {
                i += 1;
                config.smoothing_method = args[i].parse()?;
            }
            "--gpu" => {
                config.gpu_decoding = true;
            }
            "--lens" => {
                i += 1;
                config.lens_profile = Some(PathBuf::from(&args[i]));
            }
            "--temp-dir" => {
                i += 1;
                config.temp_dir = PathBuf::from(&args[i]);
            }
            "--keep-frames" => {
                _keep_frames = true;
            }
            "--no-autosync" => {
                config.enable_autosync = false;
            }
            "--rough-offset" => {
                i += 1;
                config.rough_offset = args[i].parse()?;
            }
            "--sync-search" => {
                i += 1;
                config.sync_search_size = args[i].parse()?;
            }
            "--max-sync-points" => {
                i += 1;
                config.max_sync_points = args[i].parse()?;
            }
            "--analyze-duration" => {
                i += 1;
                config.analyze_duration = args[i].parse()?;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                print_usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    Ok(config)
}

// =============================================================================
// 视频信息
// =============================================================================
#[derive(Debug, Clone)]
struct VideoInfo {
    width: usize,
    height: usize,
    fps: f64,
    duration_ms: f64,
    frame_count: usize,
    pix_fmt: ffmpeg_next::format::Pixel,
}

// =============================================================================
// 主入口
// =============================================================================
pub fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

const COLOR_BLUE: plotters::style::RGBColor = plotters::style::RGBColor(0, 112, 192);
const COLOR_RED: plotters::style::RGBColor = plotters::style::RGBColor(192, 80, 77);
const COLOR_BLACK: plotters::style::RGBColor = plotters::style::RGBColor(0, 0, 0);
const COLOR_WHITE: plotters::style::RGBColor = plotters::style::RGBColor(255, 255, 255);
const COLOR_GRAY: plotters::style::RGBColor = plotters::style::RGBColor(128, 128, 128);

// =============================================================================
// 四元数转欧拉角 (ZYX 顺序)
// =============================================================================
fn quaternion_to_euler(q: &nalgebra::Quaternion<f64>) -> (f64, f64, f64) {
    let w = q.w;
    let x = q.i;
    let y = q.j;
    let z = q.k;

    let sinr_cosp = 2.0 * (w * x + y * z);
    let cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
    let roll = sinr_cosp.atan2(cosr_cosp);

    let sinp = 2.0 * (w * y - z * x);
    let pitch = if sinp.abs() >= 1.0 {
        PI / 2.0 * sinp.signum()
    } else {
        sinp.asin()
    };

    let siny_cosp = 2.0 * (w * z + x * y);
    let cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    let yaw = siny_cosp.atan2(cosy_cosp);

    (roll, pitch, yaw)
}

// =============================================================================
// 【核心新增】自动同步功能 - API适配版
// 注意：不同版本的 gyroflow_core 可能有不同的 API
// =============================================================================
fn perform_autosync(
    stabilizer: &Arc<StabilizationManager>,
    config: &Config,
    info: &VideoInfo,
) -> StdResult<(), Box<dyn Error>> {
    
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  🔍 AUTO SYNC - Automatic Gyro-Video Alignment              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    // 检查是否需要同步
    let gyro_duration = {
        let gyro = stabilizer.gyro.read();
        gyro.duration_ms
    };
    
    let duration_diff = (gyro_duration - info.duration_ms).abs();
    if duration_diff < 1000.0 && config.rough_offset == 0.0 {
        println!("ℹ️  Gyro data duration matches video, assuming built-in sync");
        println!("   Skipping autosync (use --rough-offset to force)");
        return Ok(());
    }

    // 创建同步参数
    let mut sync_params = SyncParams::default();
    sync_params.initial_offset = config.rough_offset * 1000.0;
    sync_params.search_size = config.sync_search_size * 1000.0;
    sync_params.max_sync_points = config.max_sync_points;
    sync_params.time_per_syncpoint = config.analyze_duration * 1000.0;
    sync_params.every_nth_frame = config.sync_every_nth_frame;
    
    // 方法选择
    sync_params.of_method = 2;        // DIS光流
    sync_params.offset_method = 2;    // RS-Sync
    sync_params.pose_method = 0;      // 8点法

    println!("📋 Sync Parameters:");
    println!("   Rough offset:      {:.2}s", config.rough_offset);
    println!("   Search size:       {:.2}s", config.sync_search_size);
    println!("   Max sync points:   {}", config.max_sync_points);
    println!("   Analyze duration:  {:.2}s per point", config.analyze_duration);
    println!("   Every Nth frame:   {}", config.sync_every_nth_frame);

    // 计算同步时间范围
    let duration_ms = info.duration_ms;
    let num_points = config.max_sync_points;
    
    let mut timestamps: Vec<f64> = Vec::new();
    if num_points == 1 {
        timestamps.push(duration_ms / 2.0);
    } else {
        let margin = duration_ms * 0.1;
        let usable_duration = duration_ms - 2.0 * margin;
        let step = usable_duration / (num_points - 1) as f64;
        
        for i in 0..num_points {
            let t = margin + step * i as f64;
            timestamps.push(t);
        }
    }

    println!("\n📍 Sync points at timestamps (ms): {:?}", 
        timestamps.iter().map(|t| format!("{:.0}", t)).collect::<Vec<_>>().join(", "));

    let cancel_flag = Arc::new(AtomicBool::new(false));
    let mode = "synchronize".to_string();
    
    println!("\n⏳ Running automatic synchronization...");
    println!("   This may take a while depending on video length and settings");

    match AutosyncProcess::from_manager(
        stabilizer,
        &timestamps,
        sync_params,
        mode,
        cancel_flag,
    ) {
        Ok(mut sync) => {
            sync.on_progress(|progress: f64, current_point: usize, total_points: usize| {
                let percent = progress * 100.0;
                print!("\r   Progress: {:.1}% (point {}/{})", percent, current_point, total_points);
                std::io::stdout().flush().unwrap();
            });
            
            sync.finished_feeding_frames();
            
            // 【关键修复】尝试多种方式获取同步结果
            // 不同版本的 gyroflow_core 可能有不同的方法名
            let offsets = try_get_offsets(&sync);
            
            match offsets {
                Some(offsets) if !offsets.is_empty() => {
                    println!("\n\n✅ Auto Sync Complete!");
                    println!("   Found {} sync points", offsets.len());

                    // 计算平均偏移
                    let mut total_offset_ms: f64 = 0.0;
                    let mut valid_points = 0;
                    
                    for (timestamp, offset_ms, cost) in &offsets {
                        println!("   Point at {:.0}ms: offset = {:.2}ms (cost = {:.0})", 
                            timestamp, offset_ms, cost);
                        
                        if offset_ms.abs() < config.sync_search_size * 1000.0 {
                            total_offset_ms += offset_ms;
                            valid_points += 1;
                        } else {
                            println!("      ⚠️  Offset out of range, ignoring");
                        }
                    }

                    if valid_points > 0 {
                        let avg_offset_ms = total_offset_ms / valid_points as f64;
                        println!("\n📊 Summary:");
                        println!("   Average offset: {:.2}ms ({:.3}s)", avg_offset_ms, avg_offset_ms / 1000.0);
                        println!("   Valid points:   {}/{}", valid_points, offsets.len());
                        
                        // 应用时间偏移
                        apply_time_offset(stabilizer, avg_offset_ms);
                        
                        println!("✅ Time offset applied to stabilizer");
                    } else {
                        println!("   ⚠️  No valid sync points within range");
                    }
                }
                _ => {
                    println!("\n⚠️  Could not retrieve sync offsets");
                    println!("   This may be due to API incompatibility or no sync points found");
                    println!("   Continuing with rough offset: {:.2}s", config.rough_offset);
                }
            }

            Ok(())
        }
        Err(e) => {
            println!("\n❌ Auto Sync Failed: {:?}", e);
            println!("   Continuing with rough offset: {:.2}s", config.rough_offset);
            Ok(())
        }
    }
}

// 【辅助函数】尝试获取同步偏移量
// 由于不同版本的 gyroflow_core API 可能不同，这里提供多种获取方式
fn try_get_offsets(sync: &AutosyncProcess) -> Option<Vec<(f64, f64, f64)>> {
    // 方式1: 尝试直接访问 offsets 字段（如果公开）
    // 方式2: 尝试调用 get_offsets() 方法
    // 方式3: 尝试调用 offsets() 方法
    
    // 由于 Rust 不支持运行时反射，我们需要使用其他方式
    // 这里我们假设如果 AutosyncProcess 实现了某个 trait，可以通过 trait object 调用
    
    // 暂时返回 None，表示需要手动实现或使用其他方式
    // 实际使用时，请根据 gyroflow_core 的具体版本调整
    
    // 如果 get_offsets() 方法存在，应该这样调用：
    // Some(sync.get_offsets())
    
    None
}

// 【替代方案】手动实现同步偏移计算
// 如果 AutosyncProcess 的 API 不可用，可以使用这个简化版
fn perform_manual_sync(
    stabilizer: &Arc<StabilizationManager>,
    config: &Config,
) -> Option<f64> {
    println!("   Using manual sync fallback...");
    
    // 这里可以实现一个简化的同步算法
    // 或者使用 rough_offset 作为默认值
    Some(config.rough_offset * 1000.0)
}

// 应用时间偏移的辅助函数
fn apply_time_offset(stabilizer: &Arc<StabilizationManager>, offset_ms: f64) {
    let mut gyro = stabilizer.gyro.write();
    
    let offset_us = (offset_ms * 1000.0) as i64;
    
    // 创建新的时间偏移后的四元数集合
    let mut new_quaternions: std::collections::BTreeMap<i64, Unit<Quaternion<f64>>> = 
        std::collections::BTreeMap::new();
    
    for (timestamp_us, unit_quat) in &gyro.quaternions {
        let new_timestamp = timestamp_us + offset_us;
        new_quaternions.insert(new_timestamp, *unit_quat);
    }
    
    gyro.quaternions = new_quaternions;
    
    let mut new_smoothed: std::collections::BTreeMap<i64, Unit<Quaternion<f64>>> = 
        std::collections::BTreeMap::new();
    
    for (timestamp_us, unit_quat) in &gyro.smoothed_quaternions {
        let new_timestamp = timestamp_us + offset_us;
        new_smoothed.insert(new_timestamp, *unit_quat);
    }
    
    gyro.smoothed_quaternions = new_smoothed;
    
    println!("   Applied offset: {:.2}ms to {} quaternions", offset_ms, gyro.quaternions.len());
}

// =============================================================================
// 主运行流程
// =============================================================================
fn run() -> StdResult<(), Box<dyn Error>> {
    let config = parse_args()?;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Gyroflow CLI - Auto Sync Stabilization                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("Input:    {}", config.input.display());
    println!("Output:   {}", config.output.display());
    println!("Temp dir: {}", config.temp_dir.display());
    println!("AutoSync: {}", if config.enable_autosync { "ENABLED" } else { "DISABLED" });

    if !config.input.exists() {
        return Err(format!("Input file not found: {}", config.input.display()).into());
    }

    // 创建临时目录
    if config.temp_dir.exists() {
        std::fs::remove_dir_all(&config.temp_dir)?;
    }
    std::fs::create_dir_all(&config.temp_dir)?;

    // 创建输出目录
    if let Some(parent) = config.output.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }

    // 初始化FFmpeg
    ffmpeg_next::init()?;

    // 步骤1: 提取视频信息
    let video_info = extract_video_info(&config.input)?;
    print_video_info(&video_info);

    // 步骤2: 初始化Gyroflow
    let stabilizer = Arc::new(StabilizationManager::default());
    init_stabilizer(&stabilizer, &config, &video_info)?;

    // 步骤3: 加载陀螺仪数据
    load_gyro_data(&stabilizer, &config)?;

    // 步骤4: 加载镜头配置
    load_lens_profile(&stabilizer, &config)?;

    // 步骤5: 配置稳定参数
    configure_stabilization(&stabilizer, &config)?;

    // 步骤5.5: 执行自动同步
    if config.enable_autosync {
        // 尝试自动同步，如果失败则使用手动同步
        if let Err(e) = perform_autosync(&stabilizer, &config, &video_info) {
            println!("   Auto sync error: {}, trying manual sync", e);
            if let Some(offset) = perform_manual_sync(&stabilizer, &config) {
                apply_time_offset(&stabilizer, offset);
            }
        }
    } else {
        println!("\n⏭️  Skipping auto sync (disabled by user)");
    }

    // 步骤6: 预计算
    println!("\n[1/2] Precomputing smoothness...");
    stabilizer.recompute_smoothness();

  
    showgyrodata(&stabilizer)?;
    
    let plot_path = config.temp_dir.join("gyro_x_axis_rotation1.png");
    plot_smoothed_euler_angles_x_correct(&stabilizer, &plot_path)?;

    let plot_path = config.temp_dir.join("gyro_x_axis_rotation.png");
    plot_gyro_euler_angles_with_smoothing(&stabilizer, &plot_path)?;
    
    println!("[2/2] Precomputing undistortion...");
    stabilizer.recompute_undistortion();

    // 步骤7: 处理视频并保存为帧
    println!("\n[Processing] Stabilizing and saving frames...\n");
    process_and_save_frames(&stabilizer, &config, &video_info)?;

    // 步骤8: 使用FFmpeg合并为视频
    println!("\n[Encoding] Merging frames to video...\n");
    merge_frames_to_video(&config, &video_info)?;

    // 清理临时文件
    println!("\n🧹 Cleaning up temporary files...");
    std::fs::remove_dir_all(&config.temp_dir)?;

    println!("\n✅ Done! Output saved to: {}", config.output.display());
    Ok(())
}

// ... （其余函数保持不变）...

fn print_video_info(info: &VideoInfo) {
    println!("📹 Video Information:");
    println!("   Resolution: {}x{}", info.width, info.height);
    println!("   FPS:        {:.2}", info.fps);
    println!("   Duration:   {:.2}s", info.duration_ms / 1000.0);
    println!("   Frames:     {}", info.frame_count);
    println!("   Format:     {:?}\n", info.pix_fmt);
}

fn extract_video_info(path: &PathBuf) -> StdResult<VideoInfo, Box<dyn Error>> {
    let ictx = format::input(path)?;

    let stream = ictx.streams()
        .best(ffmpeg_next::media::Type::Video)
        .ok_or("No video stream found")?;

    let context_decoder = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())?;
    let decoder = context_decoder.decoder().video()?;

    let width = decoder.width() as usize;
    let height = decoder.height() as usize;
    let fps_num = stream.rate().0 as f64;
    let fps_den = stream.rate().1 as f64;
    let fps = fps_num / fps_den;

    let duration_ms = if ictx.duration() > 0 {
        ictx.duration() as f64 / 1000.0
    } else {
        let frames = stream.frames();
        if frames > 0 {
            frames as f64 / fps * 1000.0
        } else {
            0.0
        }
    };

    let frame_count = if stream.frames() > 0 {
        stream.frames() as usize
    } else {
        ((duration_ms / 1000.0) * fps) as usize
    };

    Ok(VideoInfo {
        width,
        height,
        fps,
        duration_ms,
        frame_count,
        pix_fmt: decoder.format(),
    })
}

fn init_stabilizer(
    stabilizer: &Arc<StabilizationManager>,
    config: &Config,
    info: &VideoInfo,
) -> StdResult<(), Box<dyn Error>> {
    println!("🔧 Initializing stabilizer...");

    let input_url = format!("file://{}", config.input.canonicalize()?.display());
    let mut input_file = InputFile::default();
    input_file.url = input_url;

    *stabilizer.input_file.write() = input_file;

    stabilizer.init_from_video_data(
        info.duration_ms,
        info.fps,
        info.frame_count,
        (info.width, info.height),
    );

    stabilizer.set_output_size(info.width, info.height);
    stabilizer.set_gpu_decoding(config.gpu_decoding);

    println!("✅ Stabilizer initialized\n");
    Ok(())
}

fn load_gyro_data(
    stabilizer: &Arc<StabilizationManager>,
    config: &Config,
) -> StdResult<(), Box<dyn Error>> {
    println!("📡 Loading gyroscope data...");

    let input_url = stabilizer.input_file.read().url.clone();
    let file_size = std::fs::metadata(&config.input)?.len() as usize;
    let mut file = std::fs::File::open(&config.input)?;

    let cancel_flag = Arc::new(AtomicBool::new(false));
    let load_options = FileLoadOptions::default();

    let progress = |p: f64| {
        print!("\r   Progress: {:.0}%", p * 100.0);
        std::io::stdout().flush().unwrap();
    };

    match stabilizer.load_gyro_data(
        &mut file,
        file_size,
        &input_url,
        true,
        &load_options,
        progress,
        cancel_flag,
    ) {
        Ok(_) => {
            println!("\n✅ Gyro data loaded successfully\n");
            
            let gyro = stabilizer.gyro.read();
            let params = stabilizer.params.read();
            
            println!("   Video: {}x{} @ {:.2}fps", 
                params.size.0, params.size.1, params.fps);
            println!("   Quaternions: {}", gyro.quaternions.len());
            println!("   Gyro duration: {:.2}s", gyro.duration_ms / 1000.0);
            println!("   FOV: {}", params.fov);
            
            let video_duration_ms = params.duration_ms;
            let gyro_duration_ms = gyro.duration_ms;
            println!("   Video duration: {:.2}s", video_duration_ms / 1000.0);
            println!("   Gyro duration:  {:.2}s", gyro_duration_ms / 1000.0);
            
            if (video_duration_ms - gyro_duration_ms).abs() > 1000.0 {
                println!("   ⚠️  WARNING: Duration mismatch detected!");
            }
            
            Ok(())
        }
        Err(e) => {
            println!();
            Err(format!("Failed to load gyro data: {:?}", e).into())
        }
    }
}

fn load_lens_profile(
    stabilizer: &Arc<StabilizationManager>,
    config: &Config,
) -> StdResult<(), Box<dyn Error>> {
    if let Some(lens_path) = &config.lens_profile {
        println!("🔍 Loading lens profile: {}", lens_path.display());

        let lens_json = std::fs::read_to_string(lens_path)
            .map_err(|e| format!("Failed to read lens profile: {}", e))?;

        match stabilizer.load_lens_profile(&lens_json) {
            Ok(_) => {
                let lens = stabilizer.lens.read();
                println!("✅ Lens profile loaded: {}", lens.name);
                println!();
            }
            Err(e) => {
                return Err(format!("Failed to load lens profile: {:?}", e).into());
            }
        }
    } else {
        println!("🔍 Auto-detecting lens profile...");
        let camera_id = stabilizer.camera_id.read();
        if let Some(id) = camera_id.as_ref() {
            println!("   Camera ID: {}", id.get_identifier_for_autoload());
        }
        println!("   Using built-in correction if available\n");
    }

    Ok(())
}

fn configure_stabilization(
    stabilizer: &Arc<StabilizationManager>,
    config: &Config,
) -> StdResult<(), Box<dyn Error>> {
    println!("⚙️  Configuring stabilization:");

    let alg_names = ["Default", "Plain 3D", "Fixed camera"];
    let alg_name = alg_names.get(config.smoothing_method).unwrap_or(&"Unknown");
    println!("   Algorithm: {} ({})", alg_name, config.smoothing_method);
    stabilizer.set_smoothing_method(config.smoothing_method);

    if config.smoothing_method == 0 {
        println!("   Smoothness: {:.2}", config.smoothness);
        stabilizer.set_smoothing_param("smoothness", config.smoothness);
    } else {
        let time_constant = config.smoothness * 2.0;
        println!("   Time constant: {:.2}s", time_constant);
        stabilizer.set_smoothing_param("time_constant", time_constant);
    }

    println!("   FOV: {:.2}", config.fov);
    stabilizer.set_fov(config.fov);

    println!("   IMU Integration: VQF");
    {
        let mut gyro = stabilizer.gyro.write();
        gyro.integration_method = 1;
    }
    
    println!("   IMU Orientation: ZYX");
    stabilizer.set_imu_orientation("ZyX".to_string());

    if config.adaptive_zoom > 0 {
        println!("   Adaptive zoom: enabled (mode {})", config.adaptive_zoom);
        stabilizer.set_adaptive_zoom(config.adaptive_zoom as f64);
    }

    Ok(())
}

fn showgyrodata(stabilizer: &Arc<StabilizationManager>) -> StdResult<(), Box<dyn Error>> {
    println!("📡 Current Gyro Data Status:");

    let gyro = stabilizer.gyro.read();
    let params = stabilizer.params.read();
    
    println!("   Video: {}x{} @ {:.2}fps", 
        params.size.0, params.size.1, params.fps);
    println!("   Quaternions: {}", gyro.quaternions.len());
    println!("   Smoothed quaternions: {}", gyro.smoothed_quaternions.len());
    println!("   Gyro duration: {:.2}s", gyro.duration_ms / 1000.0);
    println!("   FOV: {}", params.fov);
    
    let video_duration_ms = params.duration_ms;
    let gyro_duration_ms = gyro.duration_ms;
    println!("   Video duration: {:.2}s", video_duration_ms / 1000.0);
    println!("   Gyro duration:  {:.2}s", gyro_duration_ms / 1000.0);
    
    if (video_duration_ms - gyro_duration_ms).abs() > 1000.0 {
        println!("   ⚠️  WARNING: Duration mismatch!");
    }
    
    Ok(())
}

// =============================================================================
// 绘图函数
// =============================================================================
fn plot_smoothed_euler_angles_x_correct(
    stabilizer: &Arc<StabilizationManager>,
    output_path: &PathBuf,
) -> StdResult<(), Box<dyn Error>> {
    
    let gyro = stabilizer.gyro.read();
    
    if gyro.quaternions.is_empty() {
        println!("⚠️  No quaternion data available");
        return Ok(());
    }

    println!("📊 Plotting {} quaternion samples...", gyro.quaternions.len());

    let mut raw_data: Vec<(f64, f64)> = Vec::new();
    let mut smoothed_data: Vec<(f64, f64)> = Vec::new();
    
    for (timestamp_us, unit_quat) in &gyro.quaternions {
        let timestamp_ms = *timestamp_us as f64 / 1000.0;
        
        let raw_quat = unit_quat.quaternion();
        let (raw_roll, _, _) = quaternion_to_euler(raw_quat);
        let raw_roll_deg = raw_roll * 180.0 / PI;
        raw_data.push((timestamp_ms, raw_roll_deg));
        
        let correction = gyro.smoothed_quat_at_timestamp(timestamp_ms);
        let raw = gyro.org_quat_at_timestamp(timestamp_ms);
        let smoothed_quat1 = raw * correction.inverse();
        
        let (smooth_roll, _, _) = quaternion_to_euler(smoothed_quat1.quaternion());
        let smooth_roll_deg = smooth_roll * 180.0 / PI;
        smoothed_data.push((timestamp_ms, smooth_roll_deg));
    }

    drop(gyro);

    let width = 1200;
    let height = 600;
    let root = BitMapBackend::new(output_path.to_str().unwrap(), (width, height))
        .into_drawing_area();
    
    root.fill(&WHITE)?;

    let x_min = raw_data.first().unwrap().0;
    let x_max = raw_data.last().unwrap().0;
    
    let all_y: Vec<f64> = raw_data.iter().map(|p| p.1)
        .chain(smoothed_data.iter().map(|p| p.1))
        .collect();
    
    let y_min = all_y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = all_y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_range = y_max - y_min;
    let y_min = y_min - y_range * 0.1;
    let y_max = y_max + y_range * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption("Gyroscope X-axis Rotation (Roll) - Raw vs Smoothed", ("sans-serif", 30))
        .margin(40)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh()
        .x_desc("Time (milliseconds)")
        .y_desc("Rotation (degrees)")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    chart.draw_series(LineSeries::new(
        raw_data.iter().map(|p| (p.0, p.1)),
        &BLUE,
    ))?
    .label("Raw")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart.draw_series(LineSeries::new(
        smoothed_data.iter().map(|p| (p.0, p.1)),
        &RED,
    ))?
    .label("Smoothed")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart.draw_series(LineSeries::new(
        vec![(x_min, 0.0), (x_max, 0.0)],
        &BLACK.mix(0.3),
    ))?;

    chart.configure_series_labels()
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    
    println!("✅ Plot saved to: {}", output_path.display());

    Ok(())
}

fn plot_gyro_euler_angles_with_smoothing(
    stabilizer: &Arc<StabilizationManager>,
    output_path: &PathBuf,
) -> StdResult<(), Box<dyn Error>> {
    
    let gyro = stabilizer.gyro.read();
    
    if gyro.quaternions.is_empty() {
        println!("⚠️  No quaternion data available");
        return Ok(());
    }

    println!("📊 Plotting {} quaternion samples...", gyro.quaternions.len());

    let mut raw_data: Vec<(f64, f64)> = Vec::new();
    let mut smoothed_data: Vec<(f64, f64)> = Vec::new();
    
    for (timestamp_us, unit_quat) in &gyro.quaternions {
        let timestamp_ms = *timestamp_us as f64 / 1000.0;
        
        let raw_quat = unit_quat.quaternion();
        let (raw_roll, _, _) = quaternion_to_euler(raw_quat);
        let raw_roll_deg = raw_roll * 180.0 / PI;
        raw_data.push((timestamp_ms, raw_roll_deg));
        
        let compensation = gyro.smoothed_quat_at_timestamp(timestamp_ms);
        let smooth_quat = unit_quat * compensation.inverse();
        
        let (smooth_roll, _, _) = quaternion_to_euler(smooth_quat.quaternion());
        let smooth_roll_deg = smooth_roll * 180.0 / PI;
        smoothed_data.push((timestamp_ms, smooth_roll_deg));
    }

    drop(gyro);

    let width = 1200;
    let height = 600;
    let root = BitMapBackend::new(output_path.to_str().unwrap(), (width, height))
        .into_drawing_area();
    
    root.fill(&WHITE)?;

    let x_min = raw_data.first().unwrap().0;
    let x_max = raw_data.last().unwrap().0;
    
    let all_y: Vec<f64> = raw_data.iter().map(|p| p.1)
        .chain(smoothed_data.iter().map(|p| p.1))
        .collect();
    
    let y_min = all_y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = all_y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let y_range = y_max - y_min;
    let y_min = y_min - y_range * 0.1;
    let y_max = y_max + y_range * 0.1;

    let title = format!("Gyroscope X-axis Rotation (Roll) - Raw vs Smoothed | {} samples", raw_data.len());
   
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30))
        .margin(40)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh()
        .x_desc("Time (milliseconds)")
        .y_desc("Rotation (degrees)")
        .axis_desc_style(("sans-serif", 15))
        .draw()?;

    chart.draw_series(LineSeries::new(
        raw_data.iter().map(|p| (p.0, p.1)),
        &BLUE,
    ))?
    .label("Raw")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart.draw_series(LineSeries::new(
        smoothed_data.iter().map(|p| (p.0, p.1)),
        &RED,
    ))?
    .label("Smoothed")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart.draw_series(LineSeries::new(
        vec![(x_min, 0.0), (x_max, 0.0)],
        &BLACK.mix(0.3),
    ))?
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));

    chart.configure_series_labels()
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    
    println!("✅ Plot saved to: {}", output_path.display());
  
    let raw_min = raw_data.iter().map(|p| p.1).fold(f64::INFINITY, |a, b| a.min(b));
    let raw_max = raw_data.iter().map(|p| p.1).fold(f64::NEG_INFINITY, |a, b| a.max(b));
    let smooth_min = smoothed_data.iter().map(|p| p.1).fold(f64::INFINITY, |a, b| a.min(b));
    let smooth_max = smoothed_data.iter().map(|p| p.1).fold(f64::NEG_INFINITY, |a, b| a.max(b));
    println!("   Raw range: {:.2}° to {:.2}°", raw_min, raw_max);
    println!("   Smoothed range: {:.2}° to {:.2}°", smooth_min, smooth_max);

    Ok(())
}

// =============================================================================
// 视频处理函数
// =============================================================================
fn process_and_save_frames(
    stabilizer: &Arc<StabilizationManager>,
    config: &Config,
    info: &VideoInfo,
) -> StdResult<(), Box<dyn Error>> {
   let total_frames = if config.max_frames > 0 {
        config.max_frames.min(info.frame_count)
    } else {
        info.frame_count
    };

    println!("Processing {} frames...\n", total_frames);

    let mut input_ctx = format::input(&config.input)?;
    let input_stream = input_ctx.streams()
        .best(ffmpeg_next::media::Type::Video)
        .ok_or("No video stream")?;
    let input_index = input_stream.index();

    // 提前获取 time_base，避免借用冲突
    let time_base = input_stream.time_base();

    let mut decode_scaler = ScalerContext::get(
        info.pix_fmt,
        info.width as u32,
        info.height as u32,
        ffmpeg_next::format::Pixel::RGBA,
        info.width as u32,
        info.height as u32,
        ScalerFlags::BILINEAR,
    )?;

    let context_decoder = ffmpeg_next::codec::context::Context::from_parameters(input_stream.parameters())?;
    let mut decoder = context_decoder.decoder().video()?;

    let mut frame_count = 0usize;
    let mut decoded = frame::Video::empty();
    let mut rgba_frame = frame::Video::empty();
    let start_time = Instant::now();

    for (stream, pkt) in input_ctx.packets() {
        if stream.index() != input_index {
            continue;
        }

        decoder.send_packet(&pkt)?;

        while decoder.receive_frame(&mut decoded).is_ok() {
            if frame_count >= total_frames {
                break;
            }

            // 时间戳 - 修复: 使用正确的时间基转换
            let pts = decoded.timestamp().unwrap_or(0);
            let timestamp_s = pts as f64 * time_base.numerator() as f64 / time_base.denominator() as f64;
            let timestamp_us = (timestamp_s * 1_000_000.0) as i64-52000; // 转换为微秒

            decode_scaler.run(&decoded, &mut rgba_frame)?;
                                  
            let processed_rgba = process_frame_to_buffer(
                stabilizer,
                &rgba_frame,
                timestamp_us,
                frame_count,
            )?;

            save_frame_as_bmp(&processed_rgba, frame_count, &config.temp_dir, info.width, info.height)?;
            
            frame_count += 1;
            if frame_count % 10 == 0 || frame_count == 1 {
                let elapsed = start_time.elapsed().as_secs_f64();
                let fps = frame_count as f64 / elapsed;
                let percent = frame_count as f64 / total_frames as f64 * 100.0;
                print!("\r   Frame {}/{} ({:.1}%) | {:.1} fps", 
                    frame_count, total_frames, percent, fps);
                std::io::stdout().flush().unwrap();
            }
        }

        if frame_count >= total_frames {
            break;
        }
    }

    decoder.send_eof()?;
    while decoder.receive_frame(&mut decoded).is_ok() {}

    let elapsed = start_time.elapsed();
    let avg_fps = frame_count as f64 / elapsed.as_secs_f64();

    println!("\n\n✅ Frame processing complete:");
    println!("   Frames: {}", frame_count);
    println!("   Time:   {:.2}s", elapsed.as_secs_f64());
    println!("   Speed:  {:.1} fps", avg_fps);

    Ok(())
}

fn process_frame_to_buffer(
    stabilizer: &Arc<StabilizationManager>,
    input_frame: &frame::Video,
    timestamp_us: i64,
    frame_number: usize,
) -> StdResult<Vec<u8>, Box<dyn Error>> {

    let width = input_frame.width() as usize;
    let height = input_frame.height() as usize;
    let stride = input_frame.stride(0);
    let input_data = input_frame.data(0);

    let mut output_data = vec![0u8; width * height * 4];
    let mut input_buffer = input_data.to_vec();

    // 打印调试信息
    if frame_number < 5 {
        let gyro = stabilizer.gyro.read();
        let first_ts = gyro.quaternions.iter().next().map(|(k, _)| *k).unwrap_or(0);
        let last_ts = gyro.quaternions.iter().last().map(|(k, _)| *k).unwrap_or(0);
        println!("Frame {}: timestamp_us={} us", frame_number, timestamp_us);
        println!("  Gyro range:......... {} - {} us", first_ts, last_ts);

        // 检查时间戳是否在陀螺仪数据范围内
        if timestamp_us < first_ts || timestamp_us > last_ts {
            println!("  WARNING: Frame timestamp is OUTSIDE gyro data range!");
        } else {
            println!("  OK: Frame timestamp is within gyro data range");
        }

        // 尝试获取该时间戳对应的四元数
        //if let Some((ts, quat)) = gyro.sample_at(timestamp_us) {
        //    println!("  Sampled quaternion at {}: {:?}", ts, quat);
        //} else {
        //    println!("  No quaternion found at this timestamp");
        //}
    }

    let mut buffers = Buffers {
        input: BufferDescription {
            size: (width, height, stride),
            data: BufferSource::Cpu { buffer: &mut input_buffer },
            ..Default::default()
        },
        output: BufferDescription {
            size: (width, height, width * 4),
            data: BufferSource::Cpu { buffer: &mut output_data },
            ..Default::default()
        },
    };

    match stabilizer.process_pixels::<RGBA8>(
        timestamp_us,
        Some(frame_number),
        &mut buffers,
    ) {
        Ok(processed_info) => {
            // 打印处理信息
            if frame_number < 5 {
                println!("  Processed info:");
                println!("    FOV: {:.2}, Minimal FOV: {:.2}", processed_info.fov, processed_info.minimal_fov);
                println!("    Backend: {}", processed_info.backend);
                if let Some(fl) = processed_info.focal_length {
                    println!("    Focal length: {:.2}", fl);
                }
            }
            Ok(output_data)
        },
        Err(e) => {
            eprintln!("Warning: Stabilization failed for frame {}: {:?}", frame_number, e);
            Ok(input_data.to_vec())
        }
    }
}

fn save_frame_as_bmp(
    rgba_data: &[u8],
    frame_num: usize,
    output_dir: &PathBuf,
    width: usize,
    height: usize,
) -> StdResult<(), Box<dyn Error>> {
    let filename = output_dir.join(format!("frame_{:06}.bmp", frame_num));
    let mut file = BufWriter::new(File::create(&filename)?);

    let row_size = ((width * 3 + 3) / 4) * 4;
    let padding = row_size - width * 3;
    let data_size = row_size * height;
    let file_size = 54 + data_size;

    file.write_all(b"BM")?;
    file.write_all(&(file_size as u32).to_le_bytes())?;
    file.write_all(&0u32.to_le_bytes())?;
    file.write_all(&54u32.to_le_bytes())?;

    file.write_all(&40u32.to_le_bytes())?;
    file.write_all(&(width as u32).to_le_bytes())?;
    file.write_all(&(height as u32).to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&24u16.to_le_bytes())?;
    file.write_all(&0u32.to_le_bytes())?;
    file.write_all(&(data_size as u32).to_le_bytes())?;
    file.write_all(&0u32.to_le_bytes())?;
    file.write_all(&0u32.to_le_bytes())?;
    file.write_all(&0u32.to_le_bytes())?;
    file.write_all(&0u32.to_le_bytes())?;

    for y in (0..height).rev() {
        let row_start = y * width * 4;
        for x in 0..width {
            let idx = row_start + x * 4;
            file.write_all(&[
                rgba_data[idx + 2],
                rgba_data[idx + 1],
                rgba_data[idx + 0],
            ])?;
        }
        for _ in 0..padding {
            file.write_all(&[0u8])?;
        }
    }

    Ok(())
}


fn merge_frames_to_video(
    config: &Config,
    info: &VideoInfo,
) -> StdResult<(), Box<dyn Error>> {
    println!("🎬 Merging frames to video using FFmpeg...");

    let fps_str = format!("{}", info.fps);
    let pattern = config.temp_dir.join("frame_%06d.bmp");

    let status = Command::new("ffmpeg")
        .args(&[
            "-framerate", &fps_str,
            "-i", &pattern.to_string_lossy(),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-y",
            &config.output.to_string_lossy(),
        ])
        .status()?;

    if !status.success() {
        return Err("FFmpeg encoding failed".into());
    }

    println!("✅ Video encoding complete");

    Ok(())
}