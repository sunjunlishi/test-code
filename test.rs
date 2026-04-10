// =============================================================================
// Gyroflow CLI - 最简单可靠的方案
// 1. 处理视频帧并保存为PNG图像序列
// 2. 调用FFmpeg命令行合并为MP4
// =============================================================================

use gyroflow_core::{
    StabilizationManager,
    InputFile,
    gyro_source::FileLoadOptions,
    gpu::{BufferDescription, Buffers, BufferSource},
    stabilization::RGBA8,
};

use std::sync::Arc;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;
use std::io::Write;
use std::process::Command;

// 只使用ffmpeg-next解码
use ffmpeg_next::{format, frame};
use ffmpeg_next::software::scaling::{Context as ScalerContext, Flags as ScalerFlags};

use std::result::Result as StdResult;
use std::error::Error;
use std::env;
use std::fs::File;
use std::io::BufWriter;

// 使用image crate保存PNG（需要在Cargo.toml中添加 image = "0.24"）
// 或者使用简单的PPM格式（无需额外依赖）

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
    temp_dir: PathBuf,  // 临时图像目录
}

impl Default for Config {
    fn default() -> Self {
        Self {
            input: PathBuf::new(),
            output: PathBuf::new(),
            lens_profile: None,
            max_frames: 400,
            smoothing_method: 0,
            smoothness: 0.5,
            fov: 1.0,
            adaptive_zoom: 0,
            gpu_decoding: false,
            temp_dir: PathBuf::from("./gyroflow_temp_frames"),
        }
    }
}

fn print_usage() {
    println!(r#"
Gyroflow CLI - Simple & Reliable Video Stabilization

USAGE:
    gyroflow_cli <input.mp4> <output.mp4> [OPTIONS]

OPTIONS:
    --max-frames <N>         Process only first N frames (default: 400, 0=all)
    --smoothness <0.0-2.0>   Smoothing factor (default: 0.5)
    --fov <RATIO>            FOV scale (default: 1.0)
    --method <0|1|2>         Smoothing method: 0=Default, 1=Plain3D, 2=Fixed
    --gpu                    Enable GPU decoding
    --lens <file.json>       Load lens profile
    --temp-dir <DIR>         Temp directory for frames (default: ./gyroflow_temp_frames)
    --keep-frames            Keep temporary frames after processing
    --help                   Show this help

EXAMPLES:
    gyroflow_cli gopro.mp4 out.mp4
    gyroflow_cli gopro.mp4 out.mp4 --max-frames 100 --smoothness 0.8
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
    let mut keep_frames = false;

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
                keep_frames = true;
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

fn run() -> StdResult<(), Box<dyn Error>> {
    let config = parse_args()?;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Gyroflow CLI - Simple Frame-based Stabilization             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("Input:    {}", config.input.display());
    println!("Output:   {}", config.output.display());
    println!("Temp dir: {}\n", config.temp_dir.display());

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

    // 步骤6: 预计算
    println!("\n[1/2] Precomputing smoothness...");
    stabilizer.recompute_smoothness();
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

fn print_video_info(info: &VideoInfo) {
    println!("📹 Video Information:");
    println!("   Resolution: {}x{}", info.width, info.height);
    println!("   FPS:        {:.2}", info.fps);
    println!("   Duration:   {:.2}s", info.duration_ms / 1000.0);
    println!("   Frames:     {}", info.frame_count);
    println!("   Format:     {:?}\n", info.pix_fmt);
}

// =============================================================================
// 提取视频信息
// =============================================================================
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

// =============================================================================
// 初始化稳定器
// =============================================================================
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

// =============================================================================
// 加载陀螺仪数据
// =============================================================================
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
            Ok(())
        }
        Err(e) => {
            println!();
            Err(format!("Failed to load gyro data: {:?}", e).into())
        }
    }
}

// =============================================================================
// 加载镜头配置
// =============================================================================
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

// =============================================================================
// 配置稳定参数
// =============================================================================
fn configure_stabilization(
    stabilizer: &Arc<StabilizationManager>,
    config: &Config,
) -> StdResult<(), Box<dyn Error>> {
    println!("⚙️  Configuring stabilization:");

    let alg_names = ["Default", "Plain 3D", "Fixed camera"];
    let alg_name = alg_names.get(config.smoothing_method).unwrap_or(&"Unknown");
    println!("   Algorithm: {} ({})", alg_name, config.smoothing_method);
    stabilizer.set_smoothing_method(config.smoothing_method);

    let time_constant = config.smoothness * 2.0;
    println!("   Smoothness: {:.2}", config.smoothness);
    stabilizer.set_smoothing_param("time_constant", time_constant);

    println!("   FOV: {:.2}", config.fov);
    stabilizer.set_fov(config.fov);

    if config.adaptive_zoom > 0 {
        println!("   Adaptive zoom: enabled (mode {})", config.adaptive_zoom);
        stabilizer.set_adaptive_zoom(config.adaptive_zoom as f64);
    }

    Ok(())
}

// =============================================================================
// 处理视频并保存帧
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

    // 打开输入
    let mut input_ctx = format::input(&config.input)?;
    let input_stream = input_ctx.streams()
        .best(ffmpeg_next::media::Type::Video)
        .ok_or("No video stream")?;
    let input_index = input_stream.index();

    // 创建转换器（解码为RGBA）
    let mut decode_scaler = ScalerContext::get(
        info.pix_fmt,
        info.width as u32,
        info.height as u32,
        ffmpeg_next::format::Pixel::RGBA,
        info.width as u32,
        info.height as u32,
        ScalerFlags::BILINEAR,
    )?;

    // 初始化解码器 - 修复：直接使用 decoder，不调用 open()
    let context_decoder = ffmpeg_next::codec::context::Context::from_parameters(input_stream.parameters())?;
    let mut decoder = context_decoder.decoder().video()?;

    // 处理变量
    let mut frame_count = 0usize;
    let mut decoded = frame::Video::empty();  // 确保声明 decoded
    let mut rgba_frame = frame::Video::empty();
    let start_time = Instant::now();

    // 主循环
    for (stream, pkt) in input_ctx.packets() {  // 注意这里用的是 pkt，不是 packet
        if stream.index() != input_index {
            continue;
        }

        decoder.send_packet(&pkt)?;  // 使用 pkt，不是 packet

        while decoder.receive_frame(&mut decoded).is_ok() {
            if frame_count >= total_frames {
                break;
            }

            // 时间戳
            let pts = decoded.timestamp().unwrap_or(0);
            let timestamp_ms = pts as f64 / 1000.0;
            let timestamp_us = (timestamp_ms * 1000.0) as i64;

            // 解码 -> RGBA
            decode_scaler.run(&decoded, &mut rgba_frame)?;

            // Gyroflow处理
            let processed_rgba = process_frame_to_buffer(
                stabilizer,
                &rgba_frame,
                timestamp_us,
                frame_count,
            )?;

            // 保存为PPM
            save_frame_as_ppm(&processed_rgba, frame_count, &config.temp_dir, info.width, info.height)?;

            // 进度
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

// =============================================================================
// 处理单帧到缓冲区
// =============================================================================
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
        Ok(_) => Ok(output_data),
        Err(e) => {
            eprintln!("Warning: Stabilization failed for frame {}: {:?}", frame_number, e);
            Ok(input_data.to_vec())
        }
    }
}

// =============================================================================
// 保存帧为PPM格式（简单，无需额外依赖）
// =============================================================================
fn save_frame_as_ppm(
    rgba_data: &[u8],
    frame_num: usize,
    output_dir: &PathBuf,
    width: usize,
    height: usize,
) -> StdResult<(), Box<dyn Error>> {
    let filename = output_dir.join(format!("frame_{:06}.ppm", frame_num));
    let file = File::create(&filename)?;
    let mut writer = BufWriter::new(file);

    // PPM头
    writeln!(writer, "P6")?;
    writeln!(writer, "{} {}", width, height)?;
    writeln!(writer, "255")?;

    // 写入RGB数据（跳过Alpha）
    for pixel in rgba_data.chunks(4) {
        writer.write_all(&[pixel[0], pixel[1], pixel[2]])?;
    }

    Ok(())
}

// =============================================================================
// 使用FFmpeg命令行合并帧为视频
// =============================================================================
fn merge_frames_to_video(
    config: &Config,
    info: &VideoInfo,
) -> StdResult<(), Box<dyn Error>> {
    println!("🎬 Merging frames to video using FFmpeg...");

    let fps_str = format!("{}", info.fps);
    let pattern = config.temp_dir.join("frame_%06d.ppm");

    // FFmpeg命令 [^68^]
    let status = Command::new("ffmpeg")
        .args(&[
            "-framerate", &fps_str,
            "-i", &pattern.to_string_lossy(),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-y", // 覆盖输出
            &config.output.to_string_lossy(),
        ])
        .status()?;

    if !status.success() {
        return Err("FFmpeg encoding failed".into());
    }

    println!("✅ Video encoding complete");

    Ok(())
}
