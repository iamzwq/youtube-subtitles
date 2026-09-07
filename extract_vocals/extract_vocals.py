#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频/音频人声提取脚本 (基于 Demucs)

主要功能：
- 使用 Meta Demucs AI 模型从视频或音频中分离并提取人声（Human Voice / Vocals）
- 自动检测并使用 GPU（CUDA）加速，无 GPU 时自动回退到 CPU
- 支持提取人声/伴奏，输出高清 WAV 或 MP3 格式
- 简单易用：可直接在命令行传入文件路径，或直接运行后拖入文件路径

依赖安装：
    pip install demucs
    （需要系统已安装 FFmpeg）
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def check_ffmpeg() -> str:
    """检查系统是否存在 ffmpeg 可执行文件，并返回路径。"""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    # Windows 常见 WinGet / 安装路径检查
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    possible_paths = []
    if local_appdata:
        possible_paths.append(Path(local_appdata) / "Microsoft" / "WinGet" / "Packages")
    possible_paths.extend([
        Path("C:/Program Files/ffmpeg/bin"),
        Path("C:/ffmpeg/bin"),
    ])

    for base in possible_paths:
        if base.exists():
            for p in base.rglob("ffmpeg.exe"):
                return str(p)

    return ""


def check_demucs() -> bool:
    """检查 demucs 是否已安装。"""
    try:
        import demucs  # noqa: F401
        return True
    except ImportError:
        return False


def install_demucs() -> bool:
    """引导或自动安装 demucs。"""
    print("\n[!] 未检测到 demucs 库。")
    print("正在尝试通过 pip 自动安装 demucs ...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "demucs"], check=True)
        print("[✓] demucs 安装成功！\n")
        return True
    except Exception as e:
        print(f"[×] 自动安装 demucs 失败: {e}")
        print("请手动在终端运行: pip install demucs\n")
        return False


def extract_audio_from_video(video_path: Path, temp_wav_path: Path, ffmpeg_bin: str) -> bool:
    """使用 ffmpeg 从视频/音频文件中提取标准 WAV 音频。"""
    cmd = [
        ffmpeg_bin, "-y",
        "-i", str(video_path),
        "-vn",                   # 禁用视频流
        "-acodec", "pcm_s16le",  # 16-bit PCM WAV
        "-ar", "44100",          # 44.1kHz 采样率
        "-ac", "2",              # 双声道
        str(temp_wav_path)
    ]
    print(f"[*] 正在从视频中提取音频: {video_path.name} ...")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return res.returncode == 0 and temp_wav_path.exists()


def run_demucs_extraction(
    audio_path: Path,
    output_dir: Path,
    model_name: str = "htdemucs",
    device: str = "auto",
    two_stems: str = "vocals",
    audio_format: str = "wav"
) -> Path:
    """运行 demucs 提取人声。"""
    cmd = [
        sys.executable, "-m", "demucs.separate",
        "-n", model_name,
        "-o", str(output_dir),
    ]

    if two_stems:
        cmd.extend(["--two-stems", two_stems])

    if audio_format.lower() in ["mp3", "flac"]:
        cmd.extend([f"--{audio_format.lower()}"])

    if device != "auto":
        cmd.extend(["-d", device])

    cmd.append(str(audio_path))

    print(f"[*] 开始使用 Demucs 模型 ({model_name}) 提取人声，请稍候...")
    print(f"    临时输出目录: {output_dir}")

    start_time = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if proc.stdout:
        for line in proc.stdout:
            line_str = line.strip()
            if line_str:
                print(f"    {line_str}")

    proc.wait()
    elapsed = time.time() - start_time

    if proc.returncode != 0:
        raise RuntimeError(f"Demucs 处理过程出错 (退出码 {proc.returncode})")

    print(f"[✓] 人声提取计算完成！耗时: {elapsed:.1f} 秒")

    # Demucs 默认输出文件结构: <output_dir>/<model_name>/<track_name>/vocals.wav
    track_name = audio_path.stem
    separated_dir = output_dir / model_name / track_name
    vocal_file = separated_dir / f"vocals.{audio_format}"

    if not vocal_file.exists():
        candidates = list(separated_dir.glob("vocals.*"))
        if candidates:
            vocal_file = candidates[0]

    return vocal_file


def main():
    parser = argparse.ArgumentParser(
        description="使用 Demucs AI 从视频或音频中一键提取人声 (Vocals)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="输入视频或音频文件路径（如 D:\\video.mp4）。留空则交互式输入。"
    )
    parser.add_argument(
        "-o", "--output",
        help="输出文件夹或输出文件路径（默认保存在原视频所在目录）"
    )
    parser.add_argument(
        "-m", "--model",
        default="htdemucs",
        choices=["htdemucs", "htdemucs_ft", "mdx_extra", "hdemucs_mmi"],
        help="Demucs 分离模型 (默认: htdemucs，速度与效果兼优)"
    )
    parser.add_argument(
        "-f", "--format",
        default="wav",
        choices=["wav", "mp3", "flac"],
        help="输出音频格式 (默认: wav)"
    )
    parser.add_argument(
        "-d", "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="计算设备 (默认: auto 自动检测 GPU)"
    )
    parser.add_argument(
        "--keep-accompaniment",
        action="store_true",
        help="除了人声外，是否同时保留去除人声后的伴奏音频 (no_vocals)"
    )

    args = parser.parse_args()

    # 1. 获取输入文件路径
    input_file_str = args.input
    if not input_file_str:
        print("=" * 60)
        print("            Demucs 视频/音频人声提取工具")
        print("=" * 60)
        input_file_str = input("请输入视频或音频文件路径 (可直接将文件拖入此窗口): ").strip()
        input_file_str = input_file_str.strip('"' "'")

    if not input_file_str:
        print("[×] 未提供有效的输入文件路径，程序退出。")
        sys.exit(1)

    input_path = Path(input_file_str).resolve()
    if not input_path.exists() or not input_path.is_file():
        print(f"[×] 文件不存在: {input_path}")
        sys.exit(1)

    # 2. 检查依赖
    ffmpeg_bin = check_ffmpeg()
    if not ffmpeg_bin:
        print("[×] 未在系统中找到 FFmpeg！")
        print("    提取视频音频需要 FFmpeg，请确保已安装 FFmpeg 并配置环境变量。")
        sys.exit(1)

    if not check_demucs():
        if not install_demucs():
            sys.exit(1)

    # 3. 确定输出位置
    if args.output:
        out_target = Path(args.output).resolve()
        if out_target.suffix.lower() in [".wav", ".mp3", ".flac"]:
            output_dir = out_target.parent
            custom_final_name = out_target
        else:
            output_dir = out_target
            custom_final_name = None
    else:
        output_dir = input_path.parent
        custom_final_name = None

    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. 创建临时目录提取 PCM 音频
    temp_dir = output_dir / f"_demucs_temp_{int(time.time())}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_wav = temp_dir / "input_audio.wav"

    try:
        if not extract_audio_from_video(input_path, temp_wav, ffmpeg_bin):
            print(f"[×] 从文件 {input_path.name} 中提取音频失败。")
            sys.exit(1)

        extracted_vocal_file = run_demucs_extraction(
            audio_path=temp_wav,
            output_dir=temp_dir,
            model_name=args.model,
            device=args.device,
            two_stems="vocals",
            audio_format=args.format
        )

        # 5. 重命名并移动最终人声产物到目标位置
        if custom_final_name:
            final_vocal_path = custom_final_name
        else:
            final_vocal_path = output_dir / f"{input_path.stem}_vocals.{args.format}"

        if extracted_vocal_file and extracted_vocal_file.exists():
            shutil.copy2(extracted_vocal_file, final_vocal_path)
            print("=" * 60)
            print("  人声提取完毕！")
            print(f"  人声文件: {final_vocal_path}")
            print(f"  文件大小: {final_vocal_path.stat().st_size / (1024*1024):.2f} MB")

            if args.keep_accompaniment:
                track_dir = temp_dir / args.model / "input_audio"
                no_vocal_file = track_dir / f"no_vocals.{args.format}"
                if no_vocal_file.exists():
                    final_novocal_path = output_dir / f"{input_path.stem}_accompaniment.{args.format}"
                    shutil.copy2(no_vocal_file, final_novocal_path)
                    print(f"  伴奏文件: {final_novocal_path}")
            print("=" * 60)
        else:
            print("[×] 未能找到提取出的人声音频文件。")

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
