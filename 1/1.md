#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube 视频自动下载 + 词级字幕翻译 + 纯中文 SRT 字幕生成 + 中文配音

功能：
1. 从视频元数据生成字幕语言候选列表（由 yt-dlp 自动匹配实际存在的轨道）
2. 下载视频 + JSON3 词级字幕（输出目录为视频ID）
3. 解析字幕 → 按标点分句 → 记录每句首尾词时间
4. 调用 LLM 逐句翻译（标题 + 前后文滑动窗口 + 统一术语表 + 按时长换算的字数预算）
5. 时间对齐 → 生成纯中文 SRT 字幕
6. 中文配音：使用 edge-tts 或 MiMo 语音克隆 (mimo-v2.5-tts-voiceclone)；按原句起点硬锚定排布字幕与配音的时间轴
7. 字幕烧录 + 配音一步合成最终视频（--no-tts 可跳过配音）
8. 翻译结果本地缓存（逐批保存），中断后重跑自动断点续传
"""

import argparse
import asyncio
import base64
import hashlib
import json
import mimetypes
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import json_repair
import miniaudio
import numpy as np
from openai import OpenAI


# ==================== 配置加载 ====================

DEFAULT_CONFIG = {
    "global": {
        "max_retries": 3,             # 网络 API（LLM/TTS）调用失败时的最大重试次数
        "sample_rate": 48000,         # 配音音轨拼接采样率（Hz），影响音质与解码/混音精度
        "cookies_from_browser": "",   # yt-dlp 携带浏览器登录 Cookie，缓解 429 限流，如 "chrome"/"edge"/"firefox"，留空则不使用
                                       # 注意：Chrome/Edge 等 Chromium 系浏览器运行时会锁定 cookie 数据库，
                                       # 使用该方式前必须完全退出浏览器进程，否则报错 "Could not copy ... cookie database"；
                                       # 如不方便每次关闭浏览器，改用 cookies_file 更省心
        "cookies_file": "",            # 直接指定 Netscape 格式的 cookies.txt 文件路径（如用浏览器插件导出），
                                       # 不依赖浏览器进程是否运行；同时配置时优先于 cookies_from_browser
        "ytdlp_args": []               # 追加给 yt-dlp 的额外命令行参数。YouTube 现在需要 JS 运行时才能解
                                       # n challenge，yt-dlp 默认只启用 Deno；若未装 Deno 但有 Node，
                                       # 填 ["--js-runtimes", "node"] 即可。否则报错
                                       # "n challenge solving failed" / "The page needs to be reloaded"
    },
    "llm": {
        "base_url": "https://token-plan-cn.xiaomimimo.com/v1",  # LLM API 的 base_url（会自动补全 /v1 后缀）
        "api_key": "",                     # LLM API Key，必填，否则脚本启动时会报错退出
        "model": "mimo-v2.5-pro",          # 使用的模型名称
        "supports_system_role": False,     # 模型是否支持独立的 system 角色消息（不支持则合并进 user 消息）
                                           # 只有 xiaomimimo 需要设置为 False，其他模型不需要设置，保持默认即可
        "thinking": {
            "type": "disabled"             # 是否开启模型的思考/推理模式（部分模型支持，disabled 为关闭）
        },
        "batch_size": 40,          # 每批次最多翻译的句子数量
        "batch_max_chars": 8000    # 每批次原文字符数上限（与 batch_size 双重约束，防止单批过长）
    },
    "tts": {
        "enabled": True,                   # 是否生成中文配音（关闭则只输出估算时间轴的字幕）
        "engine": "edge-tts",              # TTS 引擎，可选 "edge-tts" 或 "mimo-voiceclone"
        "voice": "zh-CN-YunyangNeural",     # 配音音色（engine 为 edge-tts 时使用）
        "rate": "+0%",                     # 语速调整（相对默认语速的百分比，仅 edge-tts 支持）
        "volume": "+0%",                   # 音量调整（相对默认音量的百分比，仅 edge-tts 支持）
        "pitch": "+0Hz",                   # 音调调整（相对默认音调的 Hz 偏移，仅 edge-tts 支持）
        "mix_with_original": False,        # 是否保留原声并与配音按比例混合（否则完全替换为配音）
        "batch_size": 50,                  # 每批次并发提交生成的配音条数（用于分批写入缓存）
        "concurrency": 5,                  # 单批内实际并发请求 TTS 服务的数量
        "max_tempo": 2.0,                  # 配音超长时允许的最高加速倍速；仍放不下则淡出截断，
                                           # 绝不顺延到下一句，保证全片音画硬同步
        "min_tempo": 1.0,                  # 配音明显短于可用时长时允许的最低放慢倍速，用轻微放慢
                                           # 填满可用时长、减少生硬的大段空白（放慢幅度设了下限，
                                           # 避免语速过慢显得怪异，残余极少量空白是正常现象）
        "mimo": {
            # engine 为 "mimo-voiceclone" 时使用，基于参考人声样本复刻音色
            "base_url": "https://api.xiaomimimo.com/v1",  # MiMo TTS API 的 base_url
            "api_key": "",                       # MiMo TTS API Key；留空则复用 llm.api_key
            "model": "mimo-v2.5-tts-voiceclone",  # 语音克隆模型名称
            "reference_audio": "",   # 参考人声音频路径（如 extract_vocals.py 提取出的 *_vocals.wav）
            "style_instruction": "",  # 可选：自然语言风格指令（放入 user 消息，用于控制语气/情绪）
            "format": "wav"           # 输出音频格式，wav 或 mp3
        }
    },
    "subtitle": {
        "chars_per_sec": 4.2,      # 中文配音基准语速（字/秒）；按句子可用时长换算译文字数上限，
                                   # 让 LLM 主动精简译文，从源头避免配音放不进原时间槽
        "min_sentence_sec": 1.5,   # 原句时长低于该值时，与相邻句合并为一个翻译/配音单元，
                                   # 避免过短时间槽导致配音要么被迫拉长要么严重加速
        "max_merge_chars": 200,    # 合并时原文字符数上限，防止连续短句无限合并成过长的单元
        "max_chars_per_line": 20,  # 单条字幕最大显示字符数；超过则按标点切成多条依次显示，
                                   # 仅影响字幕展示（配音仍是整句合成，不受影响），0 表示不切分
        "min_caption_ms": 800      # 切分后每条字幕的最短显示时长（ms），过短的段会并回相邻段
    }
}

SENTENCE_END_PUNCT = {'.', '!', '?', '。', '！', '？', '…'}

def has_speakable_text(text: str) -> bool:
    """判断文本是否含有可朗读的内容（纯标点/空白/符号的文本无法合成语音）。

    用 Unicode 字母/数字判断，天然覆盖所有语言（拉丁、中日韩、韩文、西里尔等），
    避免手工枚举区段遗漏（如韩文）导致整片句子被误判为不可朗读而丢弃。
    """
    return any(ch.isalnum() for ch in (text or ""))

# 字幕中的非语音提示：
# 1. 方括号注释（如 [Music] / [Applause]）：通常全为音效/注释，全部清理；
# 2. 圆括号注释：仅匹配包含明确非语音音效词汇（如 (applause), (upbeat music), (sighs) 等）的注释，
#    避免误删正常台词中的圆括号内容（如 (like Vim) 或 (page 5)）；
# 3. 音符符号（♪ ♫ 等）。
NONSPEECH_ANNOTATION_RE = re.compile(
    r"\[[^\]]*\]|"
    r"\((?=[^)]*(?:music|applause|laughter|cheering|cheers|sigh|chuckle|gasp|groan|snicker|giggle|cough|throat|whisper|cackle|sob|grunt|scream|yawn|indistinct|chatter|scoff|snort))[^)]*\)|"
    r"[\u266a\u266b\U0001f3b5\U0001f3b6]",
    re.IGNORECASE,
)

def strip_nonspeech_annotations(text: str) -> str:
    """移除字幕里的非语音注释（[Music]、(applause)、♪ 等），合并多余空白。"""
    cleaned = NONSPEECH_ANNOTATION_RE.sub(" ", text or "")
    return re.sub(r"\s{2,}", " ", cleaned).strip()


# 口语填充词（uh/um/hmm 等）：无实际语义，保留会让译文出现"呃""嗯"这类语气词。
# 只收录纯拟声的填充词，不含 ah/oh 等可能承载语气或语义的感叹词。
FILLER_WORD_RE = re.compile(
    r"(?<![\w'-])(?:u+h+|u+m+|erm+|er|hm+|mhm+|呃+|嗯+)(?![\w'-])",
    re.IGNORECASE,
)


def strip_filler_words(text: str) -> str:
    """移除口语填充词，并清理删除后残留的多余标点与空白。"""
    cleaned = FILLER_WORD_RE.sub(" ", text or "")
    cleaned = re.sub(r"\s+(?=[,，.。!！?？;；:：])", "", cleaned)
    cleaned = re.sub(r"([,，、;；])\s*(?=[,，、;；])", "", cleaned)
    cleaned = re.sub(r"^[\s,，、;；:：.。…-]+", "", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


# 常见英文缩写表（避免分句时将 Dr., Mr., vs., Inc., e.g., etc. 等误识别为句尾标点）
ENGLISH_ABBREVIATIONS = {
    # 称谓与尊称
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "rev", "rep", "sen",
    "gov", "gen", "col", "maj", "capt", "lt", "sgt", "cmdr", "adm", "hon",
    # 常见 Latin 与通用缩写
    "vs", "v", "eg", "ie", "etc", "approx", "app", "dept", "fig", "figs",
    "no", "nos", "vol", "vols", "sec", "secs", "min", "mins", "hr", "hrs",
    "sq", "ft", "in", "lbs", "oz", "yd", "mm", "cm", "m", "km",
    # 商业与机构
    "inc", "ltd", "co", "corp", "assn", "bros", "div", "est",
    # 月份与星期
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun"
}


def is_abbreviation_or_non_sentence_period(word_text: str, next_word_text: Optional[str] = None) -> bool:
    """判断以 '.' 结尾的词是否为英文缩写、首字母缩写、版本号/数字等非断句句号。"""
    # JSON3 的词块通常带前导空格（如 " T."），必须两端都去掉后再匹配
    stripped = (word_text or "").strip()
    if not stripped.endswith('.'):
        return False

    # 去除外层引号/括号/标点后取核心词
    core = stripped.strip("\"'()[]{}«»“”‘’")
    if not core.endswith('.'):
        return False

    # 1. 常见英文缩写表（如 Dr., Mr., vs., Inc., e.g., etc.）
    stem = core[:-1].lower()
    stem_nodot = core.replace(".", "").lower()
    if stem in ENGLISH_ABBREVIATIONS or stem_nodot in ENGLISH_ABBREVIATIONS:
        return True

    # 2. 单个大写字母 + 点（人名中间名首字母，如 John F. Kennedy 中的 F.）
    if re.match(r"^[A-Z]\.$", core):
        return True

    # 3. 多点首字母缩写（如 e.g., i.e., U.S., U.K., A.M., P.M., Ph.D.）
    if re.match(r"^(?:[a-zA-Z]{1,3}\.){2,}$", core):
        return True

    # 4. 数字/版本号/小数：如 v1.0. 后接小写词，或 "3." 后紧跟数字 "14"（小数切词）
    next_clean = (next_word_text or "").lstrip()
    if next_clean:
        if (re.match(r"^\$?v?\d+(?:\.\d+)*\.$", core, re.IGNORECASE)
                and (next_clean[0].islower() or next_clean[0].isdigit())):
            return True
        if next_clean[0].isdigit() and core[:-1].isdigit():
            return True

    return False


MAX_RETRIES = DEFAULT_CONFIG["global"]["max_retries"]              # 网络 API 最大重试次数
SAMPLE_RATE = DEFAULT_CONFIG["global"]["sample_rate"]              # 拼接音轨的采样率
COOKIES_FROM_BROWSER = DEFAULT_CONFIG["global"]["cookies_from_browser"]  # yt-dlp 使用的浏览器 Cookie 来源
COOKIES_FILE = DEFAULT_CONFIG["global"]["cookies_file"]            # yt-dlp 使用的 cookies.txt 文件路径
YTDLP_EXTRA_ARGS = DEFAULT_CONFIG["global"]["ytdlp_args"]          # 追加给 yt-dlp 的额外参数


def apply_global_config(config: Dict):
    """根据加载的配置动态更新全局常量（配置已与 DEFAULT_CONFIG 合并，键必定存在）"""
    global MAX_RETRIES, SAMPLE_RATE, COOKIES_FROM_BROWSER, COOKIES_FILE, YTDLP_EXTRA_ARGS
    g = config["global"]
    MAX_RETRIES = int(g["max_retries"])
    SAMPLE_RATE = int(g["sample_rate"])
    COOKIES_FROM_BROWSER = (g["cookies_from_browser"] or "").strip()
    COOKIES_FILE = (g["cookies_file"] or "").strip()
    YTDLP_EXTRA_ARGS = [str(a) for a in (g["ytdlp_args"] or [])]


def load_config() -> Dict:
    """加载脚本同目录下的 config.json 配置文件；不存在则生成模板并退出"""
    config_path = Path(__file__).parent / "config.json"

    config = json.loads(json.dumps(DEFAULT_CONFIG))  # 深拷贝

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        _deep_update(config, user_config)
        print(f"[配置] 已加载: {config_path}")
    else:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
        print(f"[配置] 未找到配置文件，已生成模板: {config_path}")
        print("[配置] 请编辑该文件填入你的 API Key 后再运行")
        sys.exit(1)

    apply_global_config(config)

    # 验证 LLM 配置
    llm = config["llm"]
    if not llm.get("api_key"):
        print("[错误] 配置文件中缺少 llm.api_key", file=sys.stderr)
        sys.exit(1)
    if not llm.get("base_url"):
        print("[错误] 配置文件中缺少 llm.base_url", file=sys.stderr)
        sys.exit(1)
    if not llm.get("model"):
        print("[错误] 配置文件中缺少 llm.model", file=sys.stderr)
        sys.exit(1)

    return config


def _deep_update(base: Dict, update: Dict):
    """深度合并字典"""
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


# ==================== 工具函数 ====================

def ms_to_srt_time(ms: int) -> str:
    """毫秒转 SRT 时间格式 HH:MM:SS,mmm"""
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    seconds = ms // 1000
    milliseconds = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def extract_video_id(url: str) -> str:
    """从 YouTube URL 提取视频 ID"""
    patterns = [
        r'(?:v=|/v/|/embed/|/shorts/|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"无法从 URL 提取视频 ID: {url}")


def strip_code_fence(text: str) -> str:
    """剥离 LLM 输出中可能存在的 markdown 代码围栏"""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    return text


def _robust_json_loads(text: str) -> Optional[object]:
    """健壮的 JSON 解析器：先尝试标准 json.loads，解析失败时使用 json_repair 修复并解析。"""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return json_repair.loads(text)
    except Exception:
        return None


def read_json_safe(path: Path) -> Optional[dict]:
    """读取 JSON 文件并返回 dict；文件不存在、解析失败或不是对象时返回 None"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def update_json_cache(path: Path, **fields):
    """合并写入缓存 JSON，保留文件中其它已有字段"""
    data = read_json_safe(path) or {}
    data.update(fields)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


# ==================== yt-dlp 相关 ====================

def sanitize_filename(name: str, max_len: int = 60) -> str:
    """清理字符串使其可安全用作文件名（去除 Windows 非法字符并限制长度）"""
    cleaned = re.sub(r'[\\/:*?"<>|\r\n\t]', " ", name)
    cleaned = cleaned.strip().rstrip(". ")
    return cleaned[:max_len].strip()


_YTDLP_PROGRESS_RE = re.compile(
    r"\[download\]\s+(\d+(?:\.\d+)?)%\s+of\s+~?([^\s]+)(?:\s+at\s+([^\s]+))?(?:\s+ETA\s+([^\s]+)|\s+in\s+([^\s]+))?"
)
_FFMPEG_TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")
_FFMPEG_SPEED_RE = re.compile(r"speed=\s*([^\s]+)")


def _make_ascii_bar(pct: float, width: int = 20) -> str:
    """生成 ASCII 进度条，例如 [=========>----------]"""
    pct = max(0.0, min(100.0, pct))
    filled = int(round(width * pct / 100.0))
    if filled == 0:
        bar = "-" * width
    elif filled == width:
        bar = "=" * width
    else:
        bar = "=" * (filled - 1) + ">" + "-" * (width - filled)
    return f"[{bar}]"


def _format_time_s(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS 或 MM:SS"""
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def probe_duration_ms(path: Path) -> Optional[int]:
    """用 ffprobe 探测音/视频时长（ms），失败返回 None"""
    if not shutil.which("ffprobe"):
        return None
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        return int(float(result.stdout.strip()) * 1000)
    except Exception:
        return None


def _stream_subprocess(cmd: List[str], label: str,
                       total_duration_s: Optional[float] = None) -> Tuple[int, str]:
    """运行子进程并实时在控制台显示结果。

    在交互式终端 (isatty) 下，为 yt-dlp 和 FFmpeg 动态呈现单行进度条；
    非终端模式（重定向/管道）下回退为普通逐行日志。
    返回 (returncode, 尾部输出文本)。
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    tail: List[str] = []
    buf = ""
    in_progress_bar = False
    is_tty = sys.stdout.isatty()

    def clear_progress_bar():
        nonlocal in_progress_bar
        if in_progress_bar and is_tty:
            sys.stdout.write("\r" + " " * 79 + "\r")
            sys.stdout.flush()
            in_progress_bar = False

    def print_line(line: str):
        nonlocal in_progress_bar
        line = line.strip()
        if not line:
            return

        tail.append(line)
        if len(tail) > 60:
            del tail[:-60]

        if is_tty:
            if label == "yt-dlp":
                m = _YTDLP_PROGRESS_RE.search(line)
                if m:
                    pct = float(m.group(1))
                    size = m.group(2)
                    speed = m.group(3)
                    eta = m.group(4) or m.group(5)
                    bar = _make_ascii_bar(pct)
                    speed_str = f" | {speed}" if speed and speed != "Unknown" else ""
                    eta_str = f" | ETA {eta}" if eta and eta != "Unknown" else ""
                    disp = f"\r[yt-dlp] 下载进度: {bar} {pct:5.1f}% | {size}{speed_str}{eta_str}"
                    sys.stdout.write(disp.ljust(79))
                    sys.stdout.flush()
                    in_progress_bar = True
                    return

            elif label == "FFmpeg":
                m_time = _FFMPEG_TIME_RE.search(line)
                if m_time:
                    h, m_val, s_val = int(m_time.group(1)), int(m_time.group(2)), float(m_time.group(3))
                    curr_sec = h * 3600 + m_val * 60 + s_val
                    m_speed = _FFMPEG_SPEED_RE.search(line)
                    speed_str = f" | {m_speed.group(1)}" if m_speed else ""
                    curr_str = _format_time_s(curr_sec)

                    if total_duration_s and total_duration_s > 0:
                        pct = min(100.0, max(0.0, curr_sec / total_duration_s * 100.0))
                        bar = _make_ascii_bar(pct)
                        total_str = _format_time_s(total_duration_s)
                        disp = f"\r[FFmpeg] 合成进度: {bar} {pct:5.1f}% ({curr_str} / {total_str}){speed_str}"
                    else:
                        disp = f"\r[FFmpeg] 正在合成: time={curr_str}{speed_str}"

                    sys.stdout.write(disp.ljust(79))
                    sys.stdout.flush()
                    in_progress_bar = True
                    return

            # 非进度信息行：如果此前展示了进度条，先擦除
            clear_progress_bar()

        print(f"[{label}] {line}")

    while True:
        chunk = proc.stdout.read1(4096)  # 有多少读多少，保证实时
        if not chunk:
            break
        buf += chunk.decode("utf-8", errors="replace")
        *lines, buf = re.split(r"[\r\n]", buf)
        for line in lines:
            print_line(line)
    if buf.strip():
        print_line(buf)

    if in_progress_bar and is_tty:
        sys.stdout.write("\n")
        sys.stdout.flush()

    code = proc.wait()
    return code, "\n".join(tail)


def run_yt_dlp(args: List[str], stream: bool = False) -> subprocess.CompletedProcess:
    """运行 yt-dlp 命令。

    stream=True 时实时转发下载日志到控制台（返回值的 stderr 仅含尾部输出）；
    默认静默捕获全部输出（--dump-json 需要解析完整 stdout，必须用默认模式）。
    包含 429 Too Many Requests 指数退避自动重试机制。
    """
    cmd = ["yt-dlp"]
    if COOKIES_FILE:
        # cookies_file 优先于 cookies_from_browser：不依赖浏览器进程/锁文件，更稳定
        cmd += ["--cookies", COOKIES_FILE]
    elif COOKIES_FROM_BROWSER:
        cmd += ["--cookies-from-browser", COOKIES_FROM_BROWSER]
    cmd += YTDLP_EXTRA_ARGS
    cmd += args
    print(f"[yt-dlp] {' '.join(cmd)}")

    attempts = MAX_RETRIES
    last_res = None

    for attempt in range(1, attempts + 1):
        if stream:
            code, tail = _stream_subprocess(cmd, "yt-dlp")
            last_res = subprocess.CompletedProcess(cmd, code, stdout="", stderr=tail)
        else:
            last_res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

        if last_res.returncode == 0:
            return last_res

        output_text = (last_res.stderr or "") + (last_res.stdout or "")
        is_429 = "429" in output_text or "Too Many Requests" in output_text or "HTTP Error 429" in output_text

        if is_429 and attempt < attempts:
            wait_sec = 5 * (2 ** (attempt - 1))  # 5s, 10s, 20s...
            print(f"[yt-dlp] 触发 YouTube 限流 (HTTP 429 Too Many Requests)，等待 {wait_sec} 秒后进行第 {attempt + 1}/{attempts} 次重试...", file=sys.stderr)
            time.sleep(wait_sec)
        else:
            if not stream and last_res.returncode != 0:
                print(f"[yt-dlp stderr] {last_res.stderr}", file=sys.stderr)
            break

    return last_res


def get_video_metadata(url: str) -> Dict:
    """获取视频元数据"""
    result = run_yt_dlp(["--dump-json", "--skip-download", url])
    if result.returncode != 0:
        raise RuntimeError(f"无法获取视频元数据: {result.stderr}")
    first_line = result.stdout.strip().splitlines()[0]
    return json.loads(first_line)


def detect_sub_langs(metadata: Dict) -> str:
    """选择唯一字幕轨道，优先自动字幕和视频语言的精确匹配。"""
    lang = (metadata.get("language") or "").strip()
    base = lang.split("-")[0] if lang and lang not in ("none", "und") else "en"
    automatic = metadata.get("automatic_captions") or {}
    manual = metadata.get("subtitles") or {}

    def rank(track: str, source: str) -> Tuple[int, int, str]:
        is_auto = source == "automatic"
        exact = track == lang or track == base
        same_base = track.split("-")[0] == base
        return (
            0 if is_auto else 1,
            0 if exact else 1 if same_base else 2,
            track,
        )

    tracks = [(track, "automatic") for track in automatic]
    tracks += [(track, "manual") for track in manual if track not in automatic]
    if not tracks:
        print("[语言] 元数据没有字幕轨道信息，回退到 en")
        return "en"

    selected, source = min(tracks, key=lambda item: rank(item[0], item[1]))
    print(f"[语言] 选择{source}字幕轨道: {selected}")
    return selected


def find_downloaded_sub(output_dir: Path, video_id: str,
                        language: str) -> Optional[Path]:
    """查找指定语言的 json3 字幕文件（yt-dlp 模板 %(id)s 下固定为 id.lang.json3）"""
    path = output_dir / f"{video_id}.{language}.json3"
    return path if path.exists() else None


def find_local_media(output_dir: Path,
                     video_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    """查找原始视频与封面。

    只认文件名恰为视频 ID 的文件（yt-dlp 模板 %(id)s 的产物），
    避免重跑时把上次以中文标题命名的成品当成原片、叠加字幕与配音。
    """
    def first_existing(extensions: Tuple[str, ...]) -> Optional[Path]:
        return next((p for ext in extensions
                     if (p := output_dir / f"{video_id}{ext}").exists()), None)

    return (first_existing((".mp4", ".webm", ".mkv", ".mov")),
            first_existing((".jpg", ".jpeg", ".png", ".webp")))


def download_video_and_subs(url: str, output_dir: Path, sub_langs: str,
                            metadata: Dict, skip_video: bool = False
                            ) -> Tuple[Optional[Path], Path, Optional[Path]]:
    """下载视频和 JSON3 字幕（metadata 由调用方传入，避免重复执行 yt-dlp 获取元数据）"""
    output_dir.mkdir(parents=True, exist_ok=True)

    video_id = metadata["id"]
    title = metadata.get("title", "unknown")

    print(f"[下载] 视频标题: {title}")
    print(f"[下载] 视频ID: {video_id}")
    print(f"[下载] 字幕语言候选: {sub_langs}")

    def build_ytdlp_cmd(sub_flag: str) -> List[str]:
        """构造 yt-dlp 下载命令（自动字幕失败后换 --write-subs 重试）"""
        return [
            *(["--skip-download"] if skip_video
              else ["-f", "bestvideo*+bestaudio/best"]),
            sub_flag,
            "--sub-langs", sub_langs,
            "--sub-format", "json3",
            "--write-thumbnail",
            "--convert-thumbnails", "jpg",
            "--sleep-subtitles", "2",
            "-o", str(output_dir / "%(id)s"),
            url,
        ]

    result = run_yt_dlp(build_ytdlp_cmd("--write-auto-subs"), stream=True)
    if result.returncode != 0:
        print("[下载] 自动字幕下载失败，尝试手动字幕...")
        result = run_yt_dlp(build_ytdlp_cmd("--write-subs"), stream=True)
        if result.returncode != 0:
            raise RuntimeError(f"字幕下载失败: {result.stderr}")

    video_path, thumbnail_path = find_local_media(output_dir, video_id)
    sub_path = find_downloaded_sub(output_dir, video_id, sub_langs)
    if skip_video:
        # 目录里可能残留着上次下载的视频，--no-video 下不让它参与后续流程
        video_path = None
    elif video_path is None:
        raise FileNotFoundError(f"未找到下载的视频文件 (ID: {video_id})")
    if sub_path is None:
        # SystemExit 不被 main 的 except Exception 捕获，直接退出且不带堆栈
        raise SystemExit(
            f"[错误] 未找到 json3 字幕文件（语言候选: {sub_langs}）。\n"
            "       该视频可能不提供 json3 格式的字幕，而本脚本依赖词级时间戳，无法继续。")

    if video_path:
        print(f"[下载完成] 视频: {video_path}")
    print(f"[下载完成] 字幕: {sub_path}")
    if thumbnail_path:
        print(f"[下载完成] 封面: {thumbnail_path}")

    return video_path, sub_path, thumbnail_path


# ==================== JSON3 解析 ====================

def parse_json3(json3_path: Path) -> List[Dict]:
    """解析 YouTube JSON3 字幕文件"""
    with open(json3_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = []
    events = data.get("events", [])
    last_end_ms = 0  # 最后一个真实词所在 event 的结束时间

    for event in events:
        base_time = event.get("tStartMs", 0)
        segs = event.get("segs", [])
        if not segs:
            continue

        appended = False
        for seg in segs:
            text = seg.get("utf8", "")
            # JSON3 用独立的换行 seg 分隔字幕行：保留空格，但不让它参与词语计时。
            if not text:
                continue
            if not text.strip():
                if words and not words[-1]["text"].endswith((" ", "\n")):
                    words[-1]["text"] += " "
                continue
            offset = seg.get("tOffsetMs", 0)
            words.append({
                "text": text,
                "start_ms": base_time + offset,
                "end_ms": None,
            })
            appended = True

        if appended:
            last_end_ms = event.get("tStartMs", 0) + event.get("dDurationMs", 0)

    for i in range(len(words) - 1):
        # 使用下一个真实词的开始时间，避免把换行 event 的时间算成词尾。
        words[i]["end_ms"] = max(words[i + 1]["start_ms"], words[i]["start_ms"])

    if words:
        words[-1]["end_ms"] = max(last_end_ms, words[-1]["start_ms"])

    print(f"[解析] 共提取 {len(words)} 个词")
    return words


def split_into_sentences(words: List[Dict]) -> List[Dict]:
    """按标点分句（智能排除英文缩写、首字母缩写、版本号/数字小数点等非断句句号）"""
    def make_sentence(ws: List[Dict]) -> Optional[Dict]:
        """把一组词组装成句子；纯标点/非语音注释/无实际内容的片段返回 None 丢弃"""
        text = strip_filler_words(
            strip_nonspeech_annotations("".join(w["text"] for w in ws).strip()))
        if not has_speakable_text(text):
            return None
        return {
            "text": text,
            "start_ms": ws[0]["start_ms"],
            "end_ms": ws[-1]["end_ms"],
        }

    sentences: List[Dict] = []
    current_words: List[Dict] = []
    num_words = len(words)

    for i, word in enumerate(words):
        current_words.append(word)
        stripped = word["text"].rstrip()
        if stripped and stripped[-1] in SENTENCE_END_PUNCT:
            next_word_text = words[i + 1]["text"] if i + 1 < num_words else None
            if stripped[-1] == '.' and is_abbreviation_or_non_sentence_period(word["text"], next_word_text):
                continue

            sent = make_sentence(current_words)
            if sent:
                sentences.append(sent)
            # 被丢弃片段的时间轴自然归入下一句（current_words 直接清空即可）
            current_words = []

    sent = make_sentence(current_words) if current_words else None
    if sent:
        sentences.append(sent)

    print(f"[分句] 共 {len(sentences)} 句")
    return sentences


def merge_short_sentences(sentences: List[Dict], min_sentence_sec: float,
                          max_merge_chars: int) -> List[Dict]:
    """把时长过短的相邻句子合并成一个翻译/配音单元。

    原字幕（尤其自动字幕）常被切得很碎，短句独立配音时容易出现两种极端：
    时间槽太窄被迫大幅加速，或配音很快读完后留下大段空白。把时长低于
    min_sentence_sec 的句子与相邻句合并、整体翻译+配音，从源头上减少这两种
    极端情况（碎句按时长换算出的字数预算过小，译文写不下必然超时被截断）。

    合并策略：从前往后扫描，只要当前累积句的时长仍低于阈值，且合并后的原文
    字符数未超过 max_merge_chars，就继续吸收下一句；达到阈值或长度上限后
    结算成一个单元。合并单元的 start_ms/end_ms 取首尾句的跨度，text 用空格
    拼接（保留自然的单词间隔，翻译时当作一整段话处理，语义更连贯）。
    """
    if not sentences:
        return sentences

    merged: List[Dict] = []
    buffer: Optional[Dict] = None


    def duration_sec(sent: Dict) -> float:
        return max(sent["end_ms"] - sent["start_ms"], 0) / 1000.0

    def flush():
        nonlocal buffer
        if buffer:
            merged.append(buffer)
        buffer = None

    for sent in sentences:
        if buffer is None:
            buffer = dict(sent)
            continue

        buffer_short = duration_sec(buffer) < min_sentence_sec
        combined_len = len(buffer["text"]) + 1 + len(sent["text"])
        if buffer_short and combined_len <= max_merge_chars:
            # 合并进当前缓冲：文本用空格拼接，时间跨度扩展到本句结尾
            buffer["text"] = buffer["text"] + " " + sent["text"]
            buffer["end_ms"] = sent["end_ms"]
        else:
            flush()
            buffer = dict(sent)

    flush()

    if len(merged) != len(sentences):
        print(f"[分句] 短句合并: {len(sentences)} 句 -> {len(merged)} 句"
              f"（阈值 {min_sentence_sec}s）")
    return merged


# ==================== LLM 翻译 ====================

class LLMClient:
    """统一 LLM 客户端"""

    def __init__(self, config: Dict):
        self.model = config["model"]
        self.supports_system_role = config["supports_system_role"]
        self.thinking = config.get("thinking")  # 可被用户整体删除，不走默认值
        self.batch_size = max(1, int(config["batch_size"]))
        self.batch_max_chars = max(1, int(config["batch_max_chars"]))

        base_url = config["base_url"].rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"

        self.client = OpenAI(
            api_key=config["api_key"],
            base_url=base_url,
            timeout=120,
        )

        print(f"[LLM] 初始化: {base_url}")
        print(f"[LLM] 模型: {self.model}")

    def _chat(self, messages: List[Dict], temperature: float = 0.3) -> str:
        """调用聊天接口（带重试机制）"""
        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=8192,
                    **({"extra_body": {"thinking": self.thinking}}
                       if self.thinking else {}),
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_err = e
                wait = 2 ** attempt
                print(f"[翻译] 调用失败 (第 {attempt}/{MAX_RETRIES} 次): {e}，{wait}s 后重试...")
                time.sleep(wait)
        raise RuntimeError(f"LLM 调用失败（已重试 {MAX_RETRIES} 次）: {last_err}")

    def translate(self, sentences: List[Dict], title: str,
                  budgets: List[int],
                  done: Optional[Dict[int, str]] = None,
                  on_progress=None,
                  glossary: Optional[List[Dict]] = None) -> List[str]:
        """分批翻译所有句子（结构化 JSON 输出，按 id 对齐，避免错位）。

        budgets: 与 sentences 对齐的每句中文字数上限（由可用时长换算）。
        done: 已有缓存译文的 {id: 译文}，这些句子不再重复请求（断点续传）。
        on_progress: 每完成一批后的回调 on_progress(results_dict)，用于增量写缓存。
        glossary: 全片统一术语表 [{term, zh}]，注入每个批次的 prompt 保证译名一致。
        任一批次重试后仍失败将抛出异常，由调用方中断后续流程。
        """
        n_total = len(sentences)
        done = dict(done or {})
        results: Dict[int, str] = dict(done)

        # 先过滤出待翻译的句子编号，再按句数/字符数上限纯粋分批，
        # 避免把缓存跳过逻辑与分批逻辑纠缠在一起。
        pending = [i for i in range(n_total) if i not in results]
        print(f"[翻译] 共 {n_total} 句，待翻译 {len(pending)} 句（每批最多 "
              f"{self.batch_size} 句，{self.batch_max_chars} 字符）...")

        cursor = 0
        while cursor < len(pending):
            # 截取一个批次：受句数上限与字符数上限双重约束（至少 1 句）
            todo = [pending[cursor]]
            batch_chars = len(sentences[pending[cursor]]["text"])
            cursor += 1
            while cursor < len(pending) and len(todo) < self.batch_size:
                next_chars = len(sentences[pending[cursor]]["text"])
                if batch_chars + next_chars > self.batch_max_chars:
                    break
                todo.append(pending[cursor])
                batch_chars += next_chars
                cursor += 1

            # 保留全局真实编号，避免续传时错位
            # 滑动窗口：上文给已确定的中文译文，下文给尚未翻译的原文
            prev_context = [results[i] for i in range(max(todo[0] - 3, 0), todo[0])
                            if i in results]
            next_preview = [s["text"] for s in sentences[todo[-1] + 1: todo[-1] + 4]]

            system_msg = "你是一位专业的视频字幕翻译师。你只输出合法的 JSON 数组，不输出任何其他内容。"
            batch_result: Dict[int, str] = {}
            missing = list(todo)
            for attempt in range(1, MAX_RETRIES + 1):
                # 重试时只重发缺失的句子，并适度提升 temperature 引入变化避免死锁
                prompt = self._build_prompt(
                    [sentences[i] for i in missing], title, missing, glossary,
                    [budgets[i] for i in missing], prev_context, next_preview)
                if self.supports_system_role:
                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ]
                else:
                    messages = [{"role": "user", "content": system_msg + "\n\n" + prompt}]

                content = self._chat(messages,
                                     temperature=0.3 + (attempt - 1) * 0.2)
                batch_result.update(
                    self._parse_json_translations(content, set(missing)))
                missing = [i for i in todo if i not in batch_result]
                if not missing:
                    break
                print(f"[警告] 批次 {todo[0]}-{todo[-1]}: 缺少 {len(missing)} 句译文 (id: {missing[:5]}...)，第 {attempt}/{MAX_RETRIES} 次重试...")

            if missing:
                # 重试后仍失败：硬性中断，不做占位降级
                raise RuntimeError(
                    f"批次 {todo[0]}-{todo[-1]} 有 {len(missing)} 句重试后仍翻译失败"
                    f" (id: {missing[:5]}...)，已中断。"
                    f"已完成部分已写入缓存，修复后重新运行可从断点继续。"
                )

            results.update(batch_result)
            if on_progress:
                on_progress(results)
            print(f"[翻译] 进度: {len(results)}/{n_total}")

        print(f"[翻译] 完成，共 {len(results)} 句")
        return [results[i] for i in range(n_total)]

    @staticmethod
    def _parse_json_translations(content: str, wanted_ids: set) -> Dict[int, str]:
        """从 LLM 返回内容解析 [{"id": n, "zh": "..."}]，返回 {id: 译文}"""
        text = strip_code_fence(content)
        begin = text.find("[")
        end = text.rfind("]")
        if begin != -1 and end != -1 and end > begin:
            data = _robust_json_loads(text[begin:end + 1])
            if data is None:
                data = _robust_json_loads(text)
        else:
            data = _robust_json_loads(text)

        result: Dict[int, str] = {}
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                try:
                    idx = int(item["id"])
                except (KeyError, TypeError, ValueError):
                    continue
                zh = str(item.get("zh", "")).strip()
                if zh and idx in wanted_ids:
                    result[idx] = zh
        return result

    @staticmethod
    def _build_prompt(sentences: List[Dict], title: str, ids: List[int],
                      glossary: Optional[List[Dict]],
                      budgets: List[int],
                      prev_context: List[str],
                      next_preview: List[str]) -> str:
        numbered_lines = [
            f"{sentence_id}. [中文不超过 {budget} 字] {sentence['text']}"
            for sentence_id, sentence, budget in zip(ids, sentences, budgets)
        ]
        numbered_text = "\n".join(numbered_lines)

        context_block = ""
        if prev_context:
            context_block += ("\n【上文参考（已翻译完成的前几句中文，仅用于保持语境与风格连贯）】\n"
                              + "\n".join(f"- {t}" for t in prev_context) + "\n")
        if next_preview:
            context_block += ("\n【下文预览（后续原文，仅用于理解语境，不要翻译）】\n"
                              + "\n".join(f"- {t}" for t in next_preview) + "\n")

        glossary_block = ""
        if glossary:
            terms = "\n".join(f"- {item['term']} → {item['zh']}" for item in glossary)
            glossary_block = (f"\n【统一术语表（下列词条必须严格采用给定译法，译法与原文相同的词条必须原样输出、"
                              f"一个字符都不能增删改）】\n{terms}\n")

        return f"""你是一位专业的视频字幕翻译师。请将以下视频字幕翻译成中文。

视频标题：{title}
{glossary_block}{context_block}
请严格逐句翻译，不要合并或拆分句子，不要遗漏任何一句。译文应自然流畅，符合中文表达习惯，适合作为视频字幕。

字数约束（重要）：每句前的 [中文不超过 N 字] 是该句配音可用时长换算出的硬性上限。请宁简勿繁，主动意译精简：删去可有可无的定语、语气词和重复表达，保留核心信息即可，不得超出字数上限。

快捷键、命令、代码和界面文字等英文/数字字面量请原样保留，不要翻译或“纠错”。特别注意：像 zz、qq、dd 这类重复字母很可能是真实的按键序列（如 Vim 按键），不是拼写错误，不得删减重复字母。

输出要求：只输出一个 JSON 数组，每个元素格式为 {{"id": 编号, "zh": "该句中文译文"}}。编号可能与其它批次重叠或看起来不连续，但必须与原句前面的编号完全一致，一个都不能改、不能漏。不要输出 markdown 代码块标记，不要输出任何解释。

原文：
{numbered_text}

中文译文（JSON 数组）："""

    def extract_glossary(self, sentences: List[Dict]) -> List[Dict]:
        """从全部字幕原文提取需统一译法的专有名词/术语表（人名、地名、作品名、术语等）。

        全程只调用一次；超长视频按均匀间隔抽样控制输入体积。
        返回 [{"term": 原文, "zh": 统一译法}]，解析失败返回 []。
        """
        MAX_LINES = 600  # 抽样上限：600 句 × ~50 字符 ≈ 3 万字符输入
        n = len(sentences)
        if n > MAX_LINES:
            step = n / MAX_LINES
            picked = [sentences[int(i * step)] for i in range(MAX_LINES)]
        else:
            picked = sentences

        numbered = "\n".join(s["text"][:50] for s in picked)

        prompt = f"""以下是某视频的全部字幕文本（每行一句）。请提取其中需要在中文翻译里保持前后一致的内容，包括两类：

A. 需要统一译名的专有名词：人名、地名、作品名、组织机构、品牌、专业术语，给出统一的中文译法。
B. 必须原样保留、绝不能翻译或“纠错”的字面量：按键与按键序列（如 zz、qq、dd、gg、Ctrl+C、:wq）、命令与命令行参数、代码标识符、文件名、快捷键、专用缩写。
   这类词条的 zh 必须填写与 term 完全相同的字符串（包括重复字母，如 zz → zz，不得写成 z）。

普通词汇和常见词不要提取；没有可提取的内容则输出空数组 []。

输出格式：JSON 数组 [{{"term": "原文", "zh": "统一的中文译法或原样字面量"}}]，不要输出其他任何内容。

字幕文本：
{numbered}

术语表（JSON 数组）："""

        content = strip_code_fence(self._chat([{"role": "user", "content": prompt}]))
        begin, end = content.find("["), content.rfind("]")
        if begin != -1 and end != -1 and end > begin:
            data = _robust_json_loads(content[begin:end + 1])
            if data is None:
                data = _robust_json_loads(content)
        else:
            data = _robust_json_loads(content)

        result = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    term = str(item.get("term", "")).strip()
                    zh = str(item.get("zh", "")).strip()
                    if term and zh:
                        result.append({"term": term, "zh": zh})
        return result

    def translate_title(self, title: str, description: str = "",
                        content_summary: str = "") -> Optional[str]:
        """翻译/改写视频标题（用于最终视频文件命名），失败返回 None。

        欧美视频标题风格与国内自媒体差异较大，有时原标题过于宽泛、噱头化，
        或需要结合上下文才能理解，直译后中文读者难以第一时间判断视频讲了什么。
        因此这里不再要求"只翻译"，而是让模型先判断原标题是否已经能清楚地
        表明视频内容：能的话就直译/意译；不能的话，则结合视频简介和字幕内容，
        重新拟一个符合中文自媒体习惯、简洁抓重点的标题（而非逐字翻译）。
        """
        if not title or title == "Unknown":
            return None
        prompt = (
            "你是一名中文自媒体编辑，需要为下面这个视频确定最终的中文标题（用于视频文件命名和展示）。\n"
            "欧美视频标题的风格和国内自媒体常有差异：有的原标题信息量不足、过于宽泛或依赖上下文，"
            "直译成中文后读者无法一眼看出视频到底讲了什么。\n\n"
            "请按以下规则判断并处理：\n"
            "1. 如果原标题本身已经能清楚表明视频的具体内容，直接把它翻译成简洁自然的中文标题；\n"
            "2. 如果原标题比较空泛、标题党、或者需要结合简介/字幕才能看出实际内容，"
            "请不要逐字直译，而是结合下面提供的视频简介和字幕内容，重新拟一个能准确概括视频内容、"
            "符合中文自媒体表达习惯的标题（可以适度提炼重点、增加信息量，但不要夸大或编造事实）；\n"
            "3. 标题字数限制：无论是翻译还是重新拟定标题，请务必控制字数（建议控制在 15 至 25 个字以内，最长不超过 30 个字），做到精炼抓重点，便于阅读和文件名显示。\n\n"
            f"原标题：{title}\n"
        )
        desc = (description or "").strip()
        if desc:
            prompt += f"视频简介：{desc[:300]}\n"
        summary = (content_summary or "").strip()
        if summary:
            prompt += f"字幕内容节选（供理解视频实际讲的内容，仅供参考不要照抄）：\n{summary}\n"
        prompt += (
            "\n只输出最终确定的中文标题本身，不要输出判断过程、不要引号、不要解释、不要保留原文。"
        )

        content = self._chat([{"role": "user", "content": prompt}])
        # 去掉可能存在的代码围栏，取第一行，再去掉包裹的引号
        text = strip_code_fence(content)
        text = text.splitlines()[0].strip().strip('"“”').strip()
        return text or None


# ==================== 字幕后处理 ====================

def compute_char_budgets(sentences: List[Dict],
                         chars_per_sec: float) -> List[int]:
    """按每句可用时长换算中文译文字数上限，注入翻译 prompt 约束译文长度。

    可用时长取「本句起点 → 下一句起点」（含句间静音间隙），因为时间轴按
    原句起点硬锚定，本句配音最多只能占用到下一句开始之前的这段时间。
    """
    budgets: List[int] = []
    n = len(sentences)
    for i, sent in enumerate(sentences):
        if i + 1 < n:
            avail_ms = sentences[i + 1]["start_ms"] - sent["start_ms"]
        else:
            avail_ms = sent["end_ms"] - sent["start_ms"]
        budgets.append(max(int(round(max(avail_ms, 0) / 1000.0 * chars_per_sec)), 4))
    return budgets


def postprocess_subtitles(sentences: List[Dict],
                          translations: List[str]) -> List[Dict]:
    """组装配音单元：1 句原文 = 1 条译文 = 1 段 TTS 音频 = 1 条字幕。

    纯标点/无实际内容的译文（无法 TTS）会被跳过。
    """
    result = []

    for sent, trans in zip(sentences, translations):
        # 跳过纯标点/无实际内容的译文（否则 TTS 会报 NoAudioReceived）
        if not has_speakable_text(trans):
            print(f"[后处理] 跳过无可朗读内容的译文: {trans!r}")
            continue

        result.append({
            "text": trans,
            "start_ms": sent["start_ms"],
            "end_ms": sent["end_ms"],
        })

    return result


# ==================== 字幕显示切分 ====================

# 切点优先级（值越小越优先）；英文句点不作为切点，避免误切 v1.0 / e.g. 这类字面量
CAPTION_CUT_PRIORITY = {
    '。': 0, '！': 0, '？': 0, '!': 0, '?': 0, '…': 0,
    '；': 1, ';': 1,
    '，': 2, ',': 2,
    '、': 3,
}

CAPTION_PAIR_OPEN = {'“': '”', '‘': '’', '《': '》', '（': '）', '(': ')',
                     '【': '】', '[': ']'}
CAPTION_PAIR_CLOSE = {v: k for k, v in CAPTION_PAIR_OPEN.items()}

MIN_CAPTION_CHARS = 5  # 过短的片段并回相邻段，避免字幕一闪而过


def _caption_cut_candidates(text: str) -> List[Tuple[int, int]]:
    """找出可切分位置，返回 [(优先级, 切点下标)]（下标为标点之后的位置）。"""
    candidates: List[Tuple[int, int]] = []
    stack: List[str] = []
    i, n = 0, len(text)

    while i < n:
        ch = text[i]

        if ch in ('"', "'"):
            if stack and stack[-1] == ch:
                stack.pop()
            else:
                stack.append(ch)
            i += 1
            continue
        if ch in CAPTION_PAIR_OPEN:
            stack.append(ch)
            i += 1
            continue
        if ch in CAPTION_PAIR_CLOSE:
            if stack and stack[-1] == CAPTION_PAIR_CLOSE[ch]:
                stack.pop()
            i += 1
            continue

        prio = CAPTION_CUT_PRIORITY.get(ch)
        if prio is None or stack:
            i += 1
            continue

        if ch == '…':
            j = i
            while j < n and text[j] == '…':
                j += 1
            candidates.append((prio, j))
            i = j
            continue

        # 数字千分位（如 46,000）不是断句逗号
        if ch in (',', '，') and 0 < i < n - 1 \
                and text[i - 1].isdigit() and text[i + 1].isdigit():
            i += 1
            continue

        candidates.append((prio, i + 1))
        i += 1

    return candidates


def _hard_split_caption(text: str, max_chars: int) -> List[str]:
    """无可用标点时按长度硬切，尽量不切断连续的英文单词/数字。"""
    parts: List[str] = []
    rest = text
    while len(rest) > max_chars:
        cut = max_chars
        while (cut > max_chars // 2
               and rest[cut - 1].isalnum() and rest[cut].isalnum()
               and rest[cut - 1].isascii() and rest[cut].isascii()):
            cut -= 1
        cut = max(cut, 1)  # 保证每轮至少推进 1 个字符，防止死循环
        parts.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        parts.append(rest)
    return parts


def split_caption_text(text: str, max_chars: int) -> List[str]:
    """把超长译文按标点切成多条展示文本（不改变文字内容，只做切分）。

    切点优先在句子中部区间内选取，再按 。！？ > ； > ， > 、 的优先级挑选，
    避免开头一个句号就把两三个字单独切出去。
    """
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    candidates = [(p, idx) for p, idx in _caption_cut_candidates(text)
                  if 0 < idx < len(text)]
    if candidates:
        middle = len(text) / 2
        balanced = [c for c in candidates
                    if len(text) * 0.25 <= c[1] <= len(text) * 0.75]
        pool = balanced or candidates
        _, cut = min(pool, key=lambda c: (c[0], abs(c[1] - middle)))
        left, right = text[:cut].strip(), text[cut:].strip()
        if left and right:
            return (split_caption_text(left, max_chars)
                    + split_caption_text(right, max_chars))

    return _hard_split_caption(text, max_chars)


def split_clips_for_display(clips: List[Dict], max_chars: int,
                            min_caption_ms: int) -> List[Dict]:
    """把过长字幕切成多条依次显示，段内时长按字符数比例分摊。

    只作用于 SRT 渲染：配音仍是整句合成，混音仍使用原始 clips，音画不受影响。
    """
    if max_chars <= 0:
        return clips

    result: List[Dict] = []
    split_count = 0

    for clip in clips:
        start, end = clip["start_ms"], clip["end_ms"]
        total = max(end - start, 0)
        segments = split_caption_text(clip["text"], max_chars)

        # 合并过短片段：字数过少的并回相邻段，再按实际时长兜底一次
        items: List[str] = []
        for seg in segments:
            if items and (len(seg) < MIN_CAPTION_CHARS
                          or not has_speakable_text(seg)):
                items[-1] += seg
            else:
                items.append(seg)
        if len(items) > 1 and len(items[0]) < MIN_CAPTION_CHARS:
            items[1] = items[0] + items[1]
            del items[0]

        while len(items) > 1:
            total_chars = max(sum(len(s) for s in items), 1)
            durations = [total * len(s) / total_chars for s in items]
            k = min(range(len(items)), key=lambda i: durations[i])
            if durations[k] >= min_caption_ms:
                break
            j = k - 1 if k > 0 else 1
            a, b = min(j, k), max(j, k)
            items[a] += items[b]
            del items[b]

        if len(items) <= 1:
            result.append({"text": clip["text"], "start_ms": start, "end_ms": end})
            continue

        split_count += 1
        total_chars = max(sum(len(s) for s in items), 1)
        t = start
        for k, seg_text in enumerate(items):
            seg_end = (end if k == len(items) - 1
                       else t + int(total * len(seg_text) / total_chars))
            seg_end = max(seg_end, t + 1)
            result.append({"text": seg_text, "start_ms": t, "end_ms": seg_end})
            t = seg_end

    if split_count:
        print(f"[字幕] {split_count} 条超长字幕已切分显示"
              f"（{len(clips)} 条 -> {len(result)} 条，上限 {max_chars} 字）")
    return result


# ==================== SRT 生成 ====================

def generate_srt(subs: List[Dict], output_path: Path):
    """生成 SRT 文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, sub in enumerate(subs, 1):
            start = ms_to_srt_time(sub["start_ms"])
            end = ms_to_srt_time(sub["end_ms"])
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{sub['text']}\n\n")

    print(f"[SRT] 已生成: {output_path} ({len(subs)} 条字幕)")


# ==================== 中文配音（TTS） ====================

class TTSClient:
    """统一 TTS 客户端"""

    def __init__(self, config: Dict, llm_config: Optional[Dict] = None):
        self.engine = config["engine"]
        self.mix_with_original = config["mix_with_original"]
        self.batch_size = max(1, int(config["batch_size"]))
        self.concurrency = max(1, int(config["concurrency"]))
        self.max_tempo = max(1.0, float(config["max_tempo"]))
        self.min_tempo = min(1.0, max(0.5, float(config["min_tempo"])))

        # 缓存签名不纳入任何 api_key 字段（含 mimo.api_key 嵌套项），避免密钥变更导致误判缓存失效，
        # 也避免密钥被写入缓存指纹
        def _strip_api_keys(d: Dict) -> Dict:
            return {k: (_strip_api_keys(v) if isinstance(v, dict) else v)
                    for k, v in d.items() if k != "api_key"}

        self.cache_signature = hashlib.sha1(
            json.dumps(_strip_api_keys(config), sort_keys=True,
                       ensure_ascii=False).encode("utf-8")
        ).hexdigest()

        if self.engine == "edge-tts":
            self.voice = config["voice"]
            self.rate = config["rate"]
            self.volume = config["volume"]
            self.pitch = config["pitch"]
            print(f"[TTS] 引擎: edge-tts, 音色: {self.voice}")
        elif self.engine == "mimo-voiceclone":
            mimo_config = config["mimo"]
            base_url = (mimo_config["base_url"]
                        or "https://api.xiaomimimo.com/v1").rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"
            api_key = mimo_config["api_key"] or (llm_config or {}).get("api_key")
            if not api_key:
                raise ValueError(
                    "mimo-voiceclone 引擎需要 tts.mimo.api_key（或复用 llm.api_key）")
            reference_audio = mimo_config["reference_audio"]
            if not reference_audio:
                raise ValueError(
                    "mimo-voiceclone 引擎需要 tts.mimo.reference_audio 指定参考人声音频路径"
                    "（可用 extract_vocals.py 提取出的 *_vocals.wav/mp3）")
            reference_path = Path(reference_audio)
            if not reference_path.exists():
                raise ValueError(f"参考人声音频不存在: {reference_path}")

            mime_type = mimetypes.guess_type(str(reference_path))[0]
            if mime_type not in ("audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav"):
                # 按扩展名兜底（mimetypes 在部分平台可能识别不到 mp3/wav）
                suffix = reference_path.suffix.lower()
                mime_type = {"mp3": "audio/mpeg", "wav": "audio/wav"}.get(
                    suffix.lstrip("."), None)
            if mime_type is None:
                raise ValueError(
                    f"参考人声音频格式不支持（仅支持 mp3/wav）: {reference_path}")
            if mime_type in ("audio/mpeg", "audio/mp3"):
                mime_type = "audio/mpeg"

            with open(reference_path, "rb") as f:
                reference_bytes = f.read()
            reference_b64 = base64.b64encode(reference_bytes).decode("utf-8")
            if len(reference_b64) > 10 * 1024 * 1024:
                raise ValueError(
                    "参考人声音频过大：Base64 编码后不能超过 10 MB，请先裁剪或压缩")
            self.mimo_voice_data_uri = f"data:{mime_type};base64,{reference_b64}"

            self.mimo_model = mimo_config["model"]
            self.mimo_format = mimo_config["format"]
            self.mimo_style_instruction = mimo_config["style_instruction"] or ""
            self.mimo_client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)
            print(f"[TTS] 引擎: mimo-voiceclone, 模型: {self.mimo_model}, "
                  f"参考音频: {reference_path.name}")
        else:
            raise ValueError(f"不支持的 TTS 引擎: {self.engine}")

    async def generate_all(self, pieces: List[Dict], output_dir: Path
                           ) -> Tuple[List[Path], List[Optional[int]]]:
        """分批生成并缓存配音，返回与 pieces 对齐的 (音频路径, 时长毫秒)。

        时长在生成时用 ffprobe 测一次并写入 tts_cache.json，
        避免后续 build_layout 再对每条配音重复 fork ffprobe。
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        extension = "mp3" if self.engine == "edge-tts" else self.mimo_format
        manifest_path = output_dir / "tts_cache.json"
        manifest = read_json_safe(manifest_path) or {}
        entries = manifest.get("entries")
        if not isinstance(entries, list):
            entries = []

        files: List[Optional[Path]] = [None] * len(pieces)
        durations: List[Optional[int]] = [None] * len(pieces)
        missing = []

        def make_entry(idx: int) -> Dict[str, object]:
            path = output_dir / f"tts_{idx:04d}.{extension}"
            return {
                "text_sha": hashlib.sha1(
                    pieces[idx]["text"].encode("utf-8")
                ).hexdigest(),
                "signature": self.cache_signature,
                "duration_ms": probe_duration_ms(path),
            }

        def save_manifest():
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump({"entries": entries}, f, ensure_ascii=False, indent=1)

        for idx, piece in enumerate(pieces):
            text_sha = hashlib.sha1(piece["text"].encode("utf-8")).hexdigest()
            entry = entries[idx] if idx < len(entries) else None
            path = output_dir / f"tts_{idx:04d}.{extension}"
            if (isinstance(entry, dict)
                    and entry.get("text_sha") == text_sha
                    and entry.get("signature") == self.cache_signature
                    and path.exists() and path.stat().st_size > 0):
                files[idx] = path
                cached_dur = entry.get("duration_ms")
                # 旧缓存无时长字段时补测一次
                durations[idx] = (cached_dur if isinstance(cached_dur, int)
                                  else probe_duration_ms(path))
            else:
                missing.append(idx)

        print(f"[TTS] 共 {len(pieces)} 条配音，缓存命中 {len(pieces) - len(missing)} 条，"
              f"待生成 {len(missing)} 条")
        for batch_start in range(0, len(missing), self.batch_size):
            batch_indexes = missing[batch_start:batch_start + self.batch_size]
            batch_pieces = [pieces[idx] for idx in batch_indexes]
            try:
                generate_fn = (self._generate_edge_tts if self.engine == "edge-tts"
                               else self._generate_mimo_voiceclone)
                batch_files = await generate_fn(
                    batch_pieces, output_dir, batch_indexes)
            except Exception:
                entries = entries[:len(pieces)]
                entries.extend({} for _ in range(len(pieces) - len(entries)))
                for idx in batch_indexes:
                    path = output_dir / f"tts_{idx:04d}.{extension}"
                    if path.exists() and path.stat().st_size > 0:
                        entries[idx] = make_entry(idx)
                save_manifest()
                raise
            for idx, path in zip(batch_indexes, batch_files):
                files[idx] = path
            entries = entries[:len(pieces)]
            entries.extend({} for _ in range(len(pieces) - len(entries)))
            for idx in batch_indexes:
                entries[idx] = make_entry(idx)
                durations[idx] = entries[idx]["duration_ms"]
            save_manifest()
            print(f"[TTS] 进度: {min(batch_start + self.batch_size, len(missing))}/"
                  f"{len(missing)} 条待生成")

        return files, durations

    async def _generate_edge_tts(self, pieces: List[Dict], output_dir: Path,
                                 indexes: Optional[List[int]] = None) -> List[Path]:
        """使用 edge-tts 以自然语速生成配音"""
        import edge_tts

        semaphore = asyncio.Semaphore(self.concurrency)

        async def generate_one(idx: int, text: str) -> Path:
            # 注意：edge-tts 实际输出 mp3 格式（ffmpeg 会自动识别，扩展名不影响使用）
            output_path = output_dir / f"tts_{idx:04d}.mp3"
            async with semaphore:
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        communicate = edge_tts.Communicate(
                            text, self.voice,
                            rate=self.rate,
                            volume=self.volume,
                            pitch=self.pitch,
                        )
                        await communicate.save(str(output_path))
                        # 校验确实生成了音频内容（空文件说明服务端未返回音频）
                        if not output_path.exists() or output_path.stat().st_size == 0:
                            raise RuntimeError("服务端未返回音频数据")
                        return output_path
                    except Exception as e:
                        if attempt == MAX_RETRIES:
                            raise
                        wait = 2 ** attempt
                        print(f"[TTS] 第 {idx} 条生成失败 (第 {attempt}/{MAX_RETRIES} 次): {e}，{wait}s 后重试...")
                        await asyncio.sleep(wait)

        indexes = indexes or list(range(len(pieces)))
        tasks = [generate_one(idx, piece["text"])
             for idx, piece in zip(indexes, pieces)]
        # 任一条最终失败即取消其余任务并抛出异常，不让它们继续在后台请求 API
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            for t in tasks:
                t.cancel()
            raise errors[0]
        print(f"[TTS] edge-tts 配音生成完成")
        return results

    async def _generate_mimo_voiceclone(self, pieces: List[Dict], output_dir: Path,
                                        indexes: Optional[List[int]] = None) -> List[Path]:
        """使用 MiMo mimo-v2.5-tts-voiceclone 基于参考人声音频克隆音色生成配音"""
        semaphore = asyncio.Semaphore(self.concurrency)

        def generate_one_sync(text: str) -> bytes:
            messages = [
                {"role": "user", "content": self.mimo_style_instruction},
                {"role": "assistant", "content": text},
            ]
            response = self.mimo_client.chat.completions.create(
                model=self.mimo_model,
                messages=messages,
                audio={"format": self.mimo_format, "voice": self.mimo_voice_data_uri},
            )
            audio = response.choices[0].message.audio
            if audio is None or not audio.data:
                raise RuntimeError("MiMo TTS 返回内容中没有音频数据")
            return base64.b64decode(audio.data)

        async def generate_one(idx: int, text: str) -> Path:
            output_path = output_dir / f"tts_{idx:04d}.{self.mimo_format}"
            async with semaphore:
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        audio_bytes = await asyncio.to_thread(generate_one_sync, text)
                        output_path.write_bytes(audio_bytes)
                        if not output_path.exists() or output_path.stat().st_size == 0:
                            raise RuntimeError("服务端未返回音频数据")
                        return output_path
                    except Exception as e:
                        if attempt == MAX_RETRIES:
                            raise
                        wait = 2 ** attempt
                        print(f"[TTS] 第 {idx} 条生成失败 (第 {attempt}/{MAX_RETRIES} 次): {e}，{wait}s 后重试...")
                        await asyncio.sleep(wait)

        indexes = indexes or list(range(len(pieces)))
        tasks = [generate_one(idx, piece["text"])
             for idx, piece in zip(indexes, pieces)]
        # 任一条最终失败即取消其余任务并抛出异常，不让它们继续在后台请求 API
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            for t in tasks:
                t.cancel()
            raise errors[0]
        print(f"[TTS] mimo-voiceclone 配音生成完成")
        return results


def mix_tts_audio(clips: List[Dict], output_audio: Path):
    """将各配音片段按最终时间轴混入完整音轨（numpy 实现，替代 amix 滤镜）。

    clips 为 build_layout 的输出，1 条 = 1 段配音音频。
    """
    if not clips:
        return

    clips = sorted(clips, key=lambda c: c["start_ms"])
    audio_clips = [c for c in clips if c.get("file")]

    decoded = []
    total_samples = SAMPLE_RATE  # 至少留 1 秒尾部
    for clip in audio_clips:
        pcm = _decode_to_pcm16(clip["file"])
        offset = int(clip["start_ms"]) * SAMPLE_RATE // 1000
        decoded.append([offset, pcm])
        total_samples = max(total_samples, offset + len(pcm) + SAMPLE_RATE)

    # 时间轴按原句起点硬锚定，配音达到最高倍速仍超长时会压到下一句起点，
    # 这里统一截断并淡出，保证不与下一句语音重叠。
    FADE_SAMPLES = int(SAMPLE_RATE * 0.01)  # 10ms 淡出
    truncated = 0
    for i in range(len(decoded)):
        offset, pcm = decoded[i]
        if i + 1 < len(decoded):
            max_len = decoded[i + 1][0] - offset
        else:
            max_len = len(pcm)
        if len(pcm) > max_len:
            pcm = pcm[:max(max_len, 1)].copy()
            fade = min(len(pcm), FADE_SAMPLES)
            if fade > 0:
                ramp = np.linspace(1.0, 0.0, fade)
                pcm[len(pcm)-fade:] = (pcm[len(pcm)-fade:] * ramp).astype(np.int16)
            decoded[i][1] = pcm
            truncated += 1
    if truncated:
        print(f"[音频] 有 {truncated} 条配音超出下一句开始位置，已截断以避免语音重叠")

    master = np.zeros(total_samples, dtype=np.int16)
    for offset, pcm in decoded:
        seg = master[offset:offset + len(pcm)].astype(np.int32) + pcm.astype(np.int32)
        master[offset:offset + len(pcm)] = np.clip(seg, -32768, 32767).astype(np.int16)

    with wave.open(str(output_audio), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(master.tobytes())

    print(f"[音频] 已拼接 {len(audio_clips)} 条配音: {output_audio}")


def _decode_to_pcm16(path: Path) -> np.ndarray:
    """使用 miniaudio 将音频解码为单声道 s16le PCM，返回 numpy 数组（异常时回退 ffmpeg）。"""
    sr = SAMPLE_RATE

    try:
        decoded = miniaudio.decode_file(
            str(path),
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=sr,
        )
        return np.frombuffer(decoded.samples, dtype=np.int16)
    except Exception:
        # 回退 ffmpeg 单文件解码
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(path),
               "-ac", "1", "-ar", str(sr), "-f", "s16le", "pipe:1"]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"音频解码失败: {path}\n{result.stderr.decode(errors='ignore')}")
        return np.frombuffer(result.stdout, dtype=np.int16)


def build_layout(pieces: List[Dict], tts_files: Optional[List[Path]],
                 max_tempo: float, min_tempo: float, chars_per_sec: float,
                 known_durations: Optional[List[Optional[int]]] = None
                 ) -> List[Dict]:
    """为字幕计算最终时间轴（字幕与配音共用同一套时间），1 句 = 1 条。

    采用**硬锚定**排布：每句的起始时间严格等于原句起始时间，绝不因为上一句
    配音超长而顺延，从根本上杜绝音画滞后随视频推进而累积。

    单句时间预算 = 本句起点 → 下一句起点（含句间静音间隙）：
    - 配音放得下：保持原速，或在 [min_tempo, 1.0) 内轻微放慢填充空白；
    - 配音放不下：在 max_tempo 以内加速；仍放不下时不再顺延，由混音阶段
      在下一句起点处淡出截断（翻译阶段的字数预算已从源头抑制这种情况）。

    无音频（--no-tts / --no-video 模式）时按 chars_per_sec 估算时长排布。

    返回 [{text, start_ms, end_ms, file, tempo}, ...]，按时间升序。
    """
    est_ms_per_char = 1000.0 / max(chars_per_sec, 0.1)

    durations = []
    for idx, piece in enumerate(pieces):
        dur = None
        if known_durations is not None and idx < len(known_durations):
            dur = known_durations[idx]
        elif tts_files:
            dur = probe_duration_ms(tts_files[idx])
        if dur is None:
            dur = len(piece["text"]) * est_ms_per_char
        durations.append(max(int(dur), 200))  # 单句最短 200ms，防异常数据

    layout = []
    sped_count = 0
    slowed_count = 0
    overflow_count = 0
    max_tempo_seen = 1.0
    min_tempo_seen = 1.0

    n = len(pieces)
    for i, piece in enumerate(pieces):
        start = piece["start_ms"]
        dur = durations[i]

        if i + 1 < n:
            avail = max(pieces[i + 1]["start_ms"] - start, 0)
        else:
            avail = max(piece["end_ms"] - start, dur)

        tempo = 1.0
        if dur > avail > 0:
            raw = dur / avail
            tempo = min(raw, max_tempo)
            if raw > max_tempo:
                overflow_count += 1
                print(f"[警告] {ms_to_srt_time(start)} 起的字幕偏长："
                      f"配音需 {dur}ms / 可用 {avail}ms，已按最高 {max_tempo}x 加速，"
                      f"超出部分将在下一句开始处淡出截断")
        elif avail > 0 and dur < avail:
            tempo = max(dur / avail, min_tempo)

        if tempo > 1.005:
            sped_count += 1
            max_tempo_seen = max(max_tempo_seen, tempo)
        elif tempo < 0.995:
            slowed_count += 1
            min_tempo_seen = min(min_tempo_seen, tempo)

        adj = max(int(dur / tempo), 100)  # 保底 100ms 防零时长
        end = start + (min(adj, avail) if avail > 0 else adj)

        layout.append({
            "text": piece["text"],
            "start_ms": start,
            "end_ms": max(end, start + 100),
            "file": tts_files[i] if tts_files else None,
            "tempo": tempo,
        })

    if sped_count:
        print(f"[时间轴] {sped_count} 条配音已加速填充时间槽（最高 {max_tempo_seen:.2f}x）")
    if slowed_count:
        print(f"[时间轴] {slowed_count} 条配音已放慢填充时间槽（最低 {min_tempo_seen:.2f}x）")
    if overflow_count:
        print(f"[时间轴] {overflow_count} 条配音达到最高倍速仍超长，将被截断"
              f"（可调低 subtitle.chars_per_sec 让译文更精简）")

    if tts_files:
        fitted_cache: Dict[Path, Path] = {}
        for entry in layout:
            f = entry["file"]
            if f and abs(entry["tempo"] - 1.0) > 0.005:
                if f not in fitted_cache:
                    fitted_cache[f] = speed_up_audio(f, entry["tempo"])
                entry["file"] = fitted_cache[f]

    return layout


_HAS_RUBBERBAND: Optional[bool] = None


def has_rubberband_filter() -> bool:
    """检测当前 ffmpeg 是否编译了 rubberband 滤镜（需要 --enable-librubberband）。

    结果做进程内缓存，避免每次变速都重新探测。
    """
    global _HAS_RUBBERBAND
    if _HAS_RUBBERBAND is None:
        try:
            result = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                                    capture_output=True, text=True)
            _HAS_RUBBERBAND = "rubberband" in (result.stdout or "")
        except Exception:
            _HAS_RUBBERBAND = False
    return _HAS_RUBBERBAND


def _atempo_chain_filter(tempo: float) -> str:
    """构造 atempo 滤镜表达式。

    ffmpeg 的 atempo 单级只支持 0.5~2.0 倍速，超出范围需要链式串联多个 atempo
    （如 3.0x 需拆成 2.0x * 1.5x）。调用方的 min_tempo 已下限到 0.5，故只需处理加速。
    """
    if tempo <= 2.0:
        return f"atempo={tempo:.4f}"

    stages = []
    remaining = tempo
    while remaining > 2.0:
        stages.append(2.0)
        remaining /= 2.0
    stages.append(remaining)
    return ",".join(f"atempo={s:.4f}" for s in stages)


def speed_up_audio(path: Path, tempo: float) -> Path:
    """生成指定倍速的临时音频。

    优先使用 ffmpeg 的 rubberband 滤镜（相位声码器变速，音质明显优于 atempo，
    尤其在较大倍速时不易出现明显的音色失真/齿音），仅当 ffmpeg 未编译该滤镜
    或调用失败时才回退到 atempo（超出 0.5~2.0 范围时自动链式串联）。
    """
    tmp_path = path.with_suffix(".fitted" + path.suffix)

    if has_rubberband_filter():
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(path),
               "-filter:a", f"rubberband=tempo={tempo:.4f}", str(tmp_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and tmp_path.exists():
            return tmp_path
        print(f"[警告] rubberband 变速失败，回退 atempo: {result.stderr}", file=sys.stderr)
        tmp_path.unlink(missing_ok=True)

    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(path),
           "-filter:a", _atempo_chain_filter(tempo), str(tmp_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0 and tmp_path.exists():
        return tmp_path
    else:
        tmp_path.unlink(missing_ok=True)
        print(f"[警告] 音频变速失败，保留原始配音: {result.stderr}", file=sys.stderr)
        return path


def _escape_filter_path(p: Path) -> str:
    r"""转义并包裹 FFmpeg 滤镜参数中的路径。

    处理 Windows 盘符冒号（C:\）、反斜杠、空格与单引号：
    1. 优先转换为相对于当前工作目录的 POSIX 相对路径，消除盘符冒号转义的复杂度；
    2. 若无法获取相对路径（如跨盘符），转换为 POSIX 绝对路径，并将盘符冒号转义为 '\:'；
    3. 用单引号包裹路径，并将路径内部的单引号转义为 '\''。
    """
    p_abs = p.resolve()
    try:
        rel = p_abs.relative_to(Path.cwd().resolve())
        posix_str = rel.as_posix()
    except ValueError:
        posix_str = p_abs.as_posix()
        if len(posix_str) > 1 and posix_str[1] == ":":
            posix_str = posix_str[0] + r"\:" + posix_str[2:]

    escaped = posix_str.replace("'", r"'\''")
    return f"'{escaped}'"


_NVENC_AVAILABLE: Optional[bool] = None


def has_nvenc() -> bool:
    """检测当前 ffmpeg 是否支持 NVIDIA h264_nvenc 硬件编码器（结果缓存）。

    仅查编码器列表还不够（有些构建列出了但显卡/驱动不可用），
    故再跑一次极短的空转编码确认真正可用。
    """
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE

    _NVENC_AVAILABLE = False
    if not shutil.which("ffmpeg"):
        return False
    try:
        listed = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if "h264_nvenc" not in (listed.stdout or ""):
            return False
        # 用 1 帧测试图真正编码一次，确认驱动/显卡可用。
        # 尺寸不能太小：NVENC 对 H.264 有最小分辨率限制，过小的测试图会直接失败
        probe = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "nullsrc=s=1280x720",
             "-frames:v", "1", "-pix_fmt", "yuv420p",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        _NVENC_AVAILABLE = probe.returncode == 0
    except Exception:
        _NVENC_AVAILABLE = False
    return _NVENC_AVAILABLE


def _video_encode_args(use_nvenc: bool) -> List[str]:
    """返回视频编码参数：优先 NVENC 硬件加速，否则回退 CPU libx264。"""
    if use_nvenc:
        # NVENC 用 -cq 控制质量（等价于 libx264 的 -crf），p5 约等于 fast
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "23"]
    return ["-c:v", "libx264", "-preset", "fast", "-crf", "23"]


def compose_final_video(video_path: Path, srt_path: Optional[Path],
                        audio_path: Optional[Path], output_path: Path,
                        mix_with_original: bool = False,
                        allow_nvenc: bool = True) -> bool:
    """一步完成字幕烧录（可选）和配音替换/混合（可选），只做一次视频转码"""
    if not shutil.which("ffmpeg"):
        print("[警告] 未找到 ffmpeg，跳过最终合成")
        return False

    # 需要重新编码视频（烧录字幕）时才涉及编码器选择；纯替换配音走 copy
    use_nvenc = allow_nvenc and bool(srt_path) and has_nvenc()
    if bool(srt_path):
        if use_nvenc:
            print("[FFmpeg] 检测到 NVIDIA NVENC，使用 GPU 硬件加速编码 (h264_nvenc)")
        else:
            print("[FFmpeg] 使用 CPU 软件编码 (libx264)"
                  + ("" if allow_nvenc else "（已通过 --no-nvenc 禁用 GPU）"))
    video_args = _video_encode_args(use_nvenc)

    sub_style = "FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Shadow=0,MarginV=1"
    sub_filter = (f"subtitles={_escape_filter_path(srt_path)}:force_style='{sub_style}'"
                  if srt_path else "")
    cmd = ["ffmpeg", "-y", "-i", str(video_path)]

    if srt_path and audio_path:
        # 字幕 + 配音一起处理：视频走滤镜烧录，音频来自配音文件
        cmd.extend(["-i", str(audio_path)])
        if mix_with_original:
            # 保留原音，按权重混合（weights 含空格，必须加单引号）
            fc = (f"[0:v]{sub_filter}[v];"
                  f"[0:a][1:a]amix=inputs=2:duration=first:normalize=0:weights='0.3 0.7'[a]")
            cmd.extend(["-filter_complex", fc, "-map", "[v]", "-map", "[a]"])
        else:
            fc = f"[0:v]{sub_filter}[v]"
            cmd.extend(["-filter_complex", fc, "-map", "[v]", "-map", "1:a"])
        cmd.extend([*video_args, "-c:a", "aac", "-b:a", "192k"])
    elif srt_path:
        # 只烧录字幕
        cmd.extend(["-vf", sub_filter,
                    *video_args, "-c:a", "aac", "-b:a", "192k"])
    elif audio_path:
        # 只替换配音（--no-burn），无需重编码视频。
        # 不加 -shortest：配音比视频短时会把视频尾部截掉，宁可尾部留白静音
        cmd.extend(["-i", str(audio_path),
                    "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k"])
    else:
        print("[警告] 没有需要合成的内容")
        return False

    cmd.append(str(output_path))

    print("[FFmpeg] 正在合成最终视频...")
    total_ms = probe_duration_ms(video_path)
    code, tail = _stream_subprocess(
        cmd, "FFmpeg", total_duration_s=total_ms / 1000 if total_ms else None)

    if code != 0:
        print(f"[FFmpeg 错误] 输出尾部:\n{tail}", file=sys.stderr)
        return False

    print(f"[FFmpeg] 最终视频完成: {output_path}")
    return True


# ==================== 翻译阶段 ====================

def translate_sentences(llm_config: Dict, sentences: List[Dict],
                        title: str, description: str,
                        output_dir: Path, video_id: str,
                        chars_per_sec: float
                        ) -> Tuple[List[str], Optional[str]]:
    """翻译阶段：提取术语表 → 逐句翻译（带字数预算与滑动窗口）→ 翻译标题。

    所有中间结果写入 {video_id}_translations.json，中断后重跑自动断点续传。
    返回 (译文列表, 中文标题或 None)。
    """
    llm = LLMClient(llm_config)
    cache_path = output_dir / f"{video_id}_translations.json"
    source_sha = hashlib.sha1(
        "\n".join(s["text"] for s in sentences).encode("utf-8")
    ).hexdigest()

    # 提取全片统一术语表（一次调用；结果写入缓存供断点续传复用）
    g = (read_json_safe(cache_path) or {}).get("glossary")
    if (isinstance(g, list) and g
            and all(isinstance(x, dict) and x.get("term") and x.get("zh") for x in g)):
        glossary = g
        print(f"[术语表] 从缓存恢复 {len(glossary)} 条词条")
    else:
        try:
            glossary = llm.extract_glossary(sentences)
            print(f"[术语表] 提取到 {len(glossary)} 条词条")
            update_json_cache(cache_path, glossary=glossary)
        except Exception as e:
            print(f"[警告] 术语表提取失败，将以无术语表模式继续: {e}", file=sys.stderr)
            glossary = []

    def save_cache(res: Dict[int, str]):
        """把当前进度写入缓存（未完成的句子存为 null，便于断点续传）"""
        update_json_cache(
            cache_path,
            model=llm.model,
            source_sha=source_sha,
            translations=[res.get(i) for i in range(len(sentences))],
        )

    done_map: Dict[int, str] = {}
    cache = read_json_safe(cache_path)
    if (cache
            and cache.get("model") == llm.model
            and cache.get("source_sha") == source_sha
            and isinstance(cache.get("translations"), list)):
        for i, t in enumerate(cache["translations"][:len(sentences)]):
            if isinstance(t, str) and t.strip():
                done_map[i] = t
        if done_map:
            print(f"[翻译] 从缓存恢复 {len(done_map)}/{len(sentences)} 句译文: {cache_path}")

    if len(done_map) == len(sentences):
        translations = [done_map[i] for i in range(len(sentences))]
        print("[翻译] 命中完整缓存，跳过 API 调用")
    else:
        budgets = compute_char_budgets(sentences, chars_per_sec)
        translations = llm.translate(sentences, title, budgets,
                                     done=done_map, on_progress=save_cache,
                                     glossary=glossary)
        save_cache({i: t for i, t in enumerate(translations)})
        print(f"[翻译] 结果已缓存: {cache_path}")

    # 取前面若干句译文拼成摘要，供标题改写时理解视频实际内容
    summary_source = [t for t in translations if t] or [s["text"] for s in sentences]
    content_summary = " ".join(summary_source[:30])[:600]

    title_zh = translate_title_cached(llm, title, description, cache_path, content_summary)

    for i, t in enumerate(translations[:3]):
        print(f"  译{i+1}: {t[:60]}...")

    return translations, title_zh


def translate_title_cached(llm: "LLMClient", title: str, description: str,
                           cache_path: Path, content_summary: str = "") -> Optional[str]:
    """翻译/改写视频标题（用于最终视频文件命名），结果写入缓存，失败返回 None。"""
    cached = read_json_safe(cache_path) or {}
    t = cached.get("title_zh")
    if isinstance(t, str) and t.strip():
        return t.strip()

    try:
        title_zh = llm.translate_title(title, description, content_summary)
    except Exception as e:
        print(f"[警告] 标题翻译失败，将使用原标题命名: {e}", file=sys.stderr)
        return None

    if title_zh:
        update_json_cache(cache_path, title_zh=title_zh)
        print(f"[标题] 中文标题: {title_zh}")
    return title_zh


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="YouTube 视频自动下载 + 中文字幕生成 + 中文配音")
    parser.add_argument("url", help="YouTube 视频 URL 或 11位视频 ID")
    parser.add_argument("-o", "--output", default="./youtube_downloads", help="根输出目录")
    parser.add_argument("--no-video", action="store_true", help="只下载字幕，不下载视频")
    parser.add_argument("--no-tts", action="store_true", help="跳过中文配音")
    parser.add_argument("--no-nvenc", action="store_true",
                        help="禁用 NVIDIA GPU 硬件编码，强制使用 CPU (libx264)")
    parser.add_argument("--skip-download", action="store_true",
                        help="跳过视频/字幕/封面下载，直接使用本地已有的文件")
    parser.add_argument("--redo-translate", action="store_true",
                        help="清除 LLM 翻译缓存，强制重新翻译")
    parser.add_argument("--redo-tts", action="store_true",
                        help="清除 TTS 配音缓存，强制重新生成配音")

    args = parser.parse_args()

    # 加载配置（固定读取脚本同目录下的 config.json）
    config = load_config()

    llm_config = config["llm"]
    tts_config = config["tts"]
    sub_config = config["subtitle"]

    # 用视频 ID 作为输出目录
    video_id = extract_video_id(args.url)
    output_dir = Path(args.output) / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[输出目录] {output_dir}")

    try:
        # 清除指定步骤的缓存
        if args.redo_translate:
            cache_trans_path = output_dir / f"{video_id}_translations.json"
            if cache_trans_path.exists():
                cache_trans_path.unlink()
                print(f"[缓存] 已清除翻译缓存: {cache_trans_path.name}")

        if args.redo_tts:
            tts_cache_path = output_dir / "tts_cache.json"
            if tts_cache_path.exists():
                tts_cache_path.unlink()
            for f in output_dir.glob("tts_*.*"):
                f.unlink(missing_ok=True)
            print(f"[缓存] 已清除 TTS 配音缓存")

        # 步骤 1: 获取元数据 + 检测语言
        print("=" * 60)
        meta_file = output_dir / "metadata.json"
        if args.skip_download:
            print("步骤 1: 读取本地视频元数据")
            print("=" * 60)
            if meta_file.exists():
                metadata = read_json_safe(meta_file) or {}
                print(f"[元数据] 已从本地 {meta_file.name} 加载")
            else:
                try:
                    metadata = get_video_metadata(args.url)
                    with open(meta_file, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"[警告] 无法联网获取元数据，使用默认元数据: {e}")
                    metadata = {"id": video_id, "title": video_id, "description": ""}
        else:
            print("步骤 1: 获取视频元数据")
            print("=" * 60)
            metadata = get_video_metadata(args.url)
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        title = metadata.get("title", "Unknown")
        description = metadata.get("description", "")

        # 按优先级列出字幕语言候选，由 yt-dlp 自动跳过不存在的轨道
        sub_langs = detect_sub_langs(metadata)
        print(f"[语言] 字幕语言候选: {sub_langs}")

        # 提前初始化，避免 --no-video 模式下引用未定义变量导致 NameError
        video_path = None
        thumbnail_path = None

        # 步骤 2: 下载视频和字幕 / 复用本地文件
        print("\n" + "=" * 60)
        if args.skip_download:
            print("步骤 2: 免下载模式，直接读取本地视频和字幕")
            print("=" * 60)
            sub_path = find_downloaded_sub(output_dir, video_id, sub_langs)
            if sub_path is None:
                json3_files = list(output_dir.glob("*.json3"))
                if json3_files:
                    sub_path = json3_files[0]
            if sub_path is None:
                raise SystemExit(
                    f"[错误] 未在 {output_dir} 找到 .json3 字幕文件，无法继续。")

            local_video, thumbnail_path = find_local_media(output_dir, video_id)
            if not args.no_video:
                video_path = local_video
                if video_path is None:
                    print(f"[警告] 未在 {output_dir} 找到原始视频文件（需命名为 {video_id}.mp4 等），"
                          "最终将只生成 SRT 字幕。")

            print(f"[复用本地] 字幕: {sub_path}")
            if video_path:
                print(f"[复用本地] 视频: {video_path}")
            if thumbnail_path:
                print(f"[复用本地] 封面: {thumbnail_path}")
        else:
            print("步骤 2: 下载视频和 JSON3 字幕")
            print("=" * 60)
            video_path, sub_path, thumbnail_path = download_video_and_subs(
                args.url, output_dir, sub_langs, metadata, skip_video=args.no_video)

        # 步骤 3: 解析 JSON3
        print("\n" + "=" * 60)
        print("步骤 3: 解析词级字幕")
        print("=" * 60)
        words = parse_json3(sub_path)

        # 步骤 4: 按标点分句
        print("\n" + "=" * 60)
        print("步骤 4: 按标点分句")
        print("=" * 60)
        sentences = split_into_sentences(words)
        sentences = merge_short_sentences(
            sentences,
            min_sentence_sec=sub_config["min_sentence_sec"],
            max_merge_chars=sub_config["max_merge_chars"],
        )

        for i, s in enumerate(sentences[:3]):
            print(f"  句{i+1}: [{ms_to_srt_time(s['start_ms'])}] {s['text'][:60]}...")

        # 步骤 5: LLM 翻译（术语表 + 逐句翻译 + 标题翻译，带断点续传缓存）
        print("\n" + "=" * 60)
        print("步骤 5: LLM 翻译")
        print("=" * 60)
        translations, title_zh = translate_sentences(
            llm_config, sentences, title, description, output_dir, video_id,
            chars_per_sec=sub_config["chars_per_sec"])

        # 步骤 6: 组装配音单元（1 句 = 1 译文 = 1 段配音 = 1 条字幕）
        print("\n" + "=" * 60)
        print("步骤 6: 组装中文字幕单元")
        print("=" * 60)
        pieces = postprocess_subtitles(sentences, translations)

        # 步骤 7: 合成配音 + 按真实语音时长排布时间轴 + 生成 SRT
        print("\n" + "=" * 60)
        print("步骤 7: 中文配音与时间轴排布")
        print("=" * 60)
        tts = None
        tts_files = None
        tts_durations = None
        mixed_audio = None
        clips = []
        cleanup_tts_cache = False
        try:
            # 配音合成仅在需要产出视频时进行；--no-video 模式只输出估算时间轴的 SRT
            if not args.no_video and not args.no_tts and tts_config["enabled"]:
                tts = TTSClient(tts_config, llm_config)
                # 自然语速合成全部配音（此时还没有最终时间轴）
                tts_files, tts_durations = asyncio.run(
                    tts.generate_all(pieces, output_dir))

            # 有实测时长则按真实语音排布（字幕与配音天然同步）；
            # 否则按 chars_per_sec 估算排布
            clips = build_layout(pieces, tts_files,
                                 max_tempo=tts_config["max_tempo"],
                                 min_tempo=tts_config["min_tempo"],
                                 chars_per_sec=sub_config["chars_per_sec"],
                                 known_durations=tts_durations)
            srt_path = output_dir / f"{video_id}_zh.srt"
            # 字幕切分只影响展示；混音仍用整句 clips
            generate_srt(
                split_clips_for_display(
                    clips,
                    sub_config["max_chars_per_line"],
                    sub_config["min_caption_ms"],
                ),
                srt_path,
            )

            if tts_files:
                mixed_audio = output_dir / f"{video_id}_zh_dub.wav"
                mix_tts_audio(clips, mixed_audio)

            # 步骤 8: 合成最终视频（字幕烧录 + 中文配音一步完成，只做一次视频转码）
            final_path = None
            if video_path and not args.no_video:
                print("\n" + "=" * 60)
                print("步骤 8: 合成最终视频（字幕烧录 + 中文配音）")
                print("=" * 60)
                final_name = sanitize_filename(title_zh or title) or video_id
                final_path = output_dir / f"{final_name}.mp4"
                print(f"[输出] 最终视频命名: {final_name}.mp4")
                ok = compose_final_video(
                    video_path,
                    srt_path,
                    mixed_audio,
                    final_path,
                    mix_with_original=bool(tts and tts.mix_with_original),
                    allow_nvenc=not args.no_nvenc,
                )
                if not ok:
                    final_path = None
                else:
                    print(f"[完成] 最终成品: {final_path}")
                    cleanup_tts_cache = (final_path.exists()
                                         and final_path.stat().st_size > 0)
        finally:
            fitted_files = {
                entry["file"] for entry in clips
                if entry.get("file") and entry["file"] not in (tts_files or [])
            }
            for f in fitted_files:
                f.unlink(missing_ok=True)
            if cleanup_tts_cache and tts_files:
                for f in tts_files:
                    f.unlink(missing_ok=True)
                (output_dir / "tts_cache.json").unlink(missing_ok=True)
            if cleanup_tts_cache and mixed_audio:
                mixed_audio.unlink(missing_ok=True)

        # 完成
        print("\n" + "=" * 60)
        print("全部完成！")
        print("=" * 60)
        print(f"字幕文件: {srt_path}")
        if video_path:
            print(f"视频文件: {video_path}")
        if thumbnail_path:
            print(f"封面图片: {thumbnail_path}")
        if final_path:
            print(f"最终视频: {final_path}")

    except Exception as e:
        print(f"\n[错误] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
