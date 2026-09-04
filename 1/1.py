#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube 视频自动下载 + 词级字幕翻译 + 纯中文 SRT 字幕生成 + 中文配音

功能：
1. 从视频元数据生成字幕语言候选列表（由 yt-dlp 自动匹配实际存在的轨道）
2. 下载视频 + JSON3 词级字幕（输出目录为视频ID）
3. 解析字幕 → 按标点分句 → 记录每句首尾词时间
4. 调用 LLM 整篇翻译（带标题/简介上下文 + 统一术语表）
5. 时间对齐 → 生成纯中文 SRT 字幕
6. 中文配音：使用 edge-tts；按实测语音时长自动排布字幕与配音的时间轴
7. 字幕烧录 + 配音一步合成最终视频（--no-tts 可跳过配音）
8. 翻译结果本地缓存（逐批保存），中断后重跑自动断点续传
"""

import argparse
import asyncio
import hashlib
import json
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
        "target_chars_per_sec": 5.5   # 中文配音目标语速（字/秒），用于按原句时长估算译文字数预算
    },
    "llm": {
        "base_url": "https://token-plan-cn.xiaomimimo.com/v1",  # LLM API 的 base_url（会自动补全 /v1 后缀）
        "api_key": "",                     # LLM API Key，必填，否则脚本启动时会报错退出
        "model": "mimo-v2.5-pro",          # 使用的模型名称
        "supports_system_role": False,     # 模型是否支持独立的 system 角色消息（不支持则合并进 user 消息）
        "thinking": {
            "type": "disabled"             # 是否开启模型的思考/推理模式（部分模型支持，disabled 为关闭）
        },
        "batch_size": 40,          # 每批次最多翻译的句子数量
        "batch_max_chars": 8000    # 每批次原文字符数上限（与 batch_size 双重约束，防止单批过长）
    },
    "tts": {
        "enabled": True,                   # 是否生成中文配音（关闭则只输出估算时间轴的字幕）
        "engine": "edge-tts",              # TTS 引擎，目前仅支持 edge-tts
        "voice": "zh-CN-YunyangNeural",     # 配音音色
        "rate": "+0%",                     # 语速调整（相对默认语速的百分比）
        "volume": "+0%",                   # 音量调整（相对默认音量的百分比）
        "pitch": "+0Hz",                   # 音调调整（相对默认音调的 Hz 偏移）
        "mix_with_original": False,        # 是否保留原声并与配音按比例混合（否则完全替换为配音）
        "batch_size": 50,                  # 每批次并发提交生成的配音条数（用于分批写入缓存）
        "concurrency": 5,                  # 单批内实际并发请求 edge-tts 服务的数量
        "max_tempo": 3.0                   # 配音超长时允许的最高加速倍速（超出此倍速的部分会被截断）
    },
    "subtitle": {
        "max_chars_per_line": 25   # 中文字幕单行最大字符数，超过则按标点拆分为多行
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
    stripped = (word_text or "").rstrip()
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

    # 4. 数字/版本号/小数（如 v1.0.，或 "3." 后紧跟数字 "14"）
    if re.match(r"^\$?v?\d+(?:\.\d+)*\.$", core, re.IGNORECASE):
        if next_word_text:
            next_clean = next_word_text.lstrip()
            if next_clean and (next_clean[0].islower() or next_clean[0].isdigit()):
                return True

    if next_word_text:
        next_clean = next_word_text.lstrip()
        # 如 "3." 后紧跟 "14" (小数切词)
        if next_clean and next_clean[0].isdigit() and core[:-1].isdigit():
            return True

    return False


GLOBAL_CONFIG = DEFAULT_CONFIG.get("global", {})
MAX_RETRIES = int(GLOBAL_CONFIG.get("max_retries", 3))             # 网络 API 最大重试次数
SAMPLE_RATE = int(GLOBAL_CONFIG.get("sample_rate", 48000))         # 拼接音轨的采样率
TARGET_CHARS_PER_SEC = float(GLOBAL_CONFIG.get("target_chars_per_sec", 5.5))  # 中文配音目标语速（字/秒）


def apply_global_config(config: Dict):
    """根据加载的配置动态更新全局常量"""
    global MAX_RETRIES, SAMPLE_RATE, TARGET_CHARS_PER_SEC
    g = config.get("global", {})
    if "max_retries" in g:
        MAX_RETRIES = int(g["max_retries"])
    if "sample_rate" in g:
        SAMPLE_RATE = int(g["sample_rate"])
    if "target_chars_per_sec" in g:
        TARGET_CHARS_PER_SEC = float(g["target_chars_per_sec"])


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
        generate_default_config(config_path)
        print(f"[配置] 未找到配置文件，已生成模板: {config_path}")
        print("[配置] 请编辑该文件填入你的 API Key 后再运行")
        sys.exit(1)

    apply_global_config(config)

    # 验证 LLM 配置
    llm = config.get("llm", {})
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


def generate_default_config(path: Path):
    """生成默认配置文件模板"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)


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
        pass

    # 简易容错处理：移除末尾悬空逗号等
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(cleaned)
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


# ==================== yt-dlp 相关 ====================

def sanitize_filename(name: str, max_len: int = 60) -> str:
    """清理字符串使其可安全用作文件名（去除 Windows 非法字符并限制长度）"""
    cleaned = re.sub(r'[\\/:*?"<>|\r\n\t]', " ", name)
    cleaned = cleaned.strip().rstrip(". ")
    return cleaned[:max_len].strip()


def _stream_subprocess(cmd: List[str], label: str,
                       log_interval_s: Optional[float] = None) -> Tuple[int, str]:
    """运行子进程并实时把合并后的 stdout/stderr 转发到控制台。

    按 \\r 或 \\n 切分逐行打印（兼容 ffmpeg 用 \\r 刷新的进度行）。
    log_interval_s 设置后，控制台日志最多按指定间隔输出一次。
    返回 (returncode, 尾部输出文本)，供失败时报告错误详情。
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    tail: List[str] = []
    buf = ""
    last_log_at = 0.0

    def print_line(line: str, force: bool = False):
        nonlocal last_log_at
        line = line.strip()
        if not line:
            return
        now = time.monotonic()
        if (force or log_interval_s is None
                or now - last_log_at >= log_interval_s):
            print(f"[{label}] {line}")
            last_log_at = now
        tail.append(line)
        if len(tail) > 60:
            del tail[:-60]

    while True:
        chunk = proc.stdout.read1(4096)  # 有多少读多少，保证实时
        if not chunk:
            break
        buf += chunk.decode("utf-8", errors="replace")
        *lines, buf = re.split(r"[\r\n]", buf)
        for line in lines:
            print_line(line)
    if buf.strip():
        print_line(buf, force=True)
    code = proc.wait()
    return code, "\n".join(tail)


def run_yt_dlp(args: List[str], stream: bool = False) -> subprocess.CompletedProcess:
    """运行 yt-dlp 命令。

    stream=True 时实时转发下载日志到控制台（返回值的 stderr 仅含尾部输出）；
    默认静默捕获全部输出（--dump-json 需要解析完整 stdout，必须用默认模式）。
    """
    cmd = ["yt-dlp"] + args
    print(f"[yt-dlp] {' '.join(cmd)}")
    if stream:
        code, tail = _stream_subprocess(cmd, "yt-dlp")
        return subprocess.CompletedProcess(cmd, code, stdout="", stderr=tail)
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(f"[yt-dlp stderr] {result.stderr}", file=sys.stderr)
    return result


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
    """在输出目录查找指定语言的 json3 字幕文件。"""
    matches = []
    for f in sorted(output_dir.iterdir()):
        if f.suffix.lower() != ".json3" or not f.stem.startswith(video_id):
            continue
        if f.stem == f"{video_id}.{language}":
            matches.append(f)
    if not matches:
        return None

    if len(matches) > 1:
        print(f"[下载] 发现 {len(matches)} 个同语言 json3 文件，选用: {matches[0].name}")
    return matches[0]


def download_video_and_subs(url: str, output_dir: Path, sub_langs: str,
                            metadata: Dict) -> Tuple[Path, Path, Optional[Path], Dict]:
    """下载视频和 JSON3 字幕（metadata 由调用方传入，避免重复执行 yt-dlp 获取元数据）"""
    output_dir.mkdir(parents=True, exist_ok=True)

    video_id = metadata["id"]
    title = metadata.get("title", "unknown")

    print(f"[下载] 视频标题: {title}")
    print(f"[下载] 视频ID: {video_id}")
    print(f"[下载] 字幕语言候选: {sub_langs}")

    template = str(output_dir / "%(id)s")

    def build_ytdlp_cmd(sub_flag: str) -> List[str]:
        """构造 yt-dlp 下载命令（自动字幕失败后换 --write-subs 重试）"""
        return [
            "-f", "bestvideo*+bestaudio/best",
            sub_flag,
            "--sub-langs", sub_langs,
            "--sub-format", "json3",
            "--write-thumbnail",
            "--convert-thumbnails", "jpg",
            "-o", template,
            url,
        ]

    result = run_yt_dlp(build_ytdlp_cmd("--write-auto-subs"), stream=True)
    if result.returncode != 0:
        print("[下载] 自动字幕下载失败，尝试手动字幕...")
        result = run_yt_dlp(build_ytdlp_cmd("--write-subs"), stream=True)
        if result.returncode != 0:
            raise RuntimeError(f"字幕下载失败: {result.stderr}")

    video_path = None
    thumbnail_path = None

    for f in output_dir.iterdir():
        if not f.stem.startswith(video_id):
            continue
        # 排除本脚本生成的衍生文件（如 xxx_zh_final.mp4），
        # 避免重跑时把上次的成品当成原视频，导致字幕/配音叠加
        if f.stem[len(video_id):].startswith("_zh"):
            continue
        suffix = f.suffix.lower()
        if suffix in (".mp4", ".webm", ".mkv", ".mov"):
            if video_path is None or len(f.stem) < len(video_path.stem):
                # 优先选择文件名恰好等于视频 ID 的原始下载文件
                video_path = f
        elif suffix in (".jpg", ".jpeg", ".png", ".webp"):
            if thumbnail_path is None or suffix == ".jpg":
                thumbnail_path = f

    sub_path = find_downloaded_sub(output_dir, video_id, sub_langs)
    if video_path is None:
        raise FileNotFoundError(f"未找到下载的视频文件 (ID: {video_id})")
    if sub_path is None:
        # SystemExit 不被 main 的 except Exception 捕获，直接退出且不带堆栈
        raise SystemExit(
            f"[错误] 未找到 json3 字幕文件（语言候选: {sub_langs}）。\n"
            "       该视频可能不提供 json3 格式的字幕，而本脚本依赖词级时间戳，无法继续。")

    print(f"[下载完成] 视频: {video_path}")
    print(f"[下载完成] 字幕: {sub_path}")
    if thumbnail_path:
        print(f"[下载完成] 封面: {thumbnail_path}")

    return video_path, sub_path, thumbnail_path, metadata


# ==================== JSON3 解析 ====================

def parse_json3(json3_path: Path) -> List[Dict]:
    """解析 YouTube JSON3 字幕文件"""
    with open(json3_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = []
    events = data.get("events", [])
    last_word_event = None  # 记录最后一个真实词所在的 event

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
            last_word_event = event

    for i in range(len(words) - 1):
        # 使用下一个真实词的开始时间，避免把换行 event 的时间算成词尾。
        words[i]["end_ms"] = max(words[i + 1]["start_ms"], words[i]["start_ms"])

    if words and last_word_event is not None:
        last_end = (last_word_event.get("tStartMs", 0)
                    + last_word_event.get("dDurationMs", 0))
        words[-1]["end_ms"] = max(last_end, words[-1]["start_ms"])

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
            "words": ws,
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


# ==================== LLM 翻译 ====================

class LLMClient:
    """统一 LLM 客户端"""

    def __init__(self, config: Dict):
        self.model = config["model"]
        self.supports_system_role = config.get("supports_system_role", True)
        self.thinking = config.get("thinking")
        self.batch_size = max(1, int(config.get("batch_size", 40)))
        self.batch_max_chars = max(1, int(config.get("batch_max_chars", 8000)))

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

    def translate(self, sentences: List[Dict], title: str, description: str,
                  done: Optional[Dict[int, str]] = None,
                  on_progress=None,
                  glossary: Optional[List[Dict]] = None) -> List[str]:
        """分批翻译所有句子（结构化 JSON 输出，按 id 对齐，避免错位）。

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
            sub_sentences = [sentences[i] for i in todo]
            prompt = self._build_prompt(sub_sentences, title, description, todo, glossary)

            system_msg = "你是一位专业的视频字幕翻译师。你只输出合法的 JSON 数组，不输出任何其他内容。"
            if self.supports_system_role:
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [{"role": "user", "content": system_msg + "\n\n" + prompt}]

            batch_result: Dict[int, str] = {}
            for attempt in range(1, MAX_RETRIES + 1):
                # 重试时适度提升 temperature（0.3 -> 0.5 -> 0.7），引入变化避免死锁
                temp = 0.3 + (attempt - 1) * 0.2
                content = self._chat(messages, temperature=temp)
                parsed = self._parse_json_translations(
                    content, todo[0], todo[-1] + 1)
                batch_result = {i: parsed[i] for i in todo if i in parsed}
                missing = [i for i in todo if i not in batch_result]
                if not missing:
                    break
                print(f"[警告] 批次 {todo[0]}-{todo[-1]}: 缺少 {len(missing)} 句译文 (id: {missing[:5]}...)，第 {attempt}/{MAX_RETRIES} 次重试...")

            still_missing = [i for i in todo if i not in batch_result]
            if still_missing:
                # 重试后仍失败：硬性中断，不做占位降级
                raise RuntimeError(
                    f"批次 {todo[0]}-{todo[-1]} 有 {len(still_missing)} 句重试后仍翻译失败"
                    f" (id: {still_missing[:5]}...)，已中断。"
                    f"已完成部分已写入缓存，修复后重新运行可从断点继续。"
                )

            results.update(batch_result)
            if on_progress:
                on_progress(results)
            print(f"[翻译] 进度: {len(results)}/{n_total}")

        print(f"[翻译] 完成，共 {len(results)} 句")
        return [results[i] for i in range(n_total)]

    @staticmethod
    def _parse_json_translations(content: str, id_start: int, id_end: int) -> Dict[int, str]:
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
                if zh and id_start <= idx < id_end:
                    result[idx] = zh
        return result

    @staticmethod
    def _build_prompt(sentences: List[Dict], title: str, description: str, ids: List[int],
                      glossary: Optional[List[Dict]] = None) -> str:
        numbered_lines = []
        for sentence_id, sentence in zip(ids, sentences):
            # 按原句时长×目标语速算出字数预算；
            # 极短句（<1秒）按实际时长比例缩减保底字数（如 max(2, round(dur * 5.5))），
            # 避免固定保底 6 字导致配音语速极端过快（例如 0.4 秒读 6 字达 15 字/秒）。
            duration_ms = max(sentence["end_ms"] - sentence["start_ms"], 0)
            duration_sec = duration_ms / 1000.0
            if duration_sec < 1.0:
                budget = max(2, round(duration_sec * TARGET_CHARS_PER_SEC))
            else:
                budget = max(4, round(duration_sec * TARGET_CHARS_PER_SEC))
            est_zh_chars = len(sentence["text"].split()) * 1.8
            if est_zh_chars > budget:
                numbered_lines.append(
                    f"{sentence_id}. [≤{budget}字] {sentence['text']}")
            else:
                numbered_lines.append(f"{sentence_id}. {sentence['text']}")
        numbered_text = "\n".join(numbered_lines)
        description = (description or "").strip() or "（无简介）"

        glossary_block = ""
        if glossary:
            terms = "\n".join(f"- {item['term']} → {item['zh']}" for item in glossary)
            glossary_block = (f"\n统一术语表（下列词条的译文必须严格采用给定译法，不得自行改译）：\n"
                             f"{terms}\n")

        return f"""你是一位专业的视频字幕翻译师。请将以下视频字幕翻译成中文。

视频标题：{title}
视频简介：{description}
{glossary_block}
以下是视频的一段字幕文本，每句前面有编号。请严格逐句翻译，不要合并或拆分句子，不要遗漏任何一句。译文应自然流畅，符合中文表达习惯，适合作为视频字幕。

部分句子编号后带有 [≤N字] 参考字数（由配音时间预算算出，只出现在信息较密的句子上）：请尽量精简、靠近该字数，以免配音语速过快。但忠实与自然优先：若精简会损失关键信息或使中文生硬，可适当超出——绝不可为压字数而遗漏、简化或臆造信息。未标字数的句子按正常翻译即可。

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

        prompt = f"""以下是某视频的全部字幕文本（每行一句）。请提取其中需要在中文翻译里保持前后一致的内容：人名、地名、作品名、组织机构、品牌、专业术语等。

普通词汇和常见词不要提取；没有可提取的内容则输出空数组 []。

输出格式：JSON 数组 [{{"term": "原文", "zh": "统一的中文译法"}}]，不要输出其他任何内容。

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

    def translate_title(self, title: str, description: str = "") -> Optional[str]:
        """翻译视频标题（用于最终视频文件命名），失败返回 None"""
        if not title or title == "Unknown":
            return None
        prompt = (
            "请把下面的视频标题翻译成简洁自然的中文，只输出译文本身，"
            "不要引号、不要解释、不要保留原文。\n"
            f"标题：{title}\n"
        )
        desc = (description or "").strip()
        if desc:
            prompt += f"（视频简介供参考：{desc[:200]}）\n"

        content = self._chat([{"role": "user", "content": prompt}])
        # 去掉可能存在的代码围栏，取第一行，再去掉包裹的引号
        text = strip_code_fence(content)
        text = text.splitlines()[0].strip().strip('"“”').strip()
        return text or None


# ==================== 字幕后处理 ====================

def split_long_sentence(text: str, max_chars: int) -> List[str]:
    """仅在标点符号处把超长文本拆成多条（无标点的超长子句保持完整，不做硬切）。

    带有成对标点保护（《》、“”、‘’、（）等内不切断）与连续省略号（.../……）保护。
    只负责文本切分，不涉及任何时间信息；真实时间轴由后续的
    build_layout 根据 TTS 实测时长决定。
    """
    delimiters = {'，', '、', '；', ',', ';', '。', '！', '？', '…', '!', '?', '.'}

    pair_open = {'“': '”', '‘': '’', '《': '》', '（': '）', '(': ')', '【': '】', '[': ']', '"': '"', "'": "'"}
    pair_close = {'”': '“', '’': '‘', '》': '《', '）': '（', ')': '(', '】': '【', ']': '['}

    # 第一遍：按标点切成子句（带有状态机保护）
    clauses = []
    last_end = 0
    stack: List[str] = []

    i = 0
    while i < len(text):
        char = text[i]

        # 跟踪成对标点开闭
        if char in ('"', "'"):
            if stack and stack[-1] == char:
                stack.pop()
            else:
                stack.append(char)
        elif char in pair_open:
            stack.append(char)
        elif char in pair_close:
            if stack and stack[-1] == pair_close[char]:
                stack.pop()

        in_pair = len(stack) > 0
        is_delimiter = False
        cut_index = i + 1

        if not in_pair and char in delimiters and i > 5:
            # 防护 (1)：数字千分位或小数点 (如 46,000 / 3.14)
            if (char in (',', '，', '.') and i + 1 < len(text)
                    and text[i - 1].isdigit() and text[i + 1].isdigit()):
                is_delimiter = False
            # 防护 (2)：连续英文点/省略号 (如 ... 或 ……)，吃完所有连续点后再断
            elif char in ('.', '…'):
                next_i = i
                while next_i < len(text) and text[next_i] in ('.', '…'):
                    next_i += 1
                # 如果是单个点且紧跟字母/数字（如 Dr. Smith / e.g.），不作为句尾断点
                if next_i - i == 1 and char == '.' and next_i < len(text) and text[next_i].isalnum():
                    is_delimiter = False
                else:
                    is_delimiter = True
                    cut_index = next_i
                    i = next_i - 1  # 游标跳过整个省略号
            else:
                is_delimiter = True

        if is_delimiter:
            clauses.append(text[last_end:cut_index])
            last_end = cut_index

        i += 1

    if last_end < len(text):
        clauses.append(text[last_end:])

    # 第二遍：把子句打包到不超过 max_chars 的片段。
    # 注意：单个子句本身超长且内部没有标点时，不做硬切（保持整句完整）
    parts = []
    buffer = ""
    for clause in clauses:
        if len(buffer) + len(clause) <= max_chars:
            buffer += clause
        else:
            if buffer:
                parts.append(buffer)
            buffer = clause
    if buffer:
        parts.append(buffer)

    # 把纯标点片段并回相邻片段：单独一个标点无法合成语音，会让 TTS 报错
    merged: List[str] = []
    for part in parts:
        if merged and not has_speakable_text(part):
            merged[-1] += part
        else:
            merged.append(part)
    if len(merged) > 1 and not has_speakable_text(merged[0]):
        merged[1] = merged[0] + merged[1]
        del merged[0]

    return merged


def postprocess_subtitles(sentences: List[Dict], translations: List[str],
                          max_chars: int) -> List[Dict]:
    """按句聚合为配音单元（每句一个 TTS 合成单元 + 供屏幕展示的多行拆分）。

    每个单元包含：
    - text: 完整译文，作为 TTS 的合成输入（保留完整语境，修复多音字失去
      上下文导致误读的问题，如把"第一行"拆开会让 TTS 丢失"行"的读音线索）；
    - lines: 供字幕屏幕展示的多行文本（超过 max_chars 时按标点拆分），
      仅用于 SRT 渲染，不参与 TTS 合成；
    - span_start/span_end: 原句时间跨度，真实起止时间由后续 build_layout
      根据整句配音实测时长决定，再按字符比例切分给各展示行。

    纯标点/无实际内容的译文（无法 TTS）会被跳过。
    """
    result = []

    for sent, trans in zip(sentences, translations):
        # 跳过纯标点/无实际内容的译文（否则 TTS 会报 NoAudioReceived）
        if not has_speakable_text(trans):
            print(f"[后处理] 跳过无可朗读内容的译文: {trans!r}")
            continue

        if len(trans) > max_chars:
            lines = [p.strip() for p in split_long_sentence(trans, max_chars)]
        else:
            lines = [trans]

        result.append({
            "text": trans,
            "lines": lines,
            "span_start": sent["start_ms"],
            "span_end": sent["end_ms"],
        })

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


# ==================== 最终合成（字幕烧录 + 配音） ====================


# ==================== 中文配音（TTS） ====================

class TTSClient:
    """统一 TTS 客户端"""

    def __init__(self, config: Dict):
        self.engine = config.get("engine", "edge-tts")
        self.enabled = config.get("enabled", True)
        self.mix_with_original = config.get("mix_with_original", False)
        self.batch_size = max(1, int(config.get("batch_size", 50)))
        self.concurrency = max(1, int(config.get("concurrency", 5)))
        self.max_tempo = max(1.0, float(config.get("max_tempo", 3.0)))
        cache_config = {key: value for key, value in config.items()
                        if key not in {"api_key"}}
        self.cache_signature = hashlib.sha1(
            json.dumps(cache_config, sort_keys=True,
                       ensure_ascii=False).encode("utf-8")
        ).hexdigest()

        if self.engine != "edge-tts":
            raise ValueError(f"不支持的 TTS 引擎: {self.engine}")
        self.voice = config.get("voice", "zh-CN-YunyangNeural")
        self.rate = config.get("rate", "+0%")
        self.volume = config.get("volume", "+0%")
        self.pitch = config.get("pitch", "+0Hz")
        print(f"[TTS] 引擎: edge-tts, 音色: {self.voice}")

    async def generate_all(self, pieces: List[Dict], output_dir: Path
                           ) -> Tuple[List[Path], List[Optional[int]]]:
        """分批生成并缓存配音，返回与 pieces 对齐的 (音频路径, 时长毫秒)。

        时长在生成时用 ffprobe 测一次并写入 tts_cache.json，
        避免后续 build_layout 再对每条配音重复 fork ffprobe。
        """
        if not self.enabled:
            return [], []

        output_dir.mkdir(parents=True, exist_ok=True)
        extension = "mp3"
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
                json.dump({"signature": self.cache_signature, "entries": entries},
                          f, ensure_ascii=False, indent=1)

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
                batch_files = await self._generate_edge_tts(
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

        result_files: List[Path] = []
        result_durations: List[Optional[int]] = []
        for path, dur in zip(files, durations):
            if path is not None:
                result_files.append(path)
                result_durations.append(dur)
        return result_files, result_durations

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


def mix_tts_audio(clips: List[Dict], output_audio: Path):
    """将各配音片段按最终时间轴混入完整音轨（numpy 实现，替代 amix 滤镜）。

    clips: build_layout 的输出。同一整句若被拆成多行展示，会共用同一个
    音频文件（file 相同）——这里按 file 去重，只在整句的起始时间混入一次，
    避免同一段配音被重复叠加播放。
    """
    if not clips:
        return

    clips = sorted(clips, key=lambda c: c["start_ms"])

    # 按音频文件去重：同一文件只取第一次出现（即整句起始时间）
    seen_files = set()
    audio_clips = []
    for clip in clips:
        f = clip.get("file")
        if not f or f in seen_files:
            continue
        seen_files.add(f)
        audio_clips.append(clip)

    decoded = []
    total_samples = SAMPLE_RATE  # 至少留 1 秒尾部
    for clip in audio_clips:
        pcm = _decode_to_pcm16(clip["file"])
        offset = int(clip["start_ms"]) * SAMPLE_RATE // 1000
        decoded.append([offset, pcm])
        total_samples = max(total_samples, offset + len(pcm) + SAMPLE_RATE)

    # 防止语音重叠的兜底：正常情况下新时间轴不会重叠，仅当某句组触发最高倍速
    # 仍超长时才会发生。超出部分直接截断并淡出。
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

    print(f"[音频] 已拼接 {len(clips)} 条配音: {output_audio}")


def _decode_to_pcm16(path: Path, sample_rate: Optional[int] = None) -> np.ndarray:
    """使用 miniaudio 将音频解码为单声道 s16le PCM，返回 numpy 数组（异常时回退 ffmpeg）。"""
    sr = sample_rate or SAMPLE_RATE

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
                 max_tempo: float = 3.0,
                 known_durations: Optional[List[Optional[int]]] = None
                 ) -> List[Dict]:
    """为字幕计算最终时间轴（字幕与配音共用同一套时间）。

    pieces 中每项对应一句完整译文的 TTS 合成单元（含 text 全句 + lines 展示行）。
    有真实配音时按整句实测时长排布（优先用 known_durations 中生成阶段已缓存
    的时长，否则回退到当场 probe）——句子本身放不下时按 max_tempo 上限加速，
    字幕整体切换时刻与语音完全同步；随后按各展示行的字符数比例，把整句时长
    分摊给屏幕上显示的多行字幕（仅影响显示切换时刻，不影响音频，因为音频是
    整句合成，不再按行拆分，从而保留完整语境供 TTS 消歧多音字）。
    无音频（--no-tts / --no-video 模式）时退化为按 130ms/字 估算排布。

    返回 [{text, start_ms, end_ms, file, tempo}, ...]，按时间升序；
    同句的多行展示项共用同一个 file 与 tempo（音频不重复生成/加速）。
    """
    EST_MS_PER_CHAR = 130  # 无实测时长时的退化估算值

    durations = []
    for idx, piece in enumerate(pieces):
        dur = None
        if known_durations is not None and idx < len(known_durations):
            dur = known_durations[idx]
        elif tts_files:
            dur = probe_duration_ms(tts_files[idx])
        if dur is None:
            dur = len(piece["text"]) * EST_MS_PER_CHAR
        durations.append(max(int(dur), 200))  # 单句最短 200ms，防异常数据

    layout = []
    sped_count = 0
    max_tempo_seen = 1.0

    for i, piece in enumerate(pieces):
        span_start = piece["span_start"]
        span_end = piece["span_end"]
        dur = durations[i]

        # 借用到下一句开始前的静音间隙，给配音更多空间以减少加速
        # （末句无后继，保持自身跨度）
        if i + 1 < len(pieces):
            effective_end = max(span_end, pieces[i + 1]["span_start"])
        else:
            effective_end = span_end

        avail = max(effective_end - span_start, 0)
        tempo = 1.0
        if dur > avail > 0:
            raw = dur / avail
            tempo = min(raw, max_tempo)
            if raw > max_tempo:
                print(f"[警告] {ms_to_srt_time(span_start)} 起的字幕严重超长："
                      f"配音需 {dur}ms / 可用 {avail}ms，"
                      f"已按最高 {max_tempo}x 加速，超出部分可能被截断")

        total_adj = max(int(dur / tempo), 100)  # 保底 100ms 防零时长
        if tempo > 1.005:
            sped_count += 1
            max_tempo_seen = max(max_tempo_seen, tempo)

        # 按字符数比例，把整句配音时长分摊给各展示行（仅影响字幕切换时刻）
        lines = piece.get("lines") or [piece["text"]]
        char_counts = [max(len(line), 1) for line in lines]
        total_chars = sum(char_counts)

        t = span_start
        for k, line in enumerate(lines):
            is_last = (k == len(lines) - 1)
            if is_last:
                line_dur = total_adj - (t - span_start)
            else:
                line_dur = max(int(total_adj * char_counts[k] / total_chars), 1)
            line_dur = max(line_dur, 1)
            layout.append({
                "text": line,
                "start_ms": t,
                "end_ms": t + line_dur,
                "file": tts_files[i] if tts_files else None,
                "tempo": tempo,
            })
            t += line_dur

    if sped_count:
        print(f"[时间轴] {sped_count} 条配音已按所在句子统一倍速加速"
              f"（最高 {max_tempo_seen:.2f}x）")

    # 对需要加速的音频统一应用 atempo（每个 TTS 文件只加速一次，即使对应多行展示）
    if tts_files:
        fitted_cache: Dict[Path, Path] = {}
        for entry in layout:
            f = entry["file"]
            if f and entry["tempo"] > 1.005:
                if f not in fitted_cache:
                    fitted_cache[f] = speed_up_audio(f, entry["tempo"])
                entry["file"] = fitted_cache[f]

    return layout


def speed_up_audio(path: Path, tempo: float) -> Path:
    """用 ffmpeg atempo 生成指定倍速的临时音频。"""
    tmp_path = path.with_suffix(".fitted" + path.suffix)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(path),
           "-filter:a", f"atempo={tempo:.4f}", str(tmp_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0 and tmp_path.exists():
        return tmp_path
    else:
        tmp_path.unlink(missing_ok=True)
        print(f"[警告] 音频加速失败，保留原始配音: {result.stderr}", file=sys.stderr)
        return path


def probe_duration_ms(path: Path) -> Optional[int]:
    """用 ffprobe 探测音频时长（ms），失败返回 None"""
    if not shutil.which("ffprobe"):
        return None
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return int(float(result.stdout.strip()) * 1000)
    except ValueError:
        return None







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
    cmd = ["ffmpeg", "-y", "-i", str(video_path)]

    if srt_path and audio_path:
        # 字幕 + 配音一起处理：视频走滤镜烧录，音频来自配音文件
        sub_filter = f"subtitles={_escape_filter_path(srt_path)}:force_style='{sub_style}'"
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
        sub_filter = f"subtitles={_escape_filter_path(srt_path)}:force_style='{sub_style}'"
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

    print("[FFmpeg] 正在合成最终视频（实时日志如下）...")
    code, tail = _stream_subprocess(cmd, "FFmpeg", log_interval_s=5.0)

    if code != 0:
        print(f"[FFmpeg 错误] 输出尾部:\n{tail}", file=sys.stderr)
        return False

    print(f"[FFmpeg] 最终视频完成: {output_path}")
    return True


# ==================== 翻译阶段 ====================

def translate_sentences(llm_config: Dict, sentences: List[Dict],
                        title: str, description: str,
                        output_dir: Path, video_id: str
                        ) -> Tuple[List[str], Optional[str]]:
    """翻译阶段：提取术语表 → 逐句翻译 → 翻译标题。

    所有中间结果写入 {video_id}_translations.json，中断后重跑自动断点续传。
    返回 (译文列表, 中文标题或 None)。
    """
    llm = LLMClient(llm_config)
    cache_path = output_dir / f"{video_id}_translations.json"
    source_sha = hashlib.sha1(
        "\n".join(s["text"] for s in sentences).encode("utf-8")
    ).hexdigest()

    # 提取全片统一术语表（一次调用；结果写入缓存供断点续传复用）
    cached_data = read_json_safe(cache_path) or {}
    g = cached_data.get("glossary")
    if (isinstance(g, list) and g
            and all(isinstance(x, dict) and x.get("term") and x.get("zh") for x in g)):
        glossary = g
        print(f"[术语表] 从缓存恢复 {len(glossary)} 条词条")
    else:
        try:
            glossary = llm.extract_glossary(sentences)
            print(f"[术语表] 提取到 {len(glossary)} 条词条")
            cached_data["glossary"] = glossary
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cached_data, f, ensure_ascii=False, indent=1)
        except Exception as e:
            print(f"[警告] 术语表提取失败，将以无术语表模式继续: {e}", file=sys.stderr)
            glossary = []

    def save_cache(res: Dict[int, str]):
        """把当前进度写入缓存（未完成的句子存为 null，便于断点续传）。

        写入时保留旧文件中所有非本函数管理的字段（title_zh、glossary 等）。
        """
        sparse = [res.get(i) for i in range(len(sentences))]
        payload = {
            "model": llm.model,
            "source_sha": source_sha,
            "count": len(sentences),
            "complete": all(t is not None for t in sparse),
            "translations": sparse,
        }
        old = read_json_safe(cache_path) or {}
        for key, value in old.items():
            payload.setdefault(key, value)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)

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
        translations = llm.translate(sentences, title, description,
                                     done=done_map, on_progress=save_cache,
                                     glossary=glossary)
        save_cache({i: t for i, t in enumerate(translations)})
        print(f"[翻译] 结果已缓存: {cache_path}")

    title_zh = translate_title_cached(llm, title, description, cache_path)

    for i, t in enumerate(translations[:3]):
        print(f"  译{i+1}: {t[:60]}...")

    return translations, title_zh


def translate_title_cached(llm: "LLMClient", title: str, description: str,
                           cache_path: Path) -> Optional[str]:
    """翻译视频标题（用于最终视频文件命名），结果写入缓存，失败返回 None。"""
    cached = read_json_safe(cache_path) or {}
    t = cached.get("title_zh")
    if isinstance(t, str) and t.strip():
        return t.strip()

    try:
        title_zh = llm.translate_title(title, description)
    except Exception as e:
        print(f"[警告] 标题翻译失败，将使用原标题命名: {e}", file=sys.stderr)
        return None

    if title_zh:
        cache_data = read_json_safe(cache_path) or {}
        cache_data["title_zh"] = title_zh
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=1)
        print(f"[标题] 中文标题: {title_zh}")
    return title_zh


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="YouTube 视频自动下载 + 中文字幕生成 + 中文配音")
    parser.add_argument("url", help="YouTube 视频 URL")
    parser.add_argument("-o", "--output", default="./youtube_downloads", help="根输出目录")
    parser.add_argument("--no-video", action="store_true", help="只下载字幕，不下载视频")
    parser.add_argument("--no-tts", action="store_true", help="跳过中文配音")
    parser.add_argument("--no-nvenc", action="store_true",
                        help="禁用 NVIDIA GPU 硬件编码，强制使用 CPU (libx264)")

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
        # 步骤 1: 获取元数据 + 检测语言
        print("=" * 60)
        print("步骤 1: 获取视频元数据")
        print("=" * 60)
        metadata = get_video_metadata(args.url)
        title = metadata.get("title", "Unknown")
        description = metadata.get("description", "")

        # 按优先级列出字幕语言候选，由 yt-dlp 自动跳过不存在的轨道
        sub_langs = detect_sub_langs(metadata)
        print(f"[语言] 字幕语言候选: {sub_langs}")

        # 提前初始化，避免 --no-video 模式下引用未定义变量导致 NameError
        video_path = None
        thumbnail_path = None

        # 步骤 2: 下载视频和字幕
        print("\n" + "=" * 60)
        print("步骤 2: 下载视频和 JSON3 字幕")
        print("=" * 60)

        if args.no_video:
            subtitle_args = [
                "--write-auto-subs",
                "--sub-langs", sub_langs,
                "--sub-format", "json3",
                "--write-thumbnail",
                "--convert-thumbnails", "jpg",
                "--skip-download",
                "-o", str(output_dir / "%(id)s"),
                args.url,
            ]
            result = run_yt_dlp(subtitle_args, stream=True)
            if result.returncode != 0:
                print("[下载] 自动字幕下载失败，尝试手动字幕...")
                subtitle_args[0] = "--write-subs"
                result = run_yt_dlp(subtitle_args, stream=True)

            sub_path = find_downloaded_sub(output_dir, video_id, sub_langs)
            if sub_path is None:
                raise SystemExit(
                    f"[错误] 未找到 json3 字幕文件（语言候选: {sub_langs}）。\n"
                    "       该视频可能不提供 json3 格式的字幕，而本脚本依赖词级时间戳，无法继续。")
            video_path = None
        else:
            video_path, sub_path, thumbnail_path, _ = download_video_and_subs(args.url, output_dir, sub_langs, metadata)

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

        for i, s in enumerate(sentences[:3]):
            print(f"  句{i+1}: [{ms_to_srt_time(s['start_ms'])}] {s['text'][:60]}...")

        # 步骤 5: LLM 翻译（术语表 + 逐句翻译 + 标题翻译，带断点续传缓存）
        print("\n" + "=" * 60)
        print("步骤 5: LLM 翻译")
        print("=" * 60)
        translations, title_zh = translate_sentences(
            llm_config, sentences, title, description, output_dir, video_id)

        # 步骤 6: 切分中文字幕文本（纯文本规则；真实时间轴由配音实测时长决定）
        print("\n" + "=" * 60)
        print("步骤 6: 切分中文字幕文本")
        print("=" * 60)
        pieces = postprocess_subtitles(
            sentences, translations,
            sub_config["max_chars_per_line"],
        )

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
            if not args.no_video and not args.no_tts:
                tts = TTSClient(tts_config)
                if tts.enabled:
                    # 自然语速合成全部配音（此时还没有最终时间轴）
                    tts_files, tts_durations = asyncio.run(
                        tts.generate_all(pieces, output_dir))

            # 有实测时长则按真实语音排布（字幕与配音天然同步）；
            # 否则退化为按 130ms/字 估算排布
            max_tempo = tts.max_tempo if tts else 3.0
            clips = build_layout(pieces, tts_files, max_tempo,
                                 known_durations=tts_durations)
            srt_path = output_dir / f"{video_id}_zh.srt"
            generate_srt(clips, srt_path)

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
