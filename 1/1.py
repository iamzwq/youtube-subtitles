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
6. 中文配音：支持 edge-tts 或 MiMo TTS；按实测语音时长自动排布字幕与配音的时间轴
7. 字幕烧录 + 配音一步合成最终视频（--no-tts 可跳过配音）
8. 翻译结果本地缓存（逐批保存），中断后重跑自动断点续传
"""

import argparse
import asyncio
import base64
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

from openai import OpenAI


# ==================== 配置加载 ====================

DEFAULT_CONFIG = {
    "llm": {
        "provider": "mimo",
        "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
        "api_key": "",
        "model": "mimo-v2.5-pro",
        "supports_system_role": False,
        "thinking": {
            "type": "disabled"
        },
        "batch_size": 40,
        "batch_max_chars": 8000
    },
    "tts": {
        "enabled": True,
        "engine": "edge-tts",
        "voice": "zh-CN-YunyangNeural",
        "rate": "+0%",
        "volume": "+0%",
        "pitch": "+0Hz",
        "mix_with_original": False,
        "batch_size": 50,
        "concurrency": 5,
        "max_tempo": 3.0
    },
    "subtitle": {
        "max_chars_per_line": 35,
        "min_duration_ms": 800,
        "max_duration_ms": 6000
    }
}

SENTENCE_END_PUNCT = {'.', '!', '?', '。', '！', '？', '…'}

def has_speakable_text(text: str) -> bool:
    """判断文本是否含有可朗读的内容（纯标点/空白/符号的文本无法合成语音）。

    用 Unicode 字母/数字判断，天然覆盖所有语言（拉丁、中日韩、韩文、西里尔等），
    避免手工枚举区段遗漏（如韩文）导致整片句子被误判为不可朗读而丢弃。
    """
    return any(ch.isalnum() for ch in (text or ""))

# 字幕中的非语音提示：[Music]/[Applause]/(upbeat music) 等方括号/圆括号注释，及音符符号。
# 这类内容不是台词，不应翻译或显示，去除后若整句无可朗读内容则丢弃。
NONSPEECH_ANNOTATION_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)|[\u266a\u266b\U0001f3b5\U0001f3b6]")

def strip_nonspeech_annotations(text: str) -> str:
    """移除字幕里的非语音注释（[Music]、(applause)、♪ 等），合并多余空白。"""
    cleaned = NONSPEECH_ANNOTATION_RE.sub(" ", text or "")
    return re.sub(r"\s{2,}", " ", cleaned).strip()

MAX_RETRIES = 3             # 网络 API 最大重试次数
SAMPLE_RATE = 48000         # 拼接音轨的采样率


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

    # 验证 TTS 配置（如果启用）
    tts = config.get("tts", {})
    if tts.get("enabled", True) and tts.get("engine") == "mimo":
        if not tts.get("api_key") and not llm.get("api_key"):
            print("[错误] TTS 使用 mimo 引擎时需要 api_key", file=sys.stderr)
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
    """按标点分句"""
    def make_sentence(ws: List[Dict]) -> Optional[Dict]:
        """把一组词组装成句子；纯标点/非语音注释/无实际内容的片段返回 None 丢弃"""
        text = strip_nonspeech_annotations("".join(w["text"] for w in ws).strip())
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
    for word in words:
        current_words.append(word)
        stripped = word["text"].rstrip()
        if stripped and stripped[-1] in SENTENCE_END_PUNCT:
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
        self.provider = config.get("provider", "openai")
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

        print(f"[LLM] 初始化: {self.provider} @ {base_url}")
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
        print(f"[翻译] 共 {n_total} 句，分批翻译（每批最多 {self.batch_size} 句，"
                      f"{self.batch_max_chars} 字符）...")

        results: Dict[int, str] = dict(done)
        start = 0
        while start < n_total:
            batch_end = start
            batch_chars = 0
            todo = []
            while (batch_end < n_total
                   and batch_end - start < self.batch_size):
                if batch_end not in results:
                    sentence_chars = len(sentences[batch_end]["text"])
                    exceeds_limit = (todo and
                                     batch_chars + sentence_chars
                                     > self.batch_max_chars)
                    if exceeds_limit:
                        break
                    todo.append(batch_end)
                    batch_chars += sentence_chars
                batch_end += 1

            if not todo:
                print(f"[翻译] 批次 {start}-{batch_end - 1}: 全部命中缓存，跳过")
                start = batch_end
                continue

            # 只翻译缺失的句子（保留全局真实编号，避免续传时错位）
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
                content = self._chat(messages)
                batch_result = self._parse_json_translations(content, todo[0], todo[-1] + 1)
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
            print(f"[翻译] 进度: {batch_end}/{n_total}")
            start = batch_end

        print(f"[翻译] 完成，共 {len(results)} 句")
        return [results[i] for i in range(n_total)]

    @staticmethod
    def _parse_json_translations(content: str, id_start: int, id_end: int) -> Dict[int, str]:
        """从 LLM 返回内容解析 [{"id": n, "zh": "..."}]，返回 {id: 译文}"""
        text = strip_code_fence(content)
        begin = text.find("[")
        end = text.rfind("]")
        if begin == -1 or end == -1 or end <= begin:
            return {}

        try:
            data = json.loads(text[begin:end + 1])
        except json.JSONDecodeError:
            return {}

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
        numbered_lines = "\n".join(
            f"{ids[i]}. {s['text']}" for i, s in enumerate(sentences)
        )
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

输出要求：只输出一个 JSON 数组，每个元素格式为 {{"id": 编号, "zh": "该句中文译文"}}。编号可能与其它批次重叠或看起来不连续，但必须与原句前面的编号完全一致，一个都不能改、不能漏。不要输出 markdown 代码块标记，不要输出任何解释。

原文：
{numbered_lines}

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
        if begin == -1 or end == -1 or end <= begin:
            return []
        try:
            data = json.loads(content[begin:end + 1])
        except json.JSONDecodeError:
            return []

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

    只负责文本切分，不涉及任何时间信息；真实时间轴由后续的
    build_layout 根据 TTS 实测时长决定。
    """
    delimiters = ['，', '、', '；', ',', ';', '。', '！', '？', '…']

    # 第一遍：按标点切成子句
    clauses = []
    last_end = 0
    for i, char in enumerate(text):
        if char in delimiters and i > 5:
            clauses.append(text[last_end:i+1])
            last_end = i + 1
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

    return parts


def postprocess_subtitles(sentences: List[Dict], translations: List[str],
                          max_chars: int) -> List[Dict]:
    """把译文切成字幕片段（纯文本规则，不含时间信息）。

    - 超过 max_chars 的译文按标点拆成多条；
    - 纯标点/无实际内容的译文（无法 TTS）跳过；
    - 每个片段记录所属英文句子的时间跨度（span_start/span_end），
      真实起止时间由后续 build_layout 根据配音实测时长决定。
    """
    result = []

    for sent, trans in zip(sentences, translations):
        # 跳过纯标点/无实际内容的译文（否则 TTS 会报 NoAudioReceived）
        if not has_speakable_text(trans):
            print(f"[后处理] 跳过无可朗读内容的译文: {trans!r}")
            continue

        if len(trans) > max_chars:
            for part in split_long_sentence(trans, max_chars):
                result.append({
                    "text": part.strip(),
                    "span_start": sent["start_ms"],
                    "span_end": sent["end_ms"],
                })
        else:
            result.append({
                "text": trans,
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

    def __init__(self, config: Dict, llm_config: Dict):
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

        if self.engine == "edge-tts":
            self.voice = config.get("voice", "zh-CN-XiaoxiaoNeural")
            self.rate = config.get("rate", "+0%")
            self.volume = config.get("volume", "+0%")
            self.pitch = config.get("pitch", "+0Hz")
            print(f"[TTS] 引擎: edge-tts, 音色: {self.voice}")
        elif self.engine == "mimo":
            self.base_url = config.get("base_url", llm_config.get("base_url", ""))
            self.api_key = config.get("api_key", llm_config.get("api_key", ""))
            self.model = config.get("model", "mimo-v2.5-tts")
            self.voice = config.get("voice", "茉莉")
            self.style = config.get("style", "")

            base_url = self.base_url.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"

            self.client = OpenAI(api_key=self.api_key, base_url=base_url, timeout=120)
            print(f"[TTS] 引擎: MiMo TTS, 模型: {self.model}, 音色: {self.voice}")
        else:
            raise ValueError(f"不支持的 TTS 引擎: {self.engine}")

    async def generate_all(self, pieces: List[Dict], output_dir: Path) -> List[Path]:
        """分批生成并缓存配音，返回与 pieces 对齐的音频路径。"""
        if not self.enabled:
            return []

        output_dir.mkdir(parents=True, exist_ok=True)
        extension = "mp3" if self.engine == "edge-tts" else "wav"
        manifest_path = output_dir / "tts_cache.json"
        manifest = read_json_safe(manifest_path) or {}
        entries = manifest.get("entries")
        if not isinstance(entries, list):
            entries = []

        files: List[Optional[Path]] = [None] * len(pieces)
        missing = []

        def make_entry(idx: int) -> Dict[str, str]:
            return {
                "text_sha": hashlib.sha1(
                    pieces[idx]["text"].encode("utf-8")
                ).hexdigest(),
                "signature": self.cache_signature,
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
            else:
                missing.append(idx)

        print(f"[TTS] 共 {len(pieces)} 条配音，缓存命中 {len(pieces) - len(missing)} 条，"
              f"待生成 {len(missing)} 条")
        for batch_start in range(0, len(missing), self.batch_size):
            batch_indexes = missing[batch_start:batch_start + self.batch_size]
            batch_pieces = [pieces[idx] for idx in batch_indexes]
            try:
                if self.engine == "edge-tts":
                    batch_files = await self._generate_edge_tts(
                        batch_pieces, output_dir, batch_indexes)
                else:
                    batch_files = await self._generate_mimo_tts(
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
            save_manifest()
            print(f"[TTS] 进度: {min(batch_start + self.batch_size, len(missing))}/"
                  f"{len(missing)} 条待生成")

        return [path for path in files if path is not None]

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

    def _generate_mimo_one(self, idx: int, text: str, output_dir: Path) -> Path:
        """生成单条 MiMo 配音（带重试机制），在线程池中同步执行"""
        output_path = output_dir / f"tts_{idx:04d}.wav"

        messages = []
        if self.style:
            messages.append({"role": "user", "content": self.style})
        messages.append({"role": "assistant", "content": text})

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    audio={"format": "wav", "voice": self.voice},
                )

                # 提取 base64 音频数据
                audio_data = response.choices[0].message.audio.data
                audio_bytes = base64.b64decode(audio_data)
                if not audio_bytes:
                    raise RuntimeError("服务端未返回音频数据")

                with open(output_path, "wb") as f:
                    f.write(audio_bytes)

                return output_path
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise
                wait = 2 ** attempt
                print(f"[TTS] 第 {idx} 条生成失败 (第 {attempt}/{MAX_RETRIES} 次): {e}，{wait}s 后重试...")
                time.sleep(wait)

    async def _generate_mimo_tts(self, pieces: List[Dict], output_dir: Path,
                                 indexes: Optional[List[int]] = None) -> List[Path]:
        """使用 MiMo TTS 以自然语速生成配音"""
        semaphore = asyncio.Semaphore(self.concurrency)

        async def run_one(idx: int, text: str) -> Path:
            # 同步 HTTP 调用放入线程池执行，信号量限制并发
            async with semaphore:
                return await asyncio.to_thread(
                    self._generate_mimo_one, idx, text, output_dir)

        indexes = indexes or list(range(len(pieces)))
        tasks = [run_one(idx, piece["text"])
             for idx, piece in zip(indexes, pieces)]
        # 任一条最终失败即取消其余任务并抛出异常，不让它们继续在后台请求 API
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            for t in tasks:
                t.cancel()
            raise errors[0]

        print(f"[TTS] MiMo TTS 配音生成完成")
        return results


def mix_tts_audio(clips: List[Dict], output_audio: Path):
    """将各配音片段按最终时间轴混入完整音轨（numpy 实现，替代 amix 滤镜）。

    clips: build_layout 的输出，每项含 start_ms 与 file。
    """
    if not clips:
        return

    try:
        import numpy as np
    except ImportError:
        raise RuntimeError("音频拼接需要 numpy，请先安装: pip install numpy")

    clips = sorted(clips, key=lambda c: c["start_ms"])
    decoded = []
    total_samples = SAMPLE_RATE  # 至少留 1 秒尾部
    for clip in clips:
        pcm = _decode_to_pcm16(clip["file"])
        offset = int(clip["start_ms"]) * SAMPLE_RATE // 1000
        decoded.append([offset, pcm])
        total_samples = max(total_samples, offset + len(pcm) + SAMPLE_RATE)

    # 防止语音重叠的兜底：正常情况下新时间轴不会重叠，仅当某句组触发最高倍速
    # 仍超长时才会发生。超出部分直接截断并淡出。
    FADE_SAMPLES = 480  # 10ms @ 48kHz
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


def _decode_to_pcm16(path: Path):
    """用 ffmpeg 把任意音频解码为 48kHz 单声道 s16le PCM，返回 numpy 数组"""
    import numpy as np
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(path),
           "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1"]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"音频解码失败: {path}\n{result.stderr.decode(errors='ignore')}")
    return np.frombuffer(result.stdout, dtype=np.int16)


def build_layout(pieces: List[Dict], tts_files: Optional[List[Path]],
                 max_tempo: float = 3.0) -> List[Dict]:
    """为字幕片段计算最终时间轴（字幕与配音共用同一套时间）。

    有 TTS 音频时逐个测量真实时长并据此排布——同一句子的整组片段
    放不下时统一加速（上限 max_tempo），字幕切换时刻与语音完全同步；
    无音频（--no-tts / --no-video 模式）时退化为按 130ms/字 估算排布。

    返回 [{text, start_ms, end_ms, file, tempo}, ...]，按时间升序。
    """
    EST_MS_PER_CHAR = 130  # 无实测时长时的退化估算值

    durations = []
    for idx, piece in enumerate(pieces):
        dur = None
        if tts_files:
            dur = probe_duration_ms(tts_files[idx])
        if dur is None:
            dur = len(piece["text"]) * EST_MS_PER_CHAR
        durations.append(max(int(dur), 200))  # 单片最短 200ms，防异常数据

    layout = []
    i = 0
    while i < len(pieces):
        # 找出同一英文句子跨度下的连续片段（postprocess 保证了它们相邻）
        j = i
        while (j < len(pieces)
               and pieces[j]["span_start"] == pieces[i]["span_start"]
               and pieces[j]["span_end"] == pieces[i]["span_end"]):
            j += 1
        group = pieces[i:j]
        group_dur = durations[i:j]
        span_start = group[0]["span_start"]
        span_end = group[0]["span_end"]

        total = sum(group_dur)
        avail = max(span_end - span_start, 0)
        tempo = 1.0
        if total > avail > 0:
            raw = total / avail
            tempo = min(raw, max_tempo)
            if raw > max_tempo:
                print(f"[警告] {ms_to_srt_time(span_start)} 起的字幕组严重超长："
                      f"配音需 {total}ms / 可用 {avail}ms，"
                      f"已按最高 {max_tempo}x 加速，超出部分可能被截断")

        t = span_start
        for k, piece in enumerate(group):
            adj = max(int(group_dur[k] / tempo), 100)  # 保底 100ms 防零时长
            layout.append({
                "text": piece["text"],
                "start_ms": t,
                "end_ms": t + adj,
                "file": tts_files[i + k] if tts_files else None,
                "tempo": tempo,
            })
            t += adj

        i = j

    sped = [e for e in layout if e["tempo"] > 1.005]
    if sped:
        print(f"[时间轴] {len(sped)} 条配音已按所在句组的统一倍速加速"
              f"（最高 {max(e['tempo'] for e in sped):.2f}x）")

    # 对需要加速的音频统一应用 atempo（每个文件恰好对应一个片段）
    if tts_files:
        for entry in layout:
            if entry["file"] and entry["tempo"] > 1.005:
                entry["file"] = speed_up_audio(entry["file"], entry["tempo"])

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
    """转义并包裹 FFmpeg 滤镜参数中的路径（处理盘符冒号、反斜杠、空格与引号）。

    用单引号包裹后，路径中的空格、冒号都会被当作字面量处理；
    内部的单引号用 '\'' 序列转义（关闭引号→转义引号→重新开启）。
    """
    s = str(p).replace("\\", "/").replace("'", "'\\''")
    return f"'{s}'"


def compose_final_video(video_path: Path, srt_path: Optional[Path],
                        audio_path: Optional[Path], output_path: Path,
                        mix_with_original: bool = False) -> bool:
    """一步完成字幕烧录（可选）和配音替换/混合（可选），只做一次视频转码"""
    if not shutil.which("ffmpeg"):
        print("[警告] 未找到 ffmpeg，跳过最终合成")
        return False

    sub_style = "FontSize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Shadow=0,MarginV=5"
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
        cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "192k"])
    elif srt_path:
        # 只烧录字幕
        sub_filter = f"subtitles={_escape_filter_path(srt_path)}:force_style='{sub_style}'"
        cmd.extend(["-vf", sub_filter,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "192k"])
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


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="YouTube 视频自动下载 + 中文字幕生成 + 中文配音")
    parser.add_argument("url", help="YouTube 视频 URL")
    parser.add_argument("-o", "--output", default="./youtube_downloads", help="根输出目录")
    parser.add_argument("--no-video", action="store_true", help="只下载字幕，不下载视频")
    parser.add_argument("--no-tts", action="store_true", help="跳过中文配音")

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

        # 步骤 5: LLM 翻译（逐批写缓存，中断后重跑自动断点续传）
        print("\n" + "=" * 60)
        print("步骤 5: LLM 翻译")
        print("=" * 60)
        llm = LLMClient(llm_config)
        translations = None
        cache_path = output_dir / f"{video_id}_translations.json"
        source_sha = hashlib.sha1(
            "\n".join(s["text"] for s in sentences).encode("utf-8")
        ).hexdigest()

        # 步骤 5.0: 提取全片统一术语表（一次调用；结果写入缓存供断点续传复用）
        cached_data = read_json_safe(cache_path) or {}
        g = cached_data.get("glossary")
        glossary = None
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
            # 保留旧文件中所有非本函数管理的字段（如 title_zh、glossary），避免覆盖丢失
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

        if translations is None:
            translations = llm.translate(sentences, title, description,
                                         done=done_map, on_progress=save_cache,
                                         glossary=glossary)
            save_cache({i: t for i, t in enumerate(translations)})
            print(f"[翻译] 结果已缓存: {cache_path}")

        # 步骤 5.5: 翻译视频标题（用于最终视频文件命名，同样写入缓存）
        title_zh = None
        cached = read_json_safe(cache_path) or {}
        t = cached.get("title_zh")
        if isinstance(t, str) and t.strip():
            title_zh = t.strip()
        if title_zh is None:
            try:
                title_zh = llm.translate_title(title, description)
                if title_zh:
                    # 合并写入缓存文件，下次重跑不再重新翻译
                    cache_data = read_json_safe(cache_path) or {}
                    cache_data["title_zh"] = title_zh
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(cache_data, f, ensure_ascii=False, indent=1)
                    print(f"[标题] 中文标题: {title_zh}")
            except Exception as e:
                print(f"[警告] 标题翻译失败，将使用原标题命名: {e}", file=sys.stderr)
                title_zh = None

        for i, t in enumerate(translations[:3]):
            print(f"  译{i+1}: {t[:60]}...")

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
        mixed_audio = None
        clips = []
        cleanup_tts_cache = False
        try:
            # 配音合成仅在需要产出视频时进行；--no-video 模式只输出估算时间轴的 SRT
            if not args.no_video and not args.no_tts:
                tts = TTSClient(tts_config, llm_config)
                if tts.enabled:
                    # 自然语速合成全部配音（此时还没有最终时间轴）
                    tts_files = asyncio.run(tts.generate_all(pieces, output_dir))

            # 有实测时长则按真实语音排布（字幕与配音天然同步）；
            # 否则退化为按 130ms/字 估算排布
            max_tempo = tts.max_tempo if tts else 3.0
            clips = build_layout(pieces, tts_files, max_tempo)
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
