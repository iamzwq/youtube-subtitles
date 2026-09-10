# 使用 Demucs 从视频/音频中提取人声

`extract_vocals.py` 是一个基于 Meta Demucs AI 模型的人声分离脚本，可自动从输入的视频（如 `.mp4`, `.mkv`, `.mov` 等）或音频（如 `.mp3`, `.wav`, `.m4a` 等）文件中将**人声 (Vocals)** 提取并保存为高清音频文件。

---

## 1. 前置条件

1. **已安装 Python 3.8+**
2. **已安装 FFmpeg**（需加入系统环境变量，以便脚本抽取视频中的音频）
3. **安装 Demucs 依赖**：

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install demucs
```

> **提示**：如果系统配置了 NVIDIA 显卡与 CUDA 版本的 PyTorch，脚本会自动启用 GPU 加速；如果没有 GPU，会自动使用 CPU 计算。

> 检查 Python 中 PyTorch 是否支持 CUDA：
>
> python -c "import torch; print('PyTorch 版本:', torch.**version**); print('CUDA 是否可用:', torch.cuda.is_available()); print('显卡名称:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')"

---

## 2. 使用方法

### 方法一：直接拖入文件运行（推荐）

直接在终端或 CMD 中运行脚本（不带参数）：

```cmd
python extract_vocals/extract_vocals.py
```

终端会提示：

```text
============================================================
            Demucs 视频/音频人声提取工具
============================================================
请输入视频或音频文件路径 (可直接将文件拖入此窗口):
```

此时将你的视频文件直接**拖入终端窗口**，按回车即可开始提取。

---

### 方法二：命令行指定文件路径

指定视频文件路径运行：

```cmd
python extract_vocals/extract_vocals.py "D:\Videos\my_video.mp4"
```

---

## 3. 高级选项

| 参数                   | 说明                                                                             | 示例                       |
| ---------------------- | -------------------------------------------------------------------------------- | -------------------------- |
| `-o, --output`         | 指定输出文件夹或输出文件名                                                       | `-o "D:\Vocals\vocal.wav"` |
| `-f, --format`         | 输出格式，可选 `wav` / `mp3` / `flac`（默认 `wav`）                              | `-f mp3`                   |
| `-m, --model`          | 选择 Demucs 模型，可选 `htdemucs`, `htdemucs_ft`, `mdx_extra`（默认 `htdemucs_ft`） | `-m htdemucs_ft`           |
| `-d, --device`         | 指定设备 `auto` / `cuda` / `cpu`（默认 `auto`）                                  | `-d cuda`                  |
| `--keep-accompaniment` | 提取人声的同时，额外保存一份去除人声后的伴奏音频                                 | `--keep-accompaniment`     |
| `--shifts`             | 随机偏移多次推理再平均，越大质量越高但越慢（默认 `2`）                           | `--shifts 5`               |
| `--overlap`            | 分段处理重叠比例，越大分段拼接痕迹越少但越慢（默认 `0.25`）                       | `--overlap 0.5`            |
| `--denoise`            | 对分离出的人声做二次净化（ffmpeg 降噪），压制残留背景音/伴奏泄漏                 | `--denoise`                |
| `--denoise-strength`   | 二次净化的降噪强度，单位 dB（默认 `12`，越大降噪越强但可能损伤人声）             | `--denoise-strength 18`    |

### 示例命令：

1. **提取人声并输出为 MP3 格式**：

   ```cmd
   python extract_vocals/extract_vocals.py "D:\video.mp4" -f mp3
   ```

2. **同时保存人声和伴奏**：

   ```cmd
   python extract_vocals/extract_vocals.py "D:\video.mp4" --keep-accompaniment
   ```

3. **指定输出目录**：
   ```cmd
   python extract_vocals/extract_vocals.py "D:\video.mp4" -o "D:\output_dir"
   ```

4. **人声中残留背景音较明显时，追求最高质量**：
   ```cmd
   python extract_vocals/extract_vocals.py "D:\video.mp4" -m htdemucs_ft --shifts 5 --overlap 0.5 --denoise
   ```

---

## 5. 分离出的人声带背景音怎么办？

Demucs 是基于神经网络的盲分离模型，源音轨混音越复杂（多乐器、混响重、人声被压缩得很紧），
分离结果就越可能残留背景音（业内称为“泄漏”，leakage）。可按下面顺序尝试优化：

1. **换用质量更高的模型**：`htdemucs_ft`（微调版，默认已启用）比 `htdemucs` 泄漏更少，
   但推理更慢；如果人声内容本身较简单（对白、播客），也可以试试 `mdx_extra`。
2. **增大 `--shifts`**：Demucs 通过随机时间位移做多次推理再取平均，`--shifts 5~10`
   通常能明显减少泄漏，但耗时也会成倍增加。
3. **增大 `--overlap`**：默认 `0.25`，调到 `0.5~0.75` 可以减少长音频分段处理时
   拼接处的伴奏泄漏，但同样会增加计算时间。
4. **开启 `--denoise` 二次净化**：分离完成后额外用 ffmpeg 做一次频谱降噪
   （`afftdn`）+ 低频滤波（`highpass`），可以压制残留的持续性背景音；
   如果发现人声听起来发闷或有"金属声"等失真，说明降噪强度太高，
   可用 `--denoise-strength` 调低（如 `8`）。
5. **如果背景音是明显的音乐/BGM 而非其他人声**：说明分离模型本身效果已接近上限，
   可以考虑先用 `--two-stems vocals` 之外，再手动叠加一轮 `htdemucs_ft` 分离
   （即对分离出的人声文件再跑一次本脚本），但要注意二次分离可能损失部分人声细节，
   建议先用 `--denoise` 试试效果是否已经足够。

---

## 6. 输出文件

默认情况下，提取出的音频文件将自动保存在**原视频所在目录**，命名格式为：

- **人声文件**：`原视频文件名_vocals.wav` (或 `.mp3`)
- **伴奏文件**（若加了 `--keep-accompaniment`）：`原视频文件名_accompaniment.wav`

[演示视频](https://www.bilibili.com/video/BV1jUXjBuEvG)
