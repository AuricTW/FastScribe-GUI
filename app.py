import os
import sys
import glob
import tempfile
import subprocess
from typing import Tuple, Optional, Dict, Any, List

import gradio as gr
from faster_whisper import WhisperModel

# --- 全域模型快取，避免每次重載 ---
MODEL_CACHE: Dict[Tuple[str, str, str], WhisperModel] = {}

# 預設提供幾個常用的 faster-whisper model 名稱
MODEL_OPTIONS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v2",
    "large-v3",
]

DEVICE_OPTIONS = ["cpu", "cuda"]
COMPUTE_TYPE_OPTIONS = ["float16", "int8_float16", "int8"]


# --- 小工具函式 ---

def format_timestamp(seconds: float) -> str:
    """把秒數轉成 SRT 用的 00:00:00,000 格式"""
    if seconds is None:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    secs = ms // 1000
    ms %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def get_model(
    model_name: str, device: str, compute_type: str
) -> WhisperModel:
    """依照 (model_name, device, compute_type) 取出或建立 WhisperModel"""
    key = (model_name, device, compute_type)
    if key not in MODEL_CACHE:
        MODEL_CACHE[key] = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )
    return MODEL_CACHE[key]


def download_youtube_audio(url: str) -> str:
    """
    使用「命令列版」 yt-dlp 將 YouTube 內容下載成檔案，回傳檔案路徑。

    策略：
    1. 先嘗試只抓 bestaudio (m4a)
    2. 若 403 / 其他錯誤，改抓 best（整支影片 mp4/webm），
       faster-whisper 也能直接吃影片檔。
    """
    tmpdir = tempfile.mkdtemp(prefix="yt_audio_")
    outtmpl = os.path.join(tmpdir, "%(title)s.%(ext)s")

    def run_yt_dlp(fmt: str) -> None:
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "-f", fmt,
            "-o", outtmpl,
            url,
        ]

        # 若之後需要 cookies 處理受限影片，可在此加上：
        # cookie_file = r"D:\逐字稿系統\youtube_cookies.txt"
        # if os.path.exists(cookie_file):
        #     cmd.extend(["--cookies", cookie_file])

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=tmpdir,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"yt-dlp 下載失敗 (exit code {proc.returncode}):\n"
                f"{proc.stderr.strip()}"
            )

    errors: List[str] = []

    # 第一次嘗試：只抓音訊
    try:
        run_yt_dlp("bestaudio[ext=m4a]/bestaudio/best")
    except RuntimeError as e:
        errors.append(str(e))

        # 第二次嘗試：抓整支影片 (video+audio)
        try:
            run_yt_dlp("best")
        except RuntimeError as e2:
            errors.append(str(e2))
            # 兩次都失敗就一起丟出去
            raise RuntimeError("\n\n".join(errors))

    # 從 tmpdir 找下載出來的檔案（排除暫存 .part）
    files = [
        f for f in glob.glob(os.path.join(tmpdir, "*"))
        if not f.endswith(".part") and os.path.isfile(f)
    ]

    if not files:
        raise RuntimeError("yt-dlp 回傳成功但沒有找到下載後的檔案。")

    # 這裡先取第一個檔案當作音訊輸入
    audio_path = files[0]
    return audio_path


def build_srt(segments) -> str:
    """
    依照 faster-whisper 回傳的 segments 產生 SRT 文字。
    """
    srt_blocks: List[str] = []
    for i, seg in enumerate(segments, start=1):
        start = format_timestamp(seg.start)
        end = format_timestamp(seg.end)
        text = seg.text.strip()
        block = f"{i}\n{start} --> {end}\n{text}\n"
        srt_blocks.append(block)
    return "\n".join(srt_blocks).strip() + "\n"


# --- 主要轉錄函式（給 Gradio 按鈕用） ---

def transcribe(
    audio_file,          # gr.File 或 None
    youtube_url: str,    # 文字框
    model_name: str,
    device: str,
    compute_type: str,
    language: str,       # "auto" or lang code
    task: str,           # "transcribe" / "translate"
    beam_size: int,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    回傳：
    - transcript_text: 顯示在 Textbox 的完整逐字稿
    - txt_path: TXT 檔案路徑（給下載按鈕）
    - srt_path: SRT 檔案路徑（給下載按鈕）
    """
    if (audio_file is None) and (not youtube_url.strip()):
        return "請先上傳檔案或輸入 YouTube 連結。", None, None

    # 取得音訊來源路徑
    if audio_file is not None:
        # gr.File 回傳的物件通常有 .name
        audio_path = getattr(audio_file, "name", None) or audio_file
    else:
        try:
            audio_path = download_youtube_audio(youtube_url.strip())
        except Exception as e:
            # 把詳細錯誤訊息顯示在 GUI 上
            return f"YouTube 下載失敗：\n{e}", None, None

    # 載入模型
    try:
        model = get_model(model_name, device, compute_type)
    except Exception as e:
        return f"載入模型失敗：{e}", None, None

    # 語言設定
    lang_arg = None if language == "auto" else language

    try:
        segments, info = model.transcribe(
            audio_path,
            beam_size=beam_size,
            language=lang_arg,
            task=task,
        )
    except Exception as e:
        return f"轉錄過程發生錯誤：{e}", None, None

    # 整理文字 + SRT
    lines: List[str] = []
    seg_list = []
    for seg in segments:
        lines.append(seg.text.strip())
        seg_list.append(seg)

    transcript_text = "\n".join(lines).strip()
    srt_text = build_srt(seg_list)

    # 寫入暫存檔，給下載按鈕用
    out_dir = tempfile.mkdtemp(prefix="transcript_")
    txt_path = os.path.join(out_dir, "transcript.txt")
    srt_path = os.path.join(out_dir, "subtitles.srt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_text)

    return transcript_text, txt_path, srt_path


# --- Gradio 介面定義 ---

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Faster-Whisper 逐字稿工具") as demo:
        gr.Markdown(
            """
            # 🎙️ Faster-Whisper 逐字稿工具

            - 支援：本機音訊 / 影片檔、YouTube 連結
            - 模型：可自行選擇不同大小的 `faster-whisper` 模型
            - 輸出：畫面逐字稿 + TXT / SRT 字幕下載
            """
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🗂️ 輸入來源")

                audio_file = gr.File(
                    label="上傳音訊 / 影片檔 (mp3, mp4, wav, m4a...)",
                    file_types=["audio", "video"],
                )
                youtube_url = gr.Textbox(
                    label="或貼上 YouTube 影片 / 播放清單連結",
                    placeholder="https://www.youtube.com/watch?v=...",
                )

                gr.Markdown("### ⚙️ 模型與參數")
                model_name = gr.Dropdown(
                    choices=MODEL_OPTIONS,
                    value="small",
                    label="模型大小 (faster-whisper)",
                )

                device = gr.Dropdown(
                    choices=DEVICE_OPTIONS,
                    value="cuda",
                    label="運算裝置",
                    info="若沒有 GPU 就選 cpu",
                )

                compute_type = gr.Dropdown(
                    choices=COMPUTE_TYPE_OPTIONS,
                    value="float16",
                    label="compute_type",
                    info="int8 系列可減少記憶體使用量",
                )

                language = gr.Dropdown(
                    choices=["auto", "zh", "en", "ja", "ko", "fr", "de", "es"],
                    value="auto",
                    label="語言 (auto 為自動偵測)",
                )

                task = gr.Dropdown(
                    choices=["transcribe", "translate"],
                    value="transcribe",
                    label="任務",
                    info="translate 會翻譯成英文",
                )

                beam_size = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=5,
                    label="beam_size (越大越準但越慢)",
                )

                run_button = gr.Button("🚀 開始轉錄", variant="primary")

            with gr.Column():
                gr.Markdown("### 📄 輸出結果")

                transcript_box = gr.Textbox(
                    label="逐字稿",
                    lines=20,
                    show_copy_button=True,
                )

                txt_download = gr.File(label="下載 TXT 檔")
                srt_download = gr.File(label="下載 SRT 字幕檔")

        run_button.click(
            fn=transcribe,
            inputs=[
                audio_file,
                youtube_url,
                model_name,
                device,
                compute_type,
                language,
                task,
                beam_size,
            ],
            outputs=[transcript_box, txt_download, srt_download],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    # 如需固定 port / 關閉 share 可改：
    # ui.launch(server_port=7860, share=False)
    ui.launch()
