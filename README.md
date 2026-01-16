# FastScribe GUI 
基於 Faster-Whisper 的本地逐字稿小工具

FastScribe GUI 是一個使用 **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** 與 **[Gradio](https://gradio.app/)** 建立的本地逐字稿應用程式，專門用來將：

- 本機音訊檔（`mp3`, `wav`, `m4a`…）
- 本機影片檔（`mp4`, `mkv`…）
- **YouTube 影片連結**

轉換為文字逐字稿，並輸出為 **TXT / SRT 字幕檔**。  
搭配 GUI 介面，直接執行 `python app.py` 即可操作使用。

---

## 功能

-  **多來源輸入**
  - 上傳本機 **音訊 / 影片檔**
  - 貼上 **YouTube 影片或播放清單連結**（內部使用 `yt-dlp` 抓取）

-  **多種 Faster-Whisper 模型選擇**
  - 內建模型列表：`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`
  - 可手動切換：
    - 運算裝置：`cpu` / `cuda`
    - `compute_type`：`float16`, `int8_float16`, `int8`（節省 GPU 記憶體）

-  **多語言支援**
  - 語言選擇：`auto`, `zh`, `en`, `ja`, `ko`, `fr`, `de`, `es`
  - `task`：
    - `transcribe`：同語言逐字稿
    - `translate`：翻譯成英文

-  **多種輸出格式**
  - 介面中顯示完整逐字稿
  - 提供：
    - `transcript.txt`（純文字）
    - `subtitles.srt`（字幕檔，可直接丟進播放器）

-  **GPU 加速（可選）**
  - 若有 CUDA GPU，可選擇 `device = cuda` + `float16 / int8_float16` 加速轉錄
  - 若 GPU 記憶體不足，可改用 `int8` 或改用 `cpu` 模式執行

---

##  專案結構

```text
.
├─ app.py            # 主程式（Gradio GUI + Faster-Whisper + yt-dlp）
└─ requirements.txt  # 相依套件清單
