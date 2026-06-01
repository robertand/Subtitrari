# Subtitratorul PRO — User Manual

Professional audio/video transcription and translation tool with advanced subtitle editing.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Quick Start](#3-quick-start)
4. [Uploading Files](#4-uploading-files)
5. [Processing Settings](#5-processing-settings)
6. [Starting Transcription](#6-starting-transcription)
7. [Working with Results](#7-working-with-results)
8. [Timeline Management](#8-timeline-management)
9. [Translation Features](#9-translation-features)
10. [Hardcoded Subtitle OCR](#10-hardcoded-subtitle-ocr)
11. [Sound Event Detection (SDH)](#11-sound-event-detection-sdh)
12. [Subtitle Styling](#12-subtitle-styling)
13. [Export Options](#13-export-options)
14. [Project Management](#14-project-management)
15. [Keyboard Shortcuts](#15-keyboard-shortcuts)
16. [Performance Tips](#16-performance-tips)
17. [Troubleshooting](#17-troubleshooting)
18. [Supported Languages](#18-supported-languages)

---

## 1. Introduction

**Subtitratorul PRO** is a professional web-based application for transcribing audio/video files and translating subtitles. It supports multiple transcription engines, local and cloud-based LLM translation, speaker diarization, visual timeline editing, hardcoded subtitle extraction via OCR, and sound event detection.

### Key Features

- **Multiple Transcription Engines**: OpenAI Whisper, Cohere Transcribe, NVIDIA NeMo Parakeet
- **Multi-language Support**: 25+ languages for transcription and translation
- **Speaker Diarization**: Automatic speaker identification with gender detection
- **Timeline Editing**: Visual subtitle editing with drag-and-drop, resize, move
- **Selective Reprocessing**: Re-transcribe specific portions only
- **Professional Translation**: Google Translate, LLM API (Claude/OpenAI), local VLLM (Gemma, Llama), TranslateGemma
- **Translation Refinement**: LLM-based grammatical correction and context adaptation
- **Hardcoded Subtitle OCR**: Extract burned-in subtitles from video using PaddleOCR
- **Sound Event Detection (SDH)**: Detect and describe sound effects for accessibility
- **Subtitle Styling**: Font, color, effects, and position customization persisted across sessions
- **Multiple Export Formats**: SRT, DOCX (with metadata), JSON project files

---

## 2. Installation

### Prerequisites (All OS)

- Python 3.10 or 3.11
- FFmpeg (required for audio/video processing)
- At least 5–8 GB of free disk space (for ML models)
- NVIDIA GPU with CUDA recommended (for acceleration)

### Windows

```powershell
# 1. Install FFmpeg
#    Download from https://www.gyan.dev/ffmpeg/builds/ (ffmpeg-release-essentials.zip)
#    Extract and add the bin folder to your System PATH

# 2. Install Python 3.11 from https://python.org
#    During installation, check "Add python.exe to PATH"

# 3. Set up the application
git clone https://github.com/robertand/Subtitrari.git
cd Subtitrari
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip ffmpeg git -y

git clone https://github.com/robertand/Subtitrari.git
cd Subtitrari
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

### macOS

```bash
brew install python@3.11 ffmpeg git

git clone https://github.com/robertand/Subtitrari.git
cd Subtitrari
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

### Starting the Application

```bash
cd Subtitrari
source venv/bin/activate   # Windows: venv\Scripts\activate
python app.py
```

Open your browser and navigate to **http://localhost:5000**.

---

## 3. Quick Start

1. **Upload** a video or audio file via drag-and-drop or click.
2. **Select a preset** or configure settings (engine, model, language).
3. **Enable translation** if needed and choose target language.
4. Click **Start Processing** (bottom-right corner).
5. Wait for transcription to complete.
6. **Edit** segments, adjust timestamps on the timeline, or refine text.
7. **Export** as SRT or DOCX.

---

## 4. Uploading Files

### Supported Formats

| Type   | Extensions                              |
|--------|-----------------------------------------|
| Video  | MP4, AVI, MOV, MKV, WebM, MXF          |
| Audio  | MP3, WAV, M4A, FLAC, OGG               |

Maximum file size: **50 GB** per file.

### Upload Methods

**Drag & Drop**: Drag a file from your file manager onto the dashed upload area.

**Click to Upload**: Click the upload area and select a file via the file dialog.

### Upload Progress

During upload you can monitor:
- Progress percentage and progress bar
- Chunk indicators showing uploaded segments
- Speed (MB/s) and estimated time remaining (ETA)
- **Pause/Resume** and **Cancel** buttons

### After Upload

Once the upload completes:
- A video/audio player appears with a preview thumbnail.
- The **Start Processing** button appears (bottom-right corner).
- Your file is ready for transcription.

---

## 5. Processing Settings

The settings panel controls every aspect of the transcription process.

### Basic Settings

| Setting              | Description                                       |
|----------------------|---------------------------------------------------|
| Preset               | Load predefined or saved configurations           |
| Transcription Engine | Whisper, Cohere, or NVIDIA NeMo Parakeet          |
| Model                | Model size (tiny → large-v3, Turbo, Turkish, etc.)|
| Source Language      | Language of the audio (or auto-detect)            |
| Segmentation         | Min/max segment duration, max characters          |
| VAD                  | Voice Activity Detection (detects speech pauses)  |
| 1s Margin            | Adds 1-second margin after speech                 |

### Advanced Settings (toggle "Advanced Options")

| Setting             | Description                                               |
|---------------------|-----------------------------------------------------------|
| Window Size         | Window size for chunked transcription (seconds)           |
| Window Overlap      | Overlap between windows (seconds)                         |
| Voice Isolation      | Separate voice from background noise (Demucs)             |
| Remove Repetitions   | Remove duplicate consecutive phrases                      |
| Sequential Segments  | Ensure no overlapping segments                            |
| Diarization         | Identify different speakers                               |
| Multi-Pass          | Triple-pass transcription (slower, more accurate)         |
| Audio Only Mode     | Process as audio even for video files                     |

### Language-Specific Options

- **Mixed Turkish Mode** (when Turkish is selected): Combines Large V3 and Turbo.
- **Mixed Korean Mode** (when Korean is selected): Optimized for Korean.

### Region Processing

Set **Start** and **End** timestamps to process only a portion of the video. Timestamps adjust to full video on export.

### Hugging Face Token

Required for gated models (e.g., `pyannote` for diarization, some custom Whisper models).

### Presets

- Save your current configuration by entering a name and clicking **Save**.
- Your saved presets appear under "Saved by you" in the dropdown.
- Delete a user preset by selecting it and clicking the trash icon.

---

## 6. Starting Transcription

### Before Starting

- Verify your file is uploaded (player shows preview).
- Adjust settings as needed.
- Enable translation if required (see [Translation Features](#9-translation-features)).

### Start Processing

Click the floating **Start Processing** button (bottom-right). The status panel shows:
- Current step (audio extraction, transcription, diarization, translation, etc.)
- Progress percentage
- Animated progress bar

Click **Cancel** at any time to abort.

### Processing Steps

1. Audio extraction (for video files)
2. Voice isolation (if enabled)
3. Transcription (main step)
4. Diarization (if enabled)
5. Gender detection (if diarization enabled)
6. Subtitle segmentation
7. Deduplication
8. OCR extraction (if enabled)
9. SDH detection (if enabled)
10. Translation (if enabled)
11. Translation refinement (if enabled)

### Completion

- The results panel opens with transcribed segments.
- The video player displays subtitles in real-time.
- The timeline view appears for visual editing.

---

## 7. Working with Results

### Segments Panel

Each segment displays:
- Index number
- Editable text
- Timestamp (start → end)
- Speaker label with gender icon (if diarization enabled)
- Action buttons (play, edit, delete)

### Editing a Segment

- Click directly on the text, edit, then click outside or press Enter to save.
- Or click the edit button to focus the text field.

### Playing a Segment

Click the play button, or click the segment to seek to its start time.

### Deleting a Segment

Click the delete button and confirm.

### Adding a New Segment

Click **Add Segment** in the toolbar. Default duration is 2 seconds.

### Tabs

| Tab        | Description                                      |
|------------|--------------------------------------------------|
| Original   | Original transcribed text (editable)             |
| Raw        | Unprocessed raw transcript (read-only)            |
| Translation | Translated text (appears after translation)      |

Switch to the **Translation** tab to see all text in a single editable textarea

---

## 8. Timeline Management

The timeline provides visual editing with pixel-precise control.

### Components

- **Ruler**: Time markers (major and minor ticks)
- **Segments**: Color-coded blocks for each subtitle
- **Playhead**: Red vertical line showing current playback position
- **Selection Overlay**: Blue regions for selective reprocessing

### Navigation

- Scroll horizontally to move through the timeline.
- Zoom using +/- buttons or mouse wheel.
- Click anywhere on the timeline to seek.

### Selecting and Editing Segments

**Click to Select**: Click a segment block to select it (white border). The segment list also highlights it.

**Resize**: Drag left handle → change start time. Drag right handle → change end time.

**Move**: Drag from the middle of a segment.

**Delete**: Hover over a segment → click the red **×** that appears.

### Zone Selection & Reprocessing

Selectively re-transcribe specific portions:

1. Check **Selection Mode** or hold **Shift** while dragging.
2. Click and drag on the timeline to create a blue zone.
3. Release to create the zone.
4. Zones appear as tags below the timeline (click **×** to remove).
5. Click **Reprocess Selection** to re-transcribe only those zones.
6. New transcriptions replace existing content; translations are preserved.

Manual zone entry: Enter start and end times (HH:MM:SS) and click **Add Zone**.

---

## 9. Translation Features

### Enabling Translation

Check **Enable Translation**. Translation settings appear below.

### Translation Engines

| Engine            | Description                                                |
|-------------------|------------------------------------------------------------|
| Google Translate  | Fast, free, good for most languages (batch mode)           |
| LLM API           | Claude / OpenAI / Custom API (professional quality)        |
| VLLM              | Local LLM via vLLM (Gemma 4-26B, Gemma 3-12B, Llama 8B, RoMistral 7B) |
| TranslateGemma    | Google's specialized translation model (27B/12B/4B)        |

### LLM API Configuration

| Setting   | Description                                           |
|-----------|-------------------------------------------------------|
| Provider  | Claude, OpenAI, or Custom                             |
| API Key   | Your API key (stored only for current session)        |
| Model     | Model identifier (e.g., gpt-4o, claude-3-5-sonnet)   |

### Translation Refinement (Correction)

Enable **Translation Refinement (LLM)** to perform an additional LLM pass that corrects grammar, improves context logic, and ensures natural phrasing. You can select the refinement model separately from the translation engine. Supported refinement models: Gemma 3-12B, Llama 8B, RoMistral 7B.

**Content Context**: Describe the video content (genre, characters, setting) to improve translation quality.

### Multiple Target Languages

1. Click **Add Another Language**.
2. Select a language from the dropdown.
3. Remove languages with the trash button.

### Translation Process

- Translations run automatically after transcription.
- Progress appears in the status panel.
- Results appear in the **Translation** tab and Translation Results panel.
- Each language has its own export options.

---

## 10. Hardcoded Subtitle OCR

Extract burned-in (hardcoded) subtitles from video files using PaddleOCR.

### How to Use

1. Enable **Extract Hardcoded Subtitles (OCR)** in advanced settings.
2. Choose region mode:
   - **Auto**: Automatically detects the subtitle region at the bottom of the frame.
   - **Full Screen**: Scans the entire frame.
   - **Manual**: Set custom top/bottom boundaries (0.0–1.0).
3. Adjust **Confidence Threshold** (default: 70%) and **Frame Skip** (default: 2).
4. Enable **Merge with ASR** to combine OCR text with speech recognition results.

OCR works on video files (MP4, AVI, MOV, MKV, WebM, MXF) and requires GPU for best performance.

---

## 11. Sound Event Detection (SDH)

Automatically detect and describe non-speech audio events (music, applause, phone rings, etc.) for accessibility.

### How to Use

1. Enable **SDH Detection** in advanced settings.
2. Select the **SDH Language** (defaults to source language).
3. Set **Confidence Threshold** (default: 45%).
4. Enable **LLM Descriptions** to get natural language descriptions of detected sounds.

SDH segments are merged with ASR and OCR results automatically.

---

## 12. Subtitle Styling

Click the **Style** button in the player controls to open the styling panel.

### Style Options

| Setting         | Options                                                    |
|-----------------|------------------------------------------------------------|
| Font Family     | Arial, Segoe UI, Tahoma, Verdana, Courier New              |
| Size            | 12–60 px                                                   |
| Text Color      | Any color (color picker)                                   |
| Background      | Color picker + opacity slider (0–1)                        |
| Effects         | None, Drop Shadow, Outline (with outline color picker)     |
| Position X/Y    | Offset in pixels                                           |

**Toggle "Top Position"** to move subtitles to the top of the video.

All styling applies in real-time. **Settings persist across sessions** (saved to browser localStorage).

---

## 13. Export Options

### SRT Export

- Click the **Export SRT** button in the results toolbar.
- Or press **Ctrl+S**.

Standard SRT format with numbered segments and timestamps.

### DOCX Export (Professional)

Click **Export DOCX** to open the export dialog.

| Field              | Description                        |
|--------------------|------------------------------------|
| Title              | Film/episode title                 |
| Series / Episode   | Series and episode identifier      |
| Translator Name    | Translator's name                  |
| Editor Name        | Editor's name                      |
| Legacy Diacritics  | Convert ș→ş, ț→ţ for older specs  |

The DOCX output includes:
- Title page with metadata table
- Numbered segments with timestamps
- Proper line wrapping (max 40 chars per line)

### Copy Full Text

Click **Copy Text** to copy all subtitle text to the clipboard.

---

## 14. Project Management

### Saving a Project

Click **Export as Project (.json)** from the File menu. Saves:
- All segments with timestamps
- All translations
- Subtitle styling preferences
- File references

### Loading a Project

1. Click **Open Project (.json)** from the File menu.
2. Select your `.json` project file.
3. All data is restored, including the video file (if available).

**Note**: Projects are not auto-saved. Manual saving is required.

---

## 15. Keyboard Shortcuts

| Shortcut        | Action                                    |
|-----------------|-------------------------------------------|
| Space           | Play/Pause video (when no input focused)  |
| Ctrl+S          | Export SRT                                |
| Ctrl+D          | Open DOCX export dialog                   |
| Esc             | Close modal dialogs                       |
| Up Arrow        | Seek to previous segment                  |
| Down Arrow      | Seek to next segment                      |
| Delete          | Delete selected segment                   |
| Shift + Drag    | Create selection zone on timeline         |
| Mouse Wheel     | Zoom timeline (when over timeline area)   |

---

## 16. Performance Tips

### For Long Files (>1 hour)
- Use windowed transcription (30–60s windows).
- Disable diarization (resource intensive).
- Use smaller models (tiny, base, small).

### For Maximum Accuracy
- Use Large V3 model.
- Enable Multi-Pass.
- Enable voice isolation.
- Use VAD segmentation.

### For Fast Results
- Use Turbo or Small model.
- Disable diarization.
- Disable voice isolation.
- Use CPU if GPU is slow (avoids VRAM bottlenecks).

### GPU Memory Management
- If you encounter CUDA out-of-memory errors, use smaller models.
- Disable Multi-Pass mode.
- Enable windowed transcription with smaller windows.
- Voice isolation and diarization consume significant VRAM.

---

## 17. Troubleshooting

| Issue                          | Solution                                                    |
|--------------------------------|-------------------------------------------------------------|
| Upload failed / chunk error    | Check internet; ensure sufficient disk space; try a smaller file or different browser |
| Model not found / download error | Provide Hugging Face token for gated models; check internet for first-time downloads |
| GPU out of memory (OOM)        | Use smaller model; disable Multi-Pass; enable windowed mode; use CPU |
| Translation not appearing      | Verify translation was enabled before processing; check API keys for LLM API engines; try Google Translate |
| Timeline not showing           | Ensure processing completed; refresh page and reload project; check browser console (F12) |
| Video won't play               | Try a different browser; verify file format is supported; check if file was uploaded properly |
| Application log                | Check `app.log` for error details                            |

---

## 18. Supported Languages

| Code | Language       |
|------|----------------|
| ro   | Romanian       |
| en   | English        |
| fr   | French         |
| de   | German         |
| es   | Spanish        |
| it   | Italian        |
| pt   | Portuguese     |
| ru   | Russian        |
| zh   | Chinese        |
| ja   | Japanese       |
| ko   | Korean         |
| ar   | Arabic         |
| hi   | Hindi          |
| tr   | Turkish        |
| nl   | Dutch          |
| pl   | Polish         |
| sv   | Swedish        |
| da   | Danish         |
| no   | Norwegian      |
| fi   | Finnish        |
| cs   | Czech          |
| hu   | Hungarian      |
| el   | Greek          |
| he   | Hebrew         |
| th   | Thai           |
| vi   | Vietnamese     |
| id   | Indonesian     |
| ms   | Malay          |
| uk   | Ukrainian      |
| bg   | Bulgarian      |
| hr   | Croatian       |

---

*Subtitratorul PRO — Professional Audio/Video Transcription and Translation Tool*
