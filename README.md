Subtitratorul PRO - User Manual
Table of Contents

    Introduction

    Getting Started

    Uploading Files

    Processing Settings

    Starting Transcription

    Working with Results

    Timeline Management

    Translation Features

    Subtitle Styling

    Export Options

    Project Management

    Keyboard Shortcuts

    Troubleshooting

Introduction

Subtitratorul PRO is a professional-grade audio/video transcription and translation application. It supports multiple transcription engines, advanced translation features, and comprehensive subtitle editing tools.
Key Features

    Multiple Transcription Engines: OpenAI Whisper, Cohere Transcribe, NVIDIA NeMo Parakeet

    Multi-language Support: 25+ languages for transcription and translation

    Speaker Diarization: Automatic speaker identification

    Timeline Editing: Visual subtitle editing with drag-and-drop

    Selective Reprocessing: Re-transcribe specific portions only

    Professional Translation: LLM-powered translation with context awareness

    Multiple Export Formats: SRT, DOCX, JSON project files

Getting Started
System Requirements

    Browser: Chrome, Firefox, Edge, or Safari (latest versions)

    Internet: Required for model downloads and API-based translations

    Storage: At least 5GB free space for models and temporary files

First Launch

    Open your browser and navigate to http://localhost:5000

    The application will load with a modern glass-morphism interface

    Your device (CPU/GPU) will be detected automatically

Uploading Files
Supported Formats
Type	Extensions
Video	MP4, AVI, MOV, MKV, WebM, MXF
Audio	MP3, WAV, M4A, FLAC, OGG

Maximum file size: 50GB per file
Upload Methods
Drag & Drop

    Drag your file from file explorer

    Drop it onto the upload area (dashed box)

Click to Upload

    Click the upload area

    Select your file from the file dialog

Upload Progress

During upload you can monitor:

    Progress percentage and progress bar

    Chunk indicators showing uploaded segments

    Speed (MB/s) and ETA

    Pause/Resume and Cancel buttons

After Upload

    The video/audio player will appear automatically

    A Start Processing button will appear (bottom-right corner)

    Your file is now ready for processing

Processing Settings

The settings panel contains numerous options for controlling the transcription process.
Basic Settings (Visible by default)
Setting	Description
Preset	Load predefined configurations
Motor Transcriere	Select transcription engine (Whisper/Cohere/Nemo)
Model	Select model size (tiny → large-v3)
Limba Sursă	Source language (or auto-detect)
Segmentare	Min/max duration, max characters per subtitle
VAD	Voice Activity Detection (detects speech pauses)
Marjă 1s	Adds 1-second margin after speech
Advanced Settings (Toggle "Opțiuni Avansate")
Processing Options
Setting	Description
Fereastră Transcriere	Window size for chunked transcription (seconds)
Overlap Transcriere	Overlap between windows (seconds)
Izolare Voce	Separates voice from background noise
Elimină repetiții	Removes duplicate consecutive phrases
Segmente secvențiale	Ensures no overlapping segments
Diarizare	Identifies different speakers
Multi-Pass	Triple-pass transcription (slower, more accurate)
Language-Specific Options

    Mixed Turkish Mode (when Turkish is selected): Combines Large V3 and Turbo models

    Mixed Korean Mode (when Korean is selected): Optimized for Korean language

Region Processing

    Start and End fields: Process only a portion of the video

    Timestamps will be adjusted to the full video when exporting

Hugging Face Token

Required for:

    Gated models (e.g., pyannote for diarization)

    Some custom Whisper models

Saving Presets

    Configure your desired settings

    Enter a name in the "Nume preset" field

    Click Salvează

    Your preset appears in the dropdown (under "Salvate de tine")

To delete a user preset: Select it and click the trash icon.
Starting Transcription
Before Starting

    Verify your file is uploaded (player shows preview)

    Adjust settings as needed

    Enable translation if required (see Translation section)

Start Processing

    Click the Start Procesare button (floating, bottom-right)

    Processing status will appear:

        Current step message

        Progress percentage

        Progress bar animation

    Cancel at any time using the cancel button

Processing Steps

    Audio extraction (for video files)

    Voice isolation (if enabled)

    Transcription (main processing)

    Diarization (if enabled)

    Gender detection (if diarization enabled)

    Subtitle segmentation

    Deduplication

    Translation (if enabled)

Completion

    Results panel appears with transcribed segments

    Video player shows subtitles in real-time

    Timeline view appears for visual editing

Working with Results
Segments Panel

The results panel displays all transcribed segments in a scrollable list.
Segment Information

Each segment shows:

    Index number

    Text (editable)

    Timestamp (start → end)

    Speaker (if diarization enabled) with gender icon

    Action buttons (play, edit, delete)

Editing a Segment

    Click directly on the text

    Type your changes

    Click outside or press Enter to save

Or click the ✏️ edit button to focus the text field.
Playing a Segment

Click the ▶️ play button or click the segment to seek to its start time.
Deleting a Segment

Click the 🗑️ delete button and confirm.
Adding a New Segment

    Click ➕ Adaugă segment in the toolbar

    New segment is added after the selected segment (or at the end)

    Default duration is 2 seconds

    Edit text as needed

Tabs
Tab	Description
Original	Original transcribed text
Raw	Unprocessed raw transcript (read-only)
Traducere	Translated text (appears after translation)
Full Text View

Switch to Traducere tab to see all text in a single editable textarea.
Timeline Management

The timeline provides visual editing of subtitles with pixel-precise control.
Timeline Components

    Ruler: Shows time markers (major and minor ticks)

    Segments: Color-coded blocks for each subtitle

    Playhead: Red vertical line showing current playback position

    Selection Overlay: Blue regions for selective reprocessing

Navigation

    Scroll horizontally to move through the timeline

    Zoom using +/- buttons or mouse wheel

    Click anywhere on timeline to seek

Selecting and Editing Segments
Click to Select

    Click a segment block to select it

    Selected segments have white border

    Segment list also highlights selected item

Resize Segment

    Drag left handle (left edge) → change start time

    Drag right handle (right edge) → change end time

Move Segment

    Drag from middle of segment → move entire segment

Delete Segment

    Hover over segment → red × appears on top-right

    Click to delete

Zone Selection & Reprocessing

This powerful feature lets you re-transcribe only specific portions.
Mode Selection

Check "Mod Selecție" or hold Shift while dragging.
Creating Zones

    Enable selection mode

    Click and drag on the timeline

    Release to create a zone (blue highlight)

Manual Zone Entry

    Enter start time (HH:MM:SS format)

    Enter end time

    Click Adaugă zonă

Managing Zones

    Zones appear as tags below the timeline

    Click × on a tag to remove it

    Șterge selecțiile removes all zones

Reprocessing Zones

    Select one or more zones

    Click Re-procesează selecția

    New transcriptions will replace existing content in those zones

    Translations are preserved/updated automatically

Translation Features
Enabling Translation

    Check "Activează traducerea"

    Translation settings appear below

Basic Translation Settings
Setting	Description
Limba Țintă	Target language for translation
Engine Traducere	Translation engine (Google, LLM API, VLLM, NLLB)
Advanced Translation Settings
Translation Engines
Engine	Description
Google Translate	Fast, free, good for most languages
LLM API	Claude/OpenAI/Custom API (professional quality)
VLLM	Local LLM (Qwen3-235B, Llama) - requires GPU
NLLB-200	Facebook's neural translation model
LLM API Configuration (for LLM API engine)
Setting	Description
Provider	Claude, OpenAI, or Custom
API Key	Your API key (stored only for current session)
Model	Model identifier (e.g., gpt-4o, claude-3-5-sonnet)
Context & Refinement
Setting	Description
Context Conținut	Describe the content (film genre, characters) for better translation
Corectură Traducere	Additional LLM pass to refine translations
Adding Multiple Target Languages

    Click Adaugă altă limbă

    Select additional language from dropdown

    Remove with trash button

Translation Process

    Translations run automatically after transcription

    Progress shown in the status panel

    Results appear in Traducere tab and Translation Results panel

    Each translation language has its own export options

Subtitle Styling

Click 🎨 Stil in the player controls to open the styling panel.
Style Options
Setting	Options
Font Family	Arial, Segoe UI, Tahoma, Verdana, Courier New
Mărime (px)	12-60 pixels
Culoare Text	Any color (color picker)
Culoare Fundal	Background color + opacity slider
Efecte	Fără (none), Drop Shadow, Outline
Poziție X/Y	Offset in pixels
Position Toggle

Check "Subtitrare Sus" to move subtitles to the top of the video instead of bottom.

All styling is applied in real-time as you adjust controls.
Export Options
SRT Export

Method 1: Click 💾 Export SRT in results toolbar
Method 2: Press Ctrl+S

Exports subtitles in standard SRT format:
text

1
00:00:01,000 --> 00:00:03,500
Hello, world!

2
00:00:04,000 --> 00:00:06,000
This is a subtitle.

DOCX Export (Professional Format)

Click 📄 Export DOCX to open the export dialog.
DOCX Metadata Fields
Field	Description
Titlu	Film/episode title
Serie / Episod	Series and episode identifier
Nume Traducător	Translator's name
Nume Redactor	Editor's name
Legacy Diacritics

Check "Conversie diacritice legacy" to convert:

    ș → ş

    ț → ţ

This is useful for older subtitle specifications.

The DOCX output includes:

    Title page with metadata table

    Numbered segments with timestamps

    Proper line wrapping (max 40 chars per line)

Copy Full Text

Click 📋 Copiază text to copy all subtitle text to clipboard.
Project Management
Saving a Project

Click 💾 Exportă ca Proiect (.json) from the File menu.

Saves:

    All segments with timestamps

    All translations

    Subtitle styling preferences

    File references

Loading a Project

    Click 📂 Deschide Proiect (.json) from File menu

    Select your .json project file

    All data is restored including video file (if available)

Auto-save

Projects are not auto-saved. Manual saving is required.
Keyboard Shortcuts
Shortcut	Action
Space	Play/Pause video (when no input focused)
Ctrl+S	Export SRT
Ctrl+D	Open DOCX export dialog
Esc	Close modal dialogs
↑	Seek to previous segment
↓	Seek to next segment
Delete	Delete selected segment
Shift + Drag	Create selection zone on timeline
Mouse Wheel	Zoom timeline (over timeline area)
Troubleshooting
Common Issues
"Upload failed" or "Chunk error"

    Check your internet connection

    Ensure sufficient disk space

    Try a smaller file or different browser

"Model not found" or download errors

    Ensure Hugging Face token is provided for gated models

    Check internet connection for first-time model downloads

    Models are cached after first download

GPU out of memory (CUDA OOM)

    Use smaller model (tiny, base, small)

    Disable Multi-Pass mode

    Enable windowed transcription (smaller windows)

    Use CPU mode (slower but stable)

Translation not appearing

    Verify translation was enabled before processing

    Check API keys for LLM API engines

    Try Google Translate as fallback

Timeline not showing

    Ensure processing completed successfully

    Refresh the page and reload the project

    Check browser console for errors

Video won't play

    Try a different browser

    Ensure file format is supported

    Check if video file was properly uploaded

Performance Tips

    For long files (>1 hour):

        Use windowed transcription (30-60s windows)

        Disable diarization (resource intensive)

        Use smaller models

    For maximum accuracy:

        Use Large V3 model

        Enable Multi-Pass

        Enable voice isolation

        Use VAD segmentation

    For fast results:

        Use Turbo or Small model

        Disable diarization

        Disable voice isolation

        Use CPU (if GPU is slow)

Getting Help

    Check the application log file (app.log)

    Use browser developer console (F12) for JavaScript errors

    Ensure all Python dependencies are installed

Appendix: Supported Languages
Code	Language
ro	Română
en	English
fr	Français
de	Deutsch
es	Español
it	Italiano
pt	Português
ru	Русский
zh	中文
ja	日本語
ko	한국어
ar	العربية
hi	हिन्दी
tr	Türkçe
nl	Nederlands
pl	Polski
sv	Svenska
da	Dansk
no	Norsk
fi	Suomi
cs	Čeština
hu	Magyar
el	Ελληνικά
he	עברית
th	ไทย
vi	Tiếng Việt

Subtitratorul PRO - Professional Audio/Video Transcription and Translation Tool
