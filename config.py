import os
from pathlib import Path

class Config:
    # Directories
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    MODELS_DIR = BASE_DIR / 'models'
    CHUNK_UPLOAD_DIR = DATA_DIR / 'chunk_uploads'
    PROCESS_DIR = DATA_DIR / 'process'
    TEMP_DIR = DATA_DIR / 'temp'
    
    # Server
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = True
    _secret_file = Path(__file__).parent / "data" / "secret.key"
    _secret_file.parent.mkdir(parents=True, exist_ok=True)
    if _secret_file.exists():
        SECRET_KEY = _secret_file.read_text().strip()
    else:
        SECRET_KEY = os.urandom(32).hex()
        _secret_file.write_text(SECRET_KEY)
    
    # Library
    LIBRARY_DIR = DATA_DIR / 'library'
    LIBRARY_FILE = DATA_DIR / 'library.json'

    # Upload
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024 * 1024  # 50GB
    CHUNK_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'mxf', 'mp3', 'wav', 'm4a', 'flac', 'ogg'}
    
    # Processing
    DEFAULT_ENGINE = 'whisper'
    AVAILABLE_ENGINES = ['whisper', 'cohere', 'nemo']
    DEFAULT_MODEL = 'large-v3'
    AVAILABLE_MODELS = [
        'tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3', 'large-v3-turbo',
        'selimc/whisper-large-v3-turbo-turkish',
        'Farazzzzzzz/whisper-tiny_to_korean_accent2',
        'parakeet-v3', 'canary', 'nemotron-3.5'
    ]
    COHERE_MODEL = "CohereLabs/cohere-transcribe-03-2026"
    DEFAULT_TRANSCRIBE_WINDOW = 30
    DEFAULT_TRANSCRIBE_OVERLAP = 15
    DEFAULT_SEGMENT_SPACING = 2

    # LLM Models
    DEFAULT_LLM_MODEL = 'google/gemma-4-26B-A4B-it'
    AVAILABLE_LLM_MODELS = [
        'allura-forge/Llama-3.3-8B-Instruct',
        'google/gemma-3-12b-it',
        'google/gemma-4-E4B-it',
        'google/gemma-4-26B-A4B-it',
        'OpenLLM-Ro/RoMistral-7b-Instruct'
    ]
    ROMISTRAL_MODEL = 'OpenLLM-Ro/RoMistral-7b-Instruct'
    DEFAULT_TRANSLATE_GROUP = 10
    VLLM_GPU_MEMORY_UTILIZATION = 0.9  # Increased to 90% to allow loading large 70B models
    VLLM_ENFORCE_EAGER = True  # Use eager mode to save VRAM from CUDA graphs

    DEFAULT_LANGUAGE = 'auto'
    PROCESSING_TIMEOUT = 7200  # 2 hours
    HEARTBEAT_INTERVAL = 30  # seconds
    
    # Segmentation
    MIN_SEGMENT_DURATION = 1.0  # 25 FR / 1s — durata minimă absolută
    MAX_SEGMENT_DURATION = 5.0  # 200 FR / 8s — durata maximă absolută
    MAX_CHARS_PER_SEGMENT = 76  # 2 lines x 38 chars
    MIN_SEGMENT_RANGE = (0.5, 3.0)
    MAX_SEGMENT_RANGE = (3.0, 10.0)
    CHARS_RANGE = (40, 120)
    
    # Subtitle formatting
    CPR = 38                         # caractere pe rând (chars per line)
    MAX_LINES = 2                    # rânduri maxim per subtitlu
    READING_SPEED = 15               # caractere/secundă — viteză de citire
    IDEAL_DURATION_RATIO = 1.0       # 100% din durata ideală
    MIN_ACCEPTED_RATIO = 0.9         # 90% din durata ideală
    MAX_ACCEPTED_RATIO = 1.1         # 110% din durata ideală
    EXCEPTIONAL_MIN_RATIO = 0.8      # 80% în cazuri excepționale
    MAX_PROPS = 3                    # propoziții maxim per subtitlu
    DIALOG_MAX_PROPS = 2             # propoziții maxim per subtitlu cu dialog
    
    # Translation
    SUPPORTED_LANGUAGES = {
        'ro': 'Română',
        'en': 'Engleză',
        'fr': 'Franceză',
        'de': 'Germană',
        'es': 'Spaniolă',
        'it': 'Italiană',
        'pt': 'Portugheză',
        'ru': 'Rusă',
        'zh': 'Chineză',
        'ja': 'Japoneză',
        'ko': 'Coreeană',
        'ar': 'Arabă',
        'hi': 'Hindi',
        'tr': 'Turcă',
        'nl': 'Olandeză',
        'pl': 'Poloneză',
        'sv': 'Suedeză',
        'da': 'Daneză',
        'no': 'Norvegiană',
        'fi': 'Finlandeză',
        'cs': 'Cehă',
        'hu': 'Maghiară',
        'el': 'Greacă',
        'he': 'Ebraică',
        'th': 'Thailandeză',
        'vi': 'Vietnameză',
        'id': 'Indoneziană',
        'ms': 'Malaieză',
        'uk': 'Ucraineană',
        'bg': 'Bulgară',
        'hr': 'Croată'
    }
    
    # OCR
    DEFAULT_OCR_CONF = 70
    DEFAULT_OCR_SIM = 80

    # Audio extraction
    AUDIO_FORMAT = 'wav'
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_CHANNELS = 1
    
    # Cleanup
    SESSION_LIFETIME = 86400  # 24 hours
    CLEANUP_INTERVAL = 3600  # 1 hour
    
    @classmethod
    def init_directories(cls):
        for dir_path in [cls.DATA_DIR, cls.MODELS_DIR, cls.CHUNK_UPLOAD_DIR, cls.PROCESS_DIR, cls.TEMP_DIR, cls.LIBRARY_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
