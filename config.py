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
    SECRET_KEY = os.urandom(24).hex()
    
    # Upload
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024 * 1024  # 50GB
    CHUNK_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'mxf', 'mp3', 'wav', 'm4a', 'flac', 'ogg'}
    
    # Processing
    DEFAULT_ENGINE = 'whisper'
    AVAILABLE_ENGINES = ['whisper', 'cohere']
    DEFAULT_MODEL = 'turbo'
    AVAILABLE_MODELS = [
        'tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3', 'large-v3-turbo', 'turbo',
        'selimc/whisper-large-v3-turbo-turkish'
    ]
    COHERE_MODEL = "CohereLabs/cohere-transcribe-03-2026"
    DEFAULT_TRANSCRIBE_WINDOW = 30
    DEFAULT_TRANSCRIBE_OVERLAP = 10

    # LLM Models
    DEFAULT_LLM_MODEL = 'Qwen/Qwen3-235B-A22B-Instruct'
    AVAILABLE_LLM_MODELS = [
        'Qwen/Qwen3-235B-A22B-Instruct',
        'nvidia/Llama-3.3-70B-Instruct-NVFP4',
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
    MIN_SEGMENT_DURATION = 1.0
    MAX_SEGMENT_DURATION = 5.0
    MAX_CHARS_PER_SEGMENT = 80
    MIN_SEGMENT_RANGE = (0.5, 3.0)
    MAX_SEGMENT_RANGE = (3.0, 10.0)
    CHARS_RANGE = (40, 120)
    
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
    
    # Audio extraction
    AUDIO_FORMAT = 'wav'
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_CHANNELS = 1
    
    # Cleanup
    SESSION_LIFETIME = 86400  # 24 hours
    CLEANUP_INTERVAL = 3600  # 1 hour
    
    @classmethod
    def init_directories(cls):
        for dir_path in [cls.DATA_DIR, cls.MODELS_DIR, cls.CHUNK_UPLOAD_DIR, cls.PROCESS_DIR, cls.TEMP_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
