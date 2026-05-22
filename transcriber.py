import sys
import types
from unittest.mock import MagicMock

# Mock torchcodec to prevent environment crashes on load
def create_torchcodec_mock():
    mock = types.ModuleType('torchcodec')
    mock.__spec__ = MagicMock()
    mock.__spec__.name = 'torchcodec'
    mock.__spec__.loader = MagicMock()
    mock.__spec__.origin = 'mock'
    mock.__spec__.submodule_search_locations = []
    mock.__version__ = '0.0.0'
    mock.__path__ = []
    mock.__file__ = 'mock'

    # Create proper classes that can be used with isinstance()
    class AudioSamples:
        pass

    # Create decoders module with AudioDecoder class
    mock.decoders = types.ModuleType('torchcodec.decoders')

    class AudioDecoder:
        """Mock AudioDecoder class that's a proper type"""
        pass

    # Add AudioDecoder to decoders module
    mock.decoders.AudioDecoder = AudioDecoder

    # Create encoders module
    mock.encoders = types.ModuleType('torchcodec.encoders')

    # Add other mock attributes
    mock.AudioSamples = AudioSamples
    mock.load = MagicMock(return_value={})
    mock.is_available = MagicMock(return_value=False)

    # Add VideoDecoder and other common classes that might be checked
    class VideoDecoder:
        pass

    mock.decoders.VideoDecoder = VideoDecoder

    # Ensure the decoders module is also accessible directly
    # This handles cases where code does 'from torchcodec.decoders import AudioDecoder'
    sys.modules['torchcodec.decoders'] = mock.decoders
    sys.modules['torchcodec.encoders'] = mock.encoders

    return mock

# Apply mock before other imports
if 'torchcodec' not in sys.modules:
    sys.modules['torchcodec'] = create_torchcodec_mock()

import whisper
import whisperx
import torch
import numpy as np
import librosa
import noisereduce as nr
import soundfile as sf
from pathlib import Path
import logging
import re
import gc
from typing import Optional, Dict, Any, List, Tuple
import time
from config import Config

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    def __init__(self):
        self.models = {}  # Vanilla Whisper models or Transformers Pipelines
        self.models_processors = {} # Processors for custom HF models (not used for pipelines)
        self.whisperx_models = {}  # WhisperX models
        self.alignment_models = {}  # WhisperX alignment models
        self.current_model = None
        self.device = self._detect_device()
        self.cohere_processor = None
        self.alignment_model = None # For Cohere fallback alignment
        self.alignment_metadata = None
        
    def _detect_device(self) -> str:
        if torch.cuda.is_available():
            logger.info("CUDA GPU detected and available")
            return "cuda"
        logger.info("No GPU detected, using CPU")
        return "cpu"
    
    def ensure_model_downloaded(self, model_id: str, cache_dir: str = "data/models") -> str:
        """Explicitly ensure a Hugging Face model is downloaded"""
        try:
            from huggingface_hub import snapshot_download
            logger.info(f"Checking/Downloading model: {model_id}")
            # snapshot_download is smart enough to skip if already present
            path = snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir
            )
            return path
        except Exception as e:
            logger.error(f"Error downloading {model_id}: {e}")
            return model_id # Fallback to original ID

    def load_model(self, model_name: str = 'small') -> Dict[str, Any]:
        """Load Whisper model with lazy loading, supporting both OpenAI names and HF IDs"""
        try:
            if model_name in self.models:
                self.current_model = self.models[model_name]
                return {"status": "loaded", "model": model_name, "device": self.device}
            
            logger.info(f"Loading model: {model_name} on {self.device}")
            start_time = time.time()
            
            # Handle Hugging Face models (containing '/')
            if '/' in model_name:
                logger.info(f"Loading custom Hugging Face model: {model_name}")
                from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

                # Check if already loaded
                if model_name in self.models:
                    self.current_model = self.models[model_name]
                    return {"status": "loaded", "model": model_name, "device": self.device}

                # Ensure downloaded
                model_path = self.ensure_model_downloaded(model_name)

                # Load model
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    trust_remote_code=True
                ).to(self.device)

                # Load processor (with fallback to base turbo)
                try:
                    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, clean_up_tokenization_spaces=False)
                except Exception as e:
                    logger.warning(f"AutoProcessor failed for {model_name}, trying local path: {e}")
                    try:
                        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, clean_up_tokenization_spaces=False)
                    except Exception:
                        logger.warning("Local AutoProcessor failed, falling back to base turbo processor")
                        processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo", clean_up_tokenization_spaces=False)

                # Create pipeline using explicitly loaded model and processor
                # We don't pass torch_dtype here as the model is already loaded with the correct dtype
                pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    chunk_length_s=30,
                    device=0 if self.device == "cuda" else -1,
                )

                self.current_model = pipe
                self.models[model_name] = pipe
            else:
                # Map names for OpenAI Whisper compatibility
                whisper_model_name = model_name
                if model_name == 'large-v3-turbo':
                    whisper_model_name = 'turbo'

                try:
                    self.current_model = whisper.load_model(
                        whisper_model_name,
                        device=self.device,
                        download_root='data/models'
                    )
                except Exception as e:
                    logger.error(f"Failed to load Whisper {whisper_model_name}: {e}")
                    raise
            
            load_time = time.time() - start_time
            self.models[model_name] = self.current_model
            
            if len(self.models) > 2:
                oldest_model = list(self.models.keys())[0]
                del self.models[oldest_model]
                torch.cuda.empty_cache() if self.device == "cuda" else None
            
            logger.info(f"Model {model_name} loaded in {load_time:.2f}s")
            return {
                "status": "loaded", 
                "model": model_name, 
                "device": self.device,
                "load_time": load_time
            }
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            if self.device == "cuda":
                logger.info("Falling back to CPU")
                self.device = "cpu"
                return self.load_model(model_name)
            raise
    
    def transcribe_audio(
        self, 
        audio_path: str, 
        model_name: str = 'small',
        language: Optional[str] = None,
        task_id: str = None,
        progress_callback = None
    ) -> Dict[str, Any]:
        """Transcribe audio file with progress tracking"""
        try:
            model_info = self.load_model(model_name)
            
            if progress_callback:
                progress_callback(10, "Loading audio...")
            
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            audio = self._normalize_audio(audio)
            
            if progress_callback:
                progress_callback(20, "Transcribing...")

            # Handle Transformers Pipeline objects
            if hasattr(self.current_model, "__call__") and hasattr(self.current_model, "task") and self.current_model.task == "automatic-speech-recognition":
                logger.info("Using transformers pipeline for transcription")

                # Build generate_kwargs properly
                generate_kwargs = {}
                if language and language != 'auto':
                    generate_kwargs["language"] = language
                    generate_kwargs["task"] = "transcribe"

                # Don't pass max_new_tokens through generate_kwargs - let the pipeline handle it
                # The pipeline will use its own chunking mechanism

                try:
                    # First try: pipeline call with chunking and timestamps
                    pipe_res = self.current_model(
                        audio,
                        chunk_length_s=30,
                        stride_length_s=5,
                        return_timestamps=True,
                        generate_kwargs=generate_kwargs if generate_kwargs else None
                    )
                except Exception as e1:
                    logger.warning(f"First pipeline attempt failed: {e1}")
                    try:
                        # Second try: without return_timestamps but with chunking
                        pipe_res = self.current_model(
                            audio,
                            chunk_length_s=30,
                            stride_length_s=5,
                            generate_kwargs=generate_kwargs if generate_kwargs else None
                        )
                    except Exception as e2:
                        logger.warning(f"Second pipeline attempt failed: {e2}")
                        # Third try: minimal call
                        pipe_res = self.current_model(audio)

                # Parse the result
                segments = []
                if isinstance(pipe_res, dict):
                    # Handle chunked output
                    if "chunks" in pipe_res:
                        current_time = 0.0
                        for chunk in pipe_res["chunks"]:
                            text = chunk.get("text", "").strip()
                            if not text:
                                continue

                            ts = chunk.get("timestamp")
                            if ts and len(ts) == 2:
                                start = float(ts[0]) if ts[0] is not None else current_time
                                end = float(ts[1]) if ts[1] is not None else start + 2.0
                            else:
                                start = current_time
                                end = start + 2.0

                            segments.append({
                                "start": start,
                                "end": end,
                                "text": text
                            })
                            current_time = end
                    elif "text" in pipe_res:
                        # Single segment output
                        segments.append({
                            "start": 0.0,
                            "end": len(audio) / sr,
                            "text": pipe_res["text"].strip()
                        })

                    result = {
                        "text": pipe_res.get("text", ""),
                        "segments": segments,
                        "language": language or "unknown"
                    }
                elif isinstance(pipe_res, str):
                    # Pipeline returned just text
                    result = {
                        "text": pipe_res,
                        "segments": [{"start": 0.0, "end": len(audio) / sr, "text": pipe_res}],
                        "language": language or "unknown"
                    }
                else:
                    # Unknown format, try to convert
                    logger.warning(f"Unexpected pipeline output type: {type(pipe_res)}")
                    result = {
                        "text": str(pipe_res),
                        "segments": [{"start": 0.0, "end": len(audio) / sr, "text": str(pipe_res)}],
                        "language": language or "unknown"
                    }
            else:
                # Standard Whisper model
                options = {
                    "task": "transcribe",
                    "verbose": False,
                    "fp16": self.device == "cuda"
                }

                if language and language != 'auto':
                    options["language"] = language

                result = self.current_model.transcribe(audio, **options)
            
            if progress_callback:
                progress_callback(90, "Post-processing...")
            
            # Capture raw text BEFORE hallucination cleaning
            result["raw_text"] = result.get("text", "")

            result = self._clean_hallucinations(result)
            
            if progress_callback:
                progress_callback(100, "Complete!")
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def transcribe_with_windowing(
        self, 
        audio_path: str, 
        model_name: str = 'small',
        language: Optional[str] = None,
        window_size: int = 30,
        overlap: int = 10,
        progress_callback = None
    ) -> Dict[str, Any]:
        """Process large audio files in windows to save memory"""
        try:
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            total_duration = len(audio) / sr
            
            if total_duration <= window_size:
                return self.transcribe_audio(audio_path, model_name, language, progress_callback)
            
            segments = []
            window_samples = window_size * sr
            overlap_samples = overlap * sr
            step = window_samples - overlap_samples
            total_windows = int(np.ceil(len(audio) / step))
            
            for i in range(0, len(audio), step):
                window_num = i // step + 1
                if progress_callback:
                    progress = int((window_num / total_windows) * 100)
                    progress_callback(progress, f"Processing window {window_num}/{total_windows}")
                
                window_audio = audio[i:i + window_samples]
                temp_file = Path(f"data/temp/window_{window_num}.wav")
                temp_file.parent.mkdir(parents=True, exist_ok=True)
                sf.write(temp_file, window_audio, sr)
                
                result = self.transcribe_audio(str(temp_file), model_name, language)
                
                offset = i / sr
                step_duration = step / sr

                for seg in result.get('segments', []):
                    is_last_window = (i + window_samples) >= len(audio)
                    if is_last_window or seg['start'] < step_duration:
                        seg['start'] += offset
                        seg['end'] += offset
                        segments.append(seg)
                
                temp_file.unlink(missing_ok=True)
            
            return {
                'text': ' '.join([s.get('text', '') for s in segments]),
                'segments': segments,
                'language': result.get('language', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Windowed transcription error: {e}")
            raise
    
    def isolate_voice(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Isolate voice by reducing background noise and music"""
        try:
            logger.info("Isolating voice using spectral gating...")
            reduced_noise = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8)
            return reduced_noise
        except Exception as e:
            logger.error(f"Voice isolation error: {e}")
            return audio

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to improve transcription quality"""
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            target_rms = 0.1
            audio = audio * (target_rms / rms)
        return np.clip(audio, -1.0, 1.0)
    
    def _clean_hallucinations(self, result: Dict) -> Dict:
        """Remove common Whisper hallucinations"""
        hallucination_patterns = [
            r'(?i)(please like and subscribe|check out my channel|thanks for watching)',
            r'(?i)(visit our website|follow us on|subscribe to)',
            r'(?i)(background music playing|music fades|applause)',
            r'(?i)(subtitles by|amara\.org|opensubtitles)',
            r'(?i)(thank you for watching|see you in the next video)',
            r'(?i)^\s*$',
            r'(?i)(♪|♫|♬|♩|♭)',
            r'(?i)(\[.*?\])',
            r'(?i)(\*.*?\*)'
        ]
        
        if 'segments' in result:
            cleaned_segments = []
            for segment in result['segments']:
                text = segment.get('text', '').strip()
                is_hallucination = False
                
                for pattern in hallucination_patterns:
                    if re.search(pattern, text):
                        is_hallucination = True
                        break
                
                words = text.split()
                if len(words) > 4:
                    from collections import Counter
                    counts = Counter(words)
                    most_common, count = counts.most_common(1)[0]
                    if count / len(words) > 0.7:
                        is_hallucination = True

                duration = segment.get('end', 0) - segment.get('start', 0)
                if duration < 0.1 and len(text) > 10:
                    is_hallucination = True

                if not is_hallucination and text:
                    cleaned_segments.append(segment)
            
            result['segments'] = cleaned_segments
            result['text'] = ' '.join([s.get('text', '') for s in cleaned_segments])
        
        return result
    
    def extract_audio_from_video(self, video_path: str, output_path: str, start_time: float = 0, duration: Optional[float] = None) -> str:
        """Extract audio from video using ffmpeg, with optional region support"""
        import ffmpeg
        
        try:
            input_args = {}
            if start_time > 0:
                input_args['ss'] = start_time

            stream = ffmpeg.input(video_path, **input_args)

            output_args = {'acodec': 'pcm_s16le', 'ac': 1, 'ar': '16k'}
            if duration is not None:
                output_args['t'] = duration

            stream = ffmpeg.output(stream, output_path, **output_args)
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            return output_path
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            raise
    
    def transcribe_whisperx(
        self,
        audio_path: str,
        model_name: str = 'large-v3',
        language: Optional[str] = None,
        batch_size: int = 16,
        progress_callback = None
    ) -> Dict[str, Any]:
        """Transcribe audio using WhisperX for state-of-the-art alignment and accuracy"""
        try:
            device = self.device
            compute_type = "float16" if device == "cuda" else "int8"

            # Map names for WhisperX / Faster-Whisper compatibility
            wx_model_name = model_name
            if model_name == 'large-v3-turbo':
                wx_model_name = 'turbo'

            # 1. Load/Get WhisperX Model
            if wx_model_name not in self.whisperx_models:
                if progress_callback:
                    progress_callback(5, f"Loading WhisperX model {wx_model_name}...")

                logger.info(f"Loading WhisperX model: {wx_model_name} on {device}")
                try:
                    self.whisperx_models[wx_model_name] = whisperx.load_model(
                        wx_model_name,
                        device,
                        compute_type=compute_type,
                        download_root='data/models'
                    )
                except Exception as e:
                    logger.error(f"Failed to load WhisperX {wx_model_name}: {e}")
                    raise

            model = self.whisperx_models[wx_model_name]

            # 2. Transcribe
            if progress_callback:
                progress_callback(15, "Transcribing with WhisperX...")

            audio = whisperx.load_audio(audio_path)
            # WhisperX transcribe takes audio array, model, and other params
            result = model.transcribe(audio, batch_size=batch_size, language=language)

            raw_text = " ".join([seg["text"] for seg in result["segments"]])
            lang_code = result["language"]

            # 3. Align
            try:
                if progress_callback:
                    progress_callback(60, f"Aligning results ({lang_code})...")

                # Load/Get Alignment Model
                if lang_code not in self.alignment_models:
                    logger.info(f"Loading WhisperX alignment model for {lang_code}")
                    model_a, metadata = whisperx.load_align_model(language_code=lang_code, device=device)
                    self.alignment_models[lang_code] = (model_a, metadata)

                model_a, metadata = self.alignment_models[lang_code]

                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    device,
                    return_char_alignments=False
                )

                if progress_callback:
                    progress_callback(100, "WhisperX transcription and alignment complete!")

                return {
                    "segments": result["segments"],
                    "language": result["language"],
                    "text": " ".join([seg["text"] for seg in result["segments"]]),
                    "raw_text": raw_text,
                    "method": "whisperx"
                }
            except Exception as align_e:
                logger.error(f"WhisperX alignment failed: {align_e}")
                # Return unaligned segments instead of falling back to vanilla whisper
                return {
                    "segments": result["segments"],
                    "language": lang_code,
                    "text": raw_text,
                    "raw_text": raw_text,
                    "method": "whisperx_unaligned",
                    "alignment_error": str(align_e)
                }

        except Exception as e:
            logger.error(f"WhisperX transcription error: {e}")
            logger.info("Falling back to Transcribe-then-Align (Vanilla Whisper + WhisperX Align)...")

            # 1. Transcribe with Vanilla Whisper (handles more model formats)
            whisper_result = self.transcribe_audio(audio_path, model_name, language, progress_callback=progress_callback)
            raw_text = whisper_result.get("text", "")

            try:
                # 2. Align with WhisperX
                # Clear VRAM before loading alignment model if needed
                # BUT ONLY IF we are on GPU, as CPU alignment is memory-efficient
                if self.device == "cuda":
                    logger.info("Clearing VRAM for alignment phase...")
                    self.unload_model()

                logger.info("Attempting WhisperX alignment on fallback results...")
                audio = whisperx.load_audio(audio_path)
                lang_code = whisper_result.get("language", language or "en")

                if lang_code not in self.alignment_models:
                    model_a, metadata = whisperx.load_align_model(language_code=lang_code, device=self.device)
                    self.alignment_models[lang_code] = (model_a, metadata)

                model_a, metadata = self.alignment_models[lang_code]

                aligned_result = whisperx.align(
                    whisper_result["segments"],
                    model_a,
                    metadata,
                    audio,
                    self.device,
                    return_char_alignments=False
                )

                return {
                    "segments": aligned_result["segments"],
                    "language": lang_code,
                    "text": whisper_result["text"],
                    "raw_text": raw_text,
                    "method": "whisper_plus_whisperx_align"
                }
            except Exception as align_e:
                logger.error(f"Alignment fallback also failed: {align_e}")
                whisper_result["raw_text"] = raw_text
                return whisper_result

    def load_cohere_model(self):
        """Load Cohere Transcribe model and processor"""
        try:
            from transformers import AutoProcessor, CohereAsrForConditionalGeneration
            
            model_name = Config.COHERE_MODEL

            if model_name not in self.models:
                logger.info(f"Loading Cohere model: {model_name}")

                self.cohere_processor = AutoProcessor.from_pretrained(model_name)
                logger.info(f"Processor class: {type(self.cohere_processor)}")

                model = CohereAsrForConditionalGeneration.from_pretrained(
                    model_name,
                    device_map="auto" if self.device == "cuda" else None,
                    torch_dtype=torch.float32,
                )
                
                logger.info(f"Loaded model class: {type(model)}")
                logger.info(f"Has generate: {hasattr(model, 'generate')}")
                
                model.eval()
                self.models[model_name] = model

            self.current_model = self.models[model_name]
            return {"status": "loaded", "model": "cohere", "device": self.device}

        except Exception as e:
            logger.error(f"Error loading Cohere model: {e}")
            raise

    def load_alignment_model(self, language_code: str = "en"):
        """Load wav2vec2 alignment model for forced alignment of Cohere text"""
        try:
            if self.alignment_model is not None and self.alignment_metadata is not None:
                return self.alignment_model, self.alignment_metadata
            
            logger.info(f"Loading wav2vec2 alignment model for language: {language_code}")
            
            # This is the wav2vec2 phoneme recognition model used for forced alignment
            # NOT Whisper - it's specifically for aligning text to audio
            self.alignment_model, self.alignment_metadata = whisperx.load_align_model(
                language_code=language_code,
                device=self.device
            )
            
            logger.info("Wav2vec2 alignment model loaded successfully")
            return self.alignment_model, self.alignment_metadata
            
        except Exception as e:
            logger.error(f"Error loading alignment model: {e}")
            raise

    def align_cohere_transcription_with_phonemes(
        self,
        audio: np.ndarray,
        cohere_segments: List[Dict[str, Any]],
        language_code: str = "en",
        return_char_alignments: bool = False
    ) -> Dict[str, Any]:
        """
        Use phoneme-based forced alignment (wav2vec2) to precisely align Cohere's 
        transcription text to the audio timeline.
        
        Pipeline:
        - Cohere: Provides the TEXT transcription (accurate text, rough timestamps)
        - wav2vec2: Does FORCED ALIGNMENT (precise word-level timestamps)
        
        Args:
            audio: Audio array (16kHz, mono, float32)
            cohere_segments: Segments from Cohere with text but inaccurate timestamps
            language_code: Language code for phoneme model
            return_char_alignments: Whether to return character-level timestamps
            
        Returns:
            Dictionary with precisely aligned segments
        """
        try:
            logger.info("Starting wav2vec2 forced alignment for Cohere transcription...")
            
            # Load the wav2vec2 alignment model
            model_a, metadata = self.load_alignment_model(language_code)
            logger.info(f"Loaded wav2vec2 alignment model for {language_code}")
            
            # Run forced alignment
            # This takes Cohere's text and finds EXACTLY where each word occurs in the audio
            result = whisperx.align(
                cohere_segments,           # Cohere's transcription text
                model_a,                   # wav2vec2 phoneme model
                metadata,                  # Language metadata
                audio,                     # Original audio
                self.device,               # Device (cuda/cpu)
                return_char_alignments=return_char_alignments,
                print_progress=False       # Suppress debug output
            )
            
            # Add alignment metadata
            aligned_word_count = 0
            for segment in result["segments"]:
                if "words" in segment:
                    # Each word now has precise start/end timestamps
                    aligned_word_count += len(segment["words"])
                    segment["alignment_source"] = "wav2vec2_phoneme_forced"
                    
                    # Calculate per-word alignment confidence
                    word_scores = [w.get("score", 0.0) for w in segment["words"]]
                    segment["mean_alignment_score"] = np.mean(word_scores) if word_scores else 0.0
            
            logger.info(
                f"Alignment complete: {aligned_word_count} words aligned "
                f"across {len(result['segments'])} segments"
            )
            
            return {
                "segments": result["segments"],
                "aligned": True,
                "alignment_method": "wav2vec2_phoneme_forced_alignment",
                "word_count": aligned_word_count
            }
            
        except Exception as e:
            logger.error(f"Phoneme alignment failed: {e}")
            logger.info("Falling back to basic timestamp estimation")
            return self._basic_timestamp_estimation(audio, cohere_segments)

    def _basic_timestamp_estimation(
        self, 
        audio: np.ndarray, 
        segments: List[Dict[str, Any]],
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """Fallback alignment using speech activity detection"""
        logger.info("Using basic VAD-based timestamp estimation...")
        
        # Detect speech regions using energy-based VAD
        speech_intervals = librosa.effects.split(audio, top_db=30, frame_length=2048, hop_length=512)
        
        if not speech_intervals or len(segments) == 0:
            return {"segments": segments, "aligned": False}
        
        # Filter valid speech intervals
        valid_intervals = []
        for start, end in speech_intervals:
            duration = (end - start) / sample_rate
            if 0.1 <= duration <= 30:
                valid_intervals.append((start / sample_rate, end / sample_rate))
        
        if not valid_intervals:
            return {"segments": segments, "aligned": False}
        
        # Distribute segments across speech intervals proportionally
        aligned_segments = []
        total_speech_duration = sum(end - start for start, end in valid_intervals)
        segment_texts = [seg.get("text", "").strip() for seg in segments if seg.get("text", "").strip()]
        
        if not segment_texts:
            return {"segments": segments, "aligned": False}
        
        total_chars = sum(len(text) for text in segment_texts)
        char_position = 0
        
        for segment_text in segment_texts:
            segment_chars = len(segment_text)
            
            if total_chars > 0 and total_speech_duration > 0:
                start_ratio = char_position / total_chars
                end_ratio = (char_position + segment_chars) / total_chars
                
                start_time = valid_intervals[0][0] + start_ratio * total_speech_duration
                end_time = valid_intervals[0][0] + end_ratio * total_speech_duration
            else:
                start_time = 0
                end_time = 1
            
            start_time = max(0, min(start_time, valid_intervals[-1][1]))
            end_time = max(start_time + 0.1, min(end_time, valid_intervals[-1][1]))
            
            aligned_segments.append({
                "start": start_time,
                "end": end_time,
                "text": segment_text,
                "aligned": False,
                "alignment_method": "basic_vad_estimation"
            })
            
            char_position += segment_chars
        
        return {"segments": aligned_segments, "aligned": False}

    def _create_initial_segments(
        self, 
        audio: np.ndarray, 
        text: str, 
        sample_rate: int = 16000
    ) -> List[Dict[str, Any]]:
        """Create initial segments based on speech activity detection"""
        try:
            # Clean up repeated text hallucinations first
            text = self._clean_repeated_text(text)
            
            # Use VAD to find speech regions
            speech_intervals = librosa.effects.split(audio, top_db=30)
            
            # Convert to list of tuples and handle the NumPy array properly
            if isinstance(speech_intervals, np.ndarray):
                if speech_intervals.size == 0:
                    speech_intervals = []
                else:
                    speech_intervals = [(int(start), int(end)) for start, end in speech_intervals]
            elif not speech_intervals:
                speech_intervals = []
            
            # If no speech detected or no text, return single segment
            if not speech_intervals or not text or not text.strip():
                return [{
                    "start": 0,
                    "end": float(len(audio)) / sample_rate if isinstance(audio, np.ndarray) else len(audio) / sample_rate,
                    "text": text if text else ""
                }]
            
            # Filter and merge close intervals
            valid_intervals = []
            min_duration = int(0.1 * sample_rate)
            max_duration = int(15 * sample_rate)
            merge_gap = int(0.3 * sample_rate)
            
            # Ensure speech_intervals is iterable and contains valid data
            if not speech_intervals or len(speech_intervals) == 0:
                return [{
                    "start": 0,
                    "end": float(len(audio)) / sample_rate,
                    "text": text
                }]
            
            merged_start = int(speech_intervals[0][0])
            merged_end = int(speech_intervals[0][1])
            
            for start, end in speech_intervals[1:]:
                start_int = int(start)
                end_int = int(end)
                
                if start_int - merged_end < merge_gap:
                    merged_end = end_int
                else:
                    if min_duration <= (merged_end - merged_start) <= max_duration:
                        valid_intervals.append((float(merged_start) / sample_rate, float(merged_end) / sample_rate))
                    merged_start = start_int
                    merged_end = end_int
            
            # Add last interval
            if min_duration <= (merged_end - merged_start) <= max_duration:
                valid_intervals.append((float(merged_start) / sample_rate, float(merged_end) / sample_rate))
            
            if not valid_intervals:
                return [{
                    "start": 0,
                    "end": float(len(audio)) / sample_rate,
                    "text": text
                }]
            
            # Split text into sentences
            sentences = self._split_into_sentences(text)
            
            if not sentences:
                return [{
                    "start": valid_intervals[0][0],
                    "end": valid_intervals[-1][1],
                    "text": text
                }]
            
            # Distribute sentences across speech intervals
            total_speech_time = float(sum(end - start for start, end in valid_intervals))
            segments = []
            
            total_chars = sum(len(s) for s in sentences)
            if total_chars > 0 and total_speech_time > 0:
                chars_per_second = float(total_chars) / total_speech_time
            else:
                chars_per_second = 15.0
            
            current_time = float(valid_intervals[0][0])
            
            for sentence in sentences:
                if chars_per_second > 0:
                    sentence_duration = float(len(sentence)) / chars_per_second
                else:
                    sentence_duration = 2.0
                
                sentence_start = current_time
                sentence_end = min(current_time + sentence_duration, float(valid_intervals[-1][1]))
                
                if sentence_end - sentence_start < 0.1:
                    sentence_end = sentence_start + 0.1
                
                segments.append({
                    "start": float(sentence_start),
                    "end": float(sentence_end),
                    "text": str(sentence)
                })
                
                current_time = sentence_end
            
            return segments
            
        except Exception as e:
            logger.error(f"Error creating initial segments: {e}")
            # Fallback: single segment for entire audio
            audio_length = len(audio) / sample_rate if isinstance(audio, np.ndarray) else 0
            return [{
                "start": 0,
                "end": float(audio_length),
                "text": text if text else ""
            }]

    def _clean_repeated_text(self, text: str) -> str:
        """Clean up repeated text hallucinations from Cohere"""
        if not text:
            return ""
        
        # Split into sentences
        sentences = text.split('. ')
        
        # Remove consecutive duplicate sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and (not cleaned_sentences or sentence != cleaned_sentences[-1]):
                cleaned_sentences.append(sentence)
        
        # Remove duplicates that appear multiple times
        from collections import Counter
        sentence_counts = Counter(cleaned_sentences)
        
        # If the same sentence appears more than twice, it's likely a hallucination
        final_sentences = []
        for sentence in cleaned_sentences:
            if sentence_counts[sentence] <= 2:  # Allow up to 2 repetitions
                final_sentences.append(sentence)
            elif sentence not in final_sentences:  # Keep first occurrence only
                final_sentences.append(sentence)
        
        return '. '.join(final_sentences) + ('.' if final_sentences else '')

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better alignment"""
        if not text or not isinstance(text, str):
            return []
        
        try:
            # Handle Korean and other languages that might use different punctuation
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.error(f"Error splitting sentences: {e}")
            return [text] if text else []

    def _post_process_segments(
        self, 
        segments: List[Dict[str, Any]], 
        total_duration: float
    ) -> List[Dict[str, Any]]:
        """Post-process aligned segments to ensure consistency"""
        cleaned_segments = []
        
        if not isinstance(total_duration, (int, float)) or total_duration <= 0:
            total_duration = 1.0
        
        for i, segment in enumerate(segments):
            try:
                # Ensure we're working with Python floats, not NumPy types
                start = float(max(0, segment.get("start", 0)))
                end = float(min(total_duration, segment.get("end", total_duration)))
                
                if start >= end:
                    end = float(min(start + 0.1, total_duration))
                
                segment["id"] = int(i)
                segment["start"] = start
                segment["end"] = end
                segment["text"] = str(segment.get("text", "")).strip()
                
                if segment["text"]:
                    cleaned_segments.append(segment)
            except Exception as e:
                logger.error(f"Error processing segment {i}: {e}")
                continue
        
        return cleaned_segments

    def transcribe_with_cohere(
        self,
        audio_path: str,
        language: str = "en",
        progress_callback = None,
        use_forced_alignment: bool = True
    ) -> Dict[str, Any]:
        """
        Full pipeline:
        1. Cohere transcribes audio (ASR) - accurate text, rough timestamps
        2. OPTIONAL: wav2vec2 forced alignment - precise word-level timestamps
        """
        try:
            self.load_cohere_model()
            
            if progress_callback:
                progress_callback(10, "Loading audio...")
            
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            audio_float32 = audio.astype(np.float32)
            
            if progress_callback:
                progress_callback(20, "Cohere transcribing...")
            
            # Cohere transcription
            inputs = self.cohere_processor(
                audio_float32,
                sampling_rate=sr,
                return_tensors="pt",
                language=language,
                punctuation=True
            )
            
            audio_chunk_index = inputs.pop("audio_chunk_index", None)
            
            model_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    if k == 'decoder_input_ids':
                        model_inputs[k] = v.to(self.current_model.device, dtype=torch.long)
                    elif k in ['input_features', 'attention_mask']:
                        model_inputs[k] = v.to(self.current_model.device, dtype=self.current_model.dtype)
                    else:
                        model_inputs[k] = v.to(self.current_model.device)
                else:
                    model_inputs[k] = v
            
            # Add better generation parameters to avoid repetition
            with torch.no_grad():
                outputs = self.current_model.generate(
                    **model_inputs, 
                    max_new_tokens=4096, # Increased to prevent truncation
                    repetition_penalty=1.2,  # Add repetition penalty
                    no_repeat_ngram_size=3,  # Prevent repeating trigrams
                    do_sample=False  # Use greedy decoding for consistency
                )
            
            # Decode Cohere's output
            if audio_chunk_index is not None:
                result = self.cohere_processor.decode(
                    outputs,
                    skip_special_tokens=True,
                    audio_chunk_index=audio_chunk_index,
                    language=language
                )
                cohere_text = " ".join(result) if isinstance(result, list) else result
            else:
                cohere_text = self.cohere_processor.decode(
                    outputs[0] if outputs.dim() > 1 else outputs,
                    skip_special_tokens=True
                )
            
            # Clean up the text
            cohere_text = str(cohere_text).strip()
            cohere_text = self._clean_repeated_text(cohere_text)
            
            logger.info(f"Cohere transcription: {cohere_text[:200]}...")
            
            if progress_callback:
                progress_callback(40, "Creating initial segments...")
            
            initial_segments = self._create_initial_segments(audio_float32, cohere_text, sr)
            
            if progress_callback:
                progress_callback(50, "Starting forced alignment..." if use_forced_alignment else "Finalizing...")
            
            if use_forced_alignment:
                try:
                    # wav2vec2 forced alignment of Cohere's text
                    aligned_result = self.align_cohere_transcription_with_phonemes(
                        audio=audio_float32,
                        cohere_segments=initial_segments,
                        language_code=language
                    )
                    final_segments = aligned_result["segments"]
                    is_aligned = aligned_result["aligned"]
                    alignment_method = aligned_result.get("alignment_method", "none")
                except Exception as e:
                    logger.error(f"Forced alignment failed: {e}, using basic segments")
                    final_segments = initial_segments
                    is_aligned = False
                    alignment_method = "alignment_failed_basic_fallback"
            else:
                final_segments = initial_segments
                is_aligned = False
                alignment_method = "speech_activity_detection_only"
            
            if progress_callback:
                progress_callback(90, "Post-processing segments...")
            
            total_duration = float(len(audio_float32)) / float(sr)
            final_segments = self._post_process_segments(final_segments, total_duration)
            
            if progress_callback:
                progress_callback(100, "Complete!")
            
            return {
                "segments": final_segments,
                "language": language,
                "text": cohere_text,
                "raw_text": cohere_text,
                "aligned": is_aligned,
                "alignment_method": alignment_method,
                "pipeline": "cohere_asr + wav2vec2_alignment" if use_forced_alignment else "cohere_asr_only"
            }
            
        except Exception as e:
            logger.error(f"Cohere pipeline error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def unload_alignment_model(self):
        """Unload wav2vec2 alignment model to free memory"""
        if self.alignment_model is not None:
            logger.info("Unloading wav2vec2 alignment model...")
            self.alignment_model = None
            self.alignment_metadata = None
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def unload_model(self):
        """Free GPU memory by unloading all models"""
        if self.device == "cuda":
            logger.info("Unloading all transcription models to free VRAM...")
            self.current_model = None
            self.cohere_processor = None
            self.models.clear()
            self.models_processors.clear()
            self.whisperx_models.clear()
            self.alignment_models.clear()
            self.unload_alignment_model()

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(1)
            logger.info("VRAM cleared.")
