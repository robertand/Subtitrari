import sys
import types
from unittest.mock import MagicMock

# Mock torchcodec to prevent environment crashes on load
_torchcodec_mock = types.ModuleType('torchcodec')
_torchcodec_mock.__spec__ = MagicMock()
_torchcodec_mock.__spec__.name = 'torchcodec'
_torchcodec_mock.__spec__.loader = MagicMock()
_torchcodec_mock.__spec__.origin = 'mock'
_torchcodec_mock.__spec__.submodule_search_locations = []
_torchcodec_mock.__version__ = '0.0.0'
_torchcodec_mock.__path__ = []
_torchcodec_mock.__file__ = 'mock'
_torchcodec_mock.decoders = MagicMock()
_torchcodec_mock.decoders.VideoDecoder = MagicMock
_torchcodec_mock.decoders.AudioDecoder = MagicMock
_torchcodec_mock.decoders.Decoder = MagicMock
_torchcodec_mock.encoders = MagicMock()
_torchcodec_mock.encoders.VideoEncoder = MagicMock
_torchcodec_mock.encoders.AudioEncoder = MagicMock
_torchcodec_mock.load = MagicMock(return_value={})
_torchcodec_mock.dump = MagicMock()
_torchcodec_mock.is_available = MagicMock(return_value=False)
_torchcodec_mock.get_version = MagicMock(return_value="0.0.0")
_torchcodec_mock.VideoDecoder = MagicMock
_torchcodec_mock.AudioDecoder = MagicMock
_torchcodec_mock.VideoEncoder = MagicMock
_torchcodec_mock.AudioEncoder = MagicMock
_torchcodec_mock.Decoder = MagicMock
_torchcodec_mock.Encoder = MagicMock
_torchcodec_mock.StreamReader = MagicMock
_torchcodec_mock.StreamWriter = MagicMock
sys.modules['torchcodec'] = _torchcodec_mock

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
from typing import Optional, Dict, Any, List
import time
from config import Config

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.device = self._detect_device()
        self.cohere_processor = None
        
    def _detect_device(self) -> str:
        if torch.cuda.is_available():
            logger.info("CUDA GPU detected and available")
            return "cuda"
        logger.info("No GPU detected, using CPU")
        return "cpu"
    
    def load_model(self, model_name: str = 'small') -> Dict[str, Any]:
        """Load Whisper model with lazy loading"""
        try:
            if model_name in self.models:
                self.current_model = self.models[model_name]
                return {"status": "loaded", "model": model_name, "device": self.device}
            
            logger.info(f"Loading model: {model_name} on {self.device}")
            start_time = time.time()
            
            self.current_model = whisper.load_model(
                model_name,
                device=self.device,
                download_root='data/models'
            )
            
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
            
            result = self._clean_hallucinations(result)
            
            if progress_callback:
                progress_callback(100, "Complete!")
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise
    
    def transcribe_with_windowing(
        self, 
        audio_path: str, 
        model_name: str = 'small',
        language: Optional[str] = None,
        window_size: int = 30,
        overlap: int = 5,
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
    
    def extract_audio_from_video(self, video_path: str, output_path: str) -> str:
        """Extract audio from video using ffmpeg"""
        import ffmpeg
        
        try:
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(stream, output_path, acodec='pcm_s16le', ac=1, ar='16k')
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            return output_path
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            raise
    
    def transcribe_whisperx(
        self,
        audio_path: str,
        model_name: str = 'small',
        language: Optional[str] = None,
        progress_callback = None
    ) -> Dict[str, Any]:
        """Transcribe audio using WhisperX for better alignment and alternative version"""
        try:
            device = self.device
            compute_type = "float16" if device == "cuda" else "int8"

            if progress_callback:
                progress_callback(10, "Loading WhisperX model...")

            model = whisperx.load_model(model_name, device, compute_type=compute_type, download_root='data/models')

            if progress_callback:
                progress_callback(30, "Transcribing with WhisperX...")

            audio = whisperx.load_audio(audio_path)
            result = model.transcribe(audio, batch_size=16, language=language)

            if progress_callback:
                progress_callback(60, "Aligning WhisperX results...")

            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

            if progress_callback:
                progress_callback(100, "WhisperX complete!")

            return {
                "segments": result["segments"],
                "language": result["language"],
                "text": " ".join([seg["text"] for seg in result["segments"]])
            }
        except Exception as e:
            logger.error(f"WhisperX transcription error: {e}")
            raise

    def load_cohere_model(self):
        """Load Cohere Transcribe model and processor"""
        from transformers import AutoProcessor
        model_name = Config.COHERE_MODEL

        try:
            # Always ensure processor is loaded first
            if self.cohere_processor is None:
                logger.info(f"Loading Cohere processor: {model_name}")
                self.cohere_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

            if model_name in self.models:
                self.current_model = self.models[model_name]
                return {"status": "loaded", "model": "cohere", "device": self.device}

            logger.info(f"Loading Cohere model: {model_name}")

            # Try multiple import paths for the correct class
            model_loaded = False

            # Method 1: Direct import of CohereAsrForConditionalGeneration
            try:
                from transformers import CohereAsrForConditionalGeneration
                model = CohereAsrForConditionalGeneration.from_pretrained(
                    model_name,
                    device_map="auto" if self.device == "cuda" else None,
                    dtype=torch.float32,
                    trust_remote_code=True
                )
                model_loaded = True
                logger.info("Loaded via CohereAsrForConditionalGeneration")
            except (ImportError, Exception) as e:
                logger.warning(f"CohereAsrForConditionalGeneration not available directly: {e}")

            # Method 2: AutoModelForSpeechSeq2Seq
            if not model_loaded:
                try:
                    from transformers import AutoModelForSpeechSeq2Seq
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        model_name,
                        device_map="auto" if self.device == "cuda" else None,
                        dtype=torch.float32,
                        trust_remote_code=True
                    )
                    model_loaded = True
                    logger.info("Loaded via AutoModelForSpeechSeq2Seq")
                except (ImportError, Exception) as e:
                    logger.warning(f"AutoModelForSpeechSeq2Seq failed: {e}")

            # Method 3: AutoModel with proper config
            if not model_loaded:
                try:
                    from transformers import AutoModel, AutoConfig
                    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                    model = AutoModel.from_pretrained(
                        model_name,
                        config=config,
                        device_map="auto" if self.device == "cuda" else None,
                        dtype=torch.float32,
                        trust_remote_code=True
                    )
                    model_loaded = True
                    logger.info("Loaded via AutoModel with config")
                except Exception as e:
                    logger.warning(f"AutoModel with config failed: {e}")

            # Method 4: Direct import from the cached model path
            if not model_loaded:
                try:
                    import importlib.util
                    import sys
                    from huggingface_hub import snapshot_download

                    logger.info("Attempting direct import from cached model path...")
                    model_path = snapshot_download(model_name)

                    spec = importlib.util.spec_from_file_location(
                        "cohere_model_modeling",
                        f"{model_path}/modeling_cohere_asr.py"
                    )
                    cohere_module = importlib.util.module_from_spec(spec)
                    sys.modules["cohere_model_modeling"] = cohere_module
                    spec.loader.exec_module(cohere_module)

                    model = cohere_module.CohereAsrForConditionalGeneration.from_pretrained(
                        model_name,
                        device_map="auto" if self.device == "cuda" else None,
                        dtype=torch.float32,
                        trust_remote_code=True
                    )
                    model_loaded = True
                    logger.info("Loaded via direct import from model path")
                except Exception as e:
                    logger.error(f"All model loading methods failed. Direct import error: {e}")
                    raise RuntimeError(f"Could not load Cohere model {model_name} with the correct architecture.")

            model.eval()
            self.models[model_name] = model
            self.current_model = model

            logger.info(f"Loaded Cohere model class: {type(model)}")

            # Verify it's the right class/has generate
            if not hasattr(model, 'generate'):
                raise RuntimeError(f"Loaded model {type(model)} does not have generate() method. Expected a Conditional Generation model.")

            return {"status": "loaded", "model": "cohere", "device": self.device}

        except Exception as e:
            logger.error(f"Fatal error loading Cohere model/processor: {e}")
            raise

    def transcribe_with_cohere(
        self,
        audio_path: str,
        language: str = "en",
        prompt: Optional[str] = None,
        progress_callback = None
    ) -> Dict[str, Any]:
        """Transcribe audio using Cohere - with forced alignment for timestamps and prompt support"""
        try:
            self.load_cohere_model()

            if progress_callback:
                progress_callback(10, "Loading audio for Cohere...")

            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            audio_float32 = audio.astype(np.float32)

            if progress_callback:
                progress_callback(25, "Processing with Cohere...")

            # Prepare inputs
            processor_kwargs = {
                "sampling_rate": sr,
                "return_tensors": "pt",
                "language": language,
                "punctuation": True
            }

            inputs = self.cohere_processor(audio_float32, **processor_kwargs)

            # Save audio_chunk_index for decoding, but don't pass to model
            audio_chunk_index = inputs.pop("audio_chunk_index", None)

            # Move inputs to model device with CORRECT dtypes
            model_inputs = {}
            for k, v in inputs.items():
                if k == "length": # Skip unused length kwarg that causes Cohere generate to crash
                    continue
                if isinstance(v, torch.Tensor):
                    if k == 'decoder_input_ids':
                        model_inputs[k] = v.to(self.current_model.device, dtype=torch.long)
                    elif k in ['input_features', 'attention_mask']:
                        model_inputs[k] = v.to(self.current_model.device, dtype=self.current_model.dtype)
                    else:
                        model_inputs[k] = v.to(self.current_model.device)
                else:
                    model_inputs[k] = v

            # Handle prompt (forced_decoder_ids or prefix)
            if prompt:
                logger.info(f"Using transcription prompt: {prompt}")
                # For Cohere, we can use the tokenizer to get prefix tokens
                prompt_ids = self.cohere_processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
                model_inputs["decoder_input_ids"] = torch.cat([model_inputs["decoder_input_ids"], prompt_ids.to(self.current_model.device)], dim=-1)

            if progress_callback:
                progress_callback(40, "Generating transcription...")

            with torch.no_grad():
                # Greedy decoding for stability
                outputs = self.current_model.generate(
                    **model_inputs,
                    max_new_tokens=448, # Increased for potentially longer audio
                    do_sample=False
                )

            # Decode with audio_chunk_index for long audio reassembly
            if audio_chunk_index is not None:
                result = self.cohere_processor.decode(
                    outputs,
                    skip_special_tokens=True,
                    audio_chunk_index=audio_chunk_index,
                    language=language
                )
                if isinstance(result, list):
                    full_text = " ".join(result)
                else:
                    full_text = result
            else:
                full_text = self.cohere_processor.decode(
                    outputs[0] if outputs.dim() > 1 else outputs,
                    skip_special_tokens=True
                )

            full_text = full_text.strip()

            if progress_callback:
                progress_callback(70, "Aligning timestamps (Forced Aligner)...")

            # FORCED ALIGNMENT using WhisperX logic
            segments = []
            try:
                # We need to create a temporary segment structure for whisperx align
                # But whisperx align needs a model and metadata.
                # Since we already have whisperx installed, we use it.
                temp_segments = [{"text": full_text, "start": 0, "end": len(audio_float32)/sr}]

                # Load alignment model (usually wav2vec2 based)
                model_a, metadata = whisperx.load_align_model(language_code=language, device=self.device)

                # Align
                result_aligned = whisperx.align(
                    temp_segments,
                    model_a,
                    metadata,
                    audio_float32,
                    self.device,
                    return_char_alignments=False
                )
                segments = result_aligned["segments"]
                logger.info(f"Forced alignment successful. Found {len(segments)} segments.")
            except Exception as align_err:
                logger.warning(f"Forced alignment failed: {align_err}. Falling back to VAD split.")
                # Fallback to the VAD-based splitting the user provided
                speech_intervals = librosa.effects.split(audio_float32, top_db=30)
                valid_intervals = [(s, e) for s, e in speech_intervals if 0.3 <= (e-s)/sr <= 15]

                if full_text and valid_intervals:
                    words = full_text.split()
                    total_duration = sum(end - start for start, end in valid_intervals) / sr
                    word_index = 0
                    for start_sample, end_sample in valid_intervals:
                        duration = (end_sample - start_sample) / sr
                        num_words = max(1, int(len(words) * duration / total_duration))
                        segment_words = words[word_index:word_index + num_words]
                        if segment_words:
                            segments.append({
                                'start': start_sample / sr,
                                'end': end_sample / sr,
                                'text': ' '.join(segment_words)
                            })
                            word_index += num_words
                        if word_index >= len(words): break

                if not segments:
                    segments = [{"start": 0, "end": len(audio)/sr, "text": full_text}]

            if progress_callback:
                progress_callback(100, "Cohere transcription complete!")

            return {
                "segments": segments,
                "language": language,
                "text": full_text
            }

        except Exception as e:
            logger.error(f"Cohere transcription fatal error: {e}")
            raise

    def unload_model(self):
        """Free GPU memory by unloading all models"""
        if self.device == "cuda":
            logger.info("Unloading all transcription models to free VRAM...")
            self.current_model = None
            self.cohere_processor = None
            self.models.clear()

            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(1)
            logger.info("VRAM cleared.")
