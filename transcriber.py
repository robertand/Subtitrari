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
from typing import Optional, Dict, Any, List
import time

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.device = self._detect_device()
        
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
            
            # Free memory if another model was loaded
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
            # Fallback to CPU if GPU fails
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
            
            # Load and preprocess audio
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Normalize audio
            audio = self._normalize_audio(audio)
            
            if progress_callback:
                progress_callback(20, "Transcribing...")
            
            # Prepare options
            options = {
                "task": "transcribe",
                "verbose": False,
                "fp16": self.device == "cuda"
            }
            
            if language and language != 'auto':
                options["language"] = language
            
            # Transcribe
            result = self.current_model.transcribe(audio, **options)
            
            if progress_callback:
                progress_callback(90, "Post-processing...")
            
            # Clean hallucinations
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
            
            # Process in windows
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
                
                # Save window to temp file
                temp_file = Path(f"data/temp/window_{window_num}.wav")
                temp_file.parent.mkdir(parents=True, exist_ok=True)
                sf.write(temp_file, window_audio, sr)
                
                # Transcribe window
                result = self.transcribe_audio(
                    str(temp_file), 
                    model_name, 
                    language
                )
                
                # Adjust timestamps and avoid duplicates from overlapping windows
                offset = i / sr
                step_duration = step / sr

                for seg in result.get('segments', []):
                    # Only add segments that start in the unique part of this window
                    # (except for the last window where we take everything remaining)
                    is_last_window = (i + window_samples) >= len(audio)
                    if is_last_window or seg['start'] < step_duration:
                        seg['start'] += offset
                        seg['end'] += offset
                        segments.append(seg)
                
                # Cleanup temp file
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
            # Use noisereduce for spectral gating
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
            r'(?i)^\s*$',  # Empty lines
            r'(?i)(♪|♫|♬|♩|♭)',
            r'(?i)(\[.*?\])', # Brackets like [MUSIC]
            r'(?i)(\*.*?\*)'  # Stars like *laughter*
        ]
        
        if 'segments' in result:
            cleaned_segments = []
            for segment in result['segments']:
                text = segment.get('text', '').strip()
                is_hallucination = False
                
                # Check patterns
                for pattern in hallucination_patterns:
                    if re.search(pattern, text):
                        is_hallucination = True
                        break
                
                # Stutter/Repetition detection (e.g. "you you you you")
                words = text.split()
                if len(words) > 4:
                    # Check if more than 70% of words are the same
                    from collections import Counter
                    counts = Counter(words)
                    most_common, count = counts.most_common(1)[0]
                    if count / len(words) > 0.7:
                        is_hallucination = True

                # Extremely short duration hallucinations
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
            stream = ffmpeg.output(
                stream, 
                output_path,
                acodec='pcm_s16le',
                ac=1,
                ar='16k'
            )
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

            # Alignment
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

    def unload_model(self):
        """Free GPU memory"""
        if self.current_model and self.device == "cuda":
            del self.current_model
            self.current_model = None
            torch.cuda.empty_cache()