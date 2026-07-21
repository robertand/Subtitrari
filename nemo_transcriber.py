"""
nemo_transcriber.py
Integrare NVIDIA NeMo Parakeet-TDT-0.6B-v3 și Nemotron 3.5 ASR Streaming
pentru transcriere cu word-level timestamps.
Modele:
  - https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
  - https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
"""

import os
import re
import gc
import subprocess
import tempfile
import logging
import signal
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Modele suportate
NEMO_MODELS = {
    "parakeet-v3": "nvidia/parakeet-tdt-0.6b-v3",
    "canary": "nvidia/canary-1b",
    "nemotron-3.5": "nvidia/nemotron-3.5-asr-streaming-0.6b"
}

# Modele de diarizare Sortformer
SORTFORMER_MODELS = {
    "sortformer-4spk": "diar_sortformer_4spk-v1",
    "sortformer-streaming": "diar_streaming_sortformer_4spk-v2.1"
}

# Limbi suportate de Parakeet v3 (implicit)
NEMO_SUPPORTED_LANGUAGES = [
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
    "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk",
    "sl", "es", "sv", "ru", "uk"
]

# Mapare nume limbă → cod ISO pentru UI
NEMO_LANGUAGE_NAMES = {
    "bg": "Bulgară", "hr": "Croată", "cs": "Cehă", "da": "Daneză",
    "nl": "Olandeză", "en": "Engleză", "et": "Estoniană", "fi": "Finlandeză",
    "fr": "Franceză", "de": "Germană", "el": "Greacă", "hu": "Maghiară",
    "it": "Italiană", "lv": "Letonă", "lt": "Lituaniană", "mt": "Malteză",
    "pl": "Poloneză", "pt": "Portugheză", "ro": "Română", "sk": "Slovacă",
    "sl": "Slovenă", "es": "Spaniolă", "sv": "Suedeză", "ru": "Rusă",
    "uk": "Ucraineană"
}

# Limbi suportate de Nemotron 3.5 (coduri 2 litere acceptate direct de model)
# Modelul acceptă și locale BCP-47 (ex: "ro-RO"), dar și codul scurt (ex: "ro")
NEMOTRON_SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "nl", "tr", "ru", "ar",
    "hi", "ja", "ko", "vi", "uk", "pl", "sv", "cs", "nb", "da",
    "bg", "fi", "hr", "sk", "zh", "hu", "ro", "et", "el", "lt",
    "lv", "mt", "sl", "he", "th", "no", "nn"
]

# Mapare cod 2 litere → prompt exact (locale) pentru Nemotron 3.5
# Modelul acceptă și codurile direct (ex: "ro"), dar folosim locale pentru claritate
NEMOTRON_LOCALES = {
    "en": "en-US", "es": "es-ES", "fr": "fr-FR", "it": "it-IT",
    "pt": "pt-BR", "nl": "nl-NL", "de": "de-DE", "tr": "tr-TR",
    "ru": "ru-RU", "ar": "ar-AR", "hi": "hi-IN", "ja": "ja-JP",
    "ko": "ko-KR", "vi": "vi-VN", "uk": "uk-UA",
    "pl": "pl-PL", "sv": "sv-SE", "cs": "cs-CZ", "nb": "nb-NO",
    "da": "da-DK", "bg": "bg-BG", "fi": "fi-FI", "hr": "hr-HR",
    "sk": "sk-SK", "zh": "zh-CN", "hu": "hu-HU", "ro": "ro-RO",
    "et": "et-EE", "el": "el-GR", "lt": "lt-LT", "lv": "lv-LV",
    "mt": "mt-MT", "sl": "sl-SI", "he": "he-IL", "th": "th-TH",
    "no": "no-NO", "nn": "nn-NO"
}

# Dimensiuni chunk recomandate pentru Nemotron 3.5 (frame = 80ms)
# att_context_size: [56, right_context] where right_context ∈ {0,1,3,6,13}
# chunk_size = (1 + right_context) * 80ms
NEMOTRON_CHUNK_RIGHT_CONTEXT = 13  # 1.12s chunks - best accuracy

# Durata maximă audio pentru full attention (în secunde)
# Audio mai lung se procesează în chunks automat
NEMO_MAX_FULL_ATTENTION_SECONDS = 1440  # 24 minute
NEMO_CHUNK_SIZE_SECONDS = 1200          # 20 minute per chunk pentru audio lung


def check_and_install_nemo():
    """Verifică dacă NeMo e instalat, dacă nu îl instalează automat."""
    try:
        import nemo.collections.asr as nemo_asr
        return True
    except ImportError:
        import subprocess
        import sys
        print("[NeMo] Instalare nemo_toolkit[asr]...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "nemo_toolkit[asr]", "--quiet"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Nu s-a putut instala NeMo:\n{result.stderr}\n"
                "Instalează manual: pip install nemo_toolkit[asr]"
            )
        return True


class NeMoTranscriber:
    """
    Wrapper pentru NVIDIA NeMo Parakeet și Canary.
    Gestionează descărcarea modelului, conversia audio și extragerea timestamps.
    """

    _nemotron_patch_applied = False

    def __init__(self):
        self._model = None
        self._model_loaded = False
        self._current_model_name = None

    @staticmethod
    def _patch_nemotron_dataloader():
        """
        Monkey-patch pentru LhotseSpeechToTextBpeDatasetWithPromptIndex
        care utilizează default_lang din config când limba nu e specificată
        în manifest (workaround pentru NeMo bug).
        """
        if NeMoTranscriber._nemotron_patch_applied:
            return
        try:
            from nemo.collections.asr.data.audio_to_text_lhotse_prompt_index import (
                LhotseSpeechToTextBpeDatasetWithPromptIndex
            )
            original = LhotseSpeechToTextBpeDatasetWithPromptIndex._get_prompt_index_for_cut

            def patched_get(self, cut):
                lang = cut.supervisions[0].language
                if lang is None:
                    default = getattr(self, 'cfg', {}).get('default_lang', 'auto')
                    return self._get_prompt_index(default)
                return original(self, cut)

            LhotseSpeechToTextBpeDatasetWithPromptIndex._get_prompt_index_for_cut = patched_get
            NeMoTranscriber._nemotron_patch_applied = True
            logger.info("[NeMo] Nemotron dataloader patch applied (default_lang fallback)")
        except Exception as e:
            logger.warning(f"[NeMo] Nu s-a putut aplica patch-ul: {e}")

    def is_available(self) -> bool:
        """Verifică dacă NeMo e instalat."""
        try:
            import nemo.collections.asr
            return True
        except ImportError:
            return False

    def load_model(self, model_name: str = "parakeet-v3", progress_callback=None) -> bool:
        """
        Încarcă modelul NeMo. La prima rulare îl descarcă automat
        din HuggingFace.
        """
        model_id = NEMO_MODELS.get(model_name, model_name)

        if self._model_loaded and self._model is not None and self._current_model_name == model_name:
            return True

        if self._model is not None:
            self.unload_model()

        if not self.is_available():
            raise ImportError(
                "NeMo nu este instalat. Instalează cu: pip install nemo_toolkit[asr]"
            )

        try:
            import torch
            import nemo.collections.asr as nemo_asr

            if progress_callback:
                progress_callback(f"Se descarcă/încarcă modelul NeMo {model_name} (~2-4GB)...")

            logger.info(f"[NeMo] Încărcare model: {model_id}")

            if model_name == "canary":
                self._model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(
                    model_name=model_id
                )
            elif model_name == "nemotron-3.5":
                if progress_callback:
                    progress_callback("Se încarcă Nemotron 3.5 ASR (streaming, 40 limbi)...")
                self._model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=model_id
                )
                try:
                    self._model.change_attention_model(
                        att_context_size=[56, NEMOTRON_CHUNK_RIGHT_CONTEXT]
                    )
                except Exception:
                    pass
                self._patch_nemotron_dataloader()
            else:
                self._model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=model_id
                )

            self._current_model_name = model_name

            # Mută pe GPU dacă e disponibil
            if torch.cuda.is_available():
                self._model = self._model.cuda()
                logger.info("[NeMo] Model încărcat pe GPU.")
            else:
                logger.warning("[NeMo] GPU nedisponibil, se rulează pe CPU (mai lent).")

            self._model.eval()
            self._model_loaded = True

            if progress_callback:
                model_label = {"parakeet-v3": "Parakeet TDT v3", "canary": "Canary-1b", "nemotron-3.5": "Nemotron 3.5 ASR"}
                progress_callback(f"Model {model_label.get(model_name, model_name)} încărcat cu succes.")

            return True

        except Exception as e:
            logger.error(f"[NeMo] Eroare la încărcarea modelului: {e}")
            if "out of memory" in str(e).lower():
                self.kill_orphaned_gpu_processes()
                try:
                    self._model = nemo_asr.models.ASRModel.from_pretrained(
                        model_name=model_id
                    )
                    self._current_model_name = model_name
                    if torch.cuda.is_available():
                        self._model = self._model.cuda()
                    self._model.eval()
                    self._model_loaded = True
                    logger.info("[NeMo] Model încărcat cu succes după OOM recovery.")
                    if progress_callback:
                        model_label = {"parakeet-v3": "Parakeet TDT v3", "canary": "Canary-1b", "nemotron-3.5": "Nemotron 3.5 ASR"}
                        progress_callback(f"Model {model_label.get(model_name, model_name)} încărcat cu succes.")
                    return True
                except Exception as e2:
                    logger.error(f"[NeMo] Eroare la reload după OOM cleanup: {e2}")
            raise

    def prepare_audio(self, input_path: str) -> str:
        """
        Convertește audio la formatul cerut de NeMo: WAV, mono, 16kHz, 16-bit.
        Returnează calea fișierului WAV pregătit (temporar).
        """
        output_path = tempfile.mktemp(suffix="_nemo_input.wav")

        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ac", "1",          # mono
            "-ar", "16000",      # 16kHz sample rate
            "-acodec", "pcm_s16le",  # 16-bit PCM
            output_path,
            "-loglevel", "error"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg eroare la pregătirea audio: {result.stderr}")

        return output_path

    def get_audio_duration(self, wav_path: str) -> float:
        """Returnează durata audio în secunde."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            wav_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            try:
                return float(result.stdout.strip())
            except ValueError:
                pass
        return 0.0

    def split_audio_chunks(self, wav_path: str, chunk_seconds: int, overlap_seconds: int = 0) -> List[Dict]:
        """
        Împarte audio-ul lung în chunks pentru procesare, cu overlap.
        Returnează lista de {path, offset_seconds, step_duration}.
        """
        duration = self.get_audio_duration(wav_path)
        chunks = []
        start = 0.0
        step = max(1.0, chunk_seconds - overlap_seconds)

        while start < duration:
            end = min(start + chunk_seconds, duration)
            chunk_path = tempfile.mktemp(suffix=f"_nemo_chunk_{int(start)}.wav")

            cmd = [
                "ffmpeg", "-y",
                "-i", wav_path,
                "-ss", str(start),
                "-to", str(end),
                "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
                chunk_path,
                "-loglevel", "error"
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            chunks.append({
                "path": chunk_path,
                "offset": start,
                "duration": end - start,
                "step_duration": step
            })

            if end >= duration:
                break
            start += step

        return chunks

    def transcribe(
        self,
        audio_path: str,
        model_name: str = "parakeet-v3",
        language: Optional[str] = None,
        window_size: int = 30,
        overlap: int = 10,
        progress_callback=None
    ) -> List[Dict]:
        """
        Transcrie fișierul audio și returnează segmente cu timestamps.
        Folosește windowing pentru a evita OOM.
        """
        if not self._model_loaded or self._current_model_name != model_name:
            self.load_model(model_name, progress_callback)

        # Pregătire audio
        if progress_callback:
            progress_callback("[NeMo] Pregătire audio...")

        wav_path = self.prepare_audio(audio_path)
        duration = self.get_audio_duration(wav_path)

        try:
            # Procesare pe chunks (windowing) conform setărilor utilizatorului
            if progress_callback:
                progress_callback(
                    f"[NeMo] Transcriere ({duration/60:.1f} minute) în ferestre de {window_size}s..."
                )
            segments = self._transcribe_chunked(wav_path, language, window_size, overlap, progress_callback)

            return segments

        finally:
            # Curăță fișierul temporar
            if os.path.exists(wav_path):
                os.unlink(wav_path)

    def _transcribe_single(self, wav_path: str, language: Optional[str]) -> List[Dict]:
        """Transcriere directă pentru audio scurt."""

        transcribe_kwargs = {"timestamps": True, "verbose": False}

        if self._current_model_name == "canary":
            transcribe_kwargs["task"] = "asr"
            transcribe_kwargs["source_lang"] = language if language and language != "auto" else "en"
            transcribe_kwargs["target_lang"] = language if language and language != "auto" else "en"
        elif self._current_model_name == "nemotron-3.5":
            # Nemotron 3.5: setează promptul și target_lang în kwargs
            target_lang = NEMOTRON_LOCALES.get(language, "auto") if language else "auto"
            try:
                if hasattr(self._model, 'set_inference_prompt'):
                    self._model.set_inference_prompt(target_lang)
            except Exception:
                pass
            transcribe_kwargs["target_lang"] = target_lang
        else:
            if language and language in NEMO_SUPPORTED_LANGUAGES:
                try:
                    if hasattr(self._model, 'set_language'):
                        self._model.set_language(language)
                except Exception:
                    pass

        # Transcriere
        hypotheses = self._model.transcribe(
            [wav_path],
            **transcribe_kwargs
        )

        return self._parse_hypothesis(hypotheses[0])

    def _transcribe_chunked(
        self,
        wav_path: str,
        language: Optional[str],
        window_size: int,
        overlap: int,
        progress_callback=None
    ) -> List[Dict]:
        """Transcriere pe chunks (windowing) pentru a salva memorie."""

        chunks = self.split_audio_chunks(wav_path, chunk_seconds=window_size, overlap_seconds=overlap)
        all_segments = []

        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(
                    f"[NeMo] Chunk {i+1}/{len(chunks)} "
                    f"(offset: {chunk['offset']/60:.1f} min)..."
                )

            try:
                transcribe_kwargs = {"timestamps": True}

                if self._current_model_name == "canary":
                    # Canary specific arguments
                    transcribe_kwargs["task"] = "asr"
                    transcribe_kwargs["source_lang"] = language if language and language != "auto" else "en"
                    transcribe_kwargs["target_lang"] = language if language and language != "auto" else "en"
                elif self._current_model_name == "nemotron-3.5":
                    # Nemotron 3.5: setează promptul și target_lang în kwargs
                    target_lang = NEMOTRON_LOCALES.get(language, "auto") if language else "auto"
                    try:
                        if hasattr(self._model, 'set_inference_prompt'):
                            self._model.set_inference_prompt(target_lang)
                    except Exception:
                        pass
                    transcribe_kwargs["target_lang"] = target_lang
                else:
                    # Setează limba dacă e specificată (altfel auto-detect pentru Parakeet)
                    if language and language in NEMO_SUPPORTED_LANGUAGES:
                        try:
                            if hasattr(self._model, 'set_language'):
                                self._model.set_language(language)
                        except Exception:
                            pass

                hypotheses = self._model.transcribe(
                    [chunk["path"]],
                    verbose=False,
                    **transcribe_kwargs
                )

                chunk_segments = self._parse_hypothesis(hypotheses[0])

                # Ajustează timestamps cu offset-ul chunk-ului
                # Păstrăm segmentele din zona sigură + jumătate din overlap
                # pentru a evita pierderea cuvintelor la granițele dintre chunk-uri
                is_last_chunk = (i == len(chunks) - 1)
                step_duration = chunk["step_duration"]
                keep_until = chunk["duration"] if is_last_chunk else (step_duration + overlap * 0.4)

                for seg in chunk_segments:
                    if seg["start"] < keep_until:
                        seg["start"] += chunk["offset"]
                        seg["end"] += chunk["offset"]
                        if "words" in seg:
                            for word in seg["words"]:
                                word["start"] += chunk["offset"]
                                word["end"] += chunk["offset"]
                        all_segments.append(seg)

            finally:
                # Curăță chunk temporar
                if os.path.exists(chunk["path"]):
                    os.unlink(chunk["path"])

        return all_segments

    @staticmethod
    def _clean_lang_tags(text: str) -> str:
        """Elimină tag-urile de limbă adăugate de Nemotron (ex: <en-US>)."""
        return re.sub(r'<\w{2}(?:-\w{2,4}){0,2}>', '', text).strip()

    def _log_timestamp_structure(self, ts: dict, hypothesis) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        for key in ("char", "word", "segment"):
            entries = ts.get(key, [])
            if entries:
                sample = entries[0] if entries else {}
                logger.debug(
                    "[Nemotron TS] %s: %d entries, sample keys: %s",
                    key, len(entries), list(sample.keys())
                )
        text = hypothesis.text if hasattr(hypothesis, "text") and hypothesis.text else "N/A"
        logger.debug("[Nemotron TS] hyp.text preview: %s", str(text)[:120])

    def _parse_hypothesis(self, hypothesis) -> List[Dict]:
        """
        Parsează output-ul NeMo hypothesis în formatul standard al aplicației.
        """
        segments = []

        try:
            ts = hypothesis.timestamp
            if isinstance(ts, dict):
                if logger.isEnabledFor(logging.DEBUG):
                    self._log_timestamp_structure(ts, hypothesis)
                segment_timestamps = ts.get("segment", [])
                word_timestamps = ts.get("word", [])

                if word_timestamps:
                    segments = self._group_words_into_segments(word_timestamps)

                if not segments and segment_timestamps:
                    segments = self._parse_segment_timestamps(
                        segment_timestamps, word_timestamps
                    )

            else:
                text = hypothesis.text if hasattr(hypothesis, "text") else str(hypothesis)
                text = self._clean_lang_tags(text.strip())
                if text:
                    segments.append({
                        "start": 0.0,
                        "end": 0.0,
                        "text": text,
                        "words": []
                    })

        except Exception as e:
            logger.error(f"[NeMo] Eroare la parsarea hypothesis: {e}")
            try:
                text = hypothesis.text if hasattr(hypothesis, "text") else ""
                text = self._clean_lang_tags(text.strip())
                if text:
                    segments.append({
                        "start": 0.0, "end": 0.0,
                        "text": text, "words": []
                    })
            except Exception:
                pass

        return segments

    def _group_words_into_segments(
        self,
        word_timestamps: List[Dict],
        max_words: int = 15,
        max_duration: float = 8.0,
        gap_threshold: float = 1.0
    ) -> List[Dict]:
        """
        Grupează cuvintele în segmente, păstrând propozițiile întregi.
        Prioritatea 1: sfârșit de propoziție (. ! ?) — desparte AICI
        Prioritatea 2: pauză lungă (> gap_threshold)
        Prioritatea 3: limită de cuvinte/durată
        """
        if not word_timestamps:
            return []

        segments = []
        current_words = []
        current_start = None

        for i, word_info in enumerate(word_timestamps):
            word = word_info.get("word", "").strip()
            w_start = float(word_info.get("start", 0))
            w_end = float(word_info.get("end", 0))

            if not word:
                continue

            if current_start is None:
                current_start = w_start

            is_sentence_end = word[-1] in '.!?' and len(word) > 1
            if is_sentence_end and current_words:
                current_words.append({"word": word, "start": w_start, "end": w_end})
                segments.append({
                    "start": current_start,
                    "end": w_end,
                    "text": " ".join(w2.get("word", "") for w2 in current_words),
                    "words": current_words.copy()
                })
                current_words = []
                current_start = None
                continue

            should_break = False
            if current_words and (w_start - float(current_words[-1].get("end", w_start))) > gap_threshold:
                should_break = True
            elif len(current_words) >= max_words:
                should_break = True
            elif current_words and (w_end - current_start) > max_duration:
                should_break = True

            if should_break and current_words:
                segments.append({
                    "start": current_start,
                    "end": float(current_words[-1].get("end", current_start)),
                    "text": " ".join(w2.get("word", "") for w2 in current_words),
                    "words": current_words.copy()
                })
                current_words = []
                current_start = w_start

            current_words.append({"word": word, "start": w_start, "end": w_end})

        if current_words:
            segments.append({
                "start": current_start,
                "end": float(current_words[-1].get("end", current_start)),
                "text": " ".join(w2.get("word", "") for w2 in current_words),
                "words": current_words.copy()
            })

        return segments

    def _parse_segment_timestamps(
        self,
        segment_timestamps: List[Dict],
        word_timestamps: List[Dict]
    ) -> List[Dict]:
        segments = []
        for seg in segment_timestamps:
            seg_start = float(seg.get("start", 0))
            seg_end = float(seg.get("end", 0))
            seg_text = self._clean_lang_tags(seg.get("segment", "").strip())
            if not seg_text:
                continue
            seg_words = []
            for w in word_timestamps:
                w_start = float(w.get("start", 0))
                w_end = float(w.get("end", 0))
                if w_start >= seg_start - 0.05 and w_end <= seg_end + 0.05:
                    seg_words.append({
                        "word": w.get("word", ""),
                        "start": w_start,
                        "end": w_end
                    })
            segments.append({
                "start": seg_start,
                "end": seg_end,
                "text": seg_text,
                "words": seg_words
            })
        return segments

    def unload_model(self):
        """Eliberează memoria GPU/RAM."""
        import torch

        if hasattr(self, '_current_model_name') and self._current_model_name:
            try:
                if hasattr(self, '_model') and self._model is not None:
                    del self._model
            except Exception:
                pass
            self._model = None
            self._model_loaded = False
            self._current_model_name = ""

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

        logger.info("[NeMo] Model descărcat din memorie.")

    @staticmethod
    def kill_orphaned_gpu_processes(skip_current: bool = True):
        """Omoră procesele zombie care încă ocupă VRAM."""
        import torch
        if not torch.cuda.is_available():
            return
        current_pid = os.getpid()
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split(", ")
                if len(parts) < 3:
                    continue
                pid_str = parts[0].strip()
                mem_str = parts[2].strip()
                try:
                    pid = int(pid_str)
                    mem_mib = int(mem_str)
                except ValueError:
                    continue
                if skip_current and pid == current_pid:
                    continue
                if mem_mib > 500:
                    try:
                        os.kill(pid, signal.SIGTERM)
                        logger.warning(f"[NeMo] Ucis proces zombie PID {pid} ({mem_mib} MiB VRAM)")
                    except (ProcessLookupError, PermissionError):
                        pass
        except Exception as e:
            logger.warning(f"[NeMo] Eroare la curățarea proceselor GPU: {e}")


# Instanță singleton
_nemo_transcriber_instance: Optional[NeMoTranscriber] = None


def get_nemo_transcriber() -> NeMoTranscriber:
    """Returnează instanța singleton a NeMoTranscriber."""
    global _nemo_transcriber_instance
    if _nemo_transcriber_instance is None:
        _nemo_transcriber_instance = NeMoTranscriber()
    return _nemo_transcriber_instance


# ========== Sortformer Speaker Diarization ==========

def run_sortformer_diarization(
    audio_path: str,
    model_key: str = "sortformer-4spk",
    progress_callback=None
) -> List[Dict]:
    """
    Rulează Sortformer speaker diarization pe un fișier audio.
    Returnează o listă de segmente cu {start, end, speaker}.
    
    Format raw de la model: [begin_seconds, end_seconds, speaker_index]
    (ex: [0.0, 3.5, "speaker_0"])
    
    Folosește huggingface_hub pentru download (NeMo built-in from_pretrained
    poate descărca fișiere corupte în unele cazuri).
    """
    import torch

    repo_id = SORTFORMER_MODELS.get(model_key, model_key)
    # Mapare nume model scurt → HF repo
    HF_SORTFORMER_REPOS = {
        "sortformer-4spk": "nvidia/diar_sortformer_4spk-v1",
        "diar_sortformer_4spk-v1": "nvidia/diar_sortformer_4spk-v1",
        "sortformer-streaming": "nvidia/diar_streaming_sortformer_4spk-v2.1",
        "diar_streaming_sortformer_4spk-v2.1": "nvidia/diar_streaming_sortformer_4spk-v2.1",
    }
    hf_repo = HF_SORTFORMER_REPOS.get(repo_id, repo_id)

    if progress_callback:
        progress_callback("Se descarcă/încarcă Sortformer diarization model...")

    from huggingface_hub import hf_hub_download
    from nemo.collections.asr.models import SortformerEncLabelModel

    # Determinăm numele fișierului .nemo
    filename = f"{os.path.basename(hf_repo)}.nemo"
    model_path = hf_hub_download(
        repo_id=hf_repo,
        filename=filename,
    )

    model = SortformerEncLabelModel.restore_from(model_path)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    if progress_callback:
        progress_callback("Se rulează diarizarea cu Sortformer...")

    raw_result = model.diarize(audio_path, verbose=False)

    model.cpu()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    segments = []
    for entry in raw_result:
        try:
            start = float(entry[0])
            end = float(entry[1])
            speaker = str(entry[2])
        except (ValueError, TypeError, IndexError):
            parts = str(entry).split()
            if len(parts) >= 3:
                start = float(parts[0])
                end = float(parts[1])
                speaker = parts[2]
            else:
                continue
        segments.append({
            "start": start,
            "end": end,
            "speaker": speaker,
        })

    if progress_callback:
        progress_callback(f"Diarizare completă: {len(segments)} segmente vorbitor.")

    return segments


def run_pyannote_diarization(
    audio_path: str,
    progress_callback=None
) -> List[Dict]:
    """
    Rulează speaker diarization cu PyAnnote Audio (pyannote/speaker-diarization-3.1).
    Identic cu abordarea dovedită din parakeet-diarized (jfgonsalves).
    Returnează listă de segmente {start, end, speaker}.
    """
    import torch
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    from pyannote.audio import Pipeline

    hf_token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(hf_token_path) and "HF_TOKEN" not in os.environ:
        with open(hf_token_path) as f:
            os.environ["HF_TOKEN"] = f.read().strip()
    if "HUGGINGFACE_TOKEN" not in os.environ and "HF_TOKEN" in os.environ:
        os.environ["HUGGINGFACE_TOKEN"] = os.environ["HF_TOKEN"]

    if progress_callback:
        progress_callback("Se încarcă modelul de diarizare PyAnnote...")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

    if torch.cuda.is_available():
        pipeline = pipeline.to(torch.device("cuda"))

    # Ajustează pragul de clustering: mai mic = mai sensibil la diferențe fine între voci
    try:
        pipeline.instantiate({"clustering": {"threshold": 0.35}})
        logger.info("PyAnnote clustering threshold set to 0.35")
    except Exception as e:
        logger.warning(f"PyAnnote threshold adjust failed (using default): {e}")

    if progress_callback:
        progress_callback("Se rulează diarizarea cu PyAnnote...")

    audio_data, sample_rate = sf.read(audio_path)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    waveform = torch.from_numpy(audio_data).float().unsqueeze(0)

    result = pipeline({"waveform": waveform, "sample_rate": sample_rate}, min_speakers=2)

    annotation = getattr(result, "speaker_diarization", result)

    segments = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        speaker_label = f"SPEAKER_{speaker}" if not str(speaker).startswith("SPEAKER_") else str(speaker)
        segments.append({
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "speaker": speaker_label,
        })

    segments.sort(key=lambda s: s["start"])
    speakers_set = set(s["speaker"] for s in segments)
    logger.info(f"PyAnnote detected {len(speakers_set)} unique speakers: {speakers_set}")
    if segments:
        logger.info(f"PyAnnote first 3: {segments[:3]}")
        logger.info(f"PyAnnote last 3: {segments[-3:]}")

    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return segments


def _get_speaker_at_time(
    time: float,
    diar_segments: List[Dict]
) -> Optional[str]:
    """Returnează speaker-ul activ la un moment dat.
    Dacă timestamp-ul e între două segmente de diarizare, returnează cel mai apropiat."""
    best = None
    best_dist = float("inf")
    for dseg in diar_segments:
        if dseg["start"] <= time <= dseg["end"]:
            return dseg["speaker"]
        dist = min(abs(time - dseg["start"]), abs(time - dseg["end"]))
        if dist < best_dist:
            best_dist = dist
            best = dseg["speaker"]
    return best


def assign_speakers_to_segments(
    transcript_segments: List[Dict],
    diarization_segments: List[Dict]
) -> List[Dict]:
    """
    Atribuie speaker fiecărui segment pe baza overlap-ului temporal maxim
    cu segmentele de diarizare. Identic cu merge_with_transcription din repo-ul
    parakeet-diarized (jfgonsalves) — fără prag minim de overlap.
    """
    if not diarization_segments:
        return transcript_segments

    diar_segments = sorted(diarization_segments, key=lambda s: s["start"])

    for seg in transcript_segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)

        if seg_end <= seg_start:
            continue

        best_speaker = None
        best_overlap = 0.0

        for dseg in diar_segments:
            overlap_start = max(seg_start, dseg["start"])
            overlap_end = min(seg_end, dseg["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dseg["speaker"]

            if dseg["start"] > seg_end:
                break

        seg["speaker"] = best_speaker if best_speaker and best_overlap > 0 else None

    assigned = set(s.get("speaker") for s in transcript_segments if s.get("speaker"))
    unassigned = sum(1 for s in transcript_segments if not s.get("speaker"))
    logger.info(f"Speaker assignment: {len(assigned)} unique speakers ({assigned}), {unassigned} unassigned segments")
    return transcript_segments


def split_segments_by_speaker(
    transcript_segments: List[Dict],
    diarization_segments: List[Dict]
) -> List[Dict]:
    """
    Taie segmentele la granițele diarizării. Fiecare cuvânt merge la zona
    de diarizare cu care are cel mai mult overlap temporal.
    Cuvintele fără speaker sunt asignate la cea mai apropiată zonă.
    """
    if not diarization_segments or not transcript_segments:
        return transcript_segments

    diar_segments = sorted(diarization_segments, key=lambda s: s["start"])
    result = []

    for seg in transcript_segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        words = seg.get("words", [])

        zones = []
        for dseg in diar_segments:
            if dseg["end"] <= seg_start:
                continue
            if dseg["start"] >= seg_end:
                break
            z_start = max(seg_start, dseg["start"])
            z_end = min(seg_end, dseg["end"])
            if z_end > z_start:
                zones.append({
                    "start": z_start, "end": z_end,
                    "speaker": dseg["speaker"],
                })

        if not zones:
            sp = _get_speaker_at_time(seg_start, diar_segments)
            seg["speaker"] = sp
            result.append(seg)
            continue

        # Segmente fără word timestamps: despică proporțional pe zone
        if not words and len(zones) > 1:
            text = seg.get("text", "")
            words_list = text.split() if text else []
            total_dur = seg_end - seg_start if seg_end > seg_start else 1
            char_pos = 0
            for zi, z in enumerate(zones):
                z_dur = z["end"] - z["start"]
                ratio = min(1.0, z_dur / total_dur)
                n_chars = int(len(text) * ratio) if text else 0
                if zi < len(zones) - 1 and n_chars > 0:
                    while n_chars < len(text) and text[n_chars] != ' ':
                        n_chars += 1
                chunk = text[:n_chars].strip() if text else ""
                text = text[n_chars:].strip() if text else ""
                if chunk or (zi == 0 and not chunk):
                    result.append({
                        "start": z["start"], "end": z["end"],
                        "text": chunk or seg.get("text", ""),
                        "words": [], "speaker": z["speaker"],
                    })
            if text:
                result.append({
                    "start": zones[-1]["start"], "end": zones[-1]["end"],
                    "text": text, "words": [], "speaker": zones[-1]["speaker"],
                })
            continue
        elif not words:
            seg["speaker"] = zones[0]["speaker"] if zones else _get_speaker_at_time(seg_start, diar_segments)
            result.append(seg)
            continue

        word_zones = {}
        orphan_words = []
        for w in words:
            w_s = w.get("start", 0)
            w_e = w.get("end", 0)
            best_zi = None
            best_o = 0
            for zi, z in enumerate(zones):
                o = max(0, min(w_e, z["end"]) - max(w_s, z["start"]))
                if o >= best_o:
                    best_o = o
                    best_zi = zi
            if best_zi is not None and best_o > 0:
                word_zones.setdefault(best_zi, []).append(w)
            else:
                orphan_words.append(w)

        for zi, z_words in word_zones.items():
            z = zones[zi]
            result.append({
                "start": z["start"], "end": z["end"],
                "text": " ".join(w.get("word", "") for w in z_words),
                "words": z_words, "speaker": z["speaker"],
            })

        if orphan_words and zones:
            z = zones[0]
            result.append({
                "start": orphan_words[0].get("start", seg_start),
                "end": orphan_words[-1].get("end", seg_end),
                "text": " ".join(w.get("word", "") for w in orphan_words),
                "words": orphan_words, "speaker": z["speaker"],
            })

    return result
