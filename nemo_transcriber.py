"""
nemo_transcriber.py
Integrare NVIDIA NeMo Parakeet-TDT-0.6B-v3 pentru transcriere cu word-level timestamps.
Model: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
"""

import os
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Modele suportate
NEMO_MODELS = {
    "parakeet-v3": "nvidia/parakeet-tdt-0.6b-v3",
    "canary": "nvidia/canary-1b"
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

    def __init__(self):
        self._model = None
        self._model_loaded = False
        self._current_model_name = None

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
            import nemo.collections.asr as nemo_asr

            if progress_callback:
                progress_callback(f"Se descarcă/încarcă modelul NeMo {model_name} (~2-4GB)...")

            logger.info(f"[NeMo] Încărcare model: {model_id}")

            if model_name == "canary":
                # Canary requires EncDecMultiTaskModel
                self._model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(
                    model_name=model_id
                )
            else:
                # from_pretrained descarcă automat din HuggingFace dacă nu e în cache
                self._model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=model_id
                )

            self._current_model_name = model_name

            # Mută pe GPU dacă e disponibil
            import torch
            if torch.cuda.is_available():
                self._model = self._model.cuda()
                logger.info("[NeMo] Model încărcat pe GPU.")
            else:
                logger.warning("[NeMo] GPU nedisponibil, se rulează pe CPU (mai lent).")

            self._model.eval()
            self._model_loaded = True

            if progress_callback:
                progress_callback("Model NeMo Parakeet încărcat cu succes.")

            return True

        except Exception as e:
            logger.error(f"[NeMo] Eroare la încărcarea modelului: {e}")
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

                # Ajustează timestamps cu offset-ul chunk-ului și aplică logică de overlap
                is_last_chunk = (i == len(chunks) - 1)
                step_duration = chunk["step_duration"]

                for seg in chunk_segments:
                    # Menținem segmentele care încep în interiorul "pasului" curent,
                    # sau toate segmentele dacă este ultimul chunk.
                    if is_last_chunk or seg["start"] < step_duration:
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

    def _parse_hypothesis(self, hypothesis) -> List[Dict]:
        """
        Parsează output-ul NeMo hypothesis în formatul standard al aplicației.
        """
        segments = []

        try:
            segment_timestamps = hypothesis.timestamp.get("segment", [])
            word_timestamps = hypothesis.timestamp.get("word", [])

            if segment_timestamps:
                for seg in segment_timestamps:
                    seg_start = float(seg.get("start", 0))
                    seg_end = float(seg.get("end", 0))
                    seg_text = seg.get("segment", "").strip()

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

            else:
                if word_timestamps:
                    segments = self._group_words_into_segments(word_timestamps)
                else:
                    text = hypothesis.text if hasattr(hypothesis, "text") else str(hypothesis)
                    if text.strip():
                        segments.append({
                            "start": 0.0,
                            "end": 0.0,
                            "text": text.strip(),
                            "words": []
                        })

        except Exception as e:
            logger.error(f"[NeMo] Eroare la parsarea hypothesis: {e}")
            try:
                text = hypothesis.text if hasattr(hypothesis, "text") else ""
                if text.strip():
                    segments.append({
                        "start": 0.0, "end": 0.0,
                        "text": text.strip(), "words": []
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
        Grupează cuvinte în segmente de subtitrare pe baza pauzelor și lungimii.
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

            should_break = False

            if len(current_words) >= max_words:
                should_break = True
            elif current_words and (w_start - float(current_words[-1].get("end", w_start))) > gap_threshold:
                should_break = True
            elif current_words and (w_end - current_start) > max_duration:
                should_break = True

            if should_break and current_words:
                segments.append({
                    "start": current_start,
                    "end": float(current_words[-1].get("end", current_start)),
                    "text": " ".join(w.get("word", "") for w in current_words),
                    "words": current_words.copy()
                })
                current_words = []
                current_start = w_start

            current_words.append({"word": word, "start": w_start, "end": w_end})

        if current_words:
            segments.append({
                "start": current_start,
                "end": float(current_words[-1].get("end", current_start)),
                "text": " ".join(w.get("word", "") for w in current_words),
                "words": current_words.copy()
            })

        return segments

    def unload_model(self):
        """Eliberează memoria GPU/RAM."""
        if self._model is not None:
            import torch
            del self._model
            self._model = None
            self._model_loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("[NeMo] Model descărcat din memorie.")


# Instanță singleton
_nemo_transcriber_instance: Optional[NeMoTranscriber] = None


def get_nemo_transcriber() -> NeMoTranscriber:
    """Returnează instanța singleton a NeMoTranscriber."""
    global _nemo_transcriber_instance
    if _nemo_transcriber_instance is None:
        _nemo_transcriber_instance = NeMoTranscriber()
    return _nemo_transcriber_instance
