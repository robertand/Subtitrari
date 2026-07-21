"""
sdh_detector.py
Detecție evenimente audio pentru subtitrări SDH (Subtitles for Deaf and Hard-of-hearing).
Folosește CLAP (zero-shot audio classification) pentru detectarea sunetelor.
"""

import os
import subprocess
import tempfile
import logging
import numpy as np
from typing import List, Dict, Optional, Callable

logger = logging.getLogger(__name__)

CLAP_MODEL_ID = "laion/clap-htsat-unfused"

# Chunk size pentru analiza audio (în secunde)
AUDIO_CHUNK_SECONDS = 3.0
AUDIO_OVERLAP_SECONDS = 0.5

# Confidence minim pentru a raporta un eveniment
DEFAULT_CONFIDENCE_THRESHOLD = 0.35

# Praguri specifice pentru evenimente ușor confundabile (ex: zgomote metalice detectate ca geam spart)
SPECIFIC_THRESHOLDS = {
    "glass_break": 0.45,
    "clock_tick": 0.45,
    "door_knock": 0.40,
    "gunshot": 0.40,
    "door_slam": 0.40,
}

# Marja minimă între scorul 1 și 2 pentru a evita falsuri pozitive
MIN_CONFIDENCE_MARGIN = 0.03

# Evenimente audio clasificate (folosite pentru CLAP zero-shot)
SOUND_EVENTS = {
    "music_general":      "music playing",
    "music_sad":          "sad emotional music",
    "music_tense":        "tense suspenseful music",
    "music_happy":        "happy upbeat music",
    "music_dramatic":     "dramatic orchestral music",
    "music_action":       "action intense music",
    "thunder":            "thunder storm sound",
    "rain":               "rain falling sound",
    "wind":               "strong wind blowing",
    "fire":               "fire crackling burning",
    "water":              "water flowing splashing",
    "applause":           "crowd applause clapping",
    "laughter":           "people laughing",
    "crying":             "person crying sobbing",
    "screaming":          "person screaming yelling",
    "whispering":         "person whispering quietly",
    "coughing":           "person coughing",
    "explosion":          "explosion blast sound",
    "gunshot":            "gunshot firearm shooting",
    "siren":              "emergency siren alarm",
    "car_crash":          "car crash accident sound",
    "door_knock":         "knocking on door",
    "door_slam":          "door slamming shut",
    "glass_break":        "glass breaking shattering",
    "phone_ring":         "phone ringing",
    "car_engine":         "car engine running driving",
    "helicopter":         "helicopter flying sound",
    "train":              "train passing sound",
    "dog_bark":           "dog barking",
    "horse":              "horse galloping neigh",
    "birds":              "birds chirping singing",
    "silence":            "silence no sound",
    "crowd_noise":        "crowd noise ambient",
    "heartbeat":          "heartbeat sound",
    "clock_tick":         "clock ticking",
}

SDH_TRANSLATIONS = {
    "ro": {
        "music_general":  "Muzică",
        "music_sad":      "Muzică tristă",
        "music_tense":    "Muzică tensionată",
        "music_happy":    "Muzică veselă",
        "music_dramatic": "Muzică dramatică",
        "music_action":   "Muzică de acțiune",
        "thunder":        "Tunet",
        "rain":           "Ploaie",
        "wind":           "Vânt",
        "fire":           "Foc",
        "water":          "Apă curgând",
        "applause":       "Aplauze",
        "laughter":       "Râsete",
        "crying":         "Plâns",
        "screaming":      "Țipăt",
        "whispering":     "Șoapte",
        "coughing":       "Tuse",
        "explosion":      "Explozie",
        "gunshot":        "Împușcătură",
        "siren":          "Sirenă",
        "car_crash":      "Accident auto",
        "door_knock":     "Bătaie în ușă",
        "door_slam":      "Ușă trântită",
        "glass_break":    "Sticlă spartă",
        "phone_ring":     "Telefon sună",
        "car_engine":     "Motor de mașină",
        "helicopter":     "Elicopter",
        "train":          "Tren",
        "dog_bark":       "Câine latră",
        "horse":          "Cal",
        "birds":          "Păsări",
        "silence":        "Liniște",
        "crowd_noise":    "Zgomot de mulțime",
        "heartbeat":      "Bătăi de inimă",
        "clock_tick":     "Ticăit de ceas",
    },
    "en": {
        "music_general":  "Music",
        "music_sad":      "Sad music",
        "music_tense":    "Tense music",
        "music_happy":    "Upbeat music",
        "music_dramatic": "Dramatic music",
        "music_action":   "Action music",
        "thunder":        "Thunder",
        "rain":           "Rain",
        "wind":           "Wind",
        "fire":           "Fire crackling",
        "water":          "Water",
        "applause":       "Applause",
        "laughter":       "Laughter",
        "crying":         "Crying",
        "screaming":      "Screaming",
        "whispering":     "Whispering",
        "coughing":       "Coughing",
        "explosion":      "Explosion",
        "gunshot":        "Gunshot",
        "siren":          "Siren",
        "car_crash":      "Car crash",
        "door_knock":     "Knocking",
        "door_slam":      "Door slams",
        "glass_break":    "Glass breaking",
        "phone_ring":     "Phone ringing",
        "car_engine":     "Car engine",
        "helicopter":     "Helicopter",
        "train":          "Train",
        "dog_bark":       "Dog barking",
        "horse":          "Horse",
        "birds":          "Birds",
        "silence":        "Silence",
        "crowd_noise":    "Crowd noise",
        "heartbeat":      "Heartbeat",
        "clock_tick":     "Clock ticking",
    },
}


class SDHDetector:
    """
    Detectează evenimente audio non-vorbire și generează descrieri SDH.
    Folosește CLAP (zero-shot) pentru clasificare.
    """

    def __init__(self):
        self._model = None
        self._processor = None
        self._loaded = False

    def is_available(self) -> bool:
        try:
            import transformers
            return True
        except ImportError:
            return False

    def load_model(self, progress_callback=None):
        if self._loaded:
            return

        if progress_callback:
            progress_callback("[SDH] Încărcare model CLAP (~900MB la prima rulare)...")
        logger.info("[SDH] Încărcare model CLAP...")

        from transformers import ClapModel, ClapProcessor
        import torch

        self._processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
        self._model = ClapModel.from_pretrained(CLAP_MODEL_ID)

        if torch.cuda.is_available():
            self._model = self._model.cuda()

        self._model.eval()
        self._loaded = True

        if progress_callback:
            progress_callback("[SDH] Model CLAP încărcat.")
        logger.info("[SDH] Model CLAP încărcat.")

    def extract_audio_wav(self, video_path: str, start_time: Optional[float] = None, duration: Optional[float] = None) -> str:
        """Extrage audio din video în format WAV mono 48kHz (CLAP antrenat pe 48kHz)."""
        out_path = tempfile.mktemp(suffix="_sdh_audio.wav")
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ac", "1", "-ar", "48000",
            "-acodec", "pcm_s16le",
        ]
        if start_time is not None:
            cmd.extend(["-ss", str(start_time)])
        if duration is not None:
            cmd.extend(["-t", str(duration)])
        cmd.extend([out_path, "-loglevel", "error"])
        subprocess.run(cmd, check=True, capture_output=True)
        return out_path

    def detect_speech_segments(self, asr_segments: List[Dict]) -> List[tuple]:
        return [(seg["start"] - 0.1, seg["end"] + 0.1) for seg in asr_segments]

    def is_speech_interval(
        self,
        start: float,
        end: float,
        speech_intervals: List[tuple]
    ) -> bool:
        for sp_start, sp_end in speech_intervals:
            overlap = min(end, sp_end) - max(start, sp_start)
            if overlap > 0.3:
                return True
        return False

    def classify_chunk(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 48000,
        top_k: int = 3
    ) -> List[Dict]:
        import torch

        candidate_labels = list(SOUND_EVENTS.keys())
        candidate_descriptions = [SOUND_EVENTS[k] for k in candidate_labels]

        inputs = self._processor(
            audio=audio_array,
            text=candidate_descriptions,
            return_tensors="pt",
            padding=True,
            sampling_rate=sample_rate
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() if torch.is_tensor(v) else v
                     for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits_per_audio[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        top_indices = np.argsort(probs)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "event_key": candidate_labels[idx],
                "confidence": float(probs[idx])
            })

        return results

    def detect_sound_events(
        self,
        video_path: str,
        asr_segments: List[Dict],
        target_lang: str = "ro",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        chunk_seconds: float = AUDIO_CHUNK_SECONDS,
        use_llm_descriptions: bool = False,
        llm_callback: Optional[Callable] = None,
        progress_callback=None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict]:
        import librosa

        if not self._loaded:
            self.load_model(progress_callback)

        if progress_callback:
            progress_callback("[SDH] Extragere audio...")
        logger.info("[SDH] Extragere audio...")
        logger.info(f"[SDH] Confidence threshold: {confidence_threshold}, lang: {target_lang}")

        audio_duration = None
        if start_time is not None and end_time is not None and end_time > start_time:
            audio_duration = end_time - start_time
        wav_path = self.extract_audio_wav(video_path, start_time=start_time, duration=audio_duration)

        try:
            audio, sr = librosa.load(wav_path, sr=48000, mono=True)
            total_duration = len(audio) / sr

            if start_time is not None:
                asr_offset = start_time
                speech_intervals = [(max(s, start_time), min(e, end_time or total_duration))
                                    for s, e in self.detect_speech_segments(asr_segments)
                                    if e > start_time and (end_time is None or s < end_time)]
            else:
                asr_offset = 0.0
                speech_intervals = self.detect_speech_segments(asr_segments)

            sdh_segments = []
            current_time = 0.0
            chunk_samples = int(chunk_seconds * sr)
            overlap_samples = int(AUDIO_OVERLAP_SECONDS * sr)

            total_chunks = int(total_duration / chunk_seconds) + 1
            chunk_count = 0

            last_event = None
            last_event_end = 0.0

            while current_time < total_duration:
                start_sample = int(current_time * sr)
                end_sample = min(start_sample + chunk_samples, len(audio))

                chunk_audio = audio[start_sample:end_sample]
                chunk_end_time = min(current_time + chunk_seconds, total_duration)

                chunk_count += 1
                if progress_callback and chunk_count % 20 == 0:
                    pct = int(current_time / total_duration * 100)
                    progress_callback(f"[SDH] Analiză audio: {pct}%...")

                if not self.is_speech_interval(current_time, chunk_end_time, speech_intervals):
                    results = self.classify_chunk(chunk_audio, sr)

                    if results and results[0]["confidence"] >= confidence_threshold:
                        top_event = results[0]["event_key"]
                        top_conf = results[0]["confidence"]

                        if top_event != "silence":
                            # Apply event-specific threshold for confusable events
                            effective_threshold = max(
                                confidence_threshold,
                                SPECIFIC_THRESHOLDS.get(top_event, 0.0)
                            )

                            # Check margin: if top-2 is too close, model is uncertain
                            second_conf = results[1]["confidence"] if len(results) > 1 else 0.0
                            margin_pass = (top_conf - second_conf) >= MIN_CONFIDENCE_MARGIN

                            if top_conf >= effective_threshold and margin_pass:
                                description = self._get_sdh_description(
                                    top_event, target_lang, results,
                                    use_llm_descriptions, llm_callback
                                )

                                if (top_event == last_event and
                                    current_time - last_event_end < 2.0):
                                    if sdh_segments:
                                        sdh_segments[-1]["end"] = chunk_end_time + asr_offset
                                else:
                                    sdh_segments.append({
                                        "start": current_time + asr_offset,
                                        "end": chunk_end_time + asr_offset,
                                        "text": f"[{description}]",
                                        "source": "sdh",
                                        "event_key": top_event,
                                        "confidence": top_conf
                                    })

                                last_event = top_event
                                last_event_end = chunk_end_time

                current_time += chunk_seconds - AUDIO_OVERLAP_SECONDS

            sdh_segments = [s for s in sdh_segments
                           if s["end"] - s["start"] >= 0.5]

            if progress_callback:
                progress_callback(f"[SDH] Detectate {len(sdh_segments)} evenimente audio.")
            logger.info(f"[SDH] Detectate {len(sdh_segments)} evenimente audio.")

            return sdh_segments

        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)

    def _get_sdh_description(
        self,
        event_key: str,
        target_lang: str,
        clap_results: List[Dict],
        use_llm: bool,
        llm_callback: Optional[Callable]
    ) -> str:
        if use_llm and llm_callback:
            try:
                context = ", ".join([r["event_key"] for r in clap_results[:3]])
                return llm_callback(event_key, context, target_lang)
            except Exception:
                pass

        lang_dict = SDH_TRANSLATIONS.get(target_lang, SDH_TRANSLATIONS.get("en", {}))
        if event_key in lang_dict:
            return lang_dict[event_key]

        en_dict = SDH_TRANSLATIONS.get("en", {})
        if event_key in en_dict:
            return en_dict[event_key]

        return event_key.replace("_", " ").title()


def merge_all_subtitle_sources(asr_segments, ocr_segments, sdh_segments):
    all_segments = []

    for seg in asr_segments:
        seg["source"] = seg.get("source", "asr")
        all_segments.append(seg)

    for seg in ocr_segments:
        all_segments.append(seg)

    for seg in sdh_segments:
        seg["source"] = "sdh"
        all_segments.append(seg)

    all_segments.sort(key=lambda x: x["start"])
    for i, seg in enumerate(all_segments, 1):
        seg["id"] = i

    return all_segments


def make_llm_sdh_callback(llm_client, target_lang):
    def callback(event_key: str, context: str, lang: str) -> str:
        prompt = (
            f"Generate a short, natural SDH subtitle description in {lang} "
            f"for this audio event: {event_key}. "
            f"Context (other detected sounds): {context}. "
            f"Return ONLY the description text, 1-4 words, no brackets, no explanation."
        )
        try:
            response = llm_client.complete(prompt, max_tokens=20)
            return response.strip()
        except Exception:
            return event_key.replace("_", " ").title()

    return callback


_sdh_detector_instance = None

def get_sdh_detector() -> SDHDetector:
    global _sdh_detector_instance
    if _sdh_detector_instance is None:
        _sdh_detector_instance = SDHDetector()
    return _sdh_detector_instance
