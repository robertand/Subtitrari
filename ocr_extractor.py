"""
ocr_extractor.py
Extragere subtitrări hardcodate din video folosind PaddleOCR.
Returnează segmente cu timestamps compatibile cu formatul aplicației.
"""

import cv2
import numpy as np
import os
import subprocess
import tempfile
import logging
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# Zona implicită de căutare a subtitrărilor: treimea de jos a imaginii
# (0.0 = sus, 1.0 = jos)
DEFAULT_SUBTITLE_REGION_TOP = 0.70     # de la 70% în jos
DEFAULT_SUBTITLE_REGION_BOTTOM = 0.98  # până la 98%

# Parametri OCR
DEFAULT_CONF_THRESHOLD = 70   # confidence minim (0-100)
DEFAULT_SIM_THRESHOLD = 80    # similaritate pentru deduplicare linii identice
FRAMES_TO_SKIP = 2            # procesează 1 din N frame-uri (mai rapid)


def check_and_install_paddleocr():
    try:
        from paddleocr import PaddleOCR
        return True
    except ImportError:
        import subprocess, sys
        logger.info("[OCR] Instalare PaddleOCR și PaddlePaddle...")
        subprocess.run([sys.executable, "-m", "pip", "install",
                       "paddlepaddle", "paddleocr>=2.7.0", "opencv-python"],
                      check=True)
        return True


class HardcodedSubtitleExtractor:
    """
    Extrage subtitrări hardcodate (burn-in) din video folosind PaddleOCR.
    Detectează automat zona de subtitrare sau folosește zona specificată de utilizator.
    """

    def __init__(self):
        self._ocr = None
        self._ocr_loaded = False

    def is_available(self) -> bool:
        try:
            from paddleocr import PaddleOCR
            return True
        except ImportError:
            return False

    def load_ocr(self, lang: str = "en", use_gpu: bool = False):
        """
        Încarcă modelul PaddleOCR.
        La prima rulare descarcă modelele automat (~50-200MB).

        Args:
            lang: codul limbii PaddleOCR (ex: 'en', 'ch', 'latin', 'cyrillic')
                  NOTĂ: pentru limbi europene (română, italiană etc.) folosește 'latin'
            use_gpu: True pentru GPU NVIDIA
        """
        if self._ocr_loaded:
            return

        from paddleocr import PaddleOCR

        # Mapare limbă aplicație → limbă PaddleOCR
        # PaddleOCR grupează limbile în familii de script
        PADDLE_LANG_MAP = {
            "ro": "latin", "it": "latin", "fr": "latin", "de": "latin",
            "es": "latin", "pt": "latin", "nl": "latin", "pl": "latin",
            "cs": "latin", "sk": "latin", "hr": "latin", "sl": "latin",
            "da": "latin", "sv": "latin", "fi": "latin", "et": "latin",
            "lv": "latin", "lt": "latin", "hu": "latin", "mt": "latin",
            "en": "en",
            "ru": "cyrillic", "uk": "cyrillic", "bg": "cyrillic",
            "el": "greek",
            "zh": "ch", "ja": "japan", "ko": "korean", "ar": "arabic",
        }

        paddle_lang = PADDLE_LANG_MAP.get(lang, "latin")

        logger.info(f"[OCR] Încărcare PaddleOCR (limbă: {paddle_lang}, GPU: {use_gpu})...")

        self._ocr = PaddleOCR(
            use_angle_cls=True,   # detectează text rotit
            lang=paddle_lang,
            use_gpu=use_gpu,
            show_log=False,
        )
        self._ocr_loaded = True
        logger.info("[OCR] PaddleOCR încărcat.")

    def detect_subtitle_region(self, video_path: str) -> Tuple[float, float]:
        """
        Detectează automat zona din frame unde apar subtitrările,
        analizând primele N frame-uri pentru a găsi unde apare text consistent.
        Returnează (top_ratio, bottom_ratio) ca fracțiuni din înălțimea imaginii.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return DEFAULT_SUBTITLE_REGION_TOP, DEFAULT_SUBTITLE_REGION_BOTTOM

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Analizează 30 de frame-uri distribuite uniform
        sample_frames = min(30, total_frames)
        step = max(1, total_frames // sample_frames)

        text_y_positions = []

        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue

            # Căutare rapidă text cu PaddleOCR în treimea de jos
            bottom_third = frame[int(height * 0.6):, :]
            results = self._ocr.ocr(bottom_third, cls=True)

            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        bbox = line[0]
                        y_center = (bbox[0][1] + bbox[2][1]) / 2
                        # Reconvertire la coordonate relative la frame complet
                        y_relative = (int(height * 0.6) + y_center) / height
                        text_y_positions.append(y_relative)

        cap.release()

        if not text_y_positions:
            return DEFAULT_SUBTITLE_REGION_TOP, DEFAULT_SUBTITLE_REGION_BOTTOM

        # Găsește centrul clusterului principal de text
        y_positions = sorted(text_y_positions)
        median_y = y_positions[len(y_positions) // 2]

        top = max(0.0, median_y - 0.15)
        bottom = min(1.0, median_y + 0.08)

        logger.info(f"[OCR] Zonă detectată automat: {top:.2f} - {bottom:.2f}")
        return top, bottom

    def extract_subtitles(
        self,
        video_path: str,
        lang: str = "en",
        use_gpu: bool = False,
        subtitle_region: Optional[Tuple[float, float]] = None,
        conf_threshold: int = DEFAULT_CONF_THRESHOLD,
        sim_threshold: int = DEFAULT_SIM_THRESHOLD,
        frames_to_skip: int = FRAMES_TO_SKIP,
        progress_callback=None
    ) -> List[Dict]:
        """
        Extrage subtitrările hardcodate din video.

        Args:
            video_path: cale fișier video
            lang: codul limbii subtitrărilor
            use_gpu: True pentru GPU NVIDIA
            subtitle_region: (top_ratio, bottom_ratio) sau None pentru auto-detect
            conf_threshold: prag minim confidence OCR (0-100)
            sim_threshold: prag similaritate pentru deduplicare (0-100)
            frames_to_skip: procesează 1 din N frame-uri
            progress_callback: funcție callback(mesaj: str)

        Returns:
            Lista segmente: [{"start": float, "end": float, "text": str, "source": "ocr"}]
        """
        if not self._ocr_loaded:
            self.load_ocr(lang=lang, use_gpu=use_gpu)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Nu pot deschide video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Detectare automată zonă dacă nu e specificată
        if subtitle_region is None:
            if progress_callback:
                progress_callback("[OCR] Detectare automată zonă subtitrare...")
            top_ratio, bottom_ratio = self.detect_subtitle_region(video_path)
        else:
            top_ratio, bottom_ratio = subtitle_region

        top_px = int(height * top_ratio)
        bottom_px = int(height * bottom_ratio)

        if progress_callback:
            msg = f"[OCR] Procesare video ({total_frames} frame-uri)"
            if top_ratio == 0 and bottom_ratio == 1:
                msg += " - TOATĂ IMAGINEA"
            progress_callback(f"{msg}...")

        # Colectează text per frame
        frame_texts = {}  # frame_idx → text
        frame_idx = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % (frames_to_skip + 1) == 0:
                # Crop zona de subtitrare (sau full frame dacă 0-1)
                if top_ratio == 0 and bottom_ratio == 1:
                    crop = frame
                else:
                    crop = frame[top_px:bottom_px, :]

                # Preprocess pentru OCR mai bun
                crop = self._preprocess_for_ocr(crop)

                # OCR
                results = self._ocr.ocr(crop, cls=True)
                text_lines = []

                if results and results[0]:
                    for line in results[0]:
                        if not line or len(line) < 2:
                            continue
                        text_content = line[1][0]  # textul
                        confidence = line[1][1]    # confidence (0-1)

                        if confidence * 100 >= conf_threshold and text_content.strip():
                            # Curățare text (elimină caractere ciudate)
                            cleaned = text_content.strip()
                            if len(cleaned) > 1: # Ignorăm caractere singuratice/zgomot
                                text_lines.append(cleaned)

                if text_lines:
                    # Sortăm liniile după Y apoi X pentru a păstra ordinea naturală a citirii
                    # PaddleOCR returnează deja o ordine decentă, dar join-ul simplu e ok
                    frame_texts[frame_idx] = " | ".join(text_lines)

                processed += 1
                if progress_callback and processed % 100 == 0:
                    pct = int(frame_idx / total_frames * 100)
                    progress_callback(f"[OCR] Progres: {pct}% ({processed} frame-uri procesate)")

            frame_idx += 1

        cap.release()

        if not frame_texts:
            if progress_callback:
                progress_callback("[OCR] Nu s-a găsit text în video.")
            return []

        # Grupează frame-urile consecutive cu text similar în segmente
        segments = self._group_frames_to_segments(frame_texts, fps, sim_threshold)

        if progress_callback:
            progress_callback(f"[OCR] Extrase {len(segments)} subtitrări hardcodate.")

        return segments

    def _preprocess_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess imagine pentru OCR mai precis.
        Mărește contrastul, elimină zgomotul.
        """
        # Mărește imaginea pentru OCR mai precis
        scale = 2.0
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        # Versiune cu text alb evidențiat (opțional, PaddleOCR se descurcă bine și fără)
        return img

    def _group_frames_to_segments(
        self,
        frame_texts: Dict[int, str],
        fps: float,
        sim_threshold: int
    ) -> List[Dict]:
        """
        Grupează frame-urile consecutive cu text similar în segmente de subtitrare.
        Elimină duplicatele și frame-urile cu text aproape identic (subtitrare statică).
        """
        if not frame_texts:
            return []

        sorted_frames = sorted(frame_texts.keys())
        segments = []

        current_text = None
        current_start_frame = None
        current_end_frame = None
        last_added_text = None

        for frame in sorted_frames:
            text = frame_texts[frame]

            if current_text is None:
                # Primul text
                current_text = text
                current_start_frame = frame
                current_end_frame = frame
            else:
                # Verifică similaritate cu textul curent
                similarity = SequenceMatcher(None, current_text, text).ratio() * 100

                if similarity >= sim_threshold:
                    # Același subtitle — extinde segmentul
                    current_end_frame = frame
                else:
                    # Text nou — salvează segmentul anterior
                    if current_text.strip() and current_text != last_added_text:
                        seg = {
                            "start": current_start_frame / fps,
                            "end": (current_end_frame + 1) / fps,
                            "text": current_text,
                            "source": "ocr"  # marcaj că vine din OCR, nu ASR
                        }
                        segments.append(seg)
                        last_added_text = current_text

                    current_text = text
                    current_start_frame = frame
                    current_end_frame = frame

        # Ultimul segment
        if current_text and current_text.strip() and current_text != last_added_text:
            segments.append({
                "start": current_start_frame / fps,
                "end": (current_end_frame + 1) / fps,
                "text": current_text,
                "source": "ocr"
            })

        return segments


# Singleton
_ocr_extractor_instance = None

def get_ocr_extractor() -> HardcodedSubtitleExtractor:
    global _ocr_extractor_instance
    if _ocr_extractor_instance is None:
        _ocr_extractor_instance = HardcodedSubtitleExtractor()
    return _ocr_extractor_instance

def merge_ocr_and_asr_segments(asr_segments, ocr_segments):
    """
    Îmbină segmentele OCR cu cele ASR (Whisper/NeMo).
    - Dacă OCR și ASR au segmente care se suprapun temporal cu text similar → păstrează ASR
    - Dacă OCR are text în zone fără ASR → adaugă segmentele OCR
    - Dacă OCR are text complet diferit față de ASR în același interval → adaugă ambele
      (OCR poate fi un titlu de capitol sau text din scenă, diferit de dialog)
    """
    all_segments = asr_segments.copy()

    for ocr_seg in ocr_segments:
        # Verifică dacă există un segment ASR care se suprapune
        overlap_found = False
        for asr_seg in asr_segments:
            # Suprapunere temporală
            if (ocr_seg["start"] < asr_seg["end"] and
                ocr_seg["end"] > asr_seg["start"]):

                # Normalizăm textul pentru comparație (eliminăm separatorul | adăugat la OCR)
                ocr_text_clean = ocr_seg["text"].replace(" | ", " ").lower()
                asr_text_clean = asr_seg["text"].lower()

                # Verifică similaritate text
                sim = SequenceMatcher(
                    None,
                    ocr_text_clean,
                    asr_text_clean
                ).ratio()

                if sim > 0.6: # Scădem pragul pentru a evita duplicatele parțiale
                    # Text similar → ASR câștigă (mai precis pe dialog)
                    overlap_found = True
                    break

        if not overlap_found:
            # Text OCR nou (titlu, text din scenă, pancartă) → adaugă
            # Marcăm textul OCR pentru a fi identificabil în UI
            ocr_seg["text"] = f"[OCR] {ocr_seg['text']}"
            all_segments.append(ocr_seg)

    # Sortează cronologic
    all_segments.sort(key=lambda x: x["start"])

    # Re-numerotează
    for i, seg in enumerate(all_segments):
        seg["id"] = i

    return all_segments
