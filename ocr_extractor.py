"""
ocr_extractor.py
Extragere subtitrări hardcodate și text de pe ecran din video folosind PaddleOCR.
Model: PP-OCRv5 (v3.x)
"""

import cv2
import numpy as np
import os
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher

# torch e necesar pentru verificare GPU în unele medii
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# Zona implicită de căutare a subtitrărilor: treimea de jos a imaginii
# (0.0 = sus, 1.0 = jos)
DEFAULT_SUBTITLE_REGION_TOP = 0.70     # de la 70% în jos
DEFAULT_SUBTITLE_REGION_BOTTOM = 0.98  # până la 98%

# Parametri OCR
DEFAULT_CONF_THRESHOLD = 70   # confidence minim (0-100)
DEFAULT_SIM_THRESHOLD = 80    # similaritate pentru deduplicare linii identice
FRAMES_TO_SKIP = 2            # procesează 1 din N frame-uri (mai rapid)

# PaddleOCR v3.x (PP-OCRv5) — coduri limbă valide
# Limbile latine europene se pasează direct cu codul ISO
# Documentație: https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html
PADDLE_LANG_MAP = {
    # Limbi latine — cod ISO direct, PP-OCRv5 le suportă nativ
    "ro": "ro",   # Română
    "it": "it",   # Italiană
    "fr": "fr",   # Franceză
    "de": "de",   # Germană
    "es": "es",   # Spaniolă
    "pt": "pt",   # Portugheză
    "nl": "nl",   # Olandeză
    "pl": "pl",   # Poloneză
    "cs": "cs",   # Cehă
    "sk": "sk",   # Slovacă
    "hr": "hr",   # Croată
    "sl": "sl",   # Slovenă
    "da": "da",   # Daneză
    "sv": "sv",   # Suedeză
    "fi": "fi",   # Finlandeză
    "et": "et",   # Estoniană
    "lv": "lv",   # Letonă
    "lt": "lt",   # Lituaniană
    "hu": "hu",   # Maghiară
    "mt": "mt",   # Malteză
    "sq": "sq",   # Albaneză
    "en": "en",   # Engleză

    # Chirilice
    "ru": "ru",   # Rusă
    "uk": "uk",   # Ucraineană
    "bg": "bg",   # Bulgară
    "sr": "sr",   # Sârbă
    "mk": "mk",   # Macedoneană

    # Alte scripturi
    "el": "el",   # Greacă
    "zh": "ch",   # Chineză simplificată (cod special PaddleOCR)
    "ja": "japan", # Japoneză (cod special PaddleOCR)
    "ko": "korean", # Coreeană (cod special PaddleOCR)
    "ar": "ar",   # Arabă
}

# Fallback: dacă limba nu e în mapare, folosește "en"
PADDLE_LANG_FALLBACK = "en"


def _clean_paddle_modules():
    """Șterge toate modulele paddle din sys.modules pentru a forța o reimportare fresh."""
    import sys
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("paddle") or "paddlex" in mod_name:
            del sys.modules[mod_name]


def check_and_install_paddleocr(use_gpu: bool = False) -> bool:
    """
    Verifică instalarea PaddleOCR și paddlepaddle (CPU sau GPU).
    """
    import subprocess
    import sys

    PADDLEOCR_PIN = "paddleocr>=2.7.0,<2.9"
    PADDLE_CPU_PIN = "paddlepaddle>=2.6.1,<3.0"

    # 1. Verifică PaddleOCR
    try:
        from paddleocr import PaddleOCR
        try:
            from paddleocr import __version__ as _ocr_ver
            if tuple(int(x) for x in _ocr_ver.split(".")[:1]) >= (3,):
                logger.warning(
                    f"[OCR] PaddleOCR {_ocr_ver} detectat, dar codul necesită "
                    f"API 2.x. Se reinstalează cu {PADDLEOCR_PIN}..."
                )
                raise ImportError("versiune 3.x incompatibilă")
        except (ImportError, ValueError, AttributeError):
            raise ImportError("versiune necunoscută sau lipsă __version__")
    except ImportError:
        logger.info(f"[OCR] Instalare {PADDLEOCR_PIN}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "paddleocr", "-y", "--quiet"],
            check=False,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", PADDLEOCR_PIN, "--quiet"],
            check=True,
        )

    # 2. Verifică Paddle Backend (GPU sau CPU)
    # NU șterge sys.modules înainte de primul import — C extensions
    # nu pot fi reimportate în aceeași sesiune după clean_modules().
    try:
        import paddle
        backend_gpu = paddle.device.is_compiled_with_cuda()
    except (ImportError, Exception):
        _clean_paddle_modules()
        backend_gpu = False

    if use_gpu:
        if backend_gpu:
            logger.info("[OCR] Paddle GPU backend este deja funcțional.")
            return True

        cuda_version = _detect_cuda_version()
        logger.info(f"[OCR] Instalare Paddle GPU (Sistem detectat: CUDA {cuda_version})...")

        index_url = "https://www.paddlepaddle.org.cn/packages/stable/cu120/"
        if cuda_version and (cuda_version.startswith("12") or cuda_version.startswith("13")):
            pkg = "paddlepaddle-gpu==2.6.2.post120"
            index_url = "https://www.paddlepaddle.org.cn/packages/stable/cu120/"
        elif cuda_version and cuda_version.startswith("11"):
            pkg = "paddlepaddle-gpu==2.6.2.post118"
            index_url = "https://www.paddlepaddle.org.cn/packages/stable/cu118/"
        else:
            pkg = "paddlepaddle-gpu==2.6.2.post120"

        # Nu dezinstala CPU înainte — instalează GPU peste cel existent.
        # Dacă instalarea GPU eșuează, fișierele CPU rămân intacte.
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "-i", index_url],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            logger.error(f"[OCR] Eșec instalare GPU: {result.stderr.strip()}")
            return False

        # Verificare GPU printr-un subproces (nu reimporta C extensions în același proces)
        verify = subprocess.run(
            [sys.executable, "-c", "import paddle; print(paddle.device.is_compiled_with_cuda())"],
            capture_output=True, text=True
        )
        if verify.returncode == 0 and verify.stdout.strip() == "True":
            logger.info("[OCR] Paddle GPU instalat cu succes.")
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "paddlepaddle", "-y", "--quiet"], check=False)
            return True
        else:
            logger.warning(f"[OCR] GPU instalat dar verificarea a eșuat: {verify.stderr.strip()}")
            return True
    else:
        # Doar CPU
        try:
            import paddle
            return True
        except Exception:
            _clean_paddle_modules()
            # Curățăm și varianta GPU dacă există
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "paddlepaddle", "paddlepaddle-gpu", "-y", "--quiet"],
                check=False,
            )
            subprocess.run([sys.executable, "-m", "pip", "install", PADDLE_CPU_PIN, "--quiet"], check=True)
            _clean_paddle_modules()
            import paddle
            return True


def _detect_cuda_version() -> str:
    """Detectează versiunea CUDA instalată pe sistem."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    parts = line.split("release")
                    if len(parts) > 1:
                        version = parts[1].strip().split(",")[0].strip()
                        return version
    except FileNotFoundError:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return "12"  # Assume modern CUDA
    except FileNotFoundError:
        pass

    return ""


def _is_gpu_actually_available() -> bool:
    """Verifică dacă GPU e disponibil fie prin torch, fie prin paddle."""
    if TORCH_AVAILABLE:
        try:
            if torch.cuda.is_available():
                return True
        except Exception:
            pass

    try:
        import paddle
        return paddle.device.is_compiled_with_cuda()
    except Exception:
        pass

    return False


class HardcodedSubtitleExtractor:
    """
    Extrage subtitrări hardcodate (burn-in) și text de pe ecran din video folosind PaddleOCR.
    """

    def __init__(self):
        self._ocr = None
        self._ocr_loaded = False
        self._current_lang = None

    def is_available(self) -> bool:
        try:
            from paddleocr import PaddleOCR
            return True
        except ImportError:
            return False

    def load_ocr(self, lang: str = "en", use_gpu: bool = False):
        """
        Încarcă PaddleOCR cu codul de limbă corect pentru v3.x (PP-OCRv5).
        """
        if self._ocr_loaded and self._current_lang == lang:
            return

        from paddleocr import PaddleOCR

        # Mapare la codul corect PaddleOCR
        paddle_lang = PADDLE_LANG_MAP.get(lang, PADDLE_LANG_FALLBACK)

        logger.info(
            f"[OCR] Încărcare PaddleOCR PP-OCRv5 "
            f"(limbă input: '{lang}' → cod PaddleOCR: '{paddle_lang}', GPU: {use_gpu})..."
        )

        if use_gpu:
            import paddle
            paddle.set_device('gpu')
            _t = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], place='gpu')
            _t = paddle.matmul(_t, _t)
        try:
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang=paddle_lang,
                use_gpu=use_gpu,
                show_log=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
            self._ocr_loaded = True
            self._current_lang = lang
            logger.info(f"[OCR] PaddleOCR încărcat cu succes (lang='{paddle_lang}').")

        except Exception as e:
            error_msg = str(e)

            if "No models are available" in error_msg:
                logger.warning(
                    f"[OCR] Limba '{paddle_lang}' nu are model disponibil. Fallback la 'en'."
                )
                self._ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    use_gpu=use_gpu,
                    show_log=False,
                )
                self._ocr_loaded = True
                self._current_lang = lang
                logger.info("[OCR] PaddleOCR încărcat cu fallback 'en'.")

            elif "GPU" in error_msg or "cuda" in error_msg.lower():
                logger.warning(
                    f"[OCR] GPU nu e disponibil ({error_msg}). Fallback la CPU."
                )
                self._ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=paddle_lang if paddle_lang != PADDLE_LANG_FALLBACK else "en",
                    use_gpu=False,
                    show_log=False,
                )
                self._ocr_loaded = True
                self._current_lang = lang
                logger.info("[OCR] PaddleOCR încărcat pe CPU (fallback).")

            elif "paddle" in error_msg.lower() and (
                "partially initialized" in error_msg.lower()
                or "circular import" in error_msg.lower()
                or "no attribute" in error_msg.lower()
            ):
                logger.warning(
                    f"[OCR] Eroare modul Paddle: {error_msg}. Se încearcă reparația..."
                )
                check_and_install_paddleocr(use_gpu=use_gpu)
                _clean_paddle_modules()
                self._ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=paddle_lang,
                    use_gpu=use_gpu,
                    show_log=False,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                )
                self._ocr_loaded = True
                self._current_lang = lang
                logger.info("[OCR] PaddleOCR încărcat după reparație Paddle.")

            else:
                raise

    def detect_subtitle_region(self, video_path: str) -> Tuple[float, float]:
        """Detectează automat zona din frame unde apar subtitrările."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return DEFAULT_SUBTITLE_REGION_TOP, DEFAULT_SUBTITLE_REGION_BOTTOM

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        sample_frames = min(30, total_frames)
        step = max(1, total_frames // sample_frames)

        text_y_positions = []

        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue

            bottom_third = frame[int(height * 0.6):, :]
            results = self._ocr.ocr(bottom_third, cls=True)

            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        bbox = line[0]
                        y_center = (bbox[0][1] + bbox[2][1]) / 2
                        y_relative = (int(height * 0.6) + y_center) / height
                        text_y_positions.append(y_relative)

        cap.release()

        if not text_y_positions:
            return DEFAULT_SUBTITLE_REGION_TOP, DEFAULT_SUBTITLE_REGION_BOTTOM

        y_positions = sorted(text_y_positions)
        median_y = y_positions[len(y_positions) // 2]

        top = max(0.0, median_y - 0.15)
        bottom = min(1.0, median_y + 0.08)

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
        progress_callback=None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict]:
        """
        Extrage subtitrările hardcodate și orice text de pe ecran.
        Dacă start_time și end_time sunt specificate, procesează doar intervalul respectiv.
        """
        if not self._ocr_loaded or self._current_lang != lang:
            # Check if GPU requested is actually available
            actual_gpu = use_gpu and _is_gpu_actually_available()
            self.load_ocr(lang=lang, use_gpu=actual_gpu)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Nu pot deschide video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if subtitle_region is None:
            if progress_callback:
                progress_callback("[OCR] Detectare automată zonă subtitrare...")
            top_ratio, bottom_ratio = self.detect_subtitle_region(video_path)
        else:
            top_ratio, bottom_ratio = subtitle_region

        top_px = int(height * top_ratio)
        bottom_px = int(height * bottom_ratio)

        # Calculate frame range from time range
        if start_time is not None:
            start_frame = int(start_time * fps)
        else:
            start_frame = 0

        if end_time is not None:
            end_frame = int(end_time * fps)
        else:
            end_frame = total_frames

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_count = min(total_frames, end_frame) - start_frame

        if progress_callback:
            msg = f"[OCR] Procesare video ({frame_count} frame-uri)"
            if top_ratio == 0 and bottom_ratio == 1:
                msg += " - TOATĂ IMAGINEA"
            if start_time is not None:
                msg += f" ({start_time}s-{end_time}s)"
            progress_callback(f"{msg}...")

        frame_texts = {}
        frame_idx = start_frame
        processed = 0

        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % (frames_to_skip + 1) == 0:
                if top_ratio == 0 and bottom_ratio == 1:
                    crop = frame
                else:
                    crop = frame[top_px:bottom_px, :]

                crop = self._preprocess_for_ocr(crop)
                results = self._ocr.ocr(crop, cls=True)
                text_lines = []

                if results and results[0]:
                    for line in results[0]:
                        if not line or len(line) < 2:
                            continue
                        text_content = line[1][0]
                        confidence = line[1][1]

                        if confidence * 100 >= conf_threshold and text_content.strip():
                            cleaned = text_content.strip()
                            if len(cleaned) > 1:
                                text_lines.append(cleaned)

                if text_lines:
                    frame_texts[frame_idx] = " | ".join(text_lines)

                processed += 1
                if progress_callback and processed % 100 == 0:
                    total_range = frame_count
                    pct = int(processed / max(total_range // (frames_to_skip + 1), 1) * 100)
                    progress_callback(f"[OCR] Progres: {pct}% ({processed} frame-uri procesate)")

            frame_idx += 1

        cap.release()

        if not frame_texts:
            if progress_callback:
                progress_callback("[OCR] Nu s-a găsit text în video.")
            return []

        segments = self._group_frames_to_segments(frame_texts, fps, sim_threshold)

        if progress_callback:
            progress_callback(f"[OCR] Extrase {len(segments)} elemente text.")

        return segments

    def _preprocess_for_ocr(self, img: np.ndarray) -> np.ndarray:
        scale = 2.0
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        return img

    def _group_frames_to_segments(
        self,
        frame_texts: Dict[int, str],
        fps: float,
        sim_threshold: int
    ) -> List[Dict]:
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
                current_text = text
                current_start_frame = frame
                current_end_frame = frame
            else:
                similarity = SequenceMatcher(None, current_text, text).ratio() * 100

                if similarity >= sim_threshold:
                    current_end_frame = frame
                else:
                    if current_text.strip() and current_text != last_added_text:
                        segments.append({
                            "start": current_start_frame / fps,
                            "end": (current_end_frame + 1) / fps,
                            "text": current_text,
                            "source": "ocr"
                        })
                        last_added_text = current_text

                    current_text = text
                    current_start_frame = frame
                    current_end_frame = frame

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
    all_segments = asr_segments.copy()

    for ocr_seg in ocr_segments:
        overlap_found = False
        for asr_seg in asr_segments:
            if (ocr_seg["start"] < asr_seg["end"] and
                ocr_seg["end"] > asr_seg["start"]):

                ocr_text_clean = ocr_seg["text"].replace(" | ", " ").lower()
                asr_text_clean = asr_seg["text"].lower()

                sim = SequenceMatcher(None, ocr_text_clean, asr_text_clean).ratio()

                if sim > 0.6:
                    overlap_found = True
                    break

        if not overlap_found:
            all_segments.append(ocr_seg)

    all_segments.sort(key=lambda x: x["start"])
    for i, seg in enumerate(all_segments):
        seg["id"] = i

    return all_segments
