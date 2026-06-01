"""
translategemma_translator.py

Integrare TranslateGemma pentru traducere locala de subtitrari.
Modele Google dedicate traducerii, bazate pe Gemma 3.

HuggingFace:
  - google/translategemma-4b-it   (~8GB BF16 / ~4GB INT8)
  - google/translategemma-12b-it  (~24GB BF16 / ~12GB INT8)
  - google/translategemma-27b-it  (~54GB BF16 / ~27GB INT8) <- RECOMANDAT 48GB VRAM

Limbi suportate: 55 (ISO 639-1 Alpha-2 codes)
Documentatie: https://huggingface.co/google/translategemma-27b-it
"""

import logging
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)

TRANSLATE_GEMMA_MODELS = {
    "translategemma-4b": {
        "model_id":    "google/translategemma-4b-it",
        "label":       "TranslateGemma 4B (~8GB BF16 / ~4GB INT8)",
        "vram_bf16_gb": 8,
        "vram_int8_gb": 4,
        "recommended_for": "GPU-uri mici, testare rapida",
        "batch_size":  20,
    },
    "translategemma-12b": {
        "model_id":    "google/translategemma-12b-it",
        "label":       "TranslateGemma 12B (~24GB BF16 / ~12GB INT8)",
        "vram_bf16_gb": 24,
        "vram_int8_gb": 12,
        "recommended_for": "GPU-uri 24-40GB VRAM",
        "batch_size":  30,
    },
    "translategemma-27b": {
        "model_id":    "google/translategemma-27b-it",
        "label":       "TranslateGemma 27B * (~27GB INT8) — Recomandat 48GB",
        "vram_bf16_gb": 54,
        "vram_int8_gb": 27,
        "recommended_for": "48GB VRAM — cel mai bun model de traducere",
        "batch_size":  10,
    },
}

DEFAULT_MODEL_KEY = "translategemma-27b"

TRANSLATEGEMMA_SUPPORTED_LANGS = {
    "af": "Afrikaans",   "ar": "Araba",        "bg": "Bul gara",
    "bn": "Bengaleza",   "ca": "Catalana",      "cs": "Ceha",
    "cy": "Galeza",      "da": "Daneza",        "de": "Germana",
    "el": "Greaca",      "en": "Engleza",       "es": "Spaniola",
    "et": "Estoniana",   "fa": "Persana",       "fi": "Finlandeza",
    "fr": "Franceza",    "gu": "Gujarati",      "he": "Ebraica",
    "hi": "Hindi",       "hr": "Croata",        "hu": "Magiara",
    "hy": "Armeniana",   "id": "Indoneziana",   "is": "Islandeza",
    "it": "Italiana",    "ja": "Japoneza",      "ka": "Georgiana",
    "kn": "Kannada",     "ko": "Coreeana",      "lt": "Lituaniana",
    "lv": "Letona",      "mk": "Macedoneana",   "ml": "Malayalam",
    "mr": "Marathi",     "ms": "Malaieziana",   "mt": "Malteza",
    "nl": "Olandeza",    "no": "Norvegiana",    "pl": "Poloneza",
    "pt": "Portugheza",  "ro": "Romana",        "ru": "Rusa",
    "sk": "Slovaca",     "sl": "Slovena",       "sq": "Albancea",
    "sr": "Sarba",       "sv": "Suedeza",       "sw": "Swahili",
    "ta": "Tamila",      "te": "Telugu",        "th": "Tailandeza",
    "tr": "Turca",       "uk": "Ucraineana",    "ur": "Urdu",
    "vi": "Vietnameza",  "zh": "Chineza",
}

LANG_NAMES_EN = {
    "af": "Afrikaans",   "ar": "Arabic",       "bg": "Bulgarian",
    "bn": "Bengali",     "ca": "Catalan",       "cs": "Czech",
    "cy": "Welsh",       "da": "Danish",        "de": "German",
    "el": "Greek",       "en": "English",       "es": "Spanish",
    "et": "Estonian",    "fa": "Persian",       "fi": "Finnish",
    "fr": "French",      "gu": "Gujarati",      "he": "Hebrew",
    "hi": "Hindi",       "hr": "Croatian",      "hu": "Hungarian",
    "hy": "Armenian",    "id": "Indonesian",    "is": "Icelandic",
    "it": "Italian",     "ja": "Japanese",      "ka": "Georgian",
    "kn": "Kannada",     "ko": "Korean",        "lt": "Lithuanian",
    "lv": "Latvian",     "mk": "Macedonian",    "ml": "Malayalam",
    "mr": "Marathi",     "ms": "Malay",         "mt": "Maltese",
    "nl": "Dutch",       "no": "Norwegian",     "pl": "Polish",
    "pt": "Portuguese",  "ro": "Romanian",      "ru": "Russian",
    "sk": "Slovak",      "sl": "Slovenian",     "sq": "Albanian",
    "sr": "Serbian",     "sv": "Swedish",       "sw": "Swahili",
    "ta": "Tamil",       "te": "Telugu",        "th": "Thai",
    "tr": "Turkish",     "uk": "Ukrainian",     "ur": "Urdu",
    "vi": "Vietnamese",  "zh": "Chinese",
}

PIVOT_THROUGH_ENGLISH = True


class TranslateGemmaTranslator:
    def __init__(self):
        self._model     = None
        self._tokenizer = None
        self._loaded    = False
        self._model_key = None
        self._use_int8  = True

    def is_available(self) -> bool:
        try:
            import transformers
            import torch
            return True
        except ImportError:
            return False

    def get_available_vram_gb(self) -> float:
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(0)
                return free / (1024 ** 3)
        except Exception:
            pass
        return 0.0

    def recommend_model_for_vram(self, vram_gb: float) -> str:
        if vram_gb >= 28:
            return "translategemma-27b"
        elif vram_gb >= 13:
            return "translategemma-12b"
        else:
            return "translategemma-4b"

    def load_model(
        self,
        model_key:          str  = DEFAULT_MODEL_KEY,
        use_int8:           bool = True,
        progress_callback        = None
    ):
        if self._loaded and self._model_key == model_key:
            return

        if self._loaded:
            self.unload_model()

        cfg = TRANSLATE_GEMMA_MODELS.get(model_key)
        if not cfg:
            raise ValueError(
                f"Model necunoscut: {model_key}. "
                f"Disponibile: {list(TRANSLATE_GEMMA_MODELS.keys())}"
            )

        model_id    = cfg["model_id"]
        vram_needed = cfg["vram_int8_gb"] if use_int8 else cfg["vram_bf16_gb"]

        if progress_callback:
            progress_callback(
                f"[TranslateGemma] Incarcare {model_key} ({model_id})...\n"
                f"La prima rulare se descarca din HuggingFace."
            )

        # Check available VRAM before attempting load
        available_vram = self.get_available_vram_gb()
        if available_vram > 0 and available_vram < vram_needed:
            msg = (
                f"VRAM insuficient: {available_vram:.1f}GB liberi, "
                f"dar {model_key} necesita ~{vram_needed}GB "
                f"({'INT8' if use_int8 else 'BF16'}).\n"
                f"Elibereaza VRAM (opreste alte procese GPU) sau alege un model mai mic."
            )
            logger.error(f"[TranslateGemma] {msg}")
            if progress_callback:
                progress_callback(f"[TranslateGemma] EROARE: {msg}")
            raise RuntimeError(msg)

        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        # Clear any fragmented VRAM before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)

        if not torch.cuda.is_available():
            logger.warning("[TranslateGemma] CUDA indisponibil! Modelul va rula pe CPU (foarte lent).")
            if progress_callback:
                progress_callback("[TranslateGemma] ATENTIE: CUDA indisponibil! Rulare pe CPU.")

        # Try INT8 first, fall back to float16 on GPU
        model = None
        actual_int8 = False

        if use_int8:
            try:
                from transformers import BitsAndBytesConfig
                qcfg = BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=qcfg,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                )
                actual_int8 = True
            except Exception as e:
                logger.warning(f"[TranslateGemma] INT8 esuat: {e}. Incerc bfloat16.")

        if model is None:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
                )
            except Exception as e:
                logger.error(f"[TranslateGemma] Eroare incarcare model: {e}")
                raise

        self._model = model
        self._model.eval()

        self._loaded    = True
        self._model_key = model_key
        self._use_int8  = actual_int8

        # Verifica pe ce device ruleaza modelul
        param_device = next(self._model.parameters()).device
        total_params = sum(p.numel() for p in self._model.parameters())
        logger.info(
            f"[TranslateGemma] Model {model_key} incarcat pe {param_device} "
            f"({total_params/1e9:.1f}B param, {'INT8' if actual_int8 else 'bfloat16'})."
        )
        if progress_callback:
            progress_callback(
                f"[TranslateGemma] Model {model_key} incarcat pe "
                f"{'GPU' if param_device.type == 'cuda' else 'CPU'} "
                f"({'INT8' if actual_int8 else 'bfloat16'})."
            )

    def _build_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type":             "text",
                        "text":             text,
                        "source_lang_code": source_lang,
                        "target_lang_code": target_lang,
                    }
                ]
            }
        ]
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _translate_batch_inner(
        self,
        texts:        List[str],
        source_lang:  str,
        target_lang:  str,
        max_new_tokens: int = 80
    ) -> List[str]:
        import torch
        device = next(self._model.parameters()).device
        results = []
        for text in texts:
            prompt = self._build_prompt(text, source_lang, target_lang)
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                add_special_tokens=False,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            input_len = inputs["input_ids"].shape[1]
            new_tokens = outputs[0][input_len:]
            translated = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            # Take first line only (model sometimes adds "Sau:\nalternative" after newline)
            first_line = translated.strip().split("\n")[0] if translated.strip() else ""
            results.append(first_line.strip())
        return results

    def translate_batch(
        self,
        texts:                List[str],
        source_lang:          str,
        target_lang:          str,
        batch_size:           int           = 10,
        progress_callback                   = None,
        pivot_through_english: bool         = True
    ) -> List[str]:
        if not self._loaded:
            raise RuntimeError("Modelul nu este incarcat. Apeleaza load_model() mai intai.")

        if source_lang not in TRANSLATEGEMMA_SUPPORTED_LANGS:
            raise ValueError(
                f"Limba sursa '{source_lang}' nu este suportata de TranslateGemma.\n"
                f"Limbi disponibile: {list(TRANSLATEGEMMA_SUPPORTED_LANGS.keys())}"
            )

        if target_lang not in TRANSLATEGEMMA_SUPPORTED_LANGS:
            raise ValueError(
                f"Limba tinta '{target_lang}' nu este suportata de TranslateGemma.\n"
                f"Limbi disponibile: {list(TRANSLATEGEMMA_SUPPORTED_LANGS.keys())}"
            )

        needs_pivot = (
            pivot_through_english
            and source_lang != "en"
            and target_lang != "en"
        )

        if needs_pivot and progress_callback:
            progress_callback(
                f"[TranslateGemma] Traducere {source_lang} -> en -> {target_lang} "
                f"(pivot prin engleza pentru calitate mai buna)."
            )

        translated_texts = []
        total = len(texts)

        for batch_start in range(0, total, batch_size):
            batch_texts = texts[batch_start: batch_start + batch_size]

            if progress_callback:
                pct = int(batch_start / total * 100)
                progress_callback(
                    f"[TranslateGemma] Traducere: {pct}% "
                    f"({batch_start}/{total} segmente)..."
                )

            batch_texts_clean = [t.strip() for t in batch_texts]
            non_empty_indices = [i for i, t in enumerate(batch_texts_clean) if t]
            empty_indices = [i for i, t in enumerate(batch_texts_clean) if not t]

            results_for_batch = [""] * len(batch_texts_clean)

            if non_empty_indices:
                to_translate = [batch_texts_clean[i] for i in non_empty_indices]

                toks = min(300, 30 * len(to_translate))
                try:
                    if needs_pivot:
                        en_results = self._translate_batch_inner(
                            to_translate, source_lang, "en", max_new_tokens=toks
                        )
                        final_results = self._translate_batch_inner(
                            en_results, "en", target_lang, max_new_tokens=toks
                        )
                    else:
                        final_results = self._translate_batch_inner(
                            to_translate, source_lang, target_lang, max_new_tokens=toks
                        )

                    for idx, result in zip(non_empty_indices, final_results):
                        results_for_batch[idx] = result

                except Exception as e:
                    logger.error(f"[TranslateGemma] Eroare batch: {e}. Cad pe traducere individuala.")
                    for idx, text in zip(non_empty_indices, to_translate):
                        try:
                            if needs_pivot:
                                en = self._translate_batch_inner([text], source_lang, "en")
                                final = self._translate_batch_inner(en, "en", target_lang)
                                results_for_batch[idx] = final[0]
                            else:
                                final = self._translate_batch_inner([text], source_lang, target_lang)
                                results_for_batch[idx] = final[0]
                        except Exception as e2:
                            logger.error(f"[TranslateGemma] Eroare segment '{text[:50]}...': {e2}")
                            results_for_batch[idx] = text

            translated_texts.extend(results_for_batch)

        if progress_callback:
            progress_callback(
                f"[TranslateGemma] Traducere completa: {len(translated_texts)} segmente."
            )

        return translated_texts

    def unload_model(self):
        if self._model is not None:
            import torch
            del self._model
            del self._tokenizer
            self._model     = None
            self._tokenizer = None
            self._loaded    = False
            self._model_key = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("[TranslateGemma] Model descarcat din VRAM.")

    def get_model_info(self) -> Dict:
        if not self._loaded:
            return {"loaded": False}
        cfg = TRANSLATE_GEMMA_MODELS.get(self._model_key, {})
        return {
            "loaded":     True,
            "model_key":  self._model_key,
            "model_id":   cfg.get("model_id"),
            "use_int8":   self._use_int8,
            "vram_used":  f"~{cfg.get('vram_int8_gb' if self._use_int8 else 'vram_bf16_gb')}GB",
        }


_translategemma_instance: Optional[TranslateGemmaTranslator] = None

def get_translategemma_translator() -> TranslateGemmaTranslator:
    global _translategemma_instance
    if _translategemma_instance is None:
        _translategemma_instance = TranslateGemmaTranslator()
    return _translategemma_instance
