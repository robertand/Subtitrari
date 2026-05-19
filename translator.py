from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, NllbTokenizer
from deep_translator import GoogleTranslator
import torch
import gc
import numpy as np
import json
from typing import List, Dict, Optional, Any
import logging
import re
import time
from config import Config

logger = logging.getLogger(__name__)

class Translator:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Language code mappings for NLLB
        self.lang_map = {
            'ro': 'ron_Latn',  # Romanian
            'en': 'eng_Latn',  # English
            'fr': 'fra_Latn',  # French
            'de': 'deu_Latn',  # German
            'es': 'spa_Latn',  # Spanish
            'it': 'ita_Latn',  # Italian
            'pt': 'por_Latn',  # Portuguese
            'ru': 'rus_Cyrl',  # Russian
            'zh': 'zho_Hans',  # Chinese (Simplified)
            'ja': 'jpn_Jpan',  # Japanese
            'ko': 'kor_Hang',  # Korean
            'ar': 'arb_Arab',  # Arabic
            'hi': 'hin_Deva',  # Hindi
            'tr': 'tur_Latn',  # Turkish
            'nl': 'nld_Latn',  # Dutch
            'pl': 'pol_Latn',  # Polish
            'sv': 'swe_Latn',  # Swedish
            'da': 'dan_Latn',  # Danish
            'no': 'nob_Latn',  # Norwegian
            'fi': 'fin_Latn',  # Finnish
            'cs': 'ces_Latn',  # Czech
            'hu': 'hun_Latn',  # Hungarian
            'el': 'ell_Grek',  # Greek
            'he': 'heb_Hebr',  # Hebrew
            'th': 'tha_Thai',  # Thai
            'vi': 'vie_Latn',  # Vietnamese
            'id': 'ind_Latn',  # Indonesian
            'ms': 'zsm_Latn',  # Malay
            'uk': 'ukr_Cyrl',  # Ukrainian
            'bg': 'bul_Cyrl',  # Bulgarian
            'hr': 'hrv_Latn',  # Croatian
        }
        
        # Languages that need bridging through English (disabled as per request)
        self.bridge_languages = {}
    
    def load_model(self, source_lang: str, target_lang: str) -> tuple:
        """Load appropriate translation model"""
        model_name = "facebook/nllb-200-distilled-600M"
        
        if model_name not in self.models:
            logger.info(f"Loading translation model: {model_name}")
            
            # Load tokenizer with explicit NllbTokenizer
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(
                model_name,
                src_lang=self.lang_map.get(source_lang, 'eng_Latn')
            )
            
            # Load model
            self.models[model_name] = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            
            # Set model to eval mode
            self.models[model_name].eval()
        
        return self.models[model_name], self.tokenizers[model_name]
    
    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        batch_size: int = 8
    ) -> List[str]:
        """Translate a batch of texts using Google Translate (via deep-translator)"""
        try:
            logger.info(f"Translating {len(texts)} segments using Google Translate: {source_lang} -> {target_lang}")
            
            # Google Translate handles mapping internally usually, but let's be safe
            # NLLB maps 'ro' to 'ron_Latn', but Google wants 'ro'
            s_lang = source_lang if source_lang != 'auto' else 'auto'
            t_lang = target_lang

            translator = GoogleTranslator(source=s_lang, target=t_lang)

            # deep-translator can translate lists
            # Note: translate_batch in deep-translator takes a list
            translations = translator.translate_batch(texts)

            return translations

        except Exception as e:
            logger.error(f"Google Translate error: {e}")
            # Fallback to NLLB if Google fails
            return self._nllb_translate(texts, source_lang, target_lang, batch_size)

    def _nllb_translate(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        batch_size: int = 8
    ) -> List[str]:
        """NLLB Fallback translation"""
        try:
            model, tokenizer = self.load_model(source_lang, target_lang)
            src_code = self.lang_map.get(source_lang, 'eng_Latn')
            tgt_code = self.lang_map.get(target_lang, 'eng_Latn')
            translations = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                if hasattr(tokenizer, 'src_lang'):
                    tokenizer.src_lang = src_code
                
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                
                # Use target language token ID for generation
                forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)
                
                # Simple generation
                with torch.no_grad():
                    translated = model.generate(
                        **inputs,
                        max_length=512,
                        forced_bos_token_id=forced_bos_token_id
                    )
                
                decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
                translations.extend(decoded)
            return translations
        except Exception as e:
            logger.error(f"NLLB Fallback error: {e}")
            return texts
    
    def _simple_translate(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str
    ) -> List[str]:
        """Simple translation fallback using pipeline"""
        try:
            from transformers import pipeline
            
            # Use translation pipeline
            pipe = pipeline(
                "translation",
                model="facebook/nllb-200-distilled-600M",
                src_lang=self.lang_map.get(source_lang, 'eng_Latn'),
                tgt_lang=self.lang_map.get(target_lang, 'eng_Latn'),
                device=0 if self.device == "cuda" else -1,
                max_length=512
            )
            
            translations = []
            for text in texts:
                result = pipe(text, max_length=512)
                translations.append(result[0]['translation_text'])
            
            return translations
            
        except Exception as e:
            logger.error(f"Simple translation error: {e}")
            # Return original texts as last resort
            return texts
    
    def _bridge_translate(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        batch_size: int = 8
    ) -> List[str]:
        """Translate through English as bridge language"""
        logger.info(f"Using bridge translation: {source_lang} → en → {target_lang}")
        
        # Step 1: Source → English
        en_texts = self.translate_batch(texts, source_lang, 'en', batch_size)
        
        # Step 2: English → Target
        final_texts = self.translate_batch(en_texts, 'en', target_lang, batch_size)
        
        return final_texts
    
    def format_translation_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        custom_prompt: Optional[str] = None
    ) -> str:
        """Format text with translation prompt for LLM"""
        if custom_prompt:
            return custom_prompt.format(
                text=text,
                source=source_lang,
                target=target_lang
            )
        
        prompt = f"Tradu următorul text din {source_lang} în {target_lang}. "
        prompt += "Tradu ținând cont de însemnătatea cuvintelor în limba originală și tradu sensul folosind cuvinte, metafore care se potrivesc în limba de destinație.\n\n"
        prompt += f"Text de tradus:\n{text}\n\nTraducere:"
        
        return prompt
    
    def translate_with_vllm_grouped(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        model_name: str = "Qwen/Qwen3-235B-A22B-Instruct",
        group_size: int = 10
    ) -> List[str]:
        """Translate using VLLM with context grouping"""
        try:
            from vllm import LLM, SamplingParams

            # Load VLLM model
            if model_name not in self.models:
                # Eliberează memoria GPU înainte de a încărca noul model greu
                if torch.cuda.is_available():
                    logger.info("Cleaning up VRAM before loading VLLM...")
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    time.sleep(1)

                # Scurtcircuitare pentru calea exactă dacă utilizatorul a descărcat modelul
                # Căutăm orice director care începe cu numele modelului în Config.MODELS_DIR
                actual_model_to_load = model_name
                base_name = model_name.split('/')[-1]

                if Config.MODELS_DIR.exists():
                    # Căutare inteligentă: verificăm dacă există un folder care conține numele de bază
                    found_local = False
                    for item in Config.MODELS_DIR.iterdir():
                        if item.is_dir() and base_name in item.name:
                            actual_model_to_load = str(item.absolute())
                            logger.info(f"Found local model directory: {actual_model_to_load}")
                            found_local = True
                            break

                    if not found_local:
                        logger.info(f"Model {model_name} not found in {Config.MODELS_DIR}, it will be downloaded automatically.")

                logger.info(f"Loading VLLM model: {actual_model_to_load}")
                # We use pipeline-parallelism if multiple GPUs are available, otherwise 1
                # Llama 3.3 70B NVFP4 requires trust_remote_code=True and enough TP size
                self.models[model_name] = LLM(
                    model=actual_model_to_load,
                    trust_remote_code=True,
                    tensor_parallel_size=torch.cuda.device_count() or 1,
                    max_model_len=4096,
                    gpu_memory_utilization=Config.VLLM_GPU_MEMORY_UTILIZATION,
                    enforce_eager=Config.VLLM_ENFORCE_EAGER,
                    disable_log_stats=True,
                    download_dir=str(Config.MODELS_DIR)
                )

            llm = self.models[model_name]
            sampling_params = SamplingParams(
                temperature=0.3,
                top_p=0.95,
                max_tokens=2048,
                stop=["<|endoftext|>", "<|im_end|>"]
            )

            system_prompt = (
                "Ești un traducător profesionist expert în subtitrări. "
                f"Tradu textul primit din {source_lang} în {target_lang}. "
                "Cerințe CRUCIALE:\n"
                "1. Adaptează limbajul natural: metafore, nume, topică și expresii idiomatice în funcție de contextul conversației.\n"
                "2. Păstrează tonul și stilul vorbitorului.\n"
                "3. Returnează rezultatul EXCLUSIV ca un obiect JSON conținând o listă de string-uri, "
                "păstrând exact numărul și ordinea segmentelor primite.\n"
                "Exemplu format răspuns: {\"translations\": [\"text 1\", \"text 2\", ...]}"
            )

            all_translations = ["" for _ in range(len(texts))]

            # Group segments for context
            for i in range(0, len(texts), group_size):
                batch = texts[i:i + group_size]

                # Prepare prompt for the group
                user_content = "Vă rog să traduceți următoarele segmente de subtitrare consecutive:\n"
                for idx, text in enumerate(batch):
                    user_content += f"{idx + 1}. {text}\n"

                full_prompt = (
                    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n{user_content}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )

                outputs = llm.generate([full_prompt], sampling_params)
                response_text = outputs[0].outputs[0].text

                try:
                    # Parse JSON response
                    # Find JSON block in case model added fluff
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(0))
                        batch_translations = data.get('translations', [])

                        if len(batch_translations) == len(batch):
                            for j, t in enumerate(batch_translations):
                                all_translations[i + j] = t
                        else:
                            # Fallback if counts mismatch: try splitting by newline or something
                            logger.warning(f"Batch {i//group_size} translation count mismatch. Using heuristic fallback.")
                            # Very basic fallback: just split by lines if it looks like a list
                            lines = [l.strip() for l in response_text.split('\n') if l.strip() and not l.startswith('{')]
                            for j in range(min(len(lines), len(batch))):
                                all_translations[i + j] = lines[j]
                    else:
                        raise ValueError("No JSON found in VLLM response")

                except Exception as e:
                    logger.error(f"Error parsing VLLM response for batch {i}: {e}")
                    # Ultimate fallback to Google Translate for this batch if LLM fails
                    fallback_translations = self.translate_batch(batch, source_lang, target_lang)
                    for j, t in enumerate(fallback_translations):
                        all_translations[i + j] = t

            return all_translations

        except Exception as e:
            logger.error(f"VLLM translation fatal error: {e}")
            # Fallback to Google Translate for entire list
            return self.translate_batch(texts, source_lang, target_lang)

    def translate_with_llm(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        model_name: str = "google/gemma-2b-it",
        custom_prompt: Optional[str] = None
    ) -> List[str]:
        """Translate using LLM (Gemma or similar)"""
        # If user chooses 'llm' engine, they might want Qwen3 now.
        # But we keep this for smaller local models if needed.
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load model
            if model_name not in self.models:
                logger.info(f"Loading LLM: {model_name}")
                
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
                self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto"
                )
            
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            
            translations = []
            
            for text in texts:
                if not text or not text.strip():
                    translations.append("")
                    continue

                prompt = self.format_translation_prompt(
                    text, source_lang, target_lang, custom_prompt
                )
                
                inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                
                # Use safer sampling to prevent NaN/Inf errors on some architectures/quantizations
                # do_sample=False is the safest (greedy decoding)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3, # This function is for standard LLM
                    do_sample=False, # Switch to greedy to avoid probability tensor issues
                    # top_p=0.95 # Not needed for greedy
                )
                
                translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only the translation part
                if "Translation:" in translation:
                    translation = translation.split("Translation:")[-1].strip()
                
                translations.append(translation)
            
            return translations
            
        except Exception as e:
            logger.error(f"LLM translation error: {e}")
            raise
    
    def detect_language(self, text: str) -> str:
        """Simple language detection based on character sets"""
        text = text.lower().strip()
        
        # Romanian specific characters
        if re.search(r'[ăâîșț]', text):
            return 'ro'
        
        # Cyrillic characters
        if re.search(r'[а-яА-Я]', text):
            return 'ru'
        
        # Chinese characters
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'
        
        # Japanese characters
        if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            return 'ja'
        
        # Korean characters
        if re.search(r'[\uac00-\ud7af]', text):
            return 'ko'
        
        # Arabic characters
        if re.search(r'[\u0600-\u06ff]', text):
            return 'ar'
        
        # Thai characters
        if re.search(r'[\u0e00-\u0e7f]', text):
            return 'th'
        
        # Common English words
        common_en = {'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'}
        words = set(text.split()[:10])
        if words & common_en:
            return 'en'
        
        return 'en'  # Default to English
    
    def refine_segments_with_llm(self, prompt: str, model_name: str = "google/gemma-2b-it") -> Optional[List[Dict]]:
        """Refine and merge segments using LLM"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load model if not already loaded
            if model_name not in self.models:
                logger.info(f"Loading LLM for refinement: {model_name}")
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
                self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto"
                )

            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]

            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.2,
                    do_sample=True
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract JSON from response
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    logger.error("Failed to parse LLM JSON response")
                    return None

            return None

        except Exception as e:
            logger.error(f"LLM refinement error: {e}")
            return None

    def correct_with_romistral(
        self,
        texts: List[str],
        target_lang: str,
        group_size: int = 10
    ) -> List[str]:
        """Refine and correct Romanian translation using RoMistral"""
        if target_lang != 'ro':
            return texts

        try:
            from vllm import LLM, SamplingParams
            model_name = Config.ROMISTRAL_MODEL

            # Use VLLM for Romistral if possible, or Fallback to standard HF
            if model_name not in self.models:
                actual_model_to_load = model_name
                base_name = model_name.split('/')[-1]

                if Config.MODELS_DIR.exists():
                    for item in Config.MODELS_DIR.iterdir():
                        if item.is_dir() and base_name in item.name:
                            actual_model_to_load = str(item.absolute())
                            break

                logger.info(f"Loading RoMistral model: {actual_model_to_load}")
                self.models[model_name] = LLM(
                    model=actual_model_to_load,
                    trust_remote_code=True,
                    tensor_parallel_size=torch.cuda.device_count() or 1,
                    max_model_len=2048,
                    gpu_memory_utilization=Config.VLLM_GPU_MEMORY_UTILIZATION,
                    enforce_eager=Config.VLLM_ENFORCE_EAGER,
                    disable_log_stats=True
                )

            llm = self.models[model_name]
            sampling_params = SamplingParams(
                temperature=0.2,
                top_p=0.9,
                max_tokens=2048,
                stop=["<|endoftext|>", "</s>"]
            )

            system_prompt = (
                "Ești un expert în limba română specializat în corectarea și adaptarea subtitrărilor. "
                "Sarcina ta este să corectezi gramatical și să îmbunătățești logica de context a textelor primite. "
                "Cerințe:\n"
                "1. Corectează greșelile gramaticale și de punctuație.\n"
                "2. Asigură-te că topica frazei sună natural în limba română.\n"
                "3. Păstrează sensul original dar adaptează-l contextului dacă este necesar.\n"
                "4. Dacă întâlnești secvențe repetitive sau variante multiple ale aceluiași enunț, folosește contextul pentru a decide care este varianta corectă și elimină redundanțele.\n"
                "5. Returnează rezultatul EXCLUSIV ca un obiect JSON conținând o listă de string-uri sub cheia 'corrections'.\n"
                "Exemplu format răspuns: {\"corrections\": [\"text corectat 1\", \"text corectat 2\", ...]}"
            )

            all_corrections = [t for t in texts]

            for i in range(0, len(texts), group_size):
                batch = texts[i:i + group_size]
                user_content = "Corectează următoarele segmente de subtitrare:\n"
                for idx, text in enumerate(batch):
                    user_content += f"{idx + 1}. {text}\n"

                # RoMistral use Mistral format usually
                full_prompt = f"<s>[INST] {system_prompt}\n\n{user_content} [/INST]"

                outputs = llm.generate([full_prompt], sampling_params)
                response_text = outputs[0].outputs[0].text

                try:
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(0))
                        batch_corrections = data.get('corrections', [])
                        if len(batch_corrections) == len(batch):
                            for j, t in enumerate(batch_corrections):
                                all_corrections[i + j] = t
                except Exception as e:
                    logger.error(f"Error parsing RoMistral response for batch {i}: {e}")

            return all_corrections

        except Exception as e:
            logger.error(f"RoMistral correction fatal error: {e}")
            return texts

    def unload_models(self):
        """Free memory by unloading models"""
        self.models.clear()
        self.tokenizers.clear()
        if self.device == "cuda":
            torch.cuda.empty_cache()

# Helper function to test language code mapping
def test_language_codes():
    """Test if language codes are properly mapped"""
    translator = Translator()
    
    # Test some common language pairs
    test_pairs = [
        ('en', 'ro'),
        ('ro', 'en'),
        ('en', 'fr'),
        ('en', 'de'),
        ('ko', 'en'),
    ]
    
    for src, tgt in test_pairs:
        src_code = translator.lang_map.get(src, 'eng_Latn')
        tgt_code = translator.lang_map.get(tgt, 'eng_Latn')
        print(f"{src} → {tgt}: {src_code} → {tgt_code}")

if __name__ == "__main__":
    test_language_codes()