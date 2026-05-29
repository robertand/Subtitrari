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
from pathlib import Path
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
                src_lang=self.lang_map.get(source_lang, 'eng_Latn'),
                clean_up_tokenization_spaces=False
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
        batch_size: int = 15,
        context: Optional[str] = None
    ) -> List[str]:
        """Translate a batch of texts using Google Translate with batching and context injection"""
        try:
            logger.info(f"Translating {len(texts)} segments using Google Translate: {source_lang} -> {target_lang}")
            
            s_lang = source_lang if source_lang != 'auto' else 'auto'
            t_lang = target_lang
            google_translator = GoogleTranslator(source=s_lang, target=t_lang)

            separator = " ||| "
            all_translations = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                # Context injection
                if context:
                    # Prepend context to the first segment of each batch to guide the engine
                    # but ensure it's removed from output
                    batch_to_send = [f"[Context: {context}] {batch[0]}"] + batch[1:]
                else:
                    batch_to_send = batch

                batch_text = separator.join(batch_to_send)

                try:
                    translated_batch_text = google_translator.translate(batch_text)
                    translated_segments = [s.strip() for s in translated_batch_text.split(separator)]

                    # Cleanup injected context from first segment if present
                    if context and translated_segments:
                        translated_segments[0] = re.sub(r'^\[Context:.*?\]\s*', '', translated_segments[0], flags=re.IGNORECASE)

                    if len(translated_segments) == len(batch):
                        all_translations.extend(translated_segments)
                    else:
                        logger.warning(f"Google Batch mismatch at {i}: sent {len(batch)}, got {len(translated_segments)}. Falling back to individual.")
                        individual = google_translator.translate_batch(batch)
                        all_translations.extend(individual)
                except Exception as e:
                    logger.error(f"Google Batch error at {i}: {e}. Falling back to individual.")
                    individual = google_translator.translate_batch(batch)
                    all_translations.extend(individual)

                # Rate limiting
                time.sleep(0.5)

            return all_translations

        except Exception as e:
            logger.error(f"Google Translate fatal error: {e}")
            # Fallback to NLLB if Google fails
            return self._nllb_translate(texts, source_lang, target_lang, 8)

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
    
    def ensure_model_downloaded(self, model_id: str, cache_dir: Optional[Path] = None) -> str:
        """Explicitly ensure a Hugging Face model is downloaded for vLLM"""
        try:
            from huggingface_hub import snapshot_download
            logger.info(f"Checking/Downloading vLLM model: {model_id}")

            # Using Config.MODELS_DIR as default
            target_dir = cache_dir or Config.MODELS_DIR

            # Special handling for Llama-3.3-8B-Instruct as requested
            if model_id == "allura-forge/Llama-3.3-8B-Instruct":
                local_dir = target_dir / "Llama-3.3-8B-Instruct"
                path = snapshot_download(
                    repo_id=model_id,
                    local_dir=str(local_dir),
                    ignore_patterns=[".pt", ".bin"]  # only safetensors
                )
                return path

            path = snapshot_download(
                repo_id=model_id,
                cache_dir=str(target_dir)
            )
            return path
        except Exception as e:
            logger.error(f"Error downloading vLLM model {model_id}: {e}")
            return model_id

    def translate_with_vllm_grouped(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        model_name: str = "Qwen/Qwen3-235B-A22B-Instruct",
        group_size: int = 10,
        metadata: List[Dict[str, Any]] = None,
        context: Optional[str] = None
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

                # Ensure the model is downloaded locally
                actual_model_to_load = self.ensure_model_downloaded(model_name)

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
                    disable_log_stats=True
                )

            llm = self.models[model_name]

            # Adjust stop tokens based on model
            stop_tokens = ["<|endoftext|>", "<|im_end|>"]
            if "Llama-3" in model_name:
                stop_tokens = ["<|endoftext|>", "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"]

            sampling_params = SamplingParams(
                temperature=0.01,  # Extremely low for max precision
                top_p=0.95,
                max_tokens=2048,
                stop=stop_tokens
            )

            system_prompt = (
                "Ești un traducător profesionist expert în subtitrări. "
                f"Tradu textul primit din {source_lang} în {target_lang}. "
                "Cerințe CRUCIALE:\n"
                "1. Adaptează limbajul natural: metafore, nume, topică și expresii idiomatice în funcție de contextul conversației.\n"
                "2. Păstrează tonul și stilul vorbitorului.\n"
                "3. ACORD GRAMATICAL: Folosește metadatele de gen furnizate (male/female) pentru a face acordul corect al adjectivelor și verbelor în limba țintă (ex: în Română, 'obosit' vs 'obosită').\n"
                "4. NU oferi explicații, note, comentarii, observații, paranteze sau text adițional. DOAR traducerea pură.\n"
                "5. Returnează rezultatul EXCLUSIV ca un obiect JSON valid sub cheia 'translations'.\n"
                "6. Păstrează exact numărul și ordinea segmentelor.\n"
            )

            if context:
                system_prompt += f"CONTEXT CONȚINUT: {context}\n"

            system_prompt += "Exemplu format răspuns: {\"translations\": [\"Traducere 1\", \"Traducere 2\"]}"

            all_translations = ["" for _ in range(len(texts))]

            # Group segments for context
            for i in range(0, len(texts), group_size):
                batch = texts[i:i + group_size]
                batch_meta = metadata[i:i + group_size] if metadata else None

                # Prepare prompt for the group
                user_content = "Vă rog să traduceți următoarele segmente de subtitrare consecutive:\n"
                for idx, text in enumerate(batch):
                    meta_str = ""
                    if batch_meta and batch_meta[idx]:
                        m = batch_meta[idx]
                        # Put meta as a separate label to avoid inclusion in text
                        gender = m.get('gender', 'unknown')
                        speaker = m.get('speaker', 'unknown')
                        meta_str = f" [Gen: {gender}, Speaker: {speaker}]"

                    user_content += f"{idx + 1}.{meta_str} {text}\n"

                if "Llama-3" in model_name:
                    # Llama 3 Prompt Template
                    full_prompt = (
                        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
                        f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|>"
                        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                    )
                else:
                    # Default ChatML (Qwen, etc.)
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
                                all_translations[i + j] = self._clean_llm_segment(t)
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
                
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(
                    model_name,
                    clean_up_tokenization_spaces=False
                )
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
                    max_length=None,  # Explicitly remove max_length to avoid conflict warning
                    temperature=0.01,
                    do_sample=True, # Use sampling with low temp to avoid 'temperature flags' warning with greedy
                )
                
                # Decode only new tokens to avoid prompt repetition
                translation = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
                
                # Clean and extract only the translation part
                translation = self._clean_llm_segment(translation)
                
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

    def correct_with_vllm(
        self,
        texts: List[str],
        target_lang: str,
        model_name: str = "OpenLLM-Ro/RoMistral-7b-Instruct",
        group_size: int = 10,
        metadata: List[Dict[str, Any]] = None,
        context: Optional[str] = None
    ) -> List[str]:
        """Refine and correct translation using a VLLM model (RoMistral or Llama)"""
        try:
            from vllm import LLM, SamplingParams

            # Use VLLM for correction
            if model_name not in self.models:
                # Eliberează memoria GPU înainte de a încărca noul model greu
                if torch.cuda.is_available():
                    logger.info("Cleaning up VRAM before loading VLLM for correction...")
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    time.sleep(1)

                actual_model_to_load = self.ensure_model_downloaded(model_name)

                # Special check for local folders if not found by snapshot_download (e.g. RoMistral manual folder)
                if not Path(actual_model_to_load).exists():
                    base_name = model_name.split('/')[-1]
                    if Config.MODELS_DIR.exists():
                        for item in Config.MODELS_DIR.iterdir():
                            if item.is_dir() and base_name in item.name:
                                actual_model_to_load = str(item.absolute())
                                break

                logger.info(f"Loading VLLM model for correction: {actual_model_to_load}")
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

            # Adjust stop tokens based on model
            stop_tokens = ["<|endoftext|>", "</s>", "<|im_end|>"]
            if "Llama-3" in model_name:
                stop_tokens = ["<|endoftext|>", "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"]

            sampling_params = SamplingParams(
                temperature=0.01,
                top_p=0.9,
                max_tokens=2048,
                stop=stop_tokens
            )

            system_prompt = (
                f"Ești un expert în limba {target_lang} specializat în corectarea și adaptarea subtitrărilor. "
                "Sarcina ta este să corectezi gramatical și să îmbunătățești logica de context a textelor primite. "
                "Cerințe:\n"
                f"1. Corectează greșelile gramaticale și de punctuație în limba {target_lang}.\n"
                f"2. Asigură-te că topica frazei sună natural în limba {target_lang}.\n"
                "3. ACORD GRAMATICAL: Folosește metadatele de gen furnizate pentru a face acordul corect în limba română (ex: 'obosit' pentru male, 'obosita' pentru female).\n"
                "4. Păstrează sensul original dar adaptează-l contextului dacă este necesar.\n"
                "5. Dacă întâlnești secvențe repetitive sau variante multiple ale aceluiași enunț, folosește contextul pentru a decide care este varianta corectă și elimină redundanțele.\n"
                "6. NU oferi explicații, note, comentarii sau paranteze. DOAR textul corectat.\n"
                "7. Returnează rezultatul EXCLUSIV ca un obiect JSON valid sub cheia 'corrections'.\n"
            )

            if context:
                system_prompt += f"CONTEXT CONȚINUT: {context}\n"

            system_prompt += "Exemplu format răspuns: {\"corrections\": [\"Corecție 1\", \"Corecție 2\"]}"

            all_corrections = [t for t in texts]

            for i in range(0, len(texts), group_size):
                batch = texts[i:i + group_size]
                batch_meta = metadata[i:i + group_size] if metadata else None

                user_content = "Corectează următoarele segmente de subtitrare:\n"
                for idx, text in enumerate(batch):
                    meta_str = ""
                    if batch_meta and batch_meta[idx]:
                        m = batch_meta[idx]
                        meta_str = f" [Gen: {m.get('gender', 'unknown')}]"

                    user_content += f"{idx + 1}.{meta_str} {text}\n"

                # Prompt Template Selection
                if "Llama-3" in model_name:
                    full_prompt = (
                        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
                        f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|>"
                        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                    )
                elif "RoMistral" in model_name or "Mistral" in model_name:
                    full_prompt = f"<s>[INST] {system_prompt}\n\n{user_content} [/INST]"
                else:
                    full_prompt = (
                        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                        f"<|im_start|>user\n{user_content}<|im_end|>\n"
                        f"<|im_start|>assistant\n"
                    )

                outputs = llm.generate([full_prompt], sampling_params)
                response_text = outputs[0].outputs[0].text

                try:
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(0))
                        batch_corrections = data.get('corrections', [])
                        if len(batch_corrections) == len(batch):
                            for j, t in enumerate(batch_corrections):
                                all_corrections[i + j] = self._clean_llm_segment(t)
                except Exception as e:
                    logger.error(f"Error parsing VLLM response for batch {i}: {e}")

            return all_corrections

        except Exception as e:
            logger.error(f"VLLM correction fatal error: {e}")
            return texts

    def correct_with_romistral(self, texts: List[str], target_lang: str, group_size: int = 10) -> List[str]:
        """Backward compatibility for RoMistral correction"""
        return self.correct_with_vllm(texts, target_lang, Config.ROMISTRAL_MODEL, group_size)

    def _clean_speaker_none(self, text: str) -> str:
        """Fix artifacts like 'Vorbește un personaj: None' or 'Speaker: None'"""
        if not text:
            return ""
        # Remove any prefix that ends with ': None' or just 'None' at start
        text = re.sub(r'^.*?:\s*None\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^None[:\s]+', '', text, flags=re.IGNORECASE)
        return text.strip()

    def _clean_llm_segment(self, text: str) -> str:
        """Strip common LLM hallucinations and artifacts from a single segment"""
        if not text:
            return ""

        # Clean None artifacts first
        text = self._clean_speaker_none(text)

        # Strip prefixes like "Traducere:", "Nota:", etc.
        patterns = [
            r'^(Traducere|Translation|Nota|Note|Rezultat|Result|Corectie|Correction|Explicație|Explanation|Răspuns|Response|Traducerea finală|Final translation|Observații|Observație):\s*',
            r'^Segment\s*\d+:\s*',
            r'^Vorbește un personaj:\s*',
            r'^\[Gen:.*?, Speaker:.*\]\s*',
            r'^\[Gen:.*?\]\s*',
            r'^\[Vorbește un personaj:.*?\]\s*',
        ]

        # Also handle metadata injected within the text
        inline_patterns = [
            r'\[Gen:.*?, Speaker:.*\]',
            r'\[Gen:.*?\]',
            r'\[Vorbește un personaj:.*?\]',
        ]

        cleaned = text.strip()
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        for pattern in inline_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()

        # Handle "Vorbește un personaj:" anywhere in string if it's not a prefix
        cleaned = re.sub(r'Vorbește un personaj:', '', cleaned, flags=re.IGNORECASE).strip()

        # Strip common trailing artifacts
        # Improved splitting to handle multi-line explanations and repetitive filler
        cleaned = re.split(r'\n+(Nota|Note|Explicație|Explanation|Comentariu|Comment|Observații|Observație):', cleaned, flags=re.IGNORECASE)[0]
        cleaned = re.split(r'\s+(Nota|Note|Explicație|Explanation|Comentariu|Comment|Observații|Observație):', cleaned, flags=re.IGNORECASE)[0]

        # If the model starts repeating "Dacă aveți nevoie de alte traduceri..."
        cleaned = re.split(r'Dacă aveți nevoie de alte', cleaned, flags=re.IGNORECASE)[0]

        # Remove bullet points if the model starts explaining words
        if "\n-" in cleaned or "\n*" in cleaned:
             cleaned = cleaned.split("\n-")[0].split("\n*")[0]

        return cleaned.strip()

    def translate_with_api(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        api_type: str,
        api_key: str,
        model: str,
        context: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_size: int = 15,
        content_type: str = "film"
    ) -> List[str]:
        """Translate using LLM APIs (Claude, OpenAI, Custom)"""
        logger.info(f"Translating {len(texts)} segments using {api_type} API ({model})")

        all_translations = ["" for _ in range(len(texts))]

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            retry_count = 0
            success = False

            while retry_count < 2 and not success:
                try:
                    translated_batch = self._call_llm_api(
                        batch, source_lang, target_lang, api_type, api_key, model, context, base_url, content_type,
                        force_numbered=(retry_count > 0)
                    )

                    if len(translated_batch) == len(batch):
                        for j, t in enumerate(translated_batch):
                            all_translations[i + j] = t
                        success = True
                    else:
                        logger.warning(f"Batch {i//batch_size} mismatch: sent {len(batch)}, got {len(translated_batch)}. Retry {retry_count+1}")
                        retry_count += 1
                except Exception as e:
                    logger.error(f"API call error: {e}")
                    retry_count += 1

            if not success:
                logger.error(f"Batch {i//batch_size} failed after retries. Falling back to individual Google Translate.")
                fallback = self.translate_batch(batch, source_lang, target_lang, batch_size=1)
                for j, t in enumerate(fallback):
                    all_translations[i + j] = t

            # Rate limiting
            time.sleep(0.5)

        return all_translations

    def _call_llm_api(
        self,
        batch: List[str],
        source_lang: str,
        target_lang: str,
        api_type: str,
        api_key: str,
        model: str,
        context: Optional[str] = None,
        base_url: Optional[str] = None,
        content_type: str = "film",
        force_numbered: bool = False
    ) -> List[str]:
        """Internal helper for LLM API calls"""
        n = len(batch)
        system_prompt = (
            f"You are a professional subtitle translator. Translate the following subtitle segments into {target_lang}.\n"
            "Rules:\n"
            f"- Preserve the meaning, tone and style of the original dialogue\n"
            f"- Keep translations natural and idiomatic in {target_lang}, not literal\n"
            "- Each line in the input corresponds to one subtitle segment. Return ONLY the translated lines, one per line, in the same order.\n"
            "- Do not add explanations, notes or extra text.\n"
            "- If a segment is a sound description like [music] or [applause], translate or keep it appropriately.\n"
            f"- The source language is {source_lang}. This is a {content_type}.\n"
        )

        if context:
            system_prompt += f"- Context for content: {context}\n"

        system_prompt += f"- Number of input lines: {n}. Return exactly {n} lines."

        if force_numbered:
            system_prompt += "\n- RETURN THE LINES NUMBERED (1: ..., 2: ...) to ensure correct mapping."

        user_message = f"Translate these {n} subtitle segments:\n"
        for idx, text in enumerate(batch):
            user_message += f"{idx + 1}: {text}\n"

        response_text = ""

        if api_type == "claude":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model=model or "claude-3-5-sonnet-20240620",
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}]
            )
            response_text = message.content[0].text
        else: # openai or custom
            import openai
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model or "gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3
            )
            response_text = response.choices[0].message.content

        # Parsing
        lines = response_text.strip().split('\n')
        translated = []

        if force_numbered or any(re.match(r'^\d+[:.]\s*', l) for l in lines[:3]):
            # Extract text from numbered lines
            for l in lines:
                match = re.match(r'^\d+[:.]\s*(.*)', l)
                if match:
                    translated.append(match.group(1).strip())
        else:
            # Just take non-empty lines
            translated = [l.strip() for l in lines if l.strip()]

        return translated

    def validate_and_retry_translations(
        self,
        original_texts: List[str],
        translated_texts: List[str],
        source_lang: str,
        target_lang: str,
        method_name: str
    ) -> List[str]:
        """Validate language of translated segments and retry if they seem untranslated"""
        if len(original_texts) != len(translated_texts):
             return translated_texts

        final_translations = list(translated_texts)

        try:
            from langdetect import detect_langs
        except ImportError:
            # Fallback to simple identity check if langdetect not available
            for i in range(len(final_translations)):
                if final_translations[i] == original_texts[i] and len(original_texts[i]) > 3:
                    logger.info(f"Retrying segment {i} due to identity with source")
                    retry = self.translate_batch([original_texts[i]], source_lang, target_lang, batch_size=1)
                    final_translations[i] = retry[0]
            return final_translations

        for i in range(len(final_translations)):
            orig = original_texts[i]
            trans = final_translations[i]

            if not trans or len(trans) < 3:
                continue

            # Skip short segments or sounds like [music]
            if re.match(r'^\[.*\]$', trans):
                continue

            try:
                # Basic identity check first
                if trans == orig:
                    is_untranslated = True
                else:
                    # Langdetect check
                    detected = detect_langs(trans)
                    # If target_lang (e.g. 'ro') is not in top detected with reasonable prob
                    is_untranslated = True
                    for d in detected:
                        if d.lang == target_lang or (target_lang == 'en' and d.lang in ['en', 'ca']): # en fallback
                            is_untranslated = False
                            break

                if is_untranslated:
                    logger.info(f"[{method_name}] Segment {i} seems untranslated (detected: {[d.lang for d in detected]}). Retrying...")
                    # Max 2 retries handled by loop if we wanted, but let's do 1 explicit here
                    retry = self.translate_batch([orig], source_lang, target_lang, batch_size=1)
                    final_translations[i] = retry[0]
            except Exception as e:
                logger.debug(f"Langdetect failed for segment {i}: {e}")

        return final_translations

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