from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, NllbTokenizer
import torch
import numpy as np
import json
from typing import List, Dict, Optional, Any
import logging
import re

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
        """Translate a batch of texts"""
        try:
            # Check if bridging is needed
            if source_lang in self.bridge_languages and target_lang != 'en':
                return self._bridge_translate(texts, source_lang, target_lang, batch_size)
            
            model, tokenizer = self.load_model(source_lang, target_lang)
            
            # Get language codes
            src_code = self.lang_map.get(source_lang, 'eng_Latn')
            tgt_code = self.lang_map.get(target_lang, 'eng_Latn')
            
            translations = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Set source language for tokenizer
                if hasattr(tokenizer, 'src_lang'):
                    tokenizer.src_lang = src_code
                
                # Tokenize with explicit language code
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Get target language token ID
                # Try different methods to get the forced bos token id
                forced_bos_token_id = None
                
                # Method 1: Try lang_code_to_id
                if hasattr(tokenizer, 'lang_code_to_id'):
                    forced_bos_token_id = tokenizer.lang_code_to_id.get(tgt_code)
                
                # Method 2: Try convert_tokens_to_ids with language code
                if forced_bos_token_id is None and hasattr(tokenizer, 'convert_tokens_to_ids'):
                    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)
                
                # Method 3: Use tokenizer's internal language mapping
                if forced_bos_token_id is None:
                    # Get it from the tokenizer's additional_special_tokens
                    try:
                        forced_bos_token_id = tokenizer.get_lang_id(tgt_code)
                    except:
                        # Last resort: encode the language token directly
                        lang_token = tokenizer.encode(tgt_code, add_special_tokens=False)
                        if lang_token:
                            forced_bos_token_id = lang_token[0]
                
                # Generate translation
                with torch.no_grad():
                    translated = model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_token_id,
                        max_length=512,
                        num_beams=5,
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        length_penalty=0.8
                    )
                
                # Decode
                decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
                translations.extend(decoded)
                
                # Clear memory
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            
            return translations
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            # Fallback: try with a simpler approach
            return self._simple_translate(texts, source_lang, target_lang)
    
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
    
    def translate_with_llm(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        model_name: str = "google/gemma-2b-it",
        custom_prompt: Optional[str] = None
    ) -> List[str]:
        """Translate using LLM (Gemma or similar)"""
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
                prompt = self.format_translation_prompt(
                    text, source_lang, target_lang, custom_prompt
                )
                
                inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.95
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