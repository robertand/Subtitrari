import re
import json
from typing import List, Dict, Any, Optional
import numpy as np
import librosa

class SubtitleSegmenter:
    def __init__(self):
        pass
    
    def segment_by_time(
        self,
        segments: List[Dict],
        min_duration: float = 1.0,
        max_duration: float = 5.0,
        max_chars: int = 80,
        overlap: float = 0.5
    ) -> List[Dict]:
        """Segment subtitles by time constraints with optional overlap"""
        result = []
        
        for segment in segments:
            text = segment.get('text', '').strip()
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            duration = end - start
            
            if not text:
                continue
            
            # If segment is too short, try to merge with next (preserving overlap if possible)
            if duration < min_duration:
                if result:
                    # Merge with previous
                    prev = result[-1]
                    # If they already overlap significantly, just append text
                    if start < prev['end']:
                        prev['text'] += ' ' + text
                        prev['end'] = max(prev['end'], end)
                    else:
                        prev['text'] += ' ' + text
                        prev['end'] = end
                else:
                    result.append({
                        'text': text,
                        'start': start,
                        'end': end + overlap
                    })
                continue
            
            # If segment is too long, split it
            if duration > max_duration or len(text) > max_chars:
                sub_segments = self._split_segment(
                    text, start, end, max_duration, max_chars, overlap
                )
                result.extend(sub_segments)
            else:
                result.append({
                    'text': text,
                    'start': start,
                    'end': end + overlap
                })
        
        return result
    
    def segment_by_pauses(
        self,
        audio_path: str,
        segments: List[Dict],
        min_pause_duration: float = 1.0,
        max_duration: float = 5.0,
        max_chars: int = 80,
        overlap: float = 0.5
    ) -> List[Dict]:
        """Segment using Voice Activity Detection based on pauses with overlap"""
        try:
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Voice activity detection using energy
            frame_length = 2048
            hop_length = 512
            
            rms = librosa.feature.rms(
                y=audio, 
                frame_length=frame_length, 
                hop_length=hop_length
            )[0]
            
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            
            # Detect silence/pauses
            silence_threshold = -40  # dB
            is_speech = rms_db > silence_threshold
            
            # Convert frames to time
            times = librosa.frames_to_time(
                np.arange(len(rms_db)), 
                sr=sr, 
                hop_length=hop_length
            )
            
            # Find pauses in segments
            result = []
            for segment in segments:
                start = segment['start']
                end = segment['end']
                text = segment.get('text', '').strip()
                
                if not text:
                    continue
                
                # Check for pauses within segment
                mask = (times >= start) & (times <= end)
                speech_frames = is_speech[mask]
                
                if len(speech_frames) > 0:
                    speech_ratio = np.sum(speech_frames) / len(speech_frames)
                    
                    # If there are significant pauses, split segment
                    if speech_ratio < 0.6 and (end - start) > max_duration:
                        splits = self._split_by_pauses(
                            text, start, end, times[mask], speech_frames, overlap
                        )
                        result.extend(splits)
                    else:
                        result.append({'text': text, 'start': start, 'end': end + overlap})
                else:
                    result.append({'text': text, 'start': start, 'end': end + overlap})
            
            return result
            
        except Exception as e:
            # Fallback to time-based segmentation
            return self.segment_by_time(segments, min_duration=1.0, max_duration=max_duration, max_chars=max_chars, overlap=overlap)
    
    def _split_segment(
        self,
        text: str,
        start: float,
        end: float,
        max_duration: float,
        max_chars: int,
        overlap: float = 0.5
    ) -> List[Dict]:
        """Split a long segment into smaller ones with overlap"""
        words = text.split()
        if not words:
            return [{'text': text, 'start': start, 'end': end}]
        
        # Calculate approximate duration per word
        total_duration = end - start
        words_per_second = len(words) / total_duration if total_duration > 0 else 2
        
        segments = []
        current_words = []
        current_start = start
        word_index = 0
        
        while word_index < len(words):
            current_words.append(words[word_index])
            current_text = ' '.join(current_words)
            
            # Estimate current duration
            estimated_duration = len(current_words) / words_per_second
            
            if (estimated_duration >= max_duration or len(current_text) >= max_chars) and len(current_words) > 1:
                # Save current segment and start new one
                current_text = ' '.join(current_words[:-1])
                current_end = current_start + (len(current_words) - 1) / words_per_second
                
                segments.append({
                    'text': current_text,
                    'start': round(current_start, 3),
                    'end': round(current_end + overlap, 3)
                })
                
                # Start new segment
                current_words = [words[word_index]]
                current_start = current_end
            
            word_index += 1
        
        # Add remaining words
        if current_words:
            current_text = ' '.join(current_words)
            current_end = end
            segments.append({
                'text': current_text,
                'start': round(current_start, 3),
                'end': round(current_end, 3)
            })
        
        return segments
    
    def _split_by_pauses(
        self,
        text: str,
        start: float,
        end: float,
        times: np.ndarray,
        speech_frames: np.ndarray,
        overlap: float = 0.5
    ) -> List[Dict]:
        """Split segment based on detected pauses with overlap"""
        # Find silence regions
        silence_regions = []
        in_silence = False
        silence_start = 0
        
        for i, is_speech in enumerate(speech_frames):
            if not is_speech and not in_silence:
                in_silence = True
                silence_start = times[i]
            elif is_speech and in_silence:
                in_silence = False
                if times[i] - silence_start > 0.5:  # Minimum pause duration
                    silence_regions.append((silence_start, times[i]))
        
        if not silence_regions:
            return [{'text': text, 'start': start, 'end': end}]
        
        # Split text based on silence regions
        words = text.split()
        segments = []
        current_start = start
        
        for pause_start, pause_end in silence_regions:
            # Split text proportionally
            ratio = (pause_start - start) / (end - start)
            split_point = int(len(words) * ratio)
            
            if split_point > 0 and split_point < len(words):
                part1 = ' '.join(words[:split_point])
                segments.append({
                    'text': part1,
                    'start': round(current_start, 3),
                    'end': round(pause_start + overlap, 3)
                })
                
                words = words[split_point:]
                current_start = pause_end
        
        # Add remaining text
        if words:
            segments.append({
                'text': ' '.join(words),
                'start': round(current_start, 3),
                'end': round(end + overlap, 3)
            })
        
        return segments or [{'text': text, 'start': start, 'end': end}]
    
    def split_text_for_subtitle(self, text: str, max_chars_per_line: int = 40) -> str:
        """Split text into multiple lines for subtitle display"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + (1 if current_line else 0) > max_chars_per_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += len(word) + (1 if current_line else 0)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines[:2])  # Maximum 2 lines
    
    def merge_segments_similarity(self, segments: List[Dict], threshold: float = 0.6) -> List[Dict]:
        """Merge overlapping segments if word similarity exceeds threshold"""
        if not segments:
            return []

        merged = []
        i = 0
        while i < len(segments):
            current = segments[i].copy()
            j = i + 1

            while j < len(segments):
                next_seg = segments[j]

                # Check for overlap
                overlap_start = max(current['start'], next_seg['start'])
                overlap_end = min(current['end'], next_seg['end'])

                if overlap_end > overlap_start:
                    # Calculate similarity for the overlapping portion (simplified)
                    words1 = set(re.findall(r'\w+', current['text'].lower()))
                    words2 = set(re.findall(r'\w+', next_seg['text'].lower()))

                    if not words1 or not words2:
                        j += 1
                        continue

                    common = words1.intersection(words2)
                    similarity = len(common) / max(len(words1), len(words2))

                    if similarity >= threshold:
                        # Merge segments: keep the longer one or combine
                        if len(current['text']) >= len(next_seg['text']):
                            current['end'] = max(current['end'], next_seg['end'])
                        else:
                            current['text'] = next_seg['text']
                            current['start'] = min(current['start'], next_seg['start'])
                            current['end'] = max(current['end'], next_seg['end'])
                        j += 1
                    else:
                        break
                else:
                    break

            merged.append(current)
            i = j

        return merged

    def merge_segments_llm(self, segments: List[Dict], translator_obj: Any) -> List[Dict]:
        """Use LLM to refine and merge segments based on logic and context"""
        if not segments or not translator_obj:
            return segments

        # Group overlapping segments for the LLM
        groups = []
        i = 0
        while i < len(segments):
            group = [segments[i]]
            j = i + 1
            while j < len(segments):
                # If it overlaps with any segment in the current group
                overlaps = False
                for seg in group:
                    if min(seg['end'], segments[j]['end']) > max(seg['start'], segments[j]['start']):
                        overlaps = True
                        break
                if overlaps:
                    group.append(segments[j])
                    j += 1
                else:
                    break
            groups.append(group)
            i = j

        refined_segments = []
        for group in groups:
            if len(group) == 1:
                refined_segments.append(group[0])
                continue

            # Prepare prompt for LLM
            context_text = "\n".join([f"[{seg['start']}-{seg['end']}] {seg['text']}" for seg in group])

            prompt = "Următoarele segmente de subtitrare se suprapun. Te rog să deduci după logica textului și context ce rămâne și ce arunci la gunoi, retranscriind totul într-un flux coerent, păstrând timpii de început și sfârșit corespunzători segmentelor rezultate. Returnează doar segmentele în format JSON: [{\"start\": float, \"end\": float, \"text\": string}, ...]\n\n"
            prompt += context_text

            try:
                # This will be called in translator.py or app.py but here we define how we use it
                result = translator_obj.refine_segments_with_llm(prompt)
                if result:
                    refined_segments.extend(result)
                else:
                    refined_segments.extend(group) # Fallback
            except Exception:
                refined_segments.extend(group)

        return refined_segments

    def convert_diacritics(self, text: str, to_legacy: bool = True) -> str:
        """Convert between modern and legacy diacritics"""
        if to_legacy:
            # Modern → Legacy
            replacements = {
                'ș': 'ş',
                'Ș': 'Ş',
                'ț': 'ţ',
                'Ț': 'Ţ',
                'ă': 'ă',  # Keep as is
                'â': 'â',  # Keep as is
                'î': 'î'   # Keep as is
            }
        else:
            # Legacy → Modern
            replacements = {
                'ş': 'ș',
                'Ş': 'Ș',
                'ţ': 'ț',
                'Ţ': 'Ț'
            }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text