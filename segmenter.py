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
        overlap: float = 0.0
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
    
    
    def _split_segment(
        self,
        text: str,
        start: float,
        end: float,
        max_duration: float,
        max_chars: int,
        overlap: float = 0.0
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
        overlap: float = 0.0
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
    
    def merge_identical_overlapping(self, segments: List[Dict]) -> List[Dict]:
        """Merge segments with identical text that overlap into a single segment"""
        if not segments:
            return []

        # Sort by start time
        sorted_segs = sorted(segments, key=lambda x: x['start'])
        result = []

        i = 0
        while i < len(sorted_segs):
            current = sorted_segs[i].copy()
            curr_text_norm = re.sub(r'[^\w\s]', '', current['text'].lower()).strip()

            if not curr_text_norm:
                i += 1
                continue

            j = i + 1
            while j < len(sorted_segs):
                next_seg = sorted_segs[j]
                next_text_norm = re.sub(r'[^\w\s]', '', next_seg['text'].lower()).strip()

                # Check for overlap
                overlap = min(current['end'], next_seg['end']) - max(current['start'], next_seg['start'])

                if overlap > 0 and curr_text_norm == next_text_norm:
                    # Merge: extend current and skip next
                    current['start'] = min(current['start'], next_seg['start'])
                    current['end'] = max(current['end'], next_seg['end'])
                    j += 1
                else:
                    # If they don't overlap, we can stop searching for this specific identical merge
                    # because the list is sorted by start time.
                    if next_seg['start'] >= current['end']:
                        break
                    j += 1

            result.append(current)
            i = j

        return result

    def merge_segments_similarity(self, segments: List[Dict], threshold: float = 0.6) -> List[Dict]:
        """Merge overlapping segments if word similarity exceeds threshold"""
        if not segments:
            return []

        # First, handle strictly identical overlaps which are common hallucinations
        segments = self.merge_identical_overlapping(segments)

        merged = []
        i = 0
        # Sort for more predictable merging
        sorted_segs = sorted(segments, key=lambda x: x['start'])

        while i < len(sorted_segs):
            current = sorted_segs[i].copy()
            j = i + 1

            while j < len(sorted_segs):
                next_seg = sorted_segs[j]

                # Check for overlap
                overlap_start = max(current['start'], next_seg['start'])
                overlap_end = min(current['end'], next_seg['end'])

                if overlap_end > overlap_start:
                    # Calculate similarity for the overlapping portion
                    words1 = set(re.findall(r'\w+', current['text'].lower()))
                    words2 = set(re.findall(r'\w+', next_seg['text'].lower()))

                    if not words1 or not words2:
                        j += 1
                        continue

                    common = words1.intersection(words2)
                    similarity = len(common) / max(len(words1), len(words2)) if words1 or words2 else 0

                    if similarity >= threshold:
                        # Merge segments: keep the longer one or combine
                        if len(current['text']) >= len(next_seg['text']):
                            current['end'] = max(current['end'], next_seg['end'])
                            current['start'] = min(current['start'], next_seg['start'])
                        else:
                            current['text'] = next_seg['text']
                            current['start'] = min(current['start'], next_seg['start'])
                            current['end'] = max(current['end'], next_seg['end'])
                        j += 1
                    else:
                        j += 1
                else:
                    # If they don't overlap and next start is after current end, stop searching
                    if next_seg['start'] >= current['end']:
                        break
                    j += 1

            merged.append(current)
            i = j

        return merged

    def remove_repetitions(self, segments: List[Dict]) -> List[Dict]:
        """Remove consecutive identical phrases while keeping segments with background voice if text is same"""
        if not segments:
            return []

        # First, ensure we don't have identical overlapping segments
        segments = self.merge_identical_overlapping(segments)

        # Sort by start time first to ensure consecutiveness
        sorted_segments = sorted(segments, key=lambda x: x['start'])

        result = [sorted_segments[0].copy()]
        for i in range(1, len(sorted_segments)):
            curr = sorted_segments[i].copy()
            prev = result[-1]

            # Normalize text for comparison
            curr_text = re.sub(r'[^\w\s]', '', curr['text'].lower()).strip()
            prev_text = re.sub(r'[^\w\s]', '', prev['text'].lower()).strip()

            if not curr_text:
                continue

            # Check for exact matches or high similarity with significant overlap or small gap
            gap = curr['start'] - prev['end']

            if curr_text == prev_text and curr_text != "":
                if gap < 2.0: # If identical and close together
                    prev['end'] = max(prev['end'], curr['end'])
                    continue

            # Fuzzy match for near-repetitions (often caused by windowing artifacts)
            if len(curr_text) > 0 and len(prev_text) > 0:
                words1 = set(prev_text.split())
                words2 = set(curr_text.split())
                if words1 and words2:
                    common = words1.intersection(words2)
                    similarity = len(common) / max(len(words1), len(words2))
                    if similarity > 0.8 and gap < 1.0:
                        # High similarity and small gap: likely a repetition artifact
                        prev['end'] = max(prev['end'], curr['end'])
                        if len(curr['text']) > len(prev['text']):
                            prev['text'] = curr['text']
                        continue

            result.append(curr)
        return result

    def ensure_sequential(self, segments: List[Dict]) -> List[Dict]:
        """Ensure segments do not overlap: next starts exactly after previous ends"""
        if not segments:
            return []

        # Sort by start time
        sorted_segments = sorted(segments, key=lambda x: x['start'])

        result = [sorted_segments[0].copy()]
        for i in range(1, len(sorted_segments)):
            curr = sorted_segments[i].copy()
            prev = result[-1]

            if curr['start'] < prev['end']:
                # If the overlap is large, it might be a redundant segment
                if curr['end'] <= prev['end']:
                    continue # Discard fully contained segment

                curr['start'] = prev['end']

            # If the segment becomes too short after adjustment (e.g. < 0.2s), discard it
            if curr['end'] - curr['start'] < 0.2:
                continue

            result.append(curr)
        return result

    def segment_by_pauses(
        self,
        audio_path: str,
        segments: List[Dict],
        min_pause_duration: float = 1.0,
        max_duration: float = 5.0,
        max_chars: int = 80,
        overlap: float = 0.0,
        margin: float = 1.0
    ) -> List[Dict]:
        """Segment using Voice Activity Detection based on pauses with overlap and safety margin"""
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

                    # Apply margin after speech ends if it's the last part of a phrase
                    actual_end = end + overlap
                    if not speech_frames[-1]: # If ends in silence
                         # find last speech frame index in this segment
                         indices = np.where(speech_frames)[0]
                         if len(indices) > 0:
                             last_speech_idx = indices[-1]
                             speech_end_time = times[mask][last_speech_idx]
                             actual_end = min(end, speech_end_time + margin)

                    # STRICT SILENCE-BASED HALLUCINATION DETECTION
                    if speech_ratio < 0.05: # Strict threshold for hallucinations
                         continue

                    # If there are significant pauses, split segment
                    if speech_ratio < 0.6 and (end - start) > max_duration:
                        splits = self._split_by_pauses(
                            text, start, end, times[mask], speech_frames, overlap
                        )
                        # Apply margin to the end of each split if appropriate
                        result.extend(splits)
                    else:
                        result.append({'text': text, 'start': start, 'end': actual_end})
                else:
                    result.append({'text': text, 'start': start, 'end': end + overlap})

            return result

        except Exception as e:
            # Fallback to time-based segmentation
            return self.segment_by_time(segments, min_duration=1.0, max_duration=max_duration, max_chars=max_chars, overlap=overlap)

    def merge_segments_llm(self, segments: List[Dict], translator_obj: Any, whisperx_segments: List[Dict] = None) -> List[Dict]:
        """Use LLM to refine segments by comparing Whisper and WhisperX outputs"""
        if not segments or not translator_obj:
            return segments

        # If whisperx_segments is provided, we compare the two versions using a sliding window
        if whisperx_segments:
            refined_all = []
            chunk_size = 30 # Process 30 segments at a time for context

            for i in range(0, max(len(segments), len(whisperx_segments)), chunk_size):
                chunk_w = segments[i:i + chunk_size]
                chunk_wx = whisperx_segments[i:i + chunk_size]

                if not chunk_w and not chunk_wx:
                    continue

                prompt = "Am două versiuni de transcriere pentru același material audio. Prima este de la Whisper, a doua de la WhisperX. "
                prompt += "Te rog să compari ambele versiuni și să deduci care este varianta corectă pentru fiecare porțiune, bazându-te pe context și logică. "
                prompt += "Retranscrie rezultatul final într-un flux coerent de segmente de subtitrare care nu se suprapun. "
                prompt += "Păstrează continuitatea timpilor. "
                prompt += "Returnează doar segmentele în format JSON: [{\"start\": float, \"end\": float, \"text\": string}, ...]\n\n"

                prompt += "Versiunea Whisper:\n"
                prompt += "\n".join([f"[{seg['start']}-{seg['end']}] {seg['text']}" for seg in chunk_w])

                prompt += "\n\nVersiunea WhisperX:\n"
                prompt += "\n".join([f"[{seg['start']}-{seg['end']}] {seg['text']}" for seg in chunk_wx])

                try:
                    result = translator_obj.refine_segments_with_llm(prompt)
                    if result:
                        refined_all.extend(result)
                    else:
                        # If LLM fails, prefer WhisperX if available
                        refined_all.extend(chunk_wx if chunk_wx else chunk_w)
                except Exception:
                    refined_all.extend(chunk_wx if chunk_wx else chunk_w)

            return refined_all

        # Original logic for overlapping segments
        groups = []
        i = 0
        while i < len(segments):
            group = [segments[i]]
            j = i + 1
            while j < len(segments):
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

            context_text = "\n".join([f"[{seg['start']}-{seg['end']}] {seg['text']}" for seg in group])
            prompt = "Următoarele segmente de subtitrare se suprapun. Te rog să deduci după logica textului și context ce rămâne și ce arunci la gunoi, retranscriind totul într-un flux coerent, păstrând timpii de început și sfârșit corespunzători segmentelor rezultate. Returnează doar segmentele în format JSON: [{\"start\": float, \"end\": float, \"text\": string}, ...]\n\n"
            prompt += context_text

            try:
                result = translator_obj.refine_segments_with_llm(prompt)
                if result:
                    refined_segments.extend(result)
                else:
                    refined_segments.extend(group)
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

    def merge_passes_romistral(self, res1, res2, res3, translator):
        """Use RoMistral to refine and merge three transcription passes"""
        # Combine all segments for consideration
        all_segments = res1['segments'] + res2['segments'] + res3['segments']
        # Sort by start time
        all_segments.sort(key=lambda x: x['start'])

        # Implement a context-aware merging using RoMistral
        prompt = (
            "Am trei versiuni de transcriere pentru același segment audio. "
            "Trebuie să le combini într-o singură variantă finală corectă din punct de vedere gramatical și logic în limba română. "
            "Elimină redundanțele și segmentele care par a fi halucinații (zgomot de fundal interpretat ca vorbire). "
            "Dacă versiunile diferă, alege-o pe cea care are cel mai mult sens în context. "
            "Returnează rezultatul final sub formă de listă JSON de segmente: [{\"start\": float, \"end\": float, \"text\": string}].\n\n"
        )

        # We group segments into ~30s windows to avoid LLM context overflow
        final_segments = []
        window_size = 30.0
        max_time = max(s['end'] for s in all_segments) if all_segments else 0

        for t in range(0, int(max_time) + 1, int(window_size)):
            window_segments = [s for s in all_segments if s['start'] >= t and s['start'] < t + window_size]
            if not window_segments: continue

            # Format window for LLM
            window_text = "\n".join([f"[{s['start']:.2f}-{s['end']:.2f}] {s['text']}" for s in window_segments])

            try:
                # Use RoMistral specifically for Romanian refinement
                refined = translator.refine_with_romistral(prompt + window_text)
                if refined:
                    final_segments.extend(refined)
                else:
                    # Fallback to simple deduplication if LLM fails
                    final_segments.extend(self.remove_repetitions(window_segments))
            except Exception:
                final_segments.extend(self.remove_repetitions(window_segments))

        # Final pass to ensure no overlaps if desired
        return self.ensure_sequential(self.merge_identical_overlapping(final_segments))