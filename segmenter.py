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
        max_chars: int = 76,
        overlap: float = 0.0
    ) -> List[Dict]:
        """Segment subtitles by time constraints with optional overlap and speaker awareness"""
        result = []
        
        for segment in segments:
            text = segment.get('text', '').strip()
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            speaker = segment.get('speaker')
            duration = end - start
            
            if not text:
                continue
            
            # If segment is too short, try to merge with previous (if same speaker)
            if duration < min_duration:
                if result and result[-1].get('speaker') == speaker:
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
                        'end': end + overlap,
                        'speaker': speaker
                    })
                continue
            
            # If segment is too long, split it
            if duration > max_duration or len(text) > max_chars:
                sub_segments = self._split_segment(
                    text, start, end, max_duration, max_chars, overlap
                )
                # Propagate speaker to sub-segments
                for sub in sub_segments:
                    sub['speaker'] = speaker
                result.extend(sub_segments)
            else:
                result.append({
                    'text': text,
                    'start': start,
                    'end': end + overlap,
                    'speaker': speaker
                })
        
        return result
    
    
    @staticmethod
    def _get_conjunction_pattern() -> str:
        return (
            r'\b(să|și|s\-a|s\-o|că|dar|însă|deci|de|pe|la|în|cu|prin|pentru|'
            r'fără|către|spre|după|înainte|până|când|cât|unde|cum|dacă|'
            r'deoarece|fiindcă|așadar|între|sub|peste|lângă|din|dintre|'
            r'that|and|but|or|if|when|while|because|since|although|'
            r'though|unless|until|after|before|for|with|without|about|'
            r'into|through|during|between|against|under|over|where)\b'
        )

    @staticmethod
    def _get_conj_prep_set() -> set:
        return {
            'să', 'și', 's-a', 's-o', 'că', 'dar', 'însă', 'deci',
            'de', 'pe', 'la', 'în', 'cu', 'prin', 'pentru',
            'fără', 'către', 'spre', 'după', 'înainte', 'până',
            'când', 'cât', 'unde', 'cum', 'dacă',
            'deoarece', 'fiindcă', 'așadar', 'între', 'sub', 'peste',
            'lângă', 'din', 'dintre',
            'that', 'and', 'but', 'or', 'if', 'when', 'while',
            'because', 'since', 'although', 'though', 'unless',
            'until', 'after', 'before', 'for', 'with', 'without',
            'about', 'into', 'through', 'during', 'between',
            'against', 'under', 'over', 'where',
        }

    @staticmethod
    def _count_propositions(text: str) -> int:
        """Numără propozițiile dintr-un text."""
        text = text.strip()
        if not text:
            return 0
        # Numără frazele terminate cu . ! ? (inclusiv ...)
        sentences = re.split(r'[.!?]+(?:\s|$)', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 1
        # Pentru fiecare frază, vezi dacă conține conectori care introduc o nouă propoziție
        count = 0
        sub_conj = {'că', 'să', 'dacă', 'deoarece', 'fiindcă', 'deși', 'ca',
                    'that', 'if', 'because', 'since', 'although', 'though', 'unless'}
        for s in sentences:
            count += 1
            words = s.split()
            for w in words:
                w_clean = w.strip('"\'„”“(«»[{]}').lower()
                if w_clean in sub_conj:
                    count += 1
        return count

    @staticmethod
    def _has_dialog_markers(text: str) -> bool:
        """Detectează dacă textul conține marcatori de dialog (liniuțe la început de rând)."""
        return bool(re.search(r'(?:^|\n)\s*[—\-–]\s', text))

    @staticmethod
    def _smart_split(text: str, target_pos: int) -> int:
        """
        Găsește cel mai bun loc de rupere a textului aproape de target_pos,
        păstrând unitatea de sens.
        Priorități: sfârșit de propoziție > conectori de frază (înainte de) > punctuație > cuvânt.
        Din fiecare categorie alege cel mai apropiat de target_pos fără să-l depășească.
        """
        if len(text) <= target_pos:
            return len(text)

        search_start = max(0, int(target_pos * 0.55))
        search_end = min(len(text), int(target_pos * 1.1))
        region = text[search_start:search_end]

        best = None

        # 1. Sfârșit de propoziție (. ! ?) — cel mai puternic separator
        for m in re.finditer(r'(?<=[.!?])\s+', region):
            pos = search_start + m.end()
            if pos <= target_pos:
                best = max(best or 0, pos)

        # 2. Punctuație slabă (, ; :) — separator natural de frază
        if best is None:
            for m in re.finditer(r'(?<=[,;:])\s+', region):
                pos = search_start + m.end()
                if pos <= target_pos:
                    best = max(best or 0, pos)

        # 3. Conectori (prepoziții, conjuncții) — rupe ÎNAINTEA conectorului
        if best is None:
            conj = SubtitleSegmenter._get_conjunction_pattern()
            for m in re.finditer(conj, region, re.IGNORECASE):
                pos = search_start + m.start()
                if search_start < pos <= target_pos:
                    best = max(best or 0, pos)

        # 4. Ultimul spațiu înainte de target_pos
        if best is None:
            for i in range(target_pos - 1, search_start - 1, -1):
                if text[i] == ' ':
                    best = i + 1
                    break

        return best if best else target_pos

    @staticmethod
    def _ideal_duration(text: str, reading_speed: float = 15.0) -> float:
        """Calculează durata ideală pe baza numărului de caractere."""
        return len(text) / reading_speed

    @staticmethod
    def _check_duration_limits(duration: float, ideal: float) -> tuple:
        """
        Verifică limitele de durată.
        Returnează (min_accepted, max_accepted, is_exceptional_allowed).
        """
        min_acc = ideal * 0.9
        max_acc = ideal * 1.1
        exceptional_min = ideal * 0.8
        return min_acc, max_acc, exceptional_min

    def _split_segment(
        self,
        text: str,
        start: float,
        end: float,
        max_duration: float,
        max_chars: int,
        overlap: float = 0.0
    ) -> List[Dict]:
        """Despică un segment lung în mai multe, păstrând unitatea de sens."""
        words = text.split()
        if not words:
            return [{'text': text, 'start': start, 'end': end}]

        total_duration = end - start
        words_per_second = len(words) / total_duration if total_duration > 0 else 2

        segments = []
        current_start = start
        char_pos = 0

        while char_pos < len(text):
            remaining = text[char_pos:]
            if len(remaining) <= max_chars:
                seg_dur = len(remaining) / words_per_second
                seg_end = min(end, current_start + seg_dur)
                segments.append({
                    'text': remaining.strip(),
                    'start': round(current_start, 3),
                    'end': round(seg_end + overlap, 3),
                })
                break

            split_pos = self._smart_split(text, char_pos + max_chars)
            if split_pos <= char_pos:
                split_pos = char_pos + max_chars
                while split_pos < len(text) and text[split_pos] != ' ':
                    split_pos += 1
                if split_pos >= len(text):
                    split_pos = char_pos + max_chars

            chunk = text[char_pos:split_pos].strip()
            seg_dur = len(chunk.split()) / words_per_second
            seg_end = min(end, current_start + seg_dur)

            segments.append({
                'text': chunk,
                'start': round(current_start, 3),
                'end': round(seg_end + overlap, 3),
            })

            char_pos = split_pos
            current_start = seg_end

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
                if times[i] - silence_start > 0.4:  # Minimum pause duration
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
    
    def _merge_adjacent_same_speaker(
        self,
        segments: List[Dict],
        max_duration: float = 5.0,
        max_chars: int = 76
    ) -> List[Dict]:
        """Unește segmente adiacente cu același vorbitor, păstrând limitele de durată/caractere."""
        if not segments or len(segments) < 2:
            return segments

        merged = [segments[0].copy()]
        for seg in segments[1:]:
            prev = merged[-1]
            same_speaker = (
                seg.get("speaker") is not None
                and prev.get("speaker") is not None
                and seg["speaker"] == prev["speaker"]
            )
            combined_duration = max(seg["end"], prev["end"]) - min(seg["start"], prev["start"])
            combined_text = prev["text"] + " " + seg["text"]
            combined_chars = len(combined_text)

            if same_speaker and combined_duration <= max_duration and combined_chars <= max_chars:
                prev["end"] = max(prev["end"], seg["end"])
                prev["start"] = min(prev["start"], seg["start"])
                prev["text"] = combined_text
                if "words" in prev and "words" in seg:
                    prev["words"] = prev["words"] + seg["words"]
            else:
                merged.append(seg.copy())
        return merged

    @staticmethod
    def _fix_line_break(lines: List[str], max_cpr: int = 38) -> List[str]:
        """Corectează linii: mută conjuncțiile/prepozițiile de la finalul liniei la începutul următoarei."""
        if len(lines) < 2:
            return lines
        result = list(lines)
        conj_prep = SubtitleSegmenter._get_conj_prep_set()
        for i in range(len(result) - 1):
            words_i = result[i].split()
            if words_i:
                last_word = words_i[-1].strip('"\'„”“(«»[{]},.;:!?').lower()
                if last_word in conj_prep:
                    result[i] = ' '.join(words_i[:-1])
                    result[i + 1] = words_i[-1] + ' ' + result[i + 1]
        return [r for r in result if r.strip()]

    def split_text_for_subtitle(self, text: str, max_chars_per_line: int = 38) -> str:
        """Split text into multiple lines for subtitle display with unit-of-sense rules."""
        words = text.split()
        if not words:
            return ''

        if len(text) <= max_chars_per_line:
            return text

        split_pos = self._smart_split(text, max_chars_per_line)
        if split_pos <= 0 or split_pos >= len(text):
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
        else:
            lines = [text[:split_pos].strip(), text[split_pos:].strip()]

        lines = self._fix_line_break(lines, max_chars_per_line)
        return '\n'.join(lines[:2])
    
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
                same_speaker = current.get('speaker') == next_seg.get('speaker')

                if overlap > 0 and curr_text_norm == next_text_norm and same_speaker:
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

                    if similarity >= threshold and current.get('speaker') == next_seg.get('speaker'):
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
            same_speaker = curr.get('speaker') == prev.get('speaker')

            if curr_text == prev_text and curr_text != "" and same_speaker:
                if gap < 2.5: # If identical and close together (increased gap tolerance for multi-pass)
                    prev['end'] = max(prev['end'], curr['end'])
                    continue

            # Fuzzy match for near-repetitions (often caused by windowing artifacts)
            if len(curr_text) > 0 and len(prev_text) > 0 and same_speaker:
                words1 = set(prev_text.split())
                words2 = set(curr_text.split())
                if words1 and words2:
                    common = words1.intersection(words2)
                    similarity = len(common) / max(len(words1), len(words2))
                    if similarity > 0.85 and gap < 1.5:
                        # High similarity and small gap: likely a repetition artifact
                        prev['end'] = max(prev['end'], curr['end'])
                        if len(curr['text']) > len(prev['text']):
                            prev['text'] = curr['text']
                        continue

            result.append(curr)
        return result

    def ensure_sequential(self, segments: List[Dict]) -> List[Dict]:
        """Ensure segments do not overlap: next starts exactly after previous ends. Forces split on speaker change."""
        if not segments:
            return []

        # Sort by start time
        sorted_segments = sorted(segments, key=lambda x: x['start'])

        result = [sorted_segments[0].copy()]
        for i in range(1, len(sorted_segments)):
            curr = sorted_segments[i].copy()
            prev = result[-1]

            # If speaker changes, we MUST start new segment and prevent merge
            if curr.get('speaker') != prev.get('speaker'):
                # Handle potential overlap by clipping previous segment
                if prev['end'] > curr['start']:
                     prev['end'] = curr['start']
                result.append(curr)
                continue

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
        max_chars: int = 76,
        overlap: float = 0.0,
        margin: float = 1.0
    ) -> List[Dict]:
        """Segment using Voice Activity Detection based on pauses with overlap, safety margin and speaker awareness"""
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
            silence_threshold = -38  # dB
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
                speaker = segment.get('speaker')

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

                    # SILENCE-BASED HALLUCINATION DETECTION (STRICT)
                    # If the segment occurs in a very silent area (low speech ratio)
                    if speech_ratio < 0.05:
                        # Discard segments with almost no voice detected
                        continue

                    if speech_ratio < 0.1:
                        # For very low ratio, only keep if it's long and likely contains quiet speech
                        if (end - start) < 2.0:
                            continue

                    # If segment exceeds max_duration, split at natural pauses
                    if (end - start) > max_duration:
                        splits = self._split_by_pauses(
                            text, start, end, times[mask], speech_frames, overlap
                        )
                        if splits and len(splits) > 1:
                            for s in splits:
                                s['speaker'] = speaker
                            result.extend(splits)
                        else:
                            result.append({'text': text, 'start': start, 'end': actual_end, 'speaker': speaker})
                    else:
                        result.append({'text': text, 'start': start, 'end': actual_end, 'speaker': speaker})
                else:
                    result.append({'text': text, 'start': start, 'end': end + overlap, 'speaker': speaker})

            result = self._merge_adjacent_same_speaker(result, max_duration, max_chars)
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
            prompt = "Următoarele segmente de subtitrare provin din multiple treceri de transcriere și se suprapun. "
            prompt += "Folosește contextul și logica pentru a decide care este varianta corectă pentru fiecare porțiune de audio. "
            prompt += "Dacă segmentele spun același lucru cu mici variații, alege-o pe cea mai corectă gramatical și logic. "
            prompt += "Dacă sunt complet diferite, decide care se potrivește mai bine în fluxul conversației. "
            prompt += "Retranscrie totul într-un flux coerent de segmente care nu se suprapun, păstrând timpii corespunzători. "
            prompt += "Returnează rezultatul EXCLUSIV în format JSON: [{\"start\": float, \"end\": float, \"text\": string}, ...]\n\n"
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
            replacements = {
                'ș': 'ş',
                'Ș': 'Ş',
                'ț': 'ţ',
                'Ț': 'Ţ',
                'ă': 'ă',
                'â': 'â',
                'î': 'î'
            }
        else:
            replacements = {
                'ş': 'ș',
                'Ș': 'Ș',
                'ţ': 'ț',
                'Ț': 'Ț'
            }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text

    def add_segment_spacing(self, segments: List[dict], spacing: int = 2) -> List[dict]:
        """Push apart consecutive segments that touch or overlap.

        spacing in frames (1-25). 1 frame = 1/25s.
        Even spacing: half from each segment.
        Odd spacing: only from the first segment's end.
        """
        if spacing < 1:
            return segments
        frame_s = 1.0 / 25.0
        gap_s = spacing * frame_s
        half_gap = gap_s / 2.0 if spacing % 2 == 0 else 0.0
        first_only = gap_s if spacing % 2 == 1 else 0.0

        result = []
        for i in range(len(segments)):
            seg = dict(segments[i])
            if i > 0:
                prev_end = result[-1]['end']
                if prev_end >= seg['start']:
                    # Overlap or touch — push apart
                    if first_only:
                        result[-1]['end'] = max(result[-1]['start'], prev_end - first_only)
                    else:
                        result[-1]['end'] = max(result[-1]['start'], prev_end - half_gap)
                        seg['start'] = min(seg['end'], seg['start'] + half_gap)
            result.append(seg)
        return result


def detect_speech_gaps(
    audio_path: str,
    segments: List[Dict],
    min_gap_duration: float = 0.5,
    min_speech_ratio: float = 0.15
) -> List[Dict]:
    """
    Detectează zone din audio unde există vorbire dar nu există segment de subtitrare.
    Returnează lista de goluri: {start, end, duration}.
    """
    try:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        is_speech = rms_db > -38
        times = librosa.frames_to_time(np.arange(len(rms_db)), sr=sr, hop_length=hop_length)

        # Construim o mască booleana: 1 = acoperit de un segment, 0 = neacoperit
        covered = np.zeros(len(times), dtype=bool)
        for seg in segments:
            mask = (times >= seg["start"]) & (times <= seg["end"])
            covered[mask] = True

        # Găsim zone unde e vorbire DAR neacoperit
        gaps = []
        in_gap = False
        gap_start = 0.0
        speech_frames_in_gap = 0
        total_frames_in_gap = 0

        for i in range(len(times)):
            if not covered[i] and is_speech[i]:
                if not in_gap:
                    in_gap = True
                    gap_start = times[i]
                    speech_frames_in_gap = 0
                    total_frames_in_gap = 0
                speech_frames_in_gap += 1
                total_frames_in_gap += 1
            elif in_gap:
                if not is_speech[i]:
                    total_frames_in_gap += 1
                if not is_speech[i] and total_frames_in_gap > 0:
                    gap_dur = times[i] - gap_start
                    speech_ratio = speech_frames_in_gap / total_frames_in_gap if total_frames_in_gap > 0 else 0
                    if gap_dur >= min_gap_duration and speech_ratio >= min_speech_ratio:
                        gaps.append({"start": gap_start, "end": times[i], "duration": gap_dur})
                    in_gap = False

        if in_gap:
            gap_dur = times[-1] - gap_start
            speech_ratio = speech_frames_in_gap / total_frames_in_gap if total_frames_in_gap > 0 else 0
            if gap_dur >= min_gap_duration and speech_ratio >= min_speech_ratio:
                gaps.append({"start": gap_start, "end": times[-1], "duration": gap_dur})

        return gaps
    except Exception as e:
        logging.getLogger(__name__).error(f"Speech gap detection failed: {e}")
        return []


def expand_gap_windows(
    gaps: List[Dict],
    segments: List[Dict],
    padding: float = 3.0
) -> List[Dict]:
    """
    Extinde fiecare gol cu padding stânga/dreapta.
    Dacă zona extinsă atinge un segment existent, se extinde să includă segmentul.
    Îmbină ferestrele care se suprapun.
    """
    windows = []
    for gap in gaps:
        w_start = max(0, gap["start"] - padding)
        w_end = gap["end"] + padding

        # Extindem să cuprindă segmente atinse
        changed = True
        while changed:
            changed = False
            for seg in segments:
                if seg["end"] >= w_start and seg["start"] <= w_end:
                    if seg["start"] < w_start:
                        w_start = seg["start"]
                        changed = True
                    if seg["end"] > w_end:
                        w_end = seg["end"]
                        changed = True

        windows.append({"start": w_start, "end": w_end})

    # Îmbinăm ferestrele care se suprapun
    if not windows:
        return []
    windows.sort(key=lambda w: w["start"])
    merged = [windows[0]]
    for w in windows[1:]:
        if w["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], w["end"])
        else:
            merged.append(w)

    return merged
