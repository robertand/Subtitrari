import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, BinaryIO
import hashlib
import time
import logging
import json
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

class FileHandler:
    def __init__(self, config):
        self.config = config
        self.upload_sessions = {}
    
    def create_upload_session(self, filename: str, total_size: int, total_chunks: int) -> str:
        """Create a new upload session"""
        session_id = hashlib.md5(f"{filename}{time.time()}".encode()).hexdigest()[:16]
        
        session_dir = self.config.CHUNK_UPLOAD_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        self.upload_sessions[session_id] = {
            'filename': secure_filename(filename),
            'total_size': total_size,
            'total_chunks': total_chunks,
            'uploaded_chunks': [],
            'session_dir': session_dir,
            'created_at': time.time(),
            'status': 'active'
        }
        
        # Save session metadata
        with open(session_dir / 'session.json', 'w') as f:
            json.dump(self.upload_sessions[session_id], f, default=str)
        
        return session_id
    
    def save_chunk(self, session_id: str, chunk_data: BinaryIO, chunk_number: int) -> Dict:
        """Save an uploaded chunk"""
        if session_id not in self.upload_sessions:
            # Try to load from disk
            session_file = self.config.CHUNK_UPLOAD_DIR / session_id / 'session.json'
            if session_file.exists():
                with open(session_file) as f:
                    self.upload_sessions[session_id] = json.load(f)
            else:
                raise ValueError("Invalid session ID")
        
        session = self.upload_sessions[session_id]
        chunk_path = session['session_dir'] / f"chunk_{chunk_number:06d}"
        
        # Save chunk
        chunk_data.save(chunk_path)
        
        if chunk_number not in session['uploaded_chunks']:
            session['uploaded_chunks'].append(chunk_number)
        
        # Update session
        session['last_activity'] = time.time()
        
        # Save updated session
        with open(session['session_dir'] / 'session.json', 'w') as f:
            json.dump(session, f, default=str)
        
        return {
            'session_id': session_id,
            'chunk_number': chunk_number,
            'uploaded_chunks': len(session['uploaded_chunks']),
            'total_chunks': session['total_chunks'],
            'progress': (len(session['uploaded_chunks']) / session['total_chunks']) * 100
        }
    
    def assemble_file(self, session_id: str, total_chunks: Optional[int] = None) -> str:
        """Assemble chunks into final file"""
        session = self.upload_sessions.get(session_id)
        if not session:
            # Try to load from disk
            session_file = self.config.CHUNK_UPLOAD_DIR / session_id / 'session.json'
            if session_file.exists():
                with open(session_file) as f:
                    session = json.load(f)
                    # Convert paths back to Path objects
                    session['session_dir'] = Path(session['session_dir'])
            else:
                raise ValueError("Session not found")
        
        if total_chunks is not None:
            session['total_chunks'] = total_chunks

        output_dir = self.config.PROCESS_DIR / session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / session['filename']
        
        # Sort chunks and write to final file
        missing_chunks = []
        with open(output_path, 'wb') as outfile:
            for i in range(session['total_chunks']):
                chunk_path = session['session_dir'] / f"chunk_{i:06d}"
                if chunk_path.exists():
                    with open(chunk_path, 'rb') as chunk_file:
                        outfile.write(chunk_file.read())
                else:
                    missing_chunks.append(i)

        if missing_chunks:
            output_path.unlink(missing_ok=True)
            raise ValueError(f"Missing chunks: {missing_chunks}")
        
        # Cleanup chunks
        shutil.rmtree(session['session_dir'])
        del self.upload_sessions[session_id]
        
        return str(output_path)
    
    def cleanup_old_sessions(self):
        """Remove expired upload sessions"""
        current_time = time.time()
        sessions_to_remove = []
        
        for session_id, session in self.upload_sessions.items():
            if current_time - session.get('created_at', 0) > self.config.SESSION_LIFETIME:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            session_dir = self.upload_sessions[session_id]['session_dir']
            if session_dir.exists():
                shutil.rmtree(session_dir)
            del self.upload_sessions[session_id]
        
        # Also cleanup orphaned directories
        for dir_path in self.config.CHUNK_UPLOAD_DIR.iterdir():
            if dir_path.is_dir():
                session_file = dir_path / 'session.json'
                if session_file.exists():
                    with open(session_file) as f:
                        session_data = json.load(f)
                    if current_time - session_data.get('created_at', 0) > self.config.SESSION_LIFETIME:
                        shutil.rmtree(dir_path)
    
    def get_session_progress(self, session_id: str) -> Optional[Dict]:
        """Get upload progress for a session"""
        session = self.upload_sessions.get(session_id)
        if session:
            return {
                'session_id': session_id,
                'filename': session['filename'],
                'uploaded_chunks': len(session['uploaded_chunks']),
                'total_chunks': session['total_chunks'],
                'progress': (len(session['uploaded_chunks']) / session['total_chunks']) * 100,
                'status': session['status']
            }
        return None
    
    def generate_preview(self, video_path: str, output_path: str, time_offset: float = 5.0) -> str:
        """Generate a preview frame from video"""
        import ffmpeg
        
        try:
            stream = ffmpeg.input(video_path, ss=time_offset)
            stream = ffmpeg.output(stream, output_path, vframes=1)
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            return output_path
        except ffmpeg.Error as e:
            logger.error(f"Preview generation error: {e}")
            return None