"""
Audio Analysis Module

Uses librosa for audio feature extraction:
- Silence detection (find natural cut points)
- Speech detection (identify clips with speech that might conflict with voiceover)

Integrated with scene detection for audio-based cut points.
"""

import os
import json
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SilenceRegion:
    """A detected silence region in audio"""
    start_time: float  # seconds
    end_time: float    # seconds
    duration: float    # seconds


@dataclass
class SpeechRegion:
    """A detected speech region in audio"""
    start_time: float
    end_time: float
    duration: float
    confidence: float  # 0-1, how confident we are this is speech


@dataclass
class AudioAnalysis:
    """Complete audio analysis for a video"""
    video_path: str
    duration: float
    has_speech: bool
    speech_ratio: float  # 0-1, portion of audio that contains speech
    silence_regions: List[SilenceRegion]
    speech_regions: List[SpeechRegion]
    suggested_cut_points: List[float]  # Timestamps of natural cut points
    
    def to_dict(self) -> dict:
        return {
            'video_path': str(self.video_path),
            'duration': float(self.duration),
            'has_speech': bool(self.has_speech),
            'speech_ratio': round(float(self.speech_ratio), 3),
            'silence_regions': [asdict(s) for s in self.silence_regions],
            'speech_regions': [asdict(s) for s in self.speech_regions],
            'suggested_cut_points': [float(x) for x in self.suggested_cut_points]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AudioAnalysis":
        return cls(
            video_path=data['video_path'],
            duration=data['duration'],
            has_speech=data['has_speech'],
            speech_ratio=data['speech_ratio'],
            silence_regions=[SilenceRegion(**s) for s in data.get('silence_regions', [])],
            speech_regions=[SpeechRegion(**s) for s in data.get('speech_regions', [])],
            suggested_cut_points=data.get('suggested_cut_points', [])
        )


class AudioAnalyzer:
    """
    Analyze audio tracks for silence and speech detection.
    """
    
    def __init__(self, config=None):
        """
        Initialize analyzer.
        
        Args:
            config: Pipeline config (optional)
        """
        self.config = config
        
        # Get settings from config or use defaults
        if config and hasattr(config, 'audio_analysis'):
            audio_config = config.audio_analysis
            self.silence_threshold_db = getattr(audio_config, 'silence_threshold_db', -40.0)
            self.min_silence_duration = getattr(audio_config, 'min_silence_duration', 0.3)
            self.speech_threshold = getattr(audio_config, 'speech_threshold', 0.5)
            self.sample_rate = getattr(audio_config, 'sample_rate', 22050)
            self.min_speech_duration = getattr(audio_config, 'min_speech_duration', 0.2)
        else:
            # Default settings (fallback if no config)
            self.silence_threshold_db = -40.0  # dB below which is considered silence
            self.min_silence_duration = 0.3    # Minimum silence duration to detect (seconds)
            self.speech_threshold = 0.5        # Confidence threshold for speech detection
            self.sample_rate = 22050           # Sample rate for analysis
            self.min_speech_duration = 0.2     # Minimum speech duration to detect (seconds)
        
        # Try to import librosa
        try:
            import librosa
            self.librosa = librosa
            self._available = True
        except ImportError:
            logger.warning("librosa not installed. Install with: pip install librosa")
            self._available = False
    
    def is_available(self) -> bool:
        """Check if audio analysis is available"""
        return self._available
    
    def _extract_audio(self, video_path: str) -> Optional[str]:
        """
        Extract audio from video to temporary WAV file.
        Returns path to audio file.
        """
        audio_path = Path(video_path).with_suffix('.analysis.wav')
        
        # Skip if already exists
        if audio_path.exists():
            return str(audio_path)
        
        # Try GPU-accelerated decoding first (NVIDIA CUDA)
        cmd = [
            'ffmpeg',
            '-y',
            '-hwaccel', 'cuda',
            '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', str(self.sample_rate),
            '-ac', '1',
            str(audio_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if audio_path.exists() and audio_path.stat().st_size > 0:
                return str(audio_path)
            
            # Fallback to CPU decoding if GPU fails
            cmd_cpu = [
                'ffmpeg',
                '-y',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', str(self.sample_rate),
                '-ac', '1',
                str(audio_path)
            ]
            result = subprocess.run(cmd_cpu, capture_output=True, text=True, timeout=120)
            
            if audio_path.exists() and audio_path.stat().st_size > 0:
                return str(audio_path)
            return None
            
        except Exception as e:
            logger.debug(f"Failed to extract audio from {video_path}: {e}")
            return None
    
    def _detect_silence(
        self,
        audio: np.ndarray,
        sr: int
    ) -> List[SilenceRegion]:
        """
        Detect silence regions in audio.
        
        Args:
            audio: Audio samples
            sr: Sample rate
        
        Returns:
            List of SilenceRegion objects
        """
        # Convert to dB
        # Use RMS energy in short frames
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        # Compute RMS energy
        rms = self.librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # Convert to dB
        rms_db = self.librosa.amplitude_to_db(rms, ref=np.max)
        
        # Find frames below threshold
        is_silence = rms_db < self.silence_threshold_db
        
        # Convert frames to time regions
        times = self.librosa.frames_to_time(
            np.arange(len(rms_db)),
            sr=sr,
            hop_length=hop_length
        )
        
        # Find contiguous silence regions
        silence_regions = []
        in_silence = False
        silence_start = 0
        
        for i, (t, silent) in enumerate(zip(times, is_silence)):
            if silent and not in_silence:
                # Start of silence
                in_silence = True
                silence_start = t
            elif not silent and in_silence:
                # End of silence
                in_silence = False
                duration = t - silence_start
                if duration >= self.min_silence_duration:
                    silence_regions.append(SilenceRegion(
                        start_time=round(silence_start, 3),
                        end_time=round(t, 3),
                        duration=round(duration, 3)
                    ))
        
        # Handle silence at end
        if in_silence:
            duration = times[-1] - silence_start
            if duration >= self.min_silence_duration:
                silence_regions.append(SilenceRegion(
                    start_time=round(silence_start, 3),
                    end_time=round(times[-1], 3),
                    duration=round(duration, 3)
                ))
        
        return silence_regions
    
    def _detect_speech(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Tuple[List[SpeechRegion], float]:
        """
        Detect speech regions using spectral features.
        
        Uses a simple heuristic based on:
        - Spectral centroid (speech has characteristic frequency range)
        - Spectral flatness (speech is less flat than noise)
        - Zero crossing rate (speech has moderate ZCR)
        
        Returns:
            Tuple of (speech_regions, speech_ratio)
        """
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        
        # Compute features
        spectral_centroid = self.librosa.feature.spectral_centroid(
            y=audio, sr=sr, hop_length=hop_length
        )[0]
        
        spectral_flatness = self.librosa.feature.spectral_flatness(
            y=audio, hop_length=hop_length
        )[0]
        
        zcr = self.librosa.feature.zero_crossing_rate(
            y=audio, hop_length=hop_length
        )[0]
        
        rms = self.librosa.feature.rms(
            y=audio, hop_length=hop_length
        )[0]
        
        # Normalize features
        centroid_norm = spectral_centroid / (sr / 2)  # 0-1
        
        # Speech heuristics:
        # - Spectral centroid typically 500-4000 Hz for speech
        # - Spectral flatness low for voiced speech (0.0-0.3)
        # - ZCR moderate for speech
        # - Needs some energy (not silence)
        
        centroid_in_range = (spectral_centroid > 300) & (spectral_centroid < 4000)
        flatness_speech = spectral_flatness < 0.3
        zcr_moderate = (zcr > 0.02) & (zcr < 0.2)
        has_energy = rms > 0.01 * np.max(rms)
        
        # Combine heuristics
        is_speech = centroid_in_range & flatness_speech & has_energy
        speech_confidence = is_speech.astype(float)
        
        # Smooth the detection
        from scipy.ndimage import uniform_filter1d
        try:
            speech_confidence = uniform_filter1d(speech_confidence, size=10)
        except:
            pass
        
        # Convert to time
        times = self.librosa.frames_to_time(
            np.arange(len(speech_confidence)),
            sr=sr,
            hop_length=hop_length
        )
        
        # Find speech regions
        speech_regions = []
        in_speech = False
        speech_start = 0
        speech_conf_acc = []
        
        threshold = self.speech_threshold
        
        for i, (t, conf) in enumerate(zip(times, speech_confidence)):
            if conf > threshold and not in_speech:
                in_speech = True
                speech_start = t
                speech_conf_acc = [conf]
            elif conf <= threshold and in_speech:
                in_speech = False
                duration = t - speech_start
                if duration >= self.min_speech_duration:  # Use config value
                    avg_conf = np.mean(speech_conf_acc)
                    speech_regions.append(SpeechRegion(
                        start_time=round(speech_start, 3),
                        end_time=round(t, 3),
                        duration=round(duration, 3),
                        confidence=round(float(avg_conf), 3)
                    ))
                speech_conf_acc = []
            elif in_speech:
                speech_conf_acc.append(conf)
        
        # Handle speech at end
        if in_speech and speech_conf_acc:
            duration = times[-1] - speech_start
            if duration >= self.min_speech_duration:  # Use config value
                avg_conf = np.mean(speech_conf_acc)
                speech_regions.append(SpeechRegion(
                    start_time=round(speech_start, 3),
                    end_time=round(times[-1], 3),
                    duration=round(duration, 3),
                    confidence=round(float(avg_conf), 3)
                ))
        
        # Calculate speech ratio
        total_speech_time = sum(r.duration for r in speech_regions)
        total_duration = times[-1] if len(times) > 0 else 0
        speech_ratio = total_speech_time / total_duration if total_duration > 0 else 0
        
        return speech_regions, speech_ratio
    
    def _find_cut_points(
        self,
        silence_regions: List[SilenceRegion],
        duration: float
    ) -> List[float]:
        """
        Find suggested cut points based on silence regions.
        Cut points are placed at the middle of silence regions.
        """
        cut_points = []
        
        for region in silence_regions:
            # Only suggest cuts for longer silences
            if region.duration >= 0.5:
                # Place cut at middle of silence
                cut_time = region.start_time + (region.duration / 2)
                cut_points.append(round(cut_time, 3))
        
        return cut_points
    
    def analyze_video(
        self,
        video_path: str,
        cleanup_audio: bool = True
    ) -> Optional[AudioAnalysis]:
        """
        Analyze audio in a video file.
        
        Args:
            video_path: Path to video file
            cleanup_audio: Whether to delete extracted audio after analysis
        
        Returns:
            AudioAnalysis object or None if failed
        """
        if not self._available:
            logger.warning("Audio analysis not available (librosa not installed)")
            return None
        
        video_path = str(video_path)
        
        # Extract audio
        audio_path = self._extract_audio(video_path)
        if not audio_path:
            logger.warning(f"Could not extract audio from {video_path}")
            return None
        
        try:
            # Load audio
            audio, sr = self.librosa.load(audio_path, sr=self.sample_rate, mono=True)
            duration = len(audio) / sr
            
            # Detect silence
            silence_regions = self._detect_silence(audio, sr)
            
            # Detect speech
            speech_regions, speech_ratio = self._detect_speech(audio, sr)
            
            # Find cut points
            cut_points = self._find_cut_points(silence_regions, duration)
            
            # Determine if has significant speech (convert to native Python types)
            has_speech = bool(speech_ratio > 0.1)  # More than 10% speech
            
            analysis = AudioAnalysis(
                video_path=video_path,
                duration=round(float(duration), 3),
                has_speech=has_speech,
                speech_ratio=float(speech_ratio),
                silence_regions=silence_regions,
                speech_regions=speech_regions,
                suggested_cut_points=cut_points
            )
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Audio analysis failed for {video_path}: {e}")
            return None
            
        finally:
            # Cleanup temp audio file
            if cleanup_audio and audio_path:
                try:
                    Path(audio_path).unlink()
                except:
                    pass
    
    def analyze_videos(
        self,
        video_paths: List[str],
        cleanup_audio: bool = True
    ) -> Dict[str, AudioAnalysis]:
        """
        Analyze multiple videos.
        
        Returns:
            Dict mapping video path to AudioAnalysis
        """
        results = {}
        
        for video_path in video_paths:
            analysis = self.analyze_video(video_path, cleanup_audio)
            if analysis:
                results[video_path] = analysis
        
        return results


def analyze_audio(
    video_path: str,
    cleanup: bool = True
) -> Optional[AudioAnalysis]:
    """
    Convenience function to analyze audio in a video.
    
    Args:
        video_path: Path to video file
        cleanup: Whether to delete temp files
    
    Returns:
        AudioAnalysis object or None
    """
    analyzer = AudioAnalyzer()
    
    if not analyzer.is_available():
        return None
    
    return analyzer.analyze_video(video_path, cleanup)


def has_speech(video_path: str) -> bool:
    """
    Quick check if video contains speech.
    
    Returns:
        True if video contains significant speech
    """
    analysis = analyze_audio(video_path)
    return analysis.has_speech if analysis else False


def get_silence_cut_points(video_path: str) -> List[float]:
    """
    Get suggested cut points based on silence detection.
    
    Returns:
        List of timestamps (seconds) where cuts could be placed
    """
    analysis = analyze_audio(video_path)
    return analysis.suggested_cut_points if analysis else []
