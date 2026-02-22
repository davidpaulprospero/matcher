"""Tests for transcription health check functions (US-137-007)."""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.transcription.whisper_client import (
    run_transcription_health_checks,
)


class TestRunTranscriptionHealthChecks:
    """Test combined health check function."""

    @patch('src.transcription.whisper_client.check_ffmpeg_health')
    @patch('src.transcription.whisper_client.check_gpu_health')
    @patch('src.transcription.whisper_client.check_model_health')
    def test_all_checks_pass(self, mock_model, mock_gpu, mock_ffmpeg):
        """Test when all health checks pass."""
        mock_gpu.return_value = {'available': True, 'message': 'GPU OK', 'details': {}}
        mock_ffmpeg.return_value = {'available': True, 'message': 'FFmpeg OK', 'details': {}}
        mock_model.return_value = {'loadable': True, 'message': 'Model OK', 'details': {}}

        result = run_transcription_health_checks('base', 'int8', skip_model_check=False)

        assert result['all_passed'] is True
        assert len(result['checks']) == 3
        assert 'passed' in result['message'].lower()

    @patch('src.transcription.whisper_client.check_ffmpeg_health')
    @patch('src.transcription.whisper_client.check_gpu_health')
    def test_gpu_check_fails(self, mock_gpu, mock_ffmpeg):
        """Test when GPU check fails."""
        mock_gpu.return_value = {'available': False, 'message': 'No GPU', 'details': {}}
        mock_ffmpeg.return_value = {'available': True, 'message': 'FFmpeg OK', 'details': {}}

        result = run_transcription_health_checks('base', 'int8', skip_model_check=True)

        assert result['all_passed'] is False
        assert 'gpu' in result['message'].lower()

    @patch('src.transcription.whisper_client.check_ffmpeg_health')
    @patch('src.transcription.whisper_client.check_gpu_health')
    @patch('src.transcription.whisper_client.check_model_health')
    def test_skip_model_check(self, mock_model, mock_gpu, mock_ffmpeg):
        """Test skipping model check."""
        mock_gpu.return_value = {'available': True, 'message': 'GPU OK', 'details': {}}
        mock_ffmpeg.return_value = {'available': True, 'message': 'FFmpeg OK', 'details': {}}

        result = run_transcription_health_checks('base', 'int8', skip_model_check=True)

        mock_model.assert_not_called()
        assert result['all_passed'] is True
        assert len(result['checks']) == 2  # Only GPU and FFmpeg

    @patch('src.transcription.whisper_client.check_ffmpeg_health')
    @patch('src.transcription.whisper_client.check_gpu_health')
    def test_ffmpeg_check_fails(self, mock_gpu, mock_ffmpeg):
        """Test when FFmpeg check fails."""
        mock_gpu.return_value = {'available': True, 'message': 'GPU OK', 'details': {}}
        mock_ffmpeg.return_value = {'available': False, 'message': 'FFmpeg not found', 'details': {}}

        result = run_transcription_health_checks('base', 'int8', skip_model_check=True)

        assert result['all_passed'] is False
        assert 'ffmpeg' in result['message'].lower()

    @patch('src.transcription.whisper_client.check_ffmpeg_health')
    @patch('src.transcription.whisper_client.check_gpu_health')
    @patch('src.transcription.whisper_client.check_model_health')
    def test_model_check_fails(self, mock_model, mock_gpu, mock_ffmpeg):
        """Test when model check fails."""
        mock_gpu.return_value = {'available': True, 'message': 'GPU OK', 'details': {}}
        mock_ffmpeg.return_value = {'available': True, 'message': 'FFmpeg OK', 'details': {}}
        mock_model.return_value = {'loadable': False, 'message': 'Model load failed', 'details': {}}

        result = run_transcription_health_checks('base', 'int8', skip_model_check=False)

        assert result['all_passed'] is False
        assert 'model' in result['message'].lower()
