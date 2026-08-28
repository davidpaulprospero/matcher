"""Tests for src.minimax_image."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


from src.minimax_image import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    MiniMaxImageProvider,
    aspect_for_size,
    generate_image,
)


JPEG_MAGIC = b"\xff\xd8\xff\xe0\x00\x10JFIFhello"
PNG_MAGIC = b"\x89PNG\r\n\x1a\nhello"


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def _ok_response(payload_b64: list[str]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"data": {"image_base64": payload_b64}}
    resp.text = "{}"
    resp.raise_for_status = MagicMock()
    return resp


class TestApiKeyResolution:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="MINIMAX_API_KEY"):
            MiniMaxImageProvider()

    def test_constructor_api_key(self):
        provider = MiniMaxImageProvider(api_key="explicit-key")
        assert provider.api_key == "explicit-key"

    def test_env_fallback(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "env-key")
        provider = MiniMaxImageProvider()
        assert provider.api_key == "env-key"

    def test_constructor_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "env-key")
        provider = MiniMaxImageProvider(api_key="explicit")
        assert provider.api_key == "explicit"


class TestAspectMapping:
    @pytest.mark.parametrize(
        "size,expected",
        [
            ((1792, 1024), "16:9"),
            ((1024, 1024), "1:1"),
            ((1024, 1792), "9:16"),
            ((1024, 768), "4:3"),
            ((768, 1024), "3:4"),
        ],
    )
    def test_known_sizes(self, size, expected):
        assert aspect_for_size(size) == expected

    def test_unknown_size_defaults_to_16_9(self):
        assert aspect_for_size((1234, 567)) == "16:9"


class TestGenerateImage:
    def test_sends_correct_payload(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "k")
        captured = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            captured["timeout"] = timeout
            return _ok_response([_b64(b"hello")])

        with patch("src.minimax_image.requests.post", side_effect=fake_post):
            MiniMaxImageProvider().generate_image("a cat", size=(1792, 1024))

        assert captured["url"] == DEFAULT_ENDPOINT
        assert captured["json"]["model"] == DEFAULT_MODEL
        assert captured["json"]["prompt"] == "a cat"
        assert captured["json"]["aspect_ratio"] == "16:9"
        assert captured["json"]["response_format"] == "base64"
        assert captured["headers"]["Authorization"] == "Bearer k"

    def test_size_to_aspect_mapping(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "k")
        captured = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["json"] = json
            return _ok_response([_b64(b"x")])

        with patch("src.minimax_image.requests.post", side_effect=fake_post):
            MiniMaxImageProvider().generate_image("p", size=(1024, 1792))

        assert captured["json"]["aspect_ratio"] == "9:16"

    def test_aspect_override_wins_over_size(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "k")
        captured = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["json"] = json
            return _ok_response([_b64(b"x")])

        with patch("src.minimax_image.requests.post", side_effect=fake_post):
            MiniMaxImageProvider().generate_image(
                "p", size=(1024, 1792), aspect_ratio="21:9"
            )

        assert captured["json"]["aspect_ratio"] == "21:9"

    def test_empty_response_raises(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "k")
        with patch(
            "src.minimax_image.requests.post",
            return_value=_ok_response([]),
        ):
            with pytest.raises(RuntimeError, match="no images"):
                MiniMaxImageProvider().generate_image("p")

    def test_returns_decoded_bytes(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "k")
        payload = b"binary-stuff-here"
        with patch(
            "src.minimax_image.requests.post",
            return_value=_ok_response([_b64(payload)]),
        ):
            out = MiniMaxImageProvider().generate_image("p")

        assert out == payload

    def test_http_error_raises(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "k")
        resp = MagicMock()
        resp.raise_for_status.side_effect = RuntimeError("boom")
        with patch("src.minimax_image.requests.post", return_value=resp):
            with pytest.raises(RuntimeError):
                MiniMaxImageProvider().generate_image("p")


class TestGenerateImageToFile:
    def test_jpeg_bytes_get_jpg_extension(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "k")
        with patch(
            "src.minimax_image.requests.post",
            return_value=_ok_response([_b64(JPEG_MAGIC)]),
        ):
            result = MiniMaxImageProvider().generate_image_to_file(
                "p", tmp_path / "out.png"
            )

        assert result.suffix == ".jpg"
        assert result.read_bytes() == JPEG_MAGIC

    def test_png_bytes_keep_png_extension(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "k")
        with patch(
            "src.minimax_image.requests.post",
            return_value=_ok_response([_b64(PNG_MAGIC)]),
        ):
            result = MiniMaxImageProvider().generate_image_to_file(
                "p", tmp_path / "out.bin"
            )

        assert result.suffix == ".png"
        assert result.read_bytes() == PNG_MAGIC

    def test_wrong_suffix_is_corrected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "k")
        with patch(
            "src.minimax_image.requests.post",
            return_value=_ok_response([_b64(JPEG_MAGIC)]),
        ):
            result = MiniMaxImageProvider().generate_image_to_file(
                "p", tmp_path / "out"
            )

        assert result.suffix == ".jpg"


class TestModuleLevel:
    def test_generate_image_uses_env_key(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "k")
        with patch(
            "src.minimax_image.requests.post",
            return_value=_ok_response([_b64(b"bytes")]),
        ):
            result = generate_image("a prompt")
        assert result == b"bytes"

    def test_default_size_constant(self):
        from src.minimax_image import DEFAULT_SIZE

        assert DEFAULT_SIZE == (1792, 1024)
