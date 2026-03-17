"""Focused tests for the DVIDS video client and fallback orchestration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.media_sources.models import VideoResult
from src.media_sources.videos import DvidsVideoClient
from src.media_sources.videos.orchestrator import download_entity_videos


class TestDvidsVideoClient:
    """Unit tests for DVIDS search and download behavior."""

    @patch("src.media_sources.base.requests.Session")
    @pytest.mark.fast
    def test_search_returns_best_mp4_result(self, mock_session_class, tmp_path):
        """Search resolves asset files and prefers the best landscape MP4."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        search_response = MagicMock(status_code=200)
        search_response.raise_for_status = Mock()
        search_response.json.return_value = {
            "results": [
                {
                    "id": "video:999196",
                    "title": "Carrier arrival",
                    "duration": 20,
                    "url": "https://www.dvidshub.net/video/999196/carrier-arrival",
                }
            ]
        }

        asset_response = MagicMock(status_code=200)
        asset_response.raise_for_status = Mock()
        asset_response.json.return_value = {
            "results": {
                "id": "video:999196",
                "url": "https://www.dvidshub.net/video/999196/carrier-arrival",
                "duration": 20,
                "files": [
                    {
                        "src": "https://cdn.example.com/carrier-720.mp4",
                        "type": "video/mp4",
                        "width": 1280,
                        "height": 720,
                        "size": 1000,
                    },
                    {
                        "src": "https://cdn.example.com/carrier-1080.mp4",
                        "type": "video/mp4",
                        "width": 1920,
                        "height": 1080,
                        "size": 2000,
                    },
                    {
                        "src": "https://cdn.example.com/carrier-portrait.mp4",
                        "type": "video/mp4",
                        "width": 720,
                        "height": 1280,
                        "size": 3000,
                    },
                ],
            }
        }

        mock_session.get.side_effect = [search_response, asset_response]

        client = DvidsVideoClient(
            config=MagicMock(),
            output_dir=str(tmp_path),
            api_key="test_key",
        )

        results = client.search("aircraft carrier", max_results=1)

        assert len(results) == 1
        assert results[0].source == "dvids"
        assert results[0].download_url == "https://cdn.example.com/carrier-1080.mp4"
        assert results[0].width == 1920
        assert results[0].height == 1080
        assert results[0].quality == "hd"
        assert results[0].file_type == "mp4"

    @patch("src.media_sources.base.requests.Session")
    @pytest.mark.fast
    def test_search_preserves_provider_hit_order_without_hd_api_filter(self, mock_session_class, tmp_path):
        """Prefer-HD mode should not let later HD hits displace earlier provider-ranked results."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        search_response = MagicMock(status_code=200)
        search_response.raise_for_status = Mock()
        search_response.json.return_value = {
            "results": [
                {
                    "id": "video:100001",
                    "title": "Carrier arrival sd",
                    "duration": 20,
                    "url": "https://www.dvidshub.net/video/100001/carrier-arrival-sd",
                },
                {
                    "id": "video:100002",
                    "title": "Carrier arrival hd",
                    "duration": 20,
                    "url": "https://www.dvidshub.net/video/100002/carrier-arrival-hd",
                },
            ]
        }

        asset_response_sd = MagicMock(status_code=200)
        asset_response_sd.raise_for_status = Mock()
        asset_response_sd.json.return_value = {
            "results": {
                "id": "video:100001",
                "url": "https://www.dvidshub.net/video/100001/carrier-arrival-sd",
                "duration": 20,
                "files": [
                    {
                        "src": "https://cdn.example.com/carrier-sd.mp4",
                        "type": "video/mp4",
                        "width": 854,
                        "height": 480,
                        "size": 1000,
                    }
                ],
            }
        }

        asset_response_hd = MagicMock(status_code=200)
        asset_response_hd.raise_for_status = Mock()
        asset_response_hd.json.return_value = {
            "results": {
                "id": "video:100002",
                "url": "https://www.dvidshub.net/video/100002/carrier-arrival-hd",
                "duration": 20,
                "files": [
                    {
                        "src": "https://cdn.example.com/carrier-hd.mp4",
                        "type": "video/mp4",
                        "width": 1920,
                        "height": 1080,
                        "size": 2000,
                    }
                ],
            }
        }

        mock_session.get.side_effect = [search_response, asset_response_sd, asset_response_hd]

        client = DvidsVideoClient(
            config=MagicMock(),
            output_dir=str(tmp_path),
            api_key="test_key",
            prefer_hd=True,
        )

        results = client.search("aircraft carrier", max_results=1)

        search_params = mock_session.get.call_args_list[0].kwargs["params"]

        assert "hd" not in search_params
        assert len(results) == 1
        assert results[0].id == "video:100001"
        assert results[0].download_url == "https://cdn.example.com/carrier-sd.mp4"
        assert results[0].quality == "sd"

    @patch("src.media_sources.base.requests.Session")
    @pytest.mark.fast
    def test_search_returns_sd_result_when_only_sd_available_with_prefer_hd(self, mock_session_class, tmp_path):
        """Prefer-HD mode should still return SD results when HD is unavailable."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        search_response = MagicMock(status_code=200)
        search_response.raise_for_status = Mock()
        search_response.json.return_value = {
            "results": [
                {
                    "id": "video:100003",
                    "title": "Carrier arrival sd only",
                    "duration": 20,
                    "url": "https://www.dvidshub.net/video/100003/carrier-arrival-sd-only",
                }
            ]
        }

        asset_response = MagicMock(status_code=200)
        asset_response.raise_for_status = Mock()
        asset_response.json.return_value = {
            "results": {
                "id": "video:100003",
                "url": "https://www.dvidshub.net/video/100003/carrier-arrival-sd-only",
                "duration": 20,
                "files": [
                    {
                        "src": "https://cdn.example.com/carrier-sd-only.mp4",
                        "type": "video/mp4",
                        "width": 854,
                        "height": 480,
                        "size": 1000,
                    }
                ],
            }
        }

        mock_session.get.side_effect = [search_response, asset_response]

        client = DvidsVideoClient(
            config=MagicMock(),
            output_dir=str(tmp_path),
            api_key="test_key",
            prefer_hd=True,
        )

        results = client.search("aircraft carrier", max_results=1)

        search_params = mock_session.get.call_args_list[0].kwargs["params"]

        assert "hd" not in search_params
        assert len(results) == 1
        assert results[0].download_url == "https://cdn.example.com/carrier-sd-only.mp4"
        assert results[0].quality == "sd"

    @patch("src.media_sources.base.requests.Session")
    @pytest.mark.fast
    def test_search_prefer_hd_false_keeps_first_acceptable_file(self, mock_session_class, tmp_path):
        """prefer_hd=False should preserve DVIDS file order within an asset."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        search_response = MagicMock(status_code=200)
        search_response.raise_for_status = Mock()
        search_response.json.return_value = {
            "results": [
                {
                    "id": "video:100005",
                    "title": "Flight deck sequence",
                    "duration": 20,
                    "url": "https://www.dvidshub.net/video/100005/flight-deck-sequence",
                }
            ]
        }

        asset_response = MagicMock(status_code=200)
        asset_response.raise_for_status = Mock()
        asset_response.json.return_value = {
            "results": {
                "id": "video:100005",
                "url": "https://www.dvidshub.net/video/100005/flight-deck-sequence",
                "duration": 20,
                "files": [
                    {
                        "src": "https://cdn.example.com/flight-deck-sd.mp4",
                        "type": "video/mp4",
                        "width": 854,
                        "height": 480,
                        "size": 1000,
                    },
                    {
                        "src": "https://cdn.example.com/flight-deck-hd.mp4",
                        "type": "video/mp4",
                        "width": 1920,
                        "height": 1080,
                        "size": 2000,
                    },
                ],
            }
        }

        mock_session.get.side_effect = [search_response, asset_response]

        client = DvidsVideoClient(
            config=MagicMock(),
            output_dir=str(tmp_path),
            api_key="test_key",
            prefer_hd=False,
        )

        results = client.search("flight deck", max_results=1)

        assert len(results) == 1
        assert results[0].download_url == "https://cdn.example.com/flight-deck-sd.mp4"
        assert results[0].quality == "sd"

    @patch("src.media_sources.base.requests.Session")
    @pytest.mark.fast
    def test_search_same_resolution_tie_keeps_provider_order(self, mock_session_class, tmp_path):
        """Same-resolution MP4 ties should not auto-promote the largest rendition."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        search_response = MagicMock(status_code=200)
        search_response.raise_for_status = Mock()
        search_response.json.return_value = {
            "results": [
                {
                    "id": "video:100006",
                    "title": "Deck operations",
                    "duration": 20,
                    "url": "https://www.dvidshub.net/video/100006/deck-operations",
                }
            ]
        }

        asset_response = MagicMock(status_code=200)
        asset_response.raise_for_status = Mock()
        asset_response.json.return_value = {
            "results": {
                "id": "video:100006",
                "url": "https://www.dvidshub.net/video/100006/deck-operations",
                "duration": 20,
                "files": [
                    {
                        "src": "https://cdn.example.com/deck-operations-1280x720-3000k.mp4",
                        "type": "video/mp4",
                        "width": 1280,
                        "height": 720,
                        "bitrate": 3000,
                        "size": 150000000,
                    },
                    {
                        "src": "https://cdn.example.com/deck-operations-master.mp4",
                        "type": "video/mp4",
                        "width": 1280,
                        "height": 720,
                        "bitrate": 9173,
                        "size": 469000000,
                    },
                ],
            }
        }

        mock_session.get.side_effect = [search_response, asset_response]

        client = DvidsVideoClient(
            config=MagicMock(),
            output_dir=str(tmp_path),
            api_key="test_key",
            prefer_hd=True,
        )

        results = client.search("deck operations", max_results=1)

        assert len(results) == 1
        assert results[0].download_url == "https://cdn.example.com/deck-operations-1280x720-3000k.mp4"
        assert results[0].quality == "sd"

    @patch("src.media_sources.base.requests.Session")
    @pytest.mark.fast
    def test_search_revalidates_resolved_asset_duration(self, mock_session_class, tmp_path):
        """Resolved asset duration should still respect the configured duration window."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        search_response = MagicMock(status_code=200)
        search_response.raise_for_status = Mock()
        search_response.json.return_value = {
            "results": [
                {
                    "id": "video:100007",
                    "title": "Long briefing",
                    "duration": 20,
                    "url": "https://www.dvidshub.net/video/100007/long-briefing",
                }
            ]
        }

        asset_response = MagicMock(status_code=200)
        asset_response.raise_for_status = Mock()
        asset_response.json.return_value = {
            "results": {
                "id": "video:100007",
                "url": "https://www.dvidshub.net/video/100007/long-briefing",
                "duration": 90,
                "files": [
                    {
                        "src": "https://cdn.example.com/long-briefing.mp4",
                        "type": "video/mp4",
                        "width": 1920,
                        "height": 1080,
                        "size": 2000,
                    }
                ],
            }
        }

        mock_session.get.side_effect = [search_response, asset_response]

        client = DvidsVideoClient(
            config=MagicMock(),
            output_dir=str(tmp_path),
            api_key="test_key",
            min_duration=3,
            max_duration=30,
        )

        results = client.search("briefing", max_results=1)

        assert results == []

    @patch("src.media_sources.base.requests.Session")
    @pytest.mark.fast
    def test_search_quality_uses_dvids_bitrate_threshold_when_available(self, mock_session_class, tmp_path):
        """1280x720 low-bitrate renditions should not be labeled HD when bitrate is present."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        search_response = MagicMock(status_code=200)
        search_response.raise_for_status = Mock()
        search_response.json.return_value = {
            "results": [
                {
                    "id": "video:100008",
                    "title": "Pier departure",
                    "duration": 20,
                    "url": "https://www.dvidshub.net/video/100008/pier-departure",
                }
            ]
        }

        asset_response = MagicMock(status_code=200)
        asset_response.raise_for_status = Mock()
        asset_response.json.return_value = {
            "results": {
                "id": "video:100008",
                "url": "https://www.dvidshub.net/video/100008/pier-departure",
                "duration": 20,
                "files": [
                    {
                        "src": "https://cdn.example.com/pier-departure.mp4",
                        "type": "video/mp4",
                        "width": 1280,
                        "height": 720,
                        "bitrate": 3000,
                        "size": 1000,
                    }
                ],
            }
        }

        mock_session.get.side_effect = [search_response, asset_response]

        client = DvidsVideoClient(
            config=MagicMock(),
            output_dir=str(tmp_path),
            api_key="test_key",
        )

        results = client.search("pier departure", max_results=1)

        assert len(results) == 1
        assert results[0].quality == "sd"

    @patch("src.media_sources.base.requests.Session")
    @pytest.mark.fast
    def test_search_uses_landscape_filter_without_requiring_exact_16_9(self, mock_session_class, tmp_path):
        """Landscape-only mode should allow non-16:9 landscape assets."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        search_response = MagicMock(status_code=200)
        search_response.raise_for_status = Mock()
        search_response.json.return_value = {
            "results": [
                {
                    "id": "video:100004",
                    "title": "Briefing room",
                    "duration": 20,
                    "url": "https://www.dvidshub.net/video/100004/briefing-room",
                }
            ]
        }

        asset_response = MagicMock(status_code=200)
        asset_response.raise_for_status = Mock()
        asset_response.json.return_value = {
            "results": {
                "id": "video:100004",
                "url": "https://www.dvidshub.net/video/100004/briefing-room",
                "duration": 20,
                "files": [
                    {
                        "src": "https://cdn.example.com/briefing-room.mp4",
                        "type": "video/mp4",
                        "width": 1024,
                        "height": 768,
                        "size": 1000,
                    }
                ],
            }
        }

        mock_session.get.side_effect = [search_response, asset_response]

        client = DvidsVideoClient(
            config=MagicMock(),
            output_dir=str(tmp_path),
            api_key="test_key",
            landscape_only=True,
        )

        results = client.search("briefing room", max_results=1)

        search_params = mock_session.get.call_args_list[0].kwargs["params"]

        assert search_params["aspect_ratio"] == "landscape"
        assert len(results) == 1
        assert results[0].width == 1024
        assert results[0].height == 768

    @patch("src.media_sources.base.requests.Session")
    @pytest.mark.fast
    def test_search_uses_requested_max_results_within_provider_limit(self, mock_session_class, tmp_path):
        """Direct DVIDS searches should not double requests that already fit on one page."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        search_response = MagicMock(status_code=200)
        search_response.raise_for_status = Mock()
        search_response.json.return_value = {"results": []}
        mock_session.get.return_value = search_response

        client = DvidsVideoClient(
            config=MagicMock(),
            output_dir=str(tmp_path),
            api_key="test_key",
        )

        results = client.search("carrier", max_results=26)

        search_params = mock_session.get.call_args.kwargs["params"]

        assert results == []
        assert search_params["max_results"] == 26
        assert search_params["page"] == 1

    @patch("src.media_sources.base.requests.Session")
    @pytest.mark.fast
    def test_search_paginates_when_requested_results_exceed_provider_page_limit(self, mock_session_class, tmp_path):
        """Large DVIDS searches should request additional pages instead of capping at 50 hits."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        page_one_response = MagicMock(status_code=200)
        page_one_response.raise_for_status = Mock()
        page_one_response.json.return_value = {
            "page_info": {
                "page": 1,
                "results_per_page": 50,
                "total_results": 60,
            },
            "results": [
                {
                    "id": f"video:{index:06d}",
                    "title": f"Carrier clip {index}",
                    "duration": 20,
                    "url": f"https://www.dvidshub.net/video/{index:06d}/carrier-clip-{index}",
                }
                for index in range(1, 51)
            ],
        }

        page_two_response = MagicMock(status_code=200)
        page_two_response.raise_for_status = Mock()
        page_two_response.json.return_value = {
            "page_info": {
                "page": 2,
                "results_per_page": 50,
                "total_results": 60,
            },
            "results": [
                {
                    "id": f"video:{index:06d}",
                    "title": f"Carrier clip {index}",
                    "duration": 20,
                    "url": f"https://www.dvidshub.net/video/{index:06d}/carrier-clip-{index}",
                }
                for index in range(51, 61)
            ],
        }

        mock_session.get.side_effect = [page_one_response, page_two_response]

        client = DvidsVideoClient(
            config=MagicMock(),
            output_dir=str(tmp_path),
            api_key="test_key",
        )

        with patch.object(
            client,
            "_fetch_asset",
            side_effect=lambda asset_id: {
                "id": str(asset_id),
                "url": f"https://www.dvidshub.net/{asset_id}",
                "duration": 20,
                "files": [
                    {
                        "src": f"https://cdn.example.com/{asset_id}.mp4",
                        "type": "video/mp4",
                        "width": 1280,
                        "height": 720,
                        "size": 1000,
                    }
                ],
            },
        ):
            results = client.search("carrier", max_results=60)

        first_call_params = mock_session.get.call_args_list[0].kwargs["params"]
        second_call_params = mock_session.get.call_args_list[1].kwargs["params"]

        assert len(results) == 60
        assert first_call_params["max_results"] == 50
        assert first_call_params["page"] == 1
        assert second_call_params["max_results"] == 50
        assert second_call_params["page"] == 2
        assert results[0].id == "video:000001"
        assert results[-1].id == "video:000060"

    @patch("src.media_sources.base.requests.Session")
    @pytest.mark.fast
    def test_search_skips_results_without_usable_mp4(self, mock_session_class, tmp_path):
        """Search ignores assets without a usable MP4 rendition."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        search_response = MagicMock(status_code=200)
        search_response.raise_for_status = Mock()
        search_response.json.return_value = {
            "results": [
                {
                    "id": "video:123456",
                    "title": "No mp4 asset",
                    "duration": 20,
                    "url": "https://www.dvidshub.net/video/123456/no-mp4-asset",
                }
            ]
        }

        asset_response = MagicMock(status_code=200)
        asset_response.raise_for_status = Mock()
        asset_response.json.return_value = {
            "results": {
                "id": "video:123456",
                "duration": 20,
                "files": [
                    {
                        "src": "https://cdn.example.com/stream.m3u8",
                        "type": "application/vnd.apple.mpegurl",
                        "width": 1920,
                        "height": 1080,
                    }
                ],
            }
        }

        mock_session.get.side_effect = [search_response, asset_response]

        client = DvidsVideoClient(
            config=MagicMock(),
            output_dir=str(tmp_path),
            api_key="test_key",
        )

        results = client.search("carrier", max_results=1)

        assert results == []

    @patch("src.media_sources.base.requests.Session")
    @pytest.mark.fast
    def test_download_video_uses_numeric_id_suffix(self, mock_session_class, tmp_path):
        """Download filenames strip the DVIDS video: prefix for short IDs."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        download_response = MagicMock(status_code=200)
        download_response.raise_for_status = Mock()
        download_response.iter_content.return_value = [b"x" * 200000]
        mock_session.get.return_value = download_response

        client = DvidsVideoClient(
            config=MagicMock(),
            output_dir=str(tmp_path),
            api_key="test_key",
        )

        video = VideoResult(
            id="video:999196",
            source="dvids",
            url="https://www.dvidshub.net/video/999196/carrier-arrival",
            download_url="https://cdn.example.com/carrier-1080.mp4",
            width=1920,
            height=1080,
            duration=20.0,
            quality="hd",
            file_type="mp4",
        )

        file_path = client.download_video(video)

        assert file_path is not None
        assert Path(file_path).name == "d999196.mp4"
        assert Path(file_path).exists()


class TestDvidsVideoFallback:
    """Orchestrator tests for DVIDS fallback ordering."""

    @patch("src.media_sources.videos.orchestrator.build_entity_query")
    @patch("src.media_sources.videos.orchestrator.DvidsVideoClient")
    @patch("src.media_sources.videos.orchestrator.PixabayVideoClient")
    @patch("src.media_sources.videos.orchestrator.PexelsVideoClient")
    @pytest.mark.fast
    def test_dvids_fills_remaining_slots_after_stock_sources(
        self,
        mock_pexels_class,
        mock_pixabay_class,
        mock_dvids_class,
        mock_build_query,
        tmp_path,
    ):
        """DVIDS runs after Pexels/Pixabay and can fill remaining slots."""
        mock_build_query.return_value = "carrier arrival"

        mock_pexels = MagicMock(api_key="pexels_key")
        mock_pexels.search_and_download.return_value = ["/path/pexels.mp4"]
        mock_pexels_class.return_value = mock_pexels

        mock_pixabay = MagicMock(api_key="pixabay_key")
        mock_pixabay.search_and_download.return_value = []
        mock_pixabay_class.return_value = mock_pixabay

        mock_dvids = MagicMock(api_key="dvids_key")
        mock_dvids.search_and_download.return_value = ["/path/dvids.mp4"]
        mock_dvids_class.return_value = mock_dvids

        results = download_entity_videos(
            entities=[{"text": "Nimitz", "type": "ORG", "context": "aircraft carrier"}],
            output_dir=str(tmp_path),
            videos_per_entity=2,
            pexels_key="pexels_key",
            pixabay_key="pixabay_key",
            dvids_key="dvids_key",
        )

        assert "Nimitz" in results
        assert results["Nimitz"].videos == ["/path/pexels.mp4", "/path/dvids.mp4"]
        mock_pexels.search_and_download.assert_called_once()
        mock_pixabay.search_and_download.assert_called_once()
        mock_dvids.search_and_download.assert_called_once()

    @patch("src.media_sources.videos.orchestrator.build_entity_query")
    @patch("src.media_sources.videos.orchestrator.DvidsVideoClient")
    @patch("src.media_sources.videos.orchestrator.PixabayVideoClient")
    @patch("src.media_sources.videos.orchestrator.PexelsVideoClient")
    @pytest.mark.fast
    def test_dvids_only_mode_works_with_its_api_key(
        self,
        mock_pexels_class,
        mock_pixabay_class,
        mock_dvids_class,
        mock_build_query,
        tmp_path,
    ):
        """DVIDS can be used when stock provider keys are absent."""
        mock_build_query.return_value = "navy ship"

        mock_pexels_class.return_value = MagicMock(api_key=None)
        mock_pixabay_class.return_value = MagicMock(api_key=None)

        mock_dvids = MagicMock(api_key="dvids_key")
        mock_dvids.search_and_download.return_value = ["/path/dvids-only.mp4"]
        mock_dvids_class.return_value = mock_dvids

        results = download_entity_videos(
            entities=[{"text": "Destroyer", "type": "ORG", "context": "navy vessel"}],
            output_dir=str(tmp_path),
            videos_per_entity=1,
            dvids_key="dvids_key",
        )

        assert "Destroyer" in results
        assert results["Destroyer"].videos == ["/path/dvids-only.mp4"]
        mock_dvids.search_and_download.assert_called_once_with(
            query="navy ship",
            max_videos=1,
        )

    @patch("src.media_sources.videos.orchestrator.build_entity_query")
    @patch("src.media_sources.videos.orchestrator.DvidsVideoClient")
    @patch("src.media_sources.videos.orchestrator.PixabayVideoClient")
    @patch("src.media_sources.videos.orchestrator.PexelsVideoClient")
    @pytest.mark.fast
    def test_dvids_config_flag_blocks_provider_even_with_key(
        self,
        mock_pexels_class,
        mock_pixabay_class,
        mock_dvids_class,
        mock_build_query,
        tmp_path,
    ):
        """DVIDS should stay off when config disables it explicitly."""
        mock_build_query.return_value = "navy ship"

        mock_pexels_class.return_value = MagicMock(api_key=None)
        mock_pixabay_class.return_value = MagicMock(api_key=None)

        config = MagicMock()
        config.stock_footage = Mock(dvids_enabled=False)

        results = download_entity_videos(
            entities=[{"text": "Destroyer", "type": "ORG", "context": "navy vessel"}],
            output_dir=str(tmp_path),
            videos_per_entity=1,
            dvids_key="dvids_key",
            config=config,
        )

        assert results == {}
        mock_dvids_class.assert_not_called()

    @patch("src.media_sources.videos.orchestrator.build_entity_query")
    @patch("src.media_sources.videos.orchestrator.DvidsVideoClient")
    @patch("src.media_sources.videos.orchestrator.PixabayVideoClient")
    @patch("src.media_sources.videos.orchestrator.PexelsVideoClient")
    @pytest.mark.fast
    def test_stock_footage_master_flag_blocks_all_video_clients(
        self,
        mock_pexels_class,
        mock_pixabay_class,
        mock_dvids_class,
        mock_build_query,
        tmp_path,
    ):
        """The stock_footage.enabled master switch should bypass all providers."""
        mock_build_query.return_value = "navy ship"

        config = MagicMock()
        config.stock_footage = Mock(
            enabled=False,
            pexels_enabled=True,
            pixabay_enabled=True,
            dvids_enabled=True,
        )

        results = download_entity_videos(
            entities=[{"text": "Destroyer", "type": "ORG", "context": "navy vessel"}],
            output_dir=str(tmp_path),
            videos_per_entity=1,
            pexels_key="pexels_key",
            pixabay_key="pixabay_key",
            dvids_key="dvids_key",
            config=config,
        )

        assert results == {}
        mock_pexels_class.assert_not_called()
        mock_pixabay_class.assert_not_called()
        mock_dvids_class.assert_not_called()
