"""
Extended coverage tests for src/location_service.py

Covers:
- Lines 142-152: GeoLocation.from_geonames_result feature class handling
- Lines 186, 194: LocationCache serialize/deserialize
- Lines 476-483: LLM disambiguation in disambiguate()
- Lines 499-542: _disambiguate_with_llm method
- Line 625: is_parent_region name matching
- Lines 726-730: create_location_service config handling
- Lines 743-744: Exception in LLM client initialization

Created: 2026-01-11 (Session 15)
"""

import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.location_service import (
    GeoLocation,
    LocationCache,
    LocationService,
    create_location_service,
    COUNTRY_TO_CONTINENT
)
from src.cache import CacheEntry


# ============================================================================
# Test GeoLocation.from_geonames_result Feature Class Handling
# ============================================================================

class TestGeoLocationFromGeonames:
    """Test GeoLocation.from_geonames_result feature class variations"""

    def test_feature_class_a_country(self):
        """Test line 142: Administrative feature - country"""
        result = {
            "name": "France",
            "fcl": "A",
            "fcode": "PCLI",
            "countryCode": "FR",
            "countryName": "France",
            "lat": "46.0",
            "lng": "2.0"
        }
        loc = GeoLocation.from_geonames_result(result)
        assert loc.location_type == "country"
        assert loc.feature_class == "A"

    def test_feature_class_a_region(self):
        """Test line 142: Administrative feature - region"""
        result = {
            "name": "Ile-de-France",
            "fcl": "A",
            "fcode": "ADM1",
            "countryCode": "FR",
            "countryName": "France",
            "lat": "48.8",
            "lng": "2.3"
        }
        loc = GeoLocation.from_geonames_result(result)
        assert loc.location_type == "region"

    def test_feature_class_p_city(self):
        """Test lines 143-144: Populated place"""
        result = {
            "name": "Paris",
            "fcl": "P",
            "fcode": "PPLA",
            "countryCode": "FR",
            "countryName": "France",
            "lat": "48.85",
            "lng": "2.35",
            "population": "2200000"
        }
        loc = GeoLocation.from_geonames_result(result)
        assert loc.location_type == "city"

    def test_feature_class_t_natural(self):
        """Test lines 145-146: Terrain feature (mountain, hill)"""
        result = {
            "name": "Mont Blanc",
            "fcl": "T",
            "fcode": "MT",
            "countryCode": "FR",
            "countryName": "France",
            "lat": "45.83",
            "lng": "6.87"
        }
        loc = GeoLocation.from_geonames_result(result)
        assert loc.location_type == "natural_feature"

    def test_feature_class_h_water(self):
        """Test lines 147-148: Hydrographic feature (water body)"""
        result = {
            "name": "Lake Geneva",
            "fcl": "H",
            "fcode": "LK",
            "countryCode": "CH",
            "countryName": "Switzerland",
            "lat": "46.45",
            "lng": "6.53"
        }
        loc = GeoLocation.from_geonames_result(result)
        assert loc.location_type == "natural_feature"

    def test_feature_class_s_landmark(self):
        """Test lines 149-150: Spot, building, farm"""
        result = {
            "name": "Eiffel Tower",
            "fcl": "S",
            "fcode": "MNMT",
            "countryCode": "FR",
            "countryName": "France",
            "lat": "48.858",
            "lng": "2.294"
        }
        loc = GeoLocation.from_geonames_result(result)
        assert loc.location_type == "landmark"

    def test_feature_class_l_park(self):
        """Test lines 151-152: Parks, areas"""
        result = {
            "name": "Yellowstone",
            "fcl": "L",
            "fcode": "PRK",
            "countryCode": "US",
            "countryName": "United States",
            "lat": "44.4",
            "lng": "-110.5"
        }
        loc = GeoLocation.from_geonames_result(result)
        assert loc.location_type == "region"

    def test_feature_class_unknown(self):
        """Test lines 153-154: Unknown feature class"""
        result = {
            "name": "Unknown Place",
            "fcl": "X",
            "fcode": "XXX",
            "countryCode": "US",
            "countryName": "United States",
            "lat": "40.0",
            "lng": "-100.0"
        }
        loc = GeoLocation.from_geonames_result(result)
        assert loc.location_type == "other"

    def test_timezone_as_dict(self):
        """Test line 171: timezone as dict with timeZoneId"""
        result = {
            "name": "Paris",
            "fcl": "P",
            "fcode": "PPLA",
            "countryCode": "FR",
            "countryName": "France",
            "lat": "48.85",
            "lng": "2.35",
            "timezone": {"timeZoneId": "Europe/Paris"}
        }
        loc = GeoLocation.from_geonames_result(result)
        assert loc.timezone == "Europe/Paris"


# ============================================================================
# Test LocationCache Methods (Lines 186, 194)
# ============================================================================

class TestLocationCache:
    """Test LocationCache serialization/deserialization"""

    def test_serialize_entry(self, tmp_path):
        """Test line 186: _serialize_entry method"""
        cache = LocationCache(cache_dir=tmp_path, index_name="test_cache.json")

        entry = CacheEntry(
            key="test_location",
            data={"name": "Paris", "country": "FR"},
            cached_at="2026-01-11T10:00:00",
            metadata={"source": "geonames"}
        )

        serialized = cache._serialize_entry(entry)

        assert serialized['data'] == {"name": "Paris", "country": "FR"}
        assert serialized['cached_at'] == "2026-01-11T10:00:00"
        assert serialized['metadata'] == {"source": "geonames"}

    def test_deserialize_entry(self, tmp_path):
        """Test line 194: _deserialize_entry method"""
        cache = LocationCache(cache_dir=tmp_path, index_name="test_cache.json")

        data = {
            'data': {"name": "Paris", "country": "FR"},
            'cached_at': "2026-01-11T10:00:00",
            'metadata': {"source": "geonames"}
        }

        entry = cache._deserialize_entry(data)

        assert entry.data == {"name": "Paris", "country": "FR"}
        assert entry.cached_at == "2026-01-11T10:00:00"
        assert entry.metadata == {"source": "geonames"}

    def test_deserialize_entry_no_metadata(self, tmp_path):
        """Test _deserialize_entry with missing metadata"""
        cache = LocationCache(cache_dir=tmp_path, index_name="test_cache.json")

        data = {
            'data': {"name": "Paris"},
            'cached_at': "2026-01-11T10:00:00"
            # No metadata key
        }

        entry = cache._deserialize_entry(data)

        assert entry.metadata == {}

    def test_get_default_index(self, tmp_path):
        """Test _get_default_index structure"""
        cache = LocationCache(cache_dir=tmp_path, index_name="test_cache.json")

        default = cache._get_default_index()

        assert "locations" in default
        assert "disambiguations" in default
        assert isinstance(default["locations"], dict)
        assert isinstance(default["disambiguations"], dict)

    def test_count_entries(self, tmp_path):
        """Test _count_entries across both sections"""
        cache = LocationCache(cache_dir=tmp_path, index_name="test_cache.json")

        cache.index["locations"]["paris"] = {"data": {}}
        cache.index["locations"]["london"] = {"data": {}}
        cache.index["disambiguations"]["paris:france"] = "FR"

        count = cache._count_entries()
        assert count == 3


# ============================================================================
# Test LLM Disambiguation (Lines 476-483, 499-542)
# ============================================================================

class TestLLMDisambiguation:
    """Test LLM disambiguation methods"""

    def test_disambiguate_calls_llm(self, tmp_path):
        """Test lines 476-483: disambiguate() calls LLM when available"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "1"  # Select first option
        mock_llm.generate.return_value = mock_response

        service = LocationService(
            cache_dir=str(tmp_path),
            geonames_username="",
            llm_client=mock_llm
        )

        # Pre-populate cache with ambiguous location
        service._cache["locations"]["paris"] = {
            "results": [
                {
                    "name": "Paris",
                    "location_type": "city",
                    "country_code": "FR",
                    "country_name": "France",
                    "admin1": "Ile-de-France",
                    "admin2": "",
                    "coordinates": [48.85, 2.35],
                    "population": 2200000,
                    "geoname_id": 1,
                    "feature_class": "P",
                    "feature_code": "PPLA",
                    "timezone": ""
                },
                {
                    "name": "Paris",
                    "location_type": "city",
                    "country_code": "US",
                    "country_name": "United States",
                    "admin1": "Texas",
                    "admin2": "",
                    "coordinates": [33.66, -95.55],
                    "population": 25000,
                    "geoname_id": 2,
                    "feature_class": "P",
                    "feature_code": "PPL",
                    "timezone": ""
                }
            ],
            "cached_at": "2026-01-11"
        }

        # Context doesn't match any hints
        result = service.disambiguate("Paris", context="visit the beautiful city")

        # Should have called LLM
        assert mock_llm.generate.called

    def test_disambiguate_llm_exception(self, tmp_path):
        """Test lines 482-483: LLM disambiguation exception handling"""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("API error")

        service = LocationService(
            cache_dir=str(tmp_path),
            geonames_username="",
            llm_client=mock_llm
        )

        # Pre-populate cache
        service._cache["locations"]["paris"] = {
            "results": [
                {
                    "name": "Paris",
                    "location_type": "city",
                    "country_code": "FR",
                    "country_name": "France",
                    "admin1": "",
                    "admin2": "",
                    "coordinates": [48.85, 2.35],
                    "population": 2200000,
                    "geoname_id": 1,
                    "feature_class": "P",
                    "feature_code": "PPLA",
                    "timezone": ""
                },
                {
                    "name": "Paris",
                    "location_type": "city",
                    "country_code": "US",
                    "country_name": "United States",
                    "admin1": "Texas",
                    "admin2": "",
                    "coordinates": [33.66, -95.55],
                    "population": 25000,
                    "geoname_id": 2,
                    "feature_class": "P",
                    "feature_code": "PPL",
                    "timezone": ""
                }
            ],
            "cached_at": "2026-01-11"
        }

        # Should fall back to population
        result = service.disambiguate("Paris", context="some context")
        assert result is not None
        assert result.country_code == "FR"  # Larger population wins

    def test_disambiguate_with_llm_success(self, tmp_path):
        """Test lines 499-542: _disambiguate_with_llm method"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "2"  # Select second option
        mock_llm.generate.return_value = mock_response

        service = LocationService(
            cache_dir=str(tmp_path),
            geonames_username="",
            llm_client=mock_llm
        )

        candidates = [
            GeoLocation(
                name="Paris",
                location_type="city",
                country_code="FR",
                country_name="France",
                admin1="Ile-de-France",
                population=2200000
            ),
            GeoLocation(
                name="Paris",
                location_type="city",
                country_code="US",
                country_name="United States",
                admin1="Texas",
                population=25000
            )
        ]

        result = service._disambiguate_with_llm("Paris", "Texas rodeo", candidates)

        assert result is not None
        assert result.country_code == "US"  # Selected option 2

    def test_disambiguate_with_llm_invalid_response(self, tmp_path):
        """Test _disambiguate_with_llm with invalid response"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "invalid"  # No number
        mock_llm.generate.return_value = mock_response

        service = LocationService(
            cache_dir=str(tmp_path),
            geonames_username="",
            llm_client=mock_llm
        )

        candidates = [
            GeoLocation(name="Paris", location_type="city", country_code="FR", country_name="France")
        ]

        result = service._disambiguate_with_llm("Paris", "context", candidates)
        assert result is None  # Should return None for invalid response

    def test_disambiguate_with_llm_out_of_range(self, tmp_path):
        """Test _disambiguate_with_llm with out-of-range index"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "99"  # Out of range
        mock_llm.generate.return_value = mock_response

        service = LocationService(
            cache_dir=str(tmp_path),
            geonames_username="",
            llm_client=mock_llm
        )

        candidates = [
            GeoLocation(name="Paris", location_type="city", country_code="FR", country_name="France")
        ]

        result = service._disambiguate_with_llm("Paris", "context", candidates)
        assert result is None

    def test_disambiguate_with_llm_no_client(self, tmp_path):
        """Test _disambiguate_with_llm when no LLM client"""
        service = LocationService(
            cache_dir=str(tmp_path),
            geonames_username="",
            llm_client=None
        )

        candidates = [
            GeoLocation(name="Paris", location_type="city", country_code="FR", country_name="France")
        ]

        result = service._disambiguate_with_llm("Paris", "context", candidates)
        assert result is None


# ============================================================================
# Test is_parent_region (Line 625)
# ============================================================================

class TestIsParentRegion:
    """Test is_parent_region hierarchy matching"""

    def test_country_contains_city(self, tmp_path):
        """Test country is parent of city"""
        service = LocationService(cache_dir=str(tmp_path), geonames_username="")

        france = GeoLocation(
            name="France",
            location_type="country",
            country_code="FR",
            country_name="France"
        )
        paris = GeoLocation(
            name="Paris",
            location_type="city",
            country_code="FR",
            country_name="France",
            admin1="Ile-de-France"
        )

        assert service.is_parent_region(france, paris)

    def test_region_contains_city(self, tmp_path):
        """Test region is parent of city"""
        service = LocationService(cache_dir=str(tmp_path), geonames_username="")

        region = GeoLocation(
            name="Ile-de-France",
            location_type="region",
            country_code="FR",
            country_name="France",
            admin1="Ile-de-France"
        )
        paris = GeoLocation(
            name="Paris",
            location_type="city",
            country_code="FR",
            country_name="France",
            admin1="Ile-de-France"
        )

        assert service.is_parent_region(region, paris)

    def test_name_matching_in_hierarchy(self, tmp_path):
        """Test line 625: Name matching in parent_regions"""
        service = LocationService(cache_dir=str(tmp_path), geonames_username="")

        # Parent name appears in child's hierarchy
        parent = GeoLocation(
            name="Texas",
            location_type="region",
            country_code="US",
            country_name="United States",
            admin1=""  # No admin1 to trigger hierarchy name check
        )
        child = GeoLocation(
            name="Austin",
            location_type="city",
            country_code="US",
            country_name="United States",
            admin1="Texas"
        )

        # parent_regions includes admin1="Texas", so "texas" in "Texas".lower()
        assert service.is_parent_region(parent, child)

    def test_not_parent(self, tmp_path):
        """Test unrelated locations"""
        service = LocationService(cache_dir=str(tmp_path), geonames_username="")

        france = GeoLocation(
            name="France",
            location_type="country",
            country_code="FR",
            country_name="France"
        )
        london = GeoLocation(
            name="London",
            location_type="city",
            country_code="GB",
            country_name="United Kingdom",
            admin1="England"
        )

        assert not service.is_parent_region(france, london)


# ============================================================================
# Test create_location_service Factory (Lines 726-730, 743-744)
# ============================================================================

class TestCreateLocationService:
    """Test create_location_service factory function"""

    def test_config_is_none(self):
        """Test lines 724-727: location_config is None"""
        mock_config = MagicMock()
        mock_config.matching = MagicMock()
        mock_config.matching.location_matching = None
        mock_config.gemini_api_key = None

        service = create_location_service(mock_config)

        assert service is not None
        assert service.geonames_username == ""

    def test_config_is_dict(self):
        """Test lines 728-730: location_config is dict"""
        mock_config = MagicMock()
        mock_config.matching = MagicMock()
        mock_config.matching.location_matching = {
            'cache_dir': '/custom/cache',
            'geonames_username': 'testuser'
        }
        mock_config.gemini_api_key = None

        with patch('src.location_service.LocationService') as mock_service:
            mock_service.return_value = MagicMock()
            service = create_location_service(mock_config)

        mock_service.assert_called_once()
        call_kwargs = mock_service.call_args[1]
        assert call_kwargs['cache_dir'] == '/custom/cache'
        assert call_kwargs['geonames_username'] == 'testuser'

    def test_config_is_object(self):
        """Test lines 732-733: location_config is object"""
        mock_config = MagicMock()
        mock_config.matching = MagicMock()

        location_config = MagicMock()
        location_config.cache_dir = '/object/cache'
        location_config.geonames_username = 'objuser'
        mock_config.matching.location_matching = location_config
        mock_config.gemini_api_key = None

        with patch('src.location_service.LocationService') as mock_service:
            mock_service.return_value = MagicMock()
            service = create_location_service(mock_config)

        mock_service.assert_called_once()
        call_kwargs = mock_service.call_args[1]
        assert call_kwargs['cache_dir'] == '/object/cache'
        assert call_kwargs['geonames_username'] == 'objuser'

    def test_llm_client_creation_success(self):
        """Test lines 738-742: LLM client creation with API key"""
        mock_config = MagicMock()
        mock_config.matching = MagicMock()
        mock_config.matching.location_matching = None
        mock_config.gemini_api_key = "test-api-key"

        with patch('src.llm_client.create_client') as mock_create:
            mock_create.return_value = MagicMock()
            service = create_location_service(mock_config)

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args
        assert call_kwargs[0][0] == "gemini"
        assert call_kwargs[1]['api_key'] == "test-api-key"

    def test_llm_client_creation_exception(self):
        """Test lines 743-744: Exception in LLM client creation"""
        mock_config = MagicMock()
        mock_config.matching = MagicMock()
        mock_config.matching.location_matching = None
        mock_config.gemini_api_key = "test-api-key"

        with patch('src.llm_client.create_client', side_effect=Exception("API error")):
            # Should not raise, just log and continue
            service = create_location_service(mock_config)

        assert service is not None
        assert service.llm_client is None


# ============================================================================
# Test GeoLocation Properties and Methods
# ============================================================================

class TestGeoLocationProperties:
    """Test GeoLocation dataclass properties"""

    def test_continent_property_known(self):
        """Test continent property with known country"""
        loc = GeoLocation(
            name="Paris",
            location_type="city",
            country_code="FR",
            country_name="France"
        )
        assert loc.continent == "Europe"

    def test_continent_property_unknown(self):
        """Test continent property with unknown country"""
        loc = GeoLocation(
            name="Unknown City",
            location_type="city",
            country_code="XX",
            country_name="Unknown"
        )
        assert loc.continent == "Unknown"

    def test_parent_regions_full(self):
        """Test parent_regions with all levels"""
        loc = GeoLocation(
            name="Paris",
            location_type="city",
            country_code="FR",
            country_name="France",
            admin1="Ile-de-France",
            admin2="Paris"
        )
        regions = loc.parent_regions
        assert "Paris" in regions
        assert "Ile-de-France" in regions
        assert "France" in regions
        assert "Europe" in regions

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip"""
        original = GeoLocation(
            name="Paris",
            location_type="city",
            country_code="FR",
            country_name="France",
            admin1="Ile-de-France",
            admin2="Paris",
            coordinates=(48.85, 2.35),
            population=2200000,
            geoname_id=12345,
            feature_class="P",
            feature_code="PPLA",
            timezone="Europe/Paris"
        )

        data = original.to_dict()
        restored = GeoLocation.from_dict(data)

        assert restored.name == original.name
        assert restored.country_code == original.country_code
        assert restored.coordinates == original.coordinates
        assert restored.population == original.population


# ============================================================================
# Test API Error Handling
# ============================================================================

class TestAPIErrorHandling:
    """Test GeoNames API error handling"""

    def test_api_call_401_unauthorized(self, tmp_path):
        """Test lines 321-328: 401 Unauthorized handling"""
        service = LocationService(
            cache_dir=str(tmp_path),
            geonames_username="testuser"
        )

        # Create a proper RequestException with response attribute
        mock_response = MagicMock()
        mock_response.status_code = 401

        from requests.exceptions import HTTPError
        error = HTTPError("401 Unauthorized")
        error.response = mock_response

        with patch('requests.get', side_effect=error):
            result = service._call_geonames_api("searchJSON", {"q": "Paris"})

        assert result is None

    def test_api_call_json_decode_error(self, tmp_path):
        """Test lines 332-334: JSON decode error"""
        service = LocationService(
            cache_dir=str(tmp_path),
            geonames_username="testuser"
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("msg", "doc", 0)

        with patch('requests.get', return_value=mock_response):
            result = service._call_geonames_api("searchJSON", {"q": "Paris"})

        assert result is None

    def test_api_call_status_in_response(self, tmp_path):
        """Test lines 314-316: API error status in response"""
        service = LocationService(
            cache_dir=str(tmp_path),
            geonames_username="testuser"
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "status": {"message": "Rate limit exceeded", "value": 18}
        }

        with patch('requests.get', return_value=mock_response):
            result = service._call_geonames_api("searchJSON", {"q": "Paris"})

        assert result is None


# ============================================================================
# Test Distance Calculation
# ============================================================================

class TestDistanceCalculation:
    """Test Haversine distance calculation"""

    def test_distance_same_point(self, tmp_path):
        """Test distance between same coordinates"""
        service = LocationService(cache_dir=str(tmp_path), geonames_username="")

        loc = GeoLocation(
            name="Paris",
            location_type="city",
            country_code="FR",
            country_name="France",
            coordinates=(48.85, 2.35)
        )

        distance = service.distance_km(loc, loc)
        assert distance == 0.0

    def test_distance_different_cities(self, tmp_path):
        """Test distance between different cities"""
        service = LocationService(cache_dir=str(tmp_path), geonames_username="")

        paris = GeoLocation(
            name="Paris",
            location_type="city",
            country_code="FR",
            country_name="France",
            coordinates=(48.85, 2.35)
        )
        london = GeoLocation(
            name="London",
            location_type="city",
            country_code="GB",
            country_name="United Kingdom",
            coordinates=(51.51, -0.13)
        )

        distance = service.distance_km(paris, london)
        # Paris to London is roughly 340km
        assert 300 < distance < 400
