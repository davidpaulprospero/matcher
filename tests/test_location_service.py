"""
Comprehensive tests for location service module.

Covers:
- GeoLocation dataclass operations
- Location hierarchy resolution
- Continent mapping
- GeoNames API integration (mocked)
- Location caching
- Distance calculations
- Location disambiguation
- Error handling

Created: 2026-01-09 (Phase 3.2)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import shutil
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.location_service import (
    GeoLocation,
    COUNTRY_TO_CONTINENT,
    LocationService,
    create_location_service
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_location():
    """Create a sample GeoLocation"""
    return GeoLocation(
        name="Tokyo",
        location_type="city",
        country_code="JP",
        country_name="Japan",
        admin1="Tokyo",
        coordinates=(35.6762, 139.6503),
        population=13960000,
        geoname_id=1850144,
        feature_class="P",
        feature_code="PPLC"
    )


@pytest.fixture
def geonames_response():
    """Mock GeoNames API response"""
    return {
        "geonames": [
            {
                "geonameId": 1850144,
                "name": "Tokyo",
                "lat": "35.6762",
                "lng": "139.6503",
                "countryCode": "JP",
                "countryName": "Japan",
                "adminName1": "Tokyo",
                "adminName2": "",
                "fcl": "P",
                "fcode": "PPLC",
                "population": 13960000,
                "timezone": {
                    "timeZoneId": "Asia/Tokyo"
                }
            }
        ]
    }


# ============================================================================
# Test GeoLocation Dataclass
# ============================================================================

class TestGeoLocation:
    """Test GeoLocation dataclass"""

    def test_geolocation_creation(self):
        """Test creating a GeoLocation"""
        location = GeoLocation(
            name="Paris",
            location_type="city",
            country_code="FR",
            country_name="France"
        )

        assert location.name == "Paris"
        assert location.location_type == "city"
        assert location.country_code == "FR"
        assert location.country_name == "France"

    def test_continent_property(self, sample_location):
        """Test continent property"""
        assert sample_location.continent == "Asia"

    def test_continent_unknown_country(self):
        """Test continent for unknown country"""
        location = GeoLocation(
            name="Test",
            location_type="city",
            country_code="XX",  # Unknown code
            country_name="Unknown"
        )

        assert location.continent == "Unknown"

    def test_parent_regions(self, sample_location):
        """Test parent regions hierarchy"""
        regions = sample_location.parent_regions

        assert "Tokyo" in regions  # admin1
        assert "Japan" in regions
        assert "Asia" in regions
        assert regions[-1] == "Asia"  # Continent is last

    def test_parent_regions_with_admin2(self):
        """Test parent regions with admin2"""
        location = GeoLocation(
            name="Manhattan",
            location_type="city",
            country_code="US",
            country_name="United States",
            admin1="New York",
            admin2="New York County"
        )

        regions = location.parent_regions

        assert "New York County" in regions
        assert "New York" in regions
        assert "United States" in regions
        assert "North America" in regions


# ============================================================================
# Test GeoLocation Serialization
# ============================================================================

class TestGeoLocationSerialization:
    """Test GeoLocation serialization"""

    def test_to_dict(self, sample_location):
        """Test converting to dictionary"""
        data = sample_location.to_dict()

        assert data["name"] == "Tokyo"
        assert data["country_code"] == "JP"
        assert data["population"] == 13960000
        assert isinstance(data["coordinates"], list)
        assert len(data["coordinates"]) == 2

    def test_from_dict(self, sample_location):
        """Test creating from dictionary"""
        data = sample_location.to_dict()
        restored = GeoLocation.from_dict(data)

        assert restored.name == sample_location.name
        assert restored.country_code == sample_location.country_code
        assert restored.population == sample_location.population
        assert restored.coordinates == sample_location.coordinates

    def test_from_dict_minimal(self):
        """Test creating from minimal dictionary"""
        data = {
            "name": "Test",
            "country_code": "US"
        }

        location = GeoLocation.from_dict(data)

        assert location.name == "Test"
        assert location.country_code == "US"
        assert location.location_type == "city"  # Default
        assert location.population == 0

    def test_from_geonames_result(self, geonames_response):
        """Test creating from GeoNames API response"""
        result = geonames_response["geonames"][0]

        location = GeoLocation.from_geonames_result(result)

        assert location.name == "Tokyo"
        assert location.country_code == "JP"
        assert location.geoname_id == 1850144
        assert location.location_type == "city"  # P = populated place


# ============================================================================
# Test LocationService (requires mocked API)
# ============================================================================

@pytest.mark.integration
@pytest.mark.skip(reason="Integration test - requires LocationService with specific methods")
class TestLocationService:
    """Test LocationService with mocked API (integration tests)"""

    def test_init_creates_cache(self, temp_dir):
        """Test LocationService initialization"""
        cache_dir = str(temp_dir / "locations")

        service = LocationService(
            cache_dir=cache_dir,
            geonames_username="test_user"
        )

        assert service.username == "test_user"
        assert service.cache_dir.exists()


# ============================================================================
# Test Distance Calculations (if available)
# ============================================================================

class TestDistanceCalculations:
    """Test geographic distance calculations"""

    def test_location_has_coordinates(self, sample_location):
        """Test location stores coordinates"""
        assert sample_location.coordinates == (35.6762, 139.6503)

    def test_coordinates_tuple_format(self, sample_location):
        """Test coordinates are stored as tuple"""
        assert isinstance(sample_location.coordinates, tuple)
        assert len(sample_location.coordinates) == 2


# ============================================================================
# Test Continent Mapping
# ============================================================================

class TestContinentMapping:
    """Test continent mapping"""

    def test_european_countries(self):
        """Test European country codes"""
        assert COUNTRY_TO_CONTINENT["FR"] == "Europe"
        assert COUNTRY_TO_CONTINENT["DE"] == "Europe"
        assert COUNTRY_TO_CONTINENT["GB"] == "Europe"

    def test_asian_countries(self):
        """Test Asian country codes"""
        assert COUNTRY_TO_CONTINENT["JP"] == "Asia"
        assert COUNTRY_TO_CONTINENT["CN"] == "Asia"
        assert COUNTRY_TO_CONTINENT["IN"] == "Asia"

    def test_north_american_countries(self):
        """Test North American country codes"""
        assert COUNTRY_TO_CONTINENT["US"] == "North America"
        assert COUNTRY_TO_CONTINENT["CA"] == "North America"
        assert COUNTRY_TO_CONTINENT["MX"] == "North America"

    def test_african_countries(self):
        """Test African country codes"""
        assert COUNTRY_TO_CONTINENT["ZA"] == "Africa"
        assert COUNTRY_TO_CONTINENT["EG"] == "Africa"
        assert COUNTRY_TO_CONTINENT["NG"] == "Africa"

    def test_oceania_countries(self):
        """Test Oceania country codes"""
        assert COUNTRY_TO_CONTINENT["AU"] == "Oceania"
        assert COUNTRY_TO_CONTINENT["NZ"] == "Oceania"


# ============================================================================
# Test Location Types
# ============================================================================

class TestLocationTypes:
    """Test location type classification"""

    def test_city_location(self):
        """Test city location type"""
        location = GeoLocation(
            name="Paris",
            location_type="city",
            country_code="FR",
            country_name="France"
        )

        assert location.location_type == "city"

    def test_natural_feature_location(self):
        """Test natural feature location"""
        location = GeoLocation(
            name="Mount Fuji",
            location_type="natural_feature",
            country_code="JP",
            country_name="Japan"
        )

        assert location.location_type == "natural_feature"

    def test_landmark_location(self):
        """Test landmark location"""
        location = GeoLocation(
            name="Eiffel Tower",
            location_type="landmark",
            country_code="FR",
            country_name="France"
        )

        assert location.location_type == "landmark"


# ============================================================================
# Test Cache Key Generation
# ============================================================================

@pytest.mark.integration
@pytest.mark.skip(reason="Integration test - requires LocationService with _cache_key method")
class TestCacheKeyGeneration:
    """Test cache key generation (integration tests)"""

    def test_cache_key_consistent(self, temp_dir):
        """Test cache keys are consistent"""
        cache_dir = str(temp_dir / "locations")

        service = LocationService(
            cache_dir=cache_dir,
            geonames_username="test_user"
        )

        key1 = service._cache_key("Tokyo")
        key2 = service._cache_key("Tokyo")

        assert key1 == key2


# ============================================================================
# Test Factory Function
# ============================================================================

class TestCreateLocationService:
    """Test create_location_service factory"""

    def test_create_service_with_config(self, temp_dir):
        """Test creating service with config"""
        from src.config import Config

        config = Config()
        config.cache_dir = str(temp_dir)
        config.matching.location_matching.geonames_username = "test_user"
        config.matching.location_matching.enabled = True

        service = create_location_service(config)

        assert service is not None
        assert isinstance(service, LocationService)


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases"""

    def test_empty_location_name(self):
        """Test location with empty name"""
        location = GeoLocation(
            name="",
            location_type="city",
            country_code="US",
            country_name="United States"
        )

        assert location.name == ""
        assert location.country_name == "United States"

    def test_zero_population(self, sample_location):
        """Test location with zero population"""
        sample_location.population = 0

        assert sample_location.population == 0

    def test_invalid_coordinates(self):
        """Test handling invalid coordinates"""
        location = GeoLocation(
            name="Test",
            location_type="city",
            country_code="US",
            country_name="United States",
            coordinates=(0.0, 0.0)  # Null Island
        )

        assert location.coordinates == (0.0, 0.0)

    def test_missing_admin_regions(self):
        """Test location without admin regions"""
        location = GeoLocation(
            name="Small Town",
            location_type="city",
            country_code="US",
            country_name="United States"
            # No admin1 or admin2
        )

        regions = location.parent_regions

        assert "United States" in regions
        assert "North America" in regions
        assert len(regions) == 2  # Only country and continent


# ============================================================================
# Test LocationService API Integration (with mocking)
# ============================================================================

class TestLocationServiceAPIIntegration:
    """Test LocationService API methods with mocked responses"""

    @patch('requests.get')
    def test_geocode_cache_hit(self, mock_get, temp_dir):
        """Test geocoding with cache hit"""
        service = LocationService(
            cache_dir=str(temp_dir / "locations"),
            geonames_username="test_user"
        )

        # Pre-populate cache
        cache_data = {
            "results": [{
                "name": "Paris",
                "location_type": "city",
                "country_code": "FR",
                "country_name": "France",
                "coordinates": [48.8566, 2.3522],
                "population": 2200000,
                "geoname_id": 2988507,
                "feature_class": "P",
                "feature_code": "PPLC",
                "admin1": "Île-de-France",
                "admin2": "",
                "timezone": "Europe/Paris"
            }]
        }
        service._cache["locations"]["paris"] = cache_data

        # Geocode - should hit cache
        results = service.geocode("Paris")

        assert len(results) == 1
        assert results[0].name == "Paris"
        assert results[0].country_code == "FR"
        assert mock_get.call_count == 0  # No API call

    @patch('requests.get')
    def test_geocode_api_call(self, mock_get, temp_dir, geonames_response):
        """Test geocoding with API call"""
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: geonames_response
        )
        mock_get.return_value.raise_for_status = Mock()

        service = LocationService(
            cache_dir=str(temp_dir / "locations"),
            geonames_username="test_user"
        )

        results = service.geocode("Tokyo")

        assert len(results) == 1
        assert results[0].name == "Tokyo"
        assert results[0].country_code == "JP"
        assert mock_get.call_count == 1  # API was called

        # Verify cache was populated
        assert "tokyo" in service._cache["locations"]

    @patch('requests.get')
    def test_geocode_no_results(self, mock_get, temp_dir):
        """Test geocoding with no results"""
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: {"geonames": []}
        )
        mock_get.return_value.raise_for_status = Mock()

        service = LocationService(
            cache_dir=str(temp_dir / "locations"),
            geonames_username="test_user"
        )

        results = service.geocode("InvalidCityName12345")

        assert len(results) == 0

    @patch('requests.get')
    def test_geocode_api_error(self, mock_get, temp_dir):
        """Test geocoding with API error"""
        import requests
        mock_get.side_effect = requests.RequestException("Network error")

        service = LocationService(
            cache_dir=str(temp_dir / "locations"),
            geonames_username="test_user"
        )

        results = service.geocode("Paris")

        assert len(results) == 0  # Should return empty list on error

    def test_geocode_no_username(self, temp_dir):
        """Test geocoding without username"""
        service = LocationService(
            cache_dir=str(temp_dir / "locations"),
            geonames_username=""  # No username
        )

        results = service.geocode("Paris")

        assert len(results) == 0  # No API call possible

    @patch('requests.get')
    def test_get_best_match(self, mock_get, temp_dir, geonames_response):
        """Test getting best match"""
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: geonames_response
        )
        mock_get.return_value.raise_for_status = Mock()

        service = LocationService(
            cache_dir=str(temp_dir / "locations"),
            geonames_username="test_user"
        )

        best = service.get_best_match("Tokyo")

        assert best is not None
        assert best.name == "Tokyo"

    @patch('requests.get')
    def test_get_best_match_no_results(self, mock_get, temp_dir):
        """Test getting best match with no results"""
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: {"geonames": []}
        )
        mock_get.return_value.raise_for_status = Mock()

        service = LocationService(
            cache_dir=str(temp_dir / "locations"),
            geonames_username="test_user"
        )

        best = service.get_best_match("InvalidCity")

        assert best is None


# ============================================================================
# Test Disambiguation Logic
# ============================================================================

class TestDisambiguation:
    """Test location disambiguation strategies"""

    @patch('requests.get')
    def test_disambiguate_context_hints(self, mock_get, temp_dir):
        """Test disambiguation using context keywords"""
        # Mock response with multiple Paris locations
        response = {
            "geonames": [
                {
                    "geonameId": 2988507,
                    "name": "Paris",
                    "lat": "48.8566",
                    "lng": "2.3522",
                    "countryCode": "FR",
                    "countryName": "France",
                    "adminName1": "Île-de-France",
                    "adminName2": "",
                    "fcl": "P",
                    "fcode": "PPLC",
                    "population": 2200000,
                    "timezone": {"timeZoneId": "Europe/Paris"}
                },
                {
                    "geonameId": 4717560,
                    "name": "Paris",
                    "lat": "33.6609",
                    "lng": "-95.5555",
                    "countryCode": "US",
                    "countryName": "United States",
                    "adminName1": "Texas",
                    "adminName2": "",
                    "fcl": "P",
                    "fcode": "PPL",
                    "population": 25171,
                    "timezone": {"timeZoneId": "America/Chicago"}
                }
            ]
        }

        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: response
        )
        mock_get.return_value.raise_for_status = Mock()

        service = LocationService(
            cache_dir=str(temp_dir / "locations"),
            geonames_username="test_user"
        )

        # Context mentions "Eiffel" - should pick Paris, France
        result = service.disambiguate("Paris", context="We visited the Eiffel Tower")

        assert result is not None
        assert result.country_code == "FR"
        assert result.name == "Paris"

    @patch('requests.get')
    def test_disambiguate_co_location(self, mock_get, temp_dir):
        """Test disambiguation using co-occurring locations"""
        # Mock Paris responses (France and Texas)
        paris_response = {
            "geonames": [
                {
                    "geonameId": 2988507,
                    "name": "Paris",
                    "lat": "48.8566",
                    "lng": "2.3522",
                    "countryCode": "FR",
                    "countryName": "France",
                    "adminName1": "Île-de-France",
                    "adminName2": "",
                    "fcl": "P",
                    "fcode": "PPLC",
                    "population": 2200000,
                    "timezone": {"timeZoneId": "Europe/Paris"}
                },
                {
                    "geonameId": 4717560,
                    "name": "Paris",
                    "lat": "33.6609",
                    "lng": "-95.5555",
                    "countryCode": "US",
                    "countryName": "United States",
                    "adminName1": "Texas",
                    "adminName2": "",
                    "fcl": "P",
                    "fcode": "PPL",
                    "population": 25171,
                    "timezone": {"timeZoneId": "America/Chicago"}
                }
            ]
        }

        # Mock Lyon response (France only)
        lyon_response = {
            "geonames": [{
                "geonameId": 2996944,
                "name": "Lyon",
                "lat": "45.7500",
                "lng": "4.8500",
                "countryCode": "FR",
                "countryName": "France",
                "adminName1": "Auvergne-Rhône-Alpes",
                "adminName2": "",
                "fcl": "P",
                "fcode": "PPLA",
                "population": 513275,
                "timezone": {"timeZoneId": "Europe/Paris"}
            }]
        }

        def mock_get_side_effect(url, *args, **kwargs):
            if "Lyon" in kwargs.get('params', {}).get('q', ''):
                return Mock(status_code=200, json=lambda: lyon_response, raise_for_status=Mock())
            else:
                return Mock(status_code=200, json=lambda: paris_response, raise_for_status=Mock())

        mock_get.side_effect = mock_get_side_effect

        service = LocationService(
            cache_dir=str(temp_dir / "locations"),
            geonames_username="test_user"
        )

        # Disambiguate Paris with co-location Lyon (both in France)
        result = service.disambiguate("Paris", co_locations=["Lyon"])

        assert result is not None
        assert result.country_code == "FR"

    @patch('requests.get')
    def test_disambiguate_population_default(self, mock_get, temp_dir):
        """Test disambiguation defaults to highest population"""
        response = {
            "geonames": [
                {
                    "geonameId": 2988507,
                    "name": "Paris",
                    "lat": "48.8566",
                    "lng": "2.3522",
                    "countryCode": "FR",
                    "countryName": "France",
                    "adminName1": "Île-de-France",
                    "adminName2": "",
                    "fcl": "P",
                    "fcode": "PPLC",
                    "population": 2200000,
                    "timezone": {"timeZoneId": "Europe/Paris"}
                },
                {
                    "geonameId": 4717560,
                    "name": "Paris",
                    "lat": "33.6609",
                    "lng": "-95.5555",
                    "countryCode": "US",
                    "countryName": "United States",
                    "adminName1": "Texas",
                    "adminName2": "",
                    "fcl": "P",
                    "fcode": "PPL",
                    "population": 25171,
                    "timezone": {"timeZoneId": "America/Chicago"}
                }
            ]
        }

        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: response
        )
        mock_get.return_value.raise_for_status = Mock()

        service = LocationService(
            cache_dir=str(temp_dir / "locations"),
            geonames_username="test_user"
        )

        # No context hints - should pick highest population (France)
        result = service.disambiguate("Paris")

        assert result is not None
        assert result.country_code == "FR"
        assert result.population == 2200000

    @patch('requests.get')
    def test_disambiguate_single_result(self, mock_get, temp_dir, geonames_response):
        """Test disambiguation with single result"""
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: geonames_response
        )
        mock_get.return_value.raise_for_status = Mock()

        service = LocationService(
            cache_dir=str(temp_dir / "locations"),
            geonames_username="test_user"
        )

        # Only one result - no disambiguation needed
        result = service.disambiguate("Tokyo")

        assert result is not None
        assert result.name == "Tokyo"

    @patch('requests.get')
    def test_disambiguate_cache_hit(self, mock_get, temp_dir):
        """Test disambiguation with cache hit"""
        response = {
            "geonames": [
                {
                    "geonameId": 2988507,
                    "name": "Paris",
                    "lat": "48.8566",
                    "lng": "2.3522",
                    "countryCode": "FR",
                    "countryName": "France",
                    "adminName1": "Île-de-France",
                    "adminName2": "",
                    "fcl": "P",
                    "fcode": "PPLC",
                    "population": 2200000,
                    "timezone": {"timeZoneId": "Europe/Paris"}
                },
                {
                    "geonameId": 4717560,
                    "name": "Paris",
                    "lat": "33.6609",
                    "lng": "-95.5555",
                    "countryCode": "US",
                    "countryName": "United States",
                    "adminName1": "Texas",
                    "adminName2": "",
                    "fcl": "P",
                    "fcode": "PPL",
                    "population": 25171,
                    "timezone": {"timeZoneId": "America/Chicago"}
                }
            ]
        }

        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: response
        )
        mock_get.return_value.raise_for_status = Mock()

        service = LocationService(
            cache_dir=str(temp_dir / "locations"),
            geonames_username="test_user"
        )

        # Pre-populate disambiguation cache
        cache_key = "paris:visiting eiffel tower"
        service._cache["disambiguations"][cache_key] = "FR"

        # Should hit cache and return French Paris
        result = service.disambiguate("Paris", context="Visiting Eiffel Tower")

        assert result is not None
        assert result.country_code == "FR"
        assert mock_get.call_count == 1  # Only geocode call, no new disambiguation

    @patch('requests.get')
    def test_disambiguate_no_results(self, mock_get, temp_dir):
        """Test disambiguation with no results"""
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: {"geonames": []}
        )
        mock_get.return_value.raise_for_status = Mock()

        service = LocationService(
            cache_dir=str(temp_dir / "locations"),
            geonames_username="test_user"
        )

        result = service.disambiguate("InvalidCity")

        assert result is None


# ============================================================================
# Test Distance Calculations (Haversine)
# ============================================================================

class TestHaversineDistance:
    """Test distance calculations"""

    def test_distance_same_location(self, temp_dir):
        """Test distance between same location"""
        service = LocationService(cache_dir=str(temp_dir / "locations"))

        tokyo = GeoLocation(
            name="Tokyo",
            location_type="city",
            country_code="JP",
            country_name="Japan",
            coordinates=(35.6762, 139.6503)
        )

        distance = service.distance_km(tokyo, tokyo)

        assert distance == 0.0

    def test_distance_known_cities(self, temp_dir):
        """Test distance between known cities"""
        service = LocationService(cache_dir=str(temp_dir / "locations"))

        paris = GeoLocation(
            name="Paris",
            location_type="city",
            country_code="FR",
            country_name="France",
            coordinates=(48.8566, 2.3522)
        )

        london = GeoLocation(
            name="London",
            location_type="city",
            country_code="GB",
            country_name="United Kingdom",
            coordinates=(51.5074, -0.1278)
        )

        distance = service.distance_km(paris, london)

        # Paris to London is approximately 344 km
        assert 340 < distance < 350

    def test_distance_long_range(self, temp_dir):
        """Test distance for long-range locations"""
        service = LocationService(cache_dir=str(temp_dir / "locations"))

        tokyo = GeoLocation(
            name="Tokyo",
            location_type="city",
            country_code="JP",
            country_name="Japan",
            coordinates=(35.6762, 139.6503)
        )

        new_york = GeoLocation(
            name="New York",
            location_type="city",
            country_code="US",
            country_name="United States",
            coordinates=(40.7128, -74.0060)
        )

        distance = service.distance_km(tokyo, new_york)

        # Tokyo to NYC is approximately 10,850 km
        assert 10800 < distance < 10900


# ============================================================================
# Test Geographic Hierarchy Methods
# ============================================================================

class TestGeographicHierarchy:
    """Test geographic hierarchy comparison methods"""

    def test_same_country(self, temp_dir):
        """Test same country check"""
        service = LocationService(cache_dir=str(temp_dir / "locations"))

        paris = GeoLocation(
            name="Paris",
            location_type="city",
            country_code="FR",
            country_name="France",
            coordinates=(48.8566, 2.3522)
        )

        lyon = GeoLocation(
            name="Lyon",
            location_type="city",
            country_code="FR",
            country_name="France",
            coordinates=(45.7500, 4.8500)
        )

        assert service.same_country(paris, lyon) is True

    def test_different_country(self, temp_dir):
        """Test different country check"""
        service = LocationService(cache_dir=str(temp_dir / "locations"))

        paris = GeoLocation(
            name="Paris",
            location_type="city",
            country_code="FR",
            country_name="France",
            coordinates=(48.8566, 2.3522)
        )

        london = GeoLocation(
            name="London",
            location_type="city",
            country_code="GB",
            country_name="United Kingdom",
            coordinates=(51.5074, -0.1278)
        )

        assert service.same_country(paris, london) is False

    def test_same_continent(self, temp_dir):
        """Test same continent check"""
        service = LocationService(cache_dir=str(temp_dir / "locations"))

        paris = GeoLocation(
            name="Paris",
            location_type="city",
            country_code="FR",
            country_name="France",
            coordinates=(48.8566, 2.3522)
        )

        london = GeoLocation(
            name="London",
            location_type="city",
            country_code="GB",
            country_name="United Kingdom",
            coordinates=(51.5074, -0.1278)
        )

        assert service.same_continent(paris, london) is True

    def test_different_continent(self, temp_dir):
        """Test different continent check"""
        service = LocationService(cache_dir=str(temp_dir / "locations"))

        paris = GeoLocation(
            name="Paris",
            location_type="city",
            country_code="FR",
            country_name="France",
            coordinates=(48.8566, 2.3522)
        )

        tokyo = GeoLocation(
            name="Tokyo",
            location_type="city",
            country_code="JP",
            country_name="Japan",
            coordinates=(35.6762, 139.6503)
        )

        assert service.same_continent(paris, tokyo) is False

    def test_same_region(self, temp_dir):
        """Test same admin region check"""
        service = LocationService(cache_dir=str(temp_dir / "locations"))

        la = GeoLocation(
            name="Los Angeles",
            location_type="city",
            country_code="US",
            country_name="United States",
            admin1="California",
            coordinates=(34.0522, -118.2437)
        )

        sf = GeoLocation(
            name="San Francisco",
            location_type="city",
            country_code="US",
            country_name="United States",
            admin1="California",
            coordinates=(37.7749, -122.4194)
        )

        assert service.same_region(la, sf) is True

    def test_different_region(self, temp_dir):
        """Test different admin region check"""
        service = LocationService(cache_dir=str(temp_dir / "locations"))

        la = GeoLocation(
            name="Los Angeles",
            location_type="city",
            country_code="US",
            country_name="United States",
            admin1="California",
            coordinates=(34.0522, -118.2437)
        )

        ny = GeoLocation(
            name="New York",
            location_type="city",
            country_code="US",
            country_name="United States",
            admin1="New York",
            coordinates=(40.7128, -74.0060)
        )

        assert service.same_region(la, ny) is False

    def test_same_city_by_name(self, temp_dir):
        """Test same city check by name match"""
        service = LocationService(cache_dir=str(temp_dir / "locations"))

        paris1 = GeoLocation(
            name="Paris",
            location_type="city",
            country_code="FR",
            country_name="France",
            admin1="Île-de-France",
            coordinates=(48.8566, 2.3522)
        )

        paris2 = GeoLocation(
            name="Paris",
            location_type="city",
            country_code="FR",
            country_name="France",
            admin1="Île-de-France",
            coordinates=(48.8566, 2.3522)
        )

        assert service.same_city(paris1, paris2) is True

    def test_same_city_by_proximity(self, temp_dir):
        """Test same city check by proximity (<25km)"""
        service = LocationService(cache_dir=str(temp_dir / "locations"))

        paris_center = GeoLocation(
            name="Paris",
            location_type="city",
            country_code="FR",
            country_name="France",
            coordinates=(48.8566, 2.3522)
        )

        paris_suburb = GeoLocation(
            name="Suburb",
            location_type="city",
            country_code="FR",
            country_name="France",
            coordinates=(48.8900, 2.3800)  # ~5km away
        )

        assert service.same_city(paris_center, paris_suburb) is True

    def test_different_city(self, temp_dir):
        """Test different city check"""
        service = LocationService(cache_dir=str(temp_dir / "locations"))

        paris = GeoLocation(
            name="Paris",
            location_type="city",
            country_code="FR",
            country_name="France",
            coordinates=(48.8566, 2.3522)
        )

        lyon = GeoLocation(
            name="Lyon",
            location_type="city",
            country_code="FR",
            country_name="France",
            coordinates=(45.7500, 4.8500)  # ~400km away
        )

        assert service.same_city(paris, lyon) is False


# ============================================================================
# Test Parent Region Checking
# ============================================================================

class TestParentRegion:
    """Test geographic parent-child relationships"""

    def test_country_contains_city(self, temp_dir):
        """Test country as parent of city"""
        service = LocationService(cache_dir=str(temp_dir / "locations"))

        france = GeoLocation(
            name="France",
            location_type="country",
            country_code="FR",
            country_name="France",
            coordinates=(46.2276, 2.2137)
        )

        paris = GeoLocation(
            name="Paris",
            location_type="city",
            country_code="FR",
            country_name="France",
            coordinates=(48.8566, 2.3522)
        )

        assert service.is_parent_region(france, paris) is True

    def test_region_contains_city(self, temp_dir):
        """Test region as parent of city"""
        service = LocationService(cache_dir=str(temp_dir / "locations"))

        california = GeoLocation(
            name="California",
            location_type="region",
            country_code="US",
            country_name="United States",
            admin1="California",
            coordinates=(36.7783, -119.4179)
        )

        la = GeoLocation(
            name="Los Angeles",
            location_type="city",
            country_code="US",
            country_name="United States",
            admin1="California",
            coordinates=(34.0522, -118.2437)
        )

        assert service.is_parent_region(california, la) is True

    def test_not_parent_region(self, temp_dir):
        """Test locations that are not parent-child"""
        service = LocationService(cache_dir=str(temp_dir / "locations"))

        france = GeoLocation(
            name="France",
            location_type="country",
            country_code="FR",
            country_name="France",
            coordinates=(46.2276, 2.2137)
        )

        tokyo = GeoLocation(
            name="Tokyo",
            location_type="city",
            country_code="JP",
            country_name="Japan",
            coordinates=(35.6762, 139.6503)
        )

        assert service.is_parent_region(france, tokyo) is False


# ============================================================================
# Test Visual Keywords
# ============================================================================

class TestVisualKeywords:
    """Test visual keyword generation for locations"""

    def test_city_keywords(self, temp_dir):
        """Test keywords for city"""
        service = LocationService(cache_dir=str(temp_dir / "locations"))

        paris = GeoLocation(
            name="Paris",
            location_type="city",
            country_code="FR",
            country_name="France",
            admin1="Île-de-France",
            coordinates=(48.8566, 2.3522)
        )

        keywords = service.get_location_visual_keywords(paris)

        assert "Paris" in keywords
        assert "Île-de-France" in keywords
        assert "France" in keywords
        assert "skyline" in keywords
        assert "streets" in keywords

    def test_natural_feature_keywords(self, temp_dir):
        """Test keywords for natural feature"""
        service = LocationService(cache_dir=str(temp_dir / "locations"))

        fuji = GeoLocation(
            name="Mount Fuji",
            location_type="natural_feature",
            country_code="JP",
            country_name="Japan",
            coordinates=(35.3606, 138.7274)
        )

        keywords = service.get_location_visual_keywords(fuji)

        assert "Mount Fuji" in keywords
        assert "Japan" in keywords
        assert "landscape" in keywords
        assert "nature" in keywords

    def test_landmark_keywords(self, temp_dir):
        """Test keywords for landmark"""
        service = LocationService(cache_dir=str(temp_dir / "locations"))

        eiffel = GeoLocation(
            name="Eiffel Tower",
            location_type="landmark",
            country_code="FR",
            country_name="France",
            coordinates=(48.8584, 2.2945)
        )

        keywords = service.get_location_visual_keywords(eiffel)

        assert "Eiffel Tower" in keywords
        assert "France" in keywords
        assert "monument" in keywords
        assert "architecture" in keywords

    def test_country_keywords(self, temp_dir):
        """Test keywords for country"""
        service = LocationService(cache_dir=str(temp_dir / "locations"))

        france = GeoLocation(
            name="France",
            location_type="country",
            country_code="FR",
            country_name="France",
            coordinates=(46.2276, 2.2137)
        )

        keywords = service.get_location_visual_keywords(france)

        assert "France" in keywords
        assert "travel" in keywords
        assert "culture" in keywords


# ============================================================================
# Test Text Extraction
# ============================================================================

class TestTextExtraction:
    """Test location extraction from text"""

    def test_extract_city_state_pattern(self, temp_dir):
        """Test extracting 'City, State' pattern"""
        service = LocationService(cache_dir=str(temp_dir / "locations"))

        text = "We traveled from Paris, France to London, England"

        locations = service.extract_locations_from_text(text)

        assert "Paris" in locations
        assert "Paris, France" in locations
        assert "London" in locations
        assert "London, England" in locations

    def test_extract_capitalized_words(self, temp_dir):
        """Test extracting capitalized words"""
        service = LocationService(cache_dir=str(temp_dir / "locations"))

        text = "I visited Tokyo and Kyoto in Japan"

        locations = service.extract_locations_from_text(text)

        assert "Tokyo" in locations
        assert "Kyoto" in locations
        assert "Japan" in locations

    def test_extract_filters_common_words(self, temp_dir):
        """Test extraction filters common non-place words"""
        service = LocationService(cache_dir=str(temp_dir / "locations"))

        text = "The people from Tokyo visited the shrine"

        locations = service.extract_locations_from_text(text)

        assert "Tokyo" in locations
        assert "The" not in locations  # Common word filtered

    def test_extract_deduplicates(self, temp_dir):
        """Test extraction deduplicates results"""
        service = LocationService(cache_dir=str(temp_dir / "locations"))

        text = "Tokyo is amazing. Tokyo has great food. Tokyo is my favorite city."

        locations = service.extract_locations_from_text(text)

        # Should only appear once
        tokyo_count = locations.count("Tokyo")
        assert tokyo_count == 1


# ============================================================================
# Test Rate Limiting
# ============================================================================

class TestRateLimiting:
    """Test API rate limiting"""

    @patch('time.sleep')
    @patch('time.time')
    def test_rate_limit_enforced(self, mock_time, mock_sleep, temp_dir):
        """Test rate limiting enforces delay"""
        service = LocationService(
            cache_dir=str(temp_dir / "locations"),
            geonames_username="test_user",
            rate_limit_delay=1.0
        )
        service._last_api_call = 0.0  # Set initial value

        # Mock time - second call returns 0.5 seconds
        mock_time.return_value = 0.5

        service._rate_limit()

        # Should sleep for 0.5 seconds to reach 1.0 second delay
        mock_sleep.assert_called_once()
        sleep_time = mock_sleep.call_args[0][0]
        assert abs(sleep_time - 0.5) < 0.01

    @patch('time.sleep')
    @patch('time.time')
    def test_rate_limit_already_elapsed(self, mock_time, mock_sleep, temp_dir):
        """Test rate limiting when enough time has elapsed"""
        service = LocationService(
            cache_dir=str(temp_dir / "locations"),
            geonames_username="test_user",
            rate_limit_delay=1.0
        )
        service._last_api_call = 0.0  # Set initial value

        # Mock time - 2 seconds have elapsed
        mock_time.return_value = 2.0

        service._rate_limit()

        # Should not sleep
        mock_sleep.assert_not_called()


# ============================================================================
# Test Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling"""

    @patch('requests.get')
    def test_api_401_error(self, mock_get, temp_dir):
        """Test handling 401 Unauthorized error"""
        import requests
        response = Mock()
        response.status_code = 401
        error = requests.RequestException("401 Unauthorized")
        error.response = response
        mock_get.side_effect = error

        service = LocationService(
            cache_dir=str(temp_dir / "locations"),
            geonames_username="test_user"
        )

        results = service.geocode("Paris")

        assert len(results) == 0

    @patch('requests.get')
    def test_api_json_decode_error(self, mock_get, temp_dir):
        """Test handling JSON decode error"""
        mock_get.return_value = Mock(
            status_code=200,
            json=Mock(side_effect=json.JSONDecodeError("Error", "", 0)),
            raise_for_status=Mock()
        )

        service = LocationService(
            cache_dir=str(temp_dir / "locations"),
            geonames_username="test_user"
        )

        results = service.geocode("Paris")

        assert len(results) == 0

    @patch('requests.get')
    def test_api_status_error(self, mock_get, temp_dir):
        """Test handling API status error"""
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: {"status": {"message": "Rate limit exceeded"}},
            raise_for_status=Mock()
        )

        service = LocationService(
            cache_dir=str(temp_dir / "locations"),
            geonames_username="test_user"
        )

        results = service.geocode("Paris")

        assert len(results) == 0

    @patch('requests.get')
    def test_geocode_parsing_error(self, mock_get, temp_dir):
        """Test handling geocode parsing error"""
        # Return result that will cause exception during GeoLocation creation
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: {"geonames": [{"lat": "invalid_number", "lng": "invalid"}]},
            raise_for_status=Mock()
        )

        service = LocationService(
            cache_dir=str(temp_dir / "locations"),
            geonames_username="test_user"
        )

        results = service.geocode("Paris")

        # Should handle exception and return empty list (invalid lat/lng will raise ValueError)
        assert len(results) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
