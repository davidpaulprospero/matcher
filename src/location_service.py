"""
Location Service Module

Provides geocoding, location disambiguation, and geographic hierarchy resolution
using GeoNames API with local caching.

Features:
- GeoNames API integration for geocoding
- JSON-based caching (per project)
- Multi-strategy disambiguation (context, co-occurrence, LLM)
- Geographic hierarchy resolution (city → state → country → continent)
- Distance calculation for location comparison
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import requests

logger = logging.getLogger(__name__)


# Continent mapping for country codes
COUNTRY_TO_CONTINENT = {
    # Europe
    "FR": "Europe", "DE": "Europe", "GB": "Europe", "IT": "Europe", "ES": "Europe",
    "PT": "Europe", "NL": "Europe", "BE": "Europe", "CH": "Europe", "AT": "Europe",
    "PL": "Europe", "CZ": "Europe", "SE": "Europe", "NO": "Europe", "DK": "Europe",
    "FI": "Europe", "IE": "Europe", "GR": "Europe", "HU": "Europe", "RO": "Europe",
    "UA": "Europe", "RU": "Europe",
    # North America
    "US": "North America", "CA": "North America", "MX": "North America",
    # South America
    "BR": "South America", "AR": "South America", "CL": "South America",
    "CO": "South America", "PE": "South America", "VE": "South America",
    # Asia
    "CN": "Asia", "JP": "Asia", "KR": "Asia", "IN": "Asia", "ID": "Asia",
    "TH": "Asia", "VN": "Asia", "PH": "Asia", "MY": "Asia", "SG": "Asia",
    "TW": "Asia", "HK": "Asia", "PK": "Asia", "BD": "Asia", "NP": "Asia",
    "LK": "Asia", "MM": "Asia", "KH": "Asia", "LA": "Asia",
    # Middle East
    "AE": "Asia", "SA": "Asia", "IL": "Asia", "TR": "Asia", "IR": "Asia",
    "IQ": "Asia", "JO": "Asia", "LB": "Asia", "QA": "Asia", "KW": "Asia",
    # Africa
    "ZA": "Africa", "EG": "Africa", "NG": "Africa", "KE": "Africa", "MA": "Africa",
    "TN": "Africa", "GH": "Africa", "ET": "Africa", "TZ": "Africa", "UG": "Africa",
    # Oceania
    "AU": "Oceania", "NZ": "Oceania", "FJ": "Oceania", "PG": "Oceania",
}


@dataclass
class GeoLocation:
    """Resolved geographic location with hierarchy"""
    name: str                           # Display name: "Paris"
    location_type: str                  # city, country, landmark, region, natural_feature
    country_code: str                   # ISO 2-letter: "FR"
    country_name: str                   # Full name: "France"
    admin1: str = ""                    # State/Province: "Île-de-France"
    admin2: str = ""                    # County/District
    coordinates: Tuple[float, float] = (0.0, 0.0)  # (lat, lon)
    population: int = 0                 # For ranking ambiguous results
    geoname_id: int = 0                 # GeoNames reference ID
    feature_class: str = ""             # P=populated place, T=mountain, etc.
    feature_code: str = ""              # PPLA=admin center, PPL=populated place
    timezone: str = ""                  # e.g., "Europe/Paris"

    @property
    def continent(self) -> str:
        """Get continent from country code"""
        return COUNTRY_TO_CONTINENT.get(self.country_code, "Unknown")

    @property
    def parent_regions(self) -> List[str]:
        """Get geographic hierarchy from specific to general"""
        regions = []
        if self.admin2:
            regions.append(self.admin2)
        if self.admin1:
            regions.append(self.admin1)
        if self.country_name:
            regions.append(self.country_name)
        regions.append(self.continent)
        return regions

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "location_type": self.location_type,
            "country_code": self.country_code,
            "country_name": self.country_name,
            "admin1": self.admin1,
            "admin2": self.admin2,
            "coordinates": list(self.coordinates),
            "population": self.population,
            "geoname_id": self.geoname_id,
            "feature_class": self.feature_class,
            "feature_code": self.feature_code,
            "timezone": self.timezone,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GeoLocation":
        """Create from dictionary"""
        coords = data.get("coordinates", [0.0, 0.0])
        return cls(
            name=data.get("name", ""),
            location_type=data.get("location_type", "city"),
            country_code=data.get("country_code", ""),
            country_name=data.get("country_name", ""),
            admin1=data.get("admin1", ""),
            admin2=data.get("admin2", ""),
            coordinates=tuple(coords) if isinstance(coords, list) else coords,
            population=data.get("population", 0),
            geoname_id=data.get("geoname_id", 0),
            feature_class=data.get("feature_class", ""),
            feature_code=data.get("feature_code", ""),
            timezone=data.get("timezone", ""),
        )

    @classmethod
    def from_geonames_result(cls, result: dict) -> "GeoLocation":
        """Create from GeoNames API response"""
        # Determine location type from feature class/code
        feature_class = result.get("fcl", "")
        feature_code = result.get("fcode", "")

        if feature_class == "A":  # Administrative
            location_type = "country" if feature_code == "PCLI" else "region"
        elif feature_class == "P":  # Populated place
            location_type = "city"
        elif feature_class == "T":  # Mountain, hill, rock
            location_type = "natural_feature"
        elif feature_class == "H":  # Water body
            location_type = "natural_feature"
        elif feature_class == "S":  # Spot, building, farm
            location_type = "landmark"
        elif feature_class == "L":  # Parks, areas
            location_type = "region"
        else:
            location_type = "other"

        return cls(
            name=result.get("name", result.get("toponymName", "")),
            location_type=location_type,
            country_code=result.get("countryCode", ""),
            country_name=result.get("countryName", ""),
            admin1=result.get("adminName1", ""),
            admin2=result.get("adminName2", ""),
            coordinates=(
                float(result.get("lat", 0)),
                float(result.get("lng", 0))
            ),
            population=int(result.get("population", 0)),
            geoname_id=int(result.get("geonameId", 0)),
            feature_class=feature_class,
            feature_code=feature_code,
            timezone=result.get("timezone", {}).get("timeZoneId", "") if isinstance(result.get("timezone"), dict) else "",
        )


class LocationService:
    """
    Service for geocoding, disambiguation, and location comparison.

    Uses GeoNames API with local JSON caching.
    """

    GEONAMES_BASE_URL = "http://api.geonames.org"

    # Common disambiguation hints
    COUNTRY_HINTS = {
        # Place name → likely country based on common usage
        "paris": {"eiffel": "FR", "louvre": "FR", "fashion": "FR", "france": "FR", "texas": "US", "rodeo": "US"},
        "london": {"uk": "GB", "england": "GB", "thames": "GB", "ontario": "CA", "canada": "CA"},
        "rome": {"italy": "IT", "colosseum": "IT", "vatican": "IT", "georgia": "US"},
        "moscow": {"russia": "RU", "kremlin": "RU", "idaho": "US"},
        "sydney": {"australia": "AU", "opera": "AU", "harbour": "AU"},
        "melbourne": {"australia": "AU", "florida": "US"},
        "athens": {"greece": "GR", "acropolis": "GR", "georgia": "US", "ohio": "US"},
        "venice": {"italy": "IT", "gondola": "IT", "california": "US", "florida": "US"},
        "florence": {"italy": "IT", "renaissance": "IT", "south carolina": "US"},
        "dublin": {"ireland": "IE", "guinness": "IE", "ohio": "US", "california": "US"},
        "cairo": {"egypt": "EG", "pyramids": "EG", "illinois": "US"},
        "alexandria": {"egypt": "EG", "virginia": "US"},
        "barcelona": {"spain": "ES", "gaudi": "ES"},
        "munich": {"germany": "DE", "bavaria": "DE", "beer": "DE"},
        "vienna": {"austria": "AT", "mozart": "AT", "virginia": "US"},
        "birmingham": {"uk": "GB", "england": "GB", "alabama": "US"},
        "manchester": {"uk": "GB", "england": "GB", "new hampshire": "US"},
        "newcastle": {"uk": "GB", "england": "GB", "australia": "AU"},
        "perth": {"australia": "AU", "scotland": "GB"},
        "hamilton": {"new zealand": "NZ", "ontario": "CA", "bermuda": "BM"},
        "wellington": {"new zealand": "NZ"},
        "santiago": {"chile": "CL", "spain": "ES"},
        "lima": {"peru": "PE", "ohio": "US"},
        "portland": {"oregon": "US", "maine": "US"},
        "springfield": {"illinois": "US", "missouri": "US", "massachusetts": "US"},
    }

    def __init__(
        self,
        cache_dir: str = ".cache/locations",
        geonames_username: str = "",
        llm_client: Any = None,
        rate_limit_delay: float = 1.0,  # Seconds between API calls
    ):
        """
        Initialize LocationService.

        Args:
            cache_dir: Directory for location cache JSON
            geonames_username: GeoNames API username (required for API calls)
            llm_client: Optional LLM client for disambiguation
            rate_limit_delay: Delay between API calls (GeoNames rate limits)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / "location_cache.json"

        self.geonames_username = geonames_username
        self.llm_client = llm_client
        self.rate_limit_delay = rate_limit_delay
        self._last_api_call = 0.0

        self._cache: Dict[str, Any] = {"locations": {}, "disambiguations": {}}
        self._load_cache()

    def _load_cache(self):
        """Load cache from disk"""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded location cache with {len(self._cache.get('locations', {}))} entries")
            except Exception as e:
                logger.warning(f"Could not load location cache: {e}")
                self._cache = {"locations": {}, "disambiguations": {}}

    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Could not save location cache: {e}")

    def _rate_limit(self):
        """Enforce rate limiting for API calls"""
        elapsed = time.time() - self._last_api_call
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_api_call = time.time()

    def _call_geonames_api(self, endpoint: str, params: dict) -> Optional[dict]:
        """Make a rate-limited call to GeoNames API"""
        if not self.geonames_username:
            logger.warning("GeoNames username not configured - API calls disabled")
            return None

        self._rate_limit()

        params["username"] = self.geonames_username
        url = f"{self.GEONAMES_BASE_URL}/{endpoint}"

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Check for API errors
            if "status" in data:
                logger.warning(f"GeoNames API error: {data['status'].get('message', 'Unknown error')}")
                return None

            return data

        except requests.RequestException as e:
            logger.warning(f"GeoNames API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"GeoNames API response parse error: {e}")
            return None

    def geocode(self, location_name: str, max_results: int = 5) -> List[GeoLocation]:
        """
        Geocode a location name to possible GeoLocations.

        Args:
            location_name: Name to geocode (e.g., "Paris", "New York City")
            max_results: Maximum number of results to return

        Returns:
            List of possible GeoLocation matches, sorted by population
        """
        cache_key = location_name.lower().strip()

        # Check cache
        if cache_key in self._cache["locations"]:
            cached = self._cache["locations"][cache_key]
            logger.debug(f"Location cache hit: '{location_name}'")
            return [GeoLocation.from_dict(loc) for loc in cached.get("results", [])]

        # Call GeoNames API
        logger.info(f"GeoNames API: geocoding '{location_name}'")
        data = self._call_geonames_api("searchJSON", {
            "q": location_name,
            "maxRows": max_results,
            "style": "FULL",
            "orderby": "relevance",
        })

        if not data or "geonames" not in data:
            logger.warning(f"No geocoding results for: {location_name}")
            return []

        # Parse results
        locations = []
        for result in data["geonames"]:
            try:
                loc = GeoLocation.from_geonames_result(result)
                locations.append(loc)
            except Exception as e:
                logger.debug(f"Could not parse GeoNames result: {e}")

        # Sort by population (larger cities first)
        locations.sort(key=lambda x: -x.population)

        # Cache results
        self._cache["locations"][cache_key] = {
            "results": [loc.to_dict() for loc in locations],
            "cached_at": datetime.now().isoformat(),
        }
        self._save_cache()

        return locations

    def get_best_match(self, location_name: str) -> Optional[GeoLocation]:
        """
        Get the best matching location (highest population).

        Args:
            location_name: Name to geocode

        Returns:
            Best matching GeoLocation or None
        """
        results = self.geocode(location_name)
        return results[0] if results else None

    def disambiguate(
        self,
        location_name: str,
        context: str = "",
        co_locations: List[str] = None,
    ) -> Optional[GeoLocation]:
        """
        Disambiguate a location name using multiple strategies.

        Strategies (in order):
        1. Context keywords (e.g., "Eiffel Tower" → Paris, France)
        2. Co-occurring locations (e.g., "Paris and Lyon" → both France)
        3. Population (default to largest city)
        4. LLM as final tiebreaker

        Args:
            location_name: Name to disambiguate
            context: Surrounding text for context clues
            co_locations: Other locations mentioned nearby

        Returns:
            Best matching GeoLocation or None
        """
        cache_key = f"{location_name.lower()}:{context.lower()[:100]}"

        # Check disambiguation cache
        if cache_key in self._cache["disambiguations"]:
            cached_code = self._cache["disambiguations"][cache_key]
            results = self.geocode(location_name)
            for loc in results:
                if loc.country_code == cached_code:
                    return loc

        # Get all possible matches
        results = self.geocode(location_name)
        if not results:
            return None

        if len(results) == 1:
            return results[0]

        # Strategy 1: Context keyword hints
        name_lower = location_name.lower()
        context_lower = context.lower()

        if name_lower in self.COUNTRY_HINTS:
            hints = self.COUNTRY_HINTS[name_lower]
            for keyword, country_code in hints.items():
                if keyword in context_lower:
                    for loc in results:
                        if loc.country_code == country_code:
                            self._cache["disambiguations"][cache_key] = country_code
                            self._save_cache()
                            logger.debug(f"Disambiguated '{location_name}' to {country_code} via context '{keyword}'")
                            return loc

        # Strategy 2: Co-occurring locations
        if co_locations:
            co_countries = set()
            for co_loc_name in co_locations:
                co_results = self.geocode(co_loc_name, max_results=1)
                if co_results:
                    co_countries.add(co_results[0].country_code)

            # Prefer location in same country as co-locations
            for loc in results:
                if loc.country_code in co_countries:
                    self._cache["disambiguations"][cache_key] = loc.country_code
                    self._save_cache()
                    logger.debug(f"Disambiguated '{location_name}' to {loc.country_code} via co-location")
                    return loc

        # Strategy 3: LLM disambiguation (if available and ambiguous)
        if self.llm_client and len(results) > 1:
            try:
                llm_result = self._disambiguate_with_llm(location_name, context, results)
                if llm_result:
                    self._cache["disambiguations"][cache_key] = llm_result.country_code
                    self._save_cache()
                    return llm_result
            except Exception as e:
                logger.debug(f"LLM disambiguation failed: {e}")

        # Strategy 4: Default to highest population
        best = results[0]
        self._cache["disambiguations"][cache_key] = best.country_code
        self._save_cache()
        logger.debug(f"Disambiguated '{location_name}' to {best.country_code} via population")
        return best

    def _disambiguate_with_llm(
        self,
        location_name: str,
        context: str,
        candidates: List[GeoLocation],
    ) -> Optional[GeoLocation]:
        """Use LLM to disambiguate location"""
        if not self.llm_client:
            return None

        # Build prompt
        options = "\n".join([
            f"{i+1}. {loc.name}, {loc.admin1}, {loc.country_name} (pop: {loc.population:,})"
            for i, loc in enumerate(candidates[:5])
        ])

        prompt = f"""Given this context, which location is being referred to?

Location name: "{location_name}"
Context: "{context[:500]}"

Options:
{options}

Reply with ONLY the number (1-{min(5, len(candidates))}) of the correct location."""

        try:
            # Support both Gemini and Anthropic clients
            if hasattr(self.llm_client, 'generate_content'):
                # Gemini
                response = self.llm_client.generate_content(prompt)
                text = response.text.strip()
            elif hasattr(self.llm_client, 'messages'):
                # Anthropic
                response = self.llm_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=10,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = response.content[0].text.strip()
            else:
                return None

            # Parse response
            match = re.search(r'\d+', text)
            if match:
                idx = int(match.group()) - 1
                if 0 <= idx < len(candidates):
                    logger.debug(f"LLM disambiguated '{location_name}' to {candidates[idx].country_code}")
                    return candidates[idx]

        except Exception as e:
            logger.debug(f"LLM disambiguation error: {e}")

        return None

    def same_country(self, loc1: GeoLocation, loc2: GeoLocation) -> bool:
        """Check if two locations are in the same country"""
        return loc1.country_code == loc2.country_code

    def same_continent(self, loc1: GeoLocation, loc2: GeoLocation) -> bool:
        """Check if two locations are on the same continent"""
        return loc1.continent == loc2.continent

    def same_region(self, loc1: GeoLocation, loc2: GeoLocation) -> bool:
        """Check if two locations are in the same admin1 region"""
        return (
            loc1.country_code == loc2.country_code and
            loc1.admin1 == loc2.admin1 and
            loc1.admin1 != ""
        )

    def distance_km(self, loc1: GeoLocation, loc2: GeoLocation) -> float:
        """
        Calculate distance between two locations using Haversine formula.

        Args:
            loc1: First location
            loc2: Second location

        Returns:
            Distance in kilometers
        """
        lat1, lon1 = loc1.coordinates
        lat2, lon2 = loc2.coordinates

        # Earth's radius in km
        R = 6371.0

        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        # Haversine formula
        a = (
            math.sin(delta_lat / 2) ** 2 +
            math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def is_parent_region(self, parent: GeoLocation, child: GeoLocation) -> bool:
        """
        Check if parent location contains child location geographically.

        Examples:
        - France is parent of Paris
        - California is parent of Los Angeles
        """
        # Country contains city
        if parent.location_type == "country" and child.country_code == parent.country_code:
            return True

        # Region/state contains city
        if parent.location_type == "region" and parent.admin1:
            if child.country_code == parent.country_code and child.admin1 == parent.admin1:
                return True

        # Check by name matching in hierarchy
        parent_name_lower = parent.name.lower()
        for region in child.parent_regions:
            if parent_name_lower in region.lower():
                return True

        return False

    def extract_locations_from_text(self, text: str) -> List[str]:
        """
        Extract potential location names from text using simple patterns.

        This is a lightweight extraction - for more robust NER, use spaCy.

        Args:
            text: Text to extract locations from

        Returns:
            List of potential location names
        """
        locations = []

        # Pattern: "City, State" or "City, Country"
        pattern1 = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
        for match in re.finditer(pattern1, text):
            city, region = match.groups()
            locations.append(city)
            locations.append(f"{city}, {region}")

        # Pattern: Capitalized words that might be places
        # Filter common non-place words
        non_places = {
            "The", "This", "That", "There", "These", "Those", "What", "When",
            "Where", "Which", "Who", "How", "Why", "And", "But", "For", "With",
            "From", "About", "After", "Before", "During", "Through", "Between",
            "January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October", "November", "December",
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        }

        pattern2 = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b'
        for match in re.finditer(pattern2, text):
            name = match.group(1)
            if name not in non_places and len(name) > 2:
                locations.append(name)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for loc in locations:
            loc_lower = loc.lower()
            if loc_lower not in seen:
                seen.add(loc_lower)
                unique.append(loc)

        return unique

    def get_location_visual_keywords(self, location: GeoLocation) -> List[str]:
        """
        Get visual keywords associated with a location for video matching.

        Args:
            location: Resolved location

        Returns:
            List of visual keywords (landmarks, features, etc.)
        """
        keywords = []

        # Add the location name itself
        keywords.append(location.name)

        # Add hierarchy
        if location.admin1:
            keywords.append(location.admin1)
        keywords.append(location.country_name)

        # Add type-specific keywords
        if location.location_type == "city":
            keywords.extend(["skyline", "streets", "downtown", "aerial"])
        elif location.location_type == "natural_feature":
            keywords.extend(["landscape", "nature", "scenery"])
        elif location.location_type == "landmark":
            keywords.extend(["monument", "historic", "architecture"])
        elif location.location_type == "country":
            keywords.extend(["travel", "culture", "tourism"])

        return keywords


def create_location_service(config) -> LocationService:
    """
    Factory function to create LocationService from config.

    Args:
        config: Pipeline config with location_matching settings

    Returns:
        Configured LocationService instance
    """
    # Get location matching config
    location_config = getattr(config.matching, 'location_matching', None)

    if location_config is None:
        # Default settings
        cache_dir = ".cache/locations"
        geonames_username = ""
    elif isinstance(location_config, dict):
        cache_dir = location_config.get('cache_dir', '.cache/locations')
        geonames_username = location_config.get('geonames_username', '')
    else:
        cache_dir = getattr(location_config, 'cache_dir', '.cache/locations')
        geonames_username = getattr(location_config, 'geonames_username', '')

    # Try to get LLM client for disambiguation
    llm_client = None
    try:
        gemini_key = getattr(config, 'gemini_api_key', None)
        if gemini_key:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            llm_client = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        logger.debug(f"Could not initialize LLM for disambiguation: {e}")

    return LocationService(
        cache_dir=cache_dir,
        geonames_username=geonames_username,
        llm_client=llm_client,
    )
