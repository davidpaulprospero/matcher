#!/usr/bin/env python3
"""
Test proxy connectivity and YouTube access.

Usage:
    python scripts/test_proxy.py                          # Test default (127.0.0.1:1080)
    python scripts/test_proxy.py socks5://127.0.0.1:1080  # Test specific proxy
    python scripts/test_proxy.py --config                 # Test proxy from config.yaml
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_proxy(proxy_url: str = "socks5://127.0.0.1:1080") -> bool:
    """Test proxy connectivity."""
    import httpx

    print(f"\n{'='*50}")
    print(f"Testing proxy: {proxy_url}")
    print(f"{'='*50}\n")

    results = {
        "port_listening": False,
        "ip_different": False,
        "youtube_accessible": False,
    }

    # Test 1: Get direct IP
    print("1. Getting direct IP...")
    try:
        direct_ip = httpx.get("https://api.ipify.org", timeout=10).text
        print(f"   Direct IP: {direct_ip}")
    except Exception as e:
        print(f"   Could not get direct IP: {e}")
        direct_ip = None

    # Test 2: Get IP through proxy
    print("\n2. Getting IP through proxy...")
    try:
        proxy_ip = httpx.get(
            "https://api.ipify.org",
            proxy=proxy_url,
            timeout=15
        ).text
        print(f"   Proxy IP: {proxy_ip}")
        results["port_listening"] = True

        if direct_ip and proxy_ip != direct_ip:
            print(f"   ✓ IP changed! Proxy is working.")
            results["ip_different"] = True
        elif direct_ip:
            print(f"   ✗ IP same as direct. Proxy may not be routing correctly.")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        print("\n   Possible issues:")
        print("   - SSH tunnel not running")
        print("   - Wrong proxy port")
        print("   - Firewall blocking connection")
        return False

    # Test 3: Access YouTube through proxy
    print("\n3. Testing YouTube access through proxy...")
    try:
        response = httpx.get(
            "https://www.youtube.com/",
            proxy=proxy_url,
            timeout=15,
            follow_redirects=True
        )
        if response.status_code == 200:
            print(f"   ✓ YouTube accessible (HTTP {response.status_code})")
            results["youtube_accessible"] = True
        else:
            print(f"   ✗ YouTube returned HTTP {response.status_code}")
    except Exception as e:
        print(f"   ✗ YouTube access failed: {e}")

    # Test 4: Try fetching video info
    print("\n4. Testing YouTube video info fetch...")
    try:
        # Try to fetch a video page
        response = httpx.get(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            proxy=proxy_url,
            timeout=15,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        if response.status_code == 200:
            print(f"   ✓ Video page accessible")
        elif response.status_code == 429:
            print(f"   ✗ Rate limited (429) - proxy IP may be flagged")
        else:
            print(f"   ? HTTP {response.status_code}")
    except Exception as e:
        print(f"   ✗ Video fetch failed: {e}")

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    passed = sum(results.values())
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    for test, result in results.items():
        status = "✓" if result else "✗"
        print(f"  {status} {test.replace('_', ' ').title()}")

    if passed == total:
        print(f"\n✓ Proxy is working! Add to config.yaml:")
        print(f"""
download:
  fallback:
    proxy:
      enabled: true
      sources:
        - type: socks5
          url: "{proxy_url}"
""")
        return True
    else:
        print(f"\n✗ Some tests failed. Check proxy setup.")
        return False


def test_from_config() -> bool:
    """Test proxy from config.yaml."""
    try:
        from src.config import get_config
        config = get_config()

        proxy_cfg = config.download.fallback.proxy
        if not proxy_cfg.enabled:
            print("Proxy is disabled in config.yaml")
            return False

        sources = proxy_cfg.sources
        if not sources:
            print("No proxy sources configured in config.yaml")
            return False

        # Test first source
        source = sources[0]
        if isinstance(source, dict):
            proxy_url = source.get('url', '')
        else:
            proxy_url = getattr(source, 'url', '')

        if not proxy_url:
            print("No proxy URL in first source")
            return False

        return test_proxy(proxy_url)

    except Exception as e:
        print(f"Error loading config: {e}")
        return False


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "--config":
            success = test_from_config()
        else:
            success = test_proxy(sys.argv[1])
    else:
        # Default: test common SOCKS5 port
        success = test_proxy("socks5://127.0.0.1:1080")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
