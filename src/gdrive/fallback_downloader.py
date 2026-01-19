"""Google Drive file downloader with multiple fallback strategies (no auth required)."""

import io
from pathlib import Path

import gdown

from .extract import extract_file_id, construct_download_urls


def download_file(url: str, output_path: str, verbose: bool = True) -> bool:
    """
    Download file from Google Drive with multiple fallback strategies.

    Args:
        url: Google Drive URL (any format)
        output_path: Where to save the file
        verbose: Print progress

    Returns:
        True if successful, False otherwise
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Strategy 1: Extract file ID and use direct download URL
    file_id = extract_file_id(url)
    if file_id:
        if verbose:
            print(f"✓ Extracted file ID: {file_id}")
        urls_to_try = construct_download_urls(file_id)
    else:
        urls_to_try = [url]

    # Try standard gdown first
    if verbose:
        print(f"\n📥 Attempting download (Strategy 1: standard gdown)...")

    try:
        gdown.download(url, str(output_path), quiet=False, verify=False)
        if output_path.exists() and output_path.stat().st_size > 0:
            if verbose:
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"✅ Success! Downloaded {size_mb:.1f} MB")
            return True
    except Exception as e:
        if verbose:
            print(f"⚠ Strategy 1 failed: {e}")

    # Strategy 2: Try with fuzzy matching
    if verbose:
        print(f"\n📥 Attempting download (Strategy 2: fuzzy matching)...")

    try:
        if output_path.exists():
            output_path.unlink()

        gdown.download(url, str(output_path), quiet=False, verify=False, fuzzy=True)
        if output_path.exists() and output_path.stat().st_size > 0:
            if verbose:
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"✅ Success! Downloaded {size_mb:.1f} MB")
            return True
    except Exception as e:
        if verbose:
            print(f"⚠ Strategy 2 failed: {e}")

    # Strategy 3: Try direct download URLs
    if file_id:
        if verbose:
            print(f"\n📥 Attempting download (Strategy 3: direct URLs)...")

        for direct_url in urls_to_try:
            try:
                if output_path.exists():
                    output_path.unlink()

                gdown.download(direct_url, str(output_path), quiet=False, verify=False)
                if output_path.exists() and output_path.stat().st_size > 0:
                    if verbose:
                        size_mb = output_path.stat().st_size / (1024 * 1024)
                        print(f"✅ Success! Downloaded {size_mb:.1f} MB")
                    return True
            except Exception as e:
                if verbose:
                    print(f"  • {direct_url.split('?')[0]}: Failed")

    # Strategy 4: Try with requests library for confirmation page handling
    if verbose:
        print(f"\n📥 Attempting download (Strategy 4: with confirmation page handling)...")

    try:
        import requests

        if file_id:
            sess = requests.Session()

            # First request to get confirmation token
            params = {'id': file_id, 'export': 'download'}
            response = sess.get('https://drive.google.com/uc', params=params, stream=True)

            # Check for confirmation token in response
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    params['confirm'] = value
                    break

            # Download with confirmation
            response = sess.get('https://drive.google.com/uc', params=params, stream=True)

            if response.status_code == 200:
                if output_path.exists():
                    output_path.unlink()

                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                if output_path.exists() and output_path.stat().st_size > 0:
                    if verbose:
                        size_mb = output_path.stat().st_size / (1024 * 1024)
                        print(f"✅ Success! Downloaded {size_mb:.1f} MB")
                    return True
    except ImportError:
        if verbose:
            print("  ⚠ requests library not available, skipping this strategy")
    except Exception as e:
        if verbose:
            print(f"⚠ Strategy 4 failed: {e}")

    if verbose:
        print("\n❌ All strategies exhausted. Download failed.")
        print("\nTroubleshooting tips:")
        print("1. Verify the file is shared with 'Anyone with the link can view'")
        print("2. Check that the file isn't too large (>5GB)")
        print("3. Try downloading manually from Google Drive first")
        print("4. Ensure internet connection is stable")

    return False
