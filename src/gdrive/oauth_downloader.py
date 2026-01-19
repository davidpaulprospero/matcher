"""Google Drive file downloader using OAuth authentication."""

import io
import json
import pickle
import sys
from pathlib import Path

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from .extract import extract_file_id as _extract_file_id

# Google Drive API scopes
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Credentials file locations
CREDS_DIR = Path.home() / '.matcher_drive_auth'
TOKEN_FILE = CREDS_DIR / 'token.pickle'
CREDENTIALS_FILE = CREDS_DIR / 'credentials.json'


def setup_credentials_file():
    """Guide user through setting up credentials.json"""
    print("\n" + "=" * 70)
    print("SETUP: Google Drive Authentication")
    print("=" * 70)
    print(f"""
To use Google Drive authentication, you need to:

1. Go to: https://console.cloud.google.com/
2. Create a new project (or select existing)
3. Enable "Google Drive API"
4. Create OAuth 2.0 Desktop Application credentials:
   - Click "Create Credentials" > "OAuth client ID"
   - Type: Desktop Application
   - Download the JSON file
5. Save it as: {CREDENTIALS_FILE}

   Paste the file contents below (then press Ctrl+D or Ctrl+Z+Enter):
""")
    print("=" * 70)

    CREDS_DIR.mkdir(parents=True, exist_ok=True)

    if not CREDENTIALS_FILE.exists():
        print(f"\nNeed to create {CREDENTIALS_FILE}")
        print("\nOption 1: Paste JSON contents below (then press Ctrl+D):")
        try:
            content = sys.stdin.read()
            if content.strip():
                CREDENTIALS_FILE.write_text(content)
                print(f"\n✓ Saved credentials to {CREDENTIALS_FILE}")
                return True
        except EOFError:
            pass

        print("\nOption 2: Copy JSON file manually:")
        print(f"  1. Download from Google Cloud Console")
        print(f"  2. Copy to: {CREDENTIALS_FILE}")
        return False
    return True


def get_drive_service(force_reauth: bool = False):
    """
    Get authenticated Google Drive service.

    First run: Opens browser for OAuth authentication
    Later runs: Uses cached credentials
    """
    CREDS_DIR.mkdir(parents=True, exist_ok=True)

    creds = None

    # Load cached token if it exists
    if TOKEN_FILE.exists() and not force_reauth:
        print("Loading cached credentials...")
        try:
            # Try pickle format first
            with open(TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)
        except (pickle.UnpicklingError, EOFError):
            # Fall back to JSON format
            try:
                with open(TOKEN_FILE, 'r') as token:
                    token_data = json.load(token)
                    creds = Credentials.from_authorized_user_info(token_data, SCOPES)
            except Exception as e:
                print(f"⚠ Couldn't load token: {e}")
                creds = None

    # Refresh token if expired
    if creds and creds.expired and creds.refresh_token:
        try:
            print("Refreshing credentials...")
            creds.refresh(Request())
        except RefreshError:
            print("⚠ Credentials expired and couldn't refresh. Re-authenticating...")
            creds = None

    # If no valid credentials, run OAuth flow
    if not creds:
        if not CREDENTIALS_FILE.exists():
            setup_credentials_file()
            if not CREDENTIALS_FILE.exists():
                print("\n❌ credentials.json not found. Cannot proceed.")
                sys.exit(1)

        print("\nStarting Google authentication...")
        print("(A browser window will open for you to sign in)")
        print("-" * 70)

        try:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            sys.exit(1)

        # Save credentials for future use
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
        print("✓ Credentials saved for future use")

    # Build the Drive service
    service = build('drive', 'v3', credentials=creds)
    return service


def find_file_by_id(service, file_id: str):
    """Get file metadata by ID"""
    try:
        file_metadata = service.files().get(
            fileId=file_id,
            fields='id, name, mimeType, size'
        ).execute()
        return file_metadata
    except HttpError as error:
        print(f"❌ File not found: {error}")
        return None


def find_file_by_name(service, file_name: str, parent_id: str = None):
    """Search for file by name"""
    try:
        query = f"name='{file_name}' and trashed=false"
        if parent_id:
            query += f" and '{parent_id}' in parents"

        results = service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name, mimeType, size)',
            pageSize=10
        ).execute()

        files = results.get('files', [])
        if files:
            print(f"Found {len(files)} file(s) matching '{file_name}':")
            for i, f in enumerate(files, 1):
                size_mb = int(f.get('size', 0)) / (1024 * 1024)
                print(f"  {i}. {f['name']} ({size_mb:.1f} MB)")
            return files
        else:
            print(f"❌ No files found matching '{file_name}'")
            return None
    except HttpError as error:
        print(f"❌ Search failed: {error}")
        return None


def download_file(service, file_id: str, output_path: str):
    """Download file from Google Drive by ID"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Get file metadata
        file_metadata = find_file_by_id(service, file_id)
        if not file_metadata:
            return False

        file_name = file_metadata.get('name')
        file_size = int(file_metadata.get('size', 0))
        size_mb = file_size / (1024 * 1024)

        print(f"\n📥 Downloading: {file_name}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   To: {output_path}")

        # Download file
        request = service.files().get_media(fileId=file_id)
        file_stream = io.BytesIO()
        downloader = MediaIoBaseDownload(file_stream, request)

        done = False
        chunk_count = 0
        while not done:
            try:
                status, done = downloader.next_chunk()
                chunk_count += 1
                if chunk_count % 10 == 0 or done:
                    percent = 100 * status.progress()
                    print(f"   Progress: {percent:.1f}%", end='\r')
            except HttpError as error:
                print(f"\n❌ Download failed: {error}")
                return False

        # Save to file
        file_stream.seek(0)
        with open(output_path, 'wb') as f:
            f.write(file_stream.read())

        print(f"\n✅ Success! Downloaded to {output_path}")
        return True

    except Exception as error:
        print(f"❌ Error: {error}")
        return False
