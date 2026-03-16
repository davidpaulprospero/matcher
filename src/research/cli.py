#!/usr/bin/env python3
"""
Unified Research CLI - Multi-backend research using Perplexity, Gemini, and Grok.

Usage:
    python -m src.research.cli "your research query"
    python -m src.research.cli "query" --backend gemini
    python -m src.research.cli "query" --api
    python -m src.research.cli "item1, item2, item3" --compare
"""

import argparse
import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()


def get_available_backends():
    """Check which backends have API keys configured."""
    available = []
    if os.getenv("PERPLEXITY_API_KEY"):
        available.append("perplexity")
    if os.getenv("GEMINI_API_KEY"):
        available.append("gemini")
    if os.getenv("XAI_API_KEY"):
        available.append("grok")
    return available


def main():
    parser = argparse.ArgumentParser(
        description="Research using Perplexity, Gemini, or Grok AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # General research (default: perplexity)
  python -m src.research.cli "DaVinci Resolve Python API limitations"

  # Use specific backend
  python -m src.research.cli "PySceneDetect accuracy" --backend gemini
  python -m src.research.cli "latest AI news" --backend grok

  # Quick summary (shorter output)
  python -m src.research.cli "WhisperX vs faster-whisper" --quick

  # API documentation research
  python -m src.research.cli "DaVinci Resolve Timeline API" --api

  # Compare items
  python -m src.research.cli "Whisper, faster-whisper, WhisperX" --compare

  # Save output to file
  python -m src.research.cli "query" -o docs/research/output.md

Environment variables:
  PERPLEXITY_API_KEY - For Perplexity (sonar-pro)
  GEMINI_API_KEY     - For Gemini with Google Search
  XAI_API_KEY        - For Grok with X/Twitter search
"""
    )

    parser.add_argument("query", nargs="?", help="Research query or topic")
    parser.add_argument(
        "--backend", "-b",
        choices=["perplexity", "gemini", "grok", "auto"],
        default="auto",
        help="Research backend (default: auto - uses first available)"
    )
    parser.add_argument(
        "--depth", "-d",
        choices=["quick", "comprehensive"],
        default="comprehensive",
        help="Research depth (default: comprehensive)"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick summary (500-800 words)"
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Research as API documentation"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare items (comma-separated query)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List available backends and exit"
    )

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING)

    # Check available backends
    available = get_available_backends()

    if args.list_backends:
        print("Available backends:")
        for b in ["perplexity", "gemini", "grok"]:
            status = "OK" if b in available else "NO API KEY"
            key_name = {"perplexity": "PERPLEXITY_API_KEY", "gemini": "GEMINI_API_KEY", "grok": "XAI_API_KEY"}[b]
            print(f"  {b:12} [{status:10}] ({key_name})")
        return 0

    if not available:
        print("Error: No API keys configured.")
        print("Set one of: PERPLEXITY_API_KEY, GEMINI_API_KEY, XAI_API_KEY")
        return 1

    # Query is required for research
    if not args.query:
        parser.error("query is required (use --list-backends to see available backends)")
        return 1

    # Select backend
    backend = args.backend
    if backend == "auto":
        # Priority: perplexity > gemini > grok
        for b in ["perplexity", "gemini", "grok"]:
            if b in available:
                backend = b
                break
        print(f"Using backend: {backend}")

    if backend not in available:
        key_name = {"perplexity": "PERPLEXITY_API_KEY", "gemini": "GEMINI_API_KEY", "grok": "XAI_API_KEY"}[backend]
        print(f"Error: {backend} requires {key_name}")
        print(f"Available backends: {', '.join(available)}")
        return 1

    # Determine depth
    depth = "quick" if args.quick else args.depth

    try:
        # Import here to avoid loading all clients unnecessarily
        from src.research import create_client

        client = create_client(backend)

        # Execute research
        if args.api:
            result = client.api_docs(args.query)
        elif args.compare:
            items = [item.strip() for item in args.query.split(",")]
            result = client.compare(items)
        else:
            result = client.research(args.query, depth=depth)

        # Format output
        output = f"# Research: {result.topic}\n"
        output += f"**Backend:** {backend} ({result.model})\n\n"
        output += result.content

        # Add sources
        if result.sources:
            output += "\n\n## Sources\n"
            for s in result.sources:
                if isinstance(s, dict):
                    url = s.get('url', '')
                    title = s.get('title', url)
                    output += f"- [{title}]({url})\n" if url else f"- {title}\n"
                else:
                    output += f"- {s}\n"

        # Output
        if args.output:
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"Saved to: {args.output}")
        else:
            print(output)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
