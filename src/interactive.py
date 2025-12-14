"""
Interactive CLI Module for Voiceover-Matcher

Provides interactive prompts for:
- Voiceover file selection
- Keyword review and editing
- Face detection preference
- Clip grading after matching
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# KEYBOARD INPUT UTILITIES
# =============================================================================

def getch():
    """Get a single character from stdin without requiring Enter."""
    try:
        # Windows
        import msvcrt
        return msvcrt.getch().decode('utf-8', errors='ignore')
    except ImportError:
        # Unix
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_colored(text: str, color: str = None):
    """Print with ANSI colors (works in most terminals)."""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'bold': '\033[1m',
        'reset': '\033[0m'
    }
    
    if color and color in colors:
        print(f"{colors[color]}{text}{colors['reset']}")
    else:
        print(text)


# =============================================================================
# VOICEOVER FILE SELECTION
# =============================================================================

def select_voiceover_file(project_dir: Path = None, default_subdir: str = "voiceover") -> Optional[str]:
    """
    Interactive voiceover file selection.
    
    Scans for audio/video/srt files and lets user select one.
    
    Args:
        project_dir: Project directory to search in
        default_subdir: Subdirectory to look for voiceover files
        
    Returns:
        Selected file path or None if cancelled
    """
    # Supported formats
    audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac', '.opus'}
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.wmv', '.flv', '.m4v'}
    srt_extensions = {'.srt'}
    all_extensions = audio_extensions | video_extensions | srt_extensions
    
    # Determine search directories
    search_dirs = []
    if project_dir:
        voiceover_dir = project_dir / default_subdir
        if voiceover_dir.exists():
            search_dirs.append(voiceover_dir)
        search_dirs.append(project_dir)
    else:
        search_dirs.append(Path.cwd())
        voiceover_dir = Path.cwd() / default_subdir
        if voiceover_dir.exists():
            search_dirs.insert(0, voiceover_dir)
    
    # Find all candidate files
    candidates = []
    seen = set()
    
    for search_dir in search_dirs:
        for ext in all_extensions:
            for f in search_dir.glob(f"*{ext}"):
                if f.name not in seen and not f.name.startswith('.'):
                    candidates.append(f)
                    seen.add(f.name)
            # Also check immediate subdirectories
            for f in search_dir.glob(f"*/*{ext}"):
                if f.name not in seen and not f.name.startswith('.'):
                    candidates.append(f)
                    seen.add(f.name)
    
    if not candidates:
        print("\n  ⚠ No voiceover files found.")
        print(f"    Supported formats: {', '.join(sorted(all_extensions))}")
        print(f"    Search locations: {', '.join(str(d) for d in search_dirs)}")
        return None
    
    # Sort by type priority (SRT first, then audio, then video)
    def sort_key(f):
        ext = f.suffix.lower()
        if ext in srt_extensions:
            return (0, f.name.lower())
        elif ext in audio_extensions:
            return (1, f.name.lower())
        else:
            return (2, f.name.lower())
    
    candidates.sort(key=sort_key)
    
    # Display selection menu
    print("\n" + "─" * 60)
    print("  SELECT VOICEOVER FILE")
    print("─" * 60)
    
    for i, f in enumerate(candidates, 1):
        ext = f.suffix.lower()
        if ext in srt_extensions:
            icon = "📄"
            type_label = "SRT"
        elif ext in audio_extensions:
            icon = "🎵"
            type_label = "Audio"
        else:
            icon = "🎬"
            type_label = "Video"
        
        # Show relative path if in project
        try:
            if project_dir:
                display_path = f.relative_to(project_dir)
            else:
                display_path = f.name
        except ValueError:
            display_path = f.name
        
        print(f"  {i:2}. {icon} [{type_label:5}] {display_path}")
    
    print(f"\n  0. Enter custom path")
    print(f"  q. Cancel")
    
    # Get selection
    while True:
        try:
            choice = input(f"\n  Select [1-{len(candidates)}]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return None
        
        if choice == 'q':
            return None
        
        if choice == '0':
            custom_path = input("  Enter full path: ").strip().strip('"\'')
            if custom_path and Path(custom_path).exists():
                return custom_path
            print("  ⚠ File not found.")
            continue
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(candidates):
                selected = candidates[idx]
                print(f"\n  ✓ Selected: {selected.name}")
                return str(selected)
        except ValueError:
            pass
        
        print(f"  Invalid choice. Enter 1-{len(candidates)}, 0, or q.")
    
    return None


# =============================================================================
# KEYWORD REVIEW AND EDITING
# =============================================================================

@dataclass
class KeywordItem:
    """Represents a keyword with metadata."""
    text: str
    category: str = "general"  # general, entity, custom
    entity_type: str = None    # PERSON, GPE, DATE, etc.
    enabled: bool = True
    search_query: str = None   # Generated search query
    
    def __str__(self):
        status = "✓" if self.enabled else "✗"
        if self.entity_type:
            return f"[{status}] {self.text} ({self.entity_type})"
        return f"[{status}] {self.text}"


class KeywordReviewer:
    """Interactive keyword review and editing interface."""
    
    def __init__(self, keywords: List[str], entities: List[Dict] = None, topic: str = ""):
        """
        Initialize reviewer with keywords and entities.
        
        Args:
            keywords: List of keyword strings
            entities: List of entity dicts with 'text', 'type' keys
            topic: Main topic for context
        """
        self.topic = topic
        self.items: List[KeywordItem] = []
        self.cursor = 0
        self.removed_keywords_file = None
        
        # Add entities first (higher priority)
        if entities:
            for ent in entities:
                self.items.append(KeywordItem(
                    text=ent.get('text', ent.get('name', str(ent))),
                    category="entity",
                    entity_type=ent.get('type', ent.get('label', 'ENTITY')),
                    enabled=True,
                    search_query=self._generate_entity_query(ent)
                ))
        
        # Add general keywords
        entity_texts = {item.text.lower() for item in self.items}
        for kw in keywords:
            if kw.lower() not in entity_texts:
                self.items.append(KeywordItem(
                    text=kw,
                    category="general",
                    enabled=True
                ))
    
    def _generate_entity_query(self, entity: Dict) -> str:
        """Generate search query for an entity."""
        text = entity.get('text', entity.get('name', ''))
        etype = entity.get('type', entity.get('label', ''))
        
        if etype in ('PERSON', 'PER'):
            return f"{text} {self.topic} footage"
        elif etype in ('GPE', 'LOC', 'LOCATION'):
            return f"{text} {self.topic} aerial footage"
        elif etype == 'DATE':
            return f"{text} {self.topic} news footage"
        elif etype in ('EVENT', 'FAC'):
            return f"{text} footage"
        elif etype == 'ORG':
            return f"{text} {self.topic} operation"
        else:
            return f"{text} {self.topic}"
    
    def set_removed_keywords_file(self, filepath: str):
        """Set file to save/load removed keywords."""
        self.removed_keywords_file = Path(filepath)
        self._load_removed_keywords()
    
    def _load_removed_keywords(self):
        """Load previously removed keywords."""
        if self.removed_keywords_file and self.removed_keywords_file.exists():
            try:
                with open(self.removed_keywords_file, 'r') as f:
                    removed = json.load(f)
                
                removed_set = set(removed.get('removed', []))
                for item in self.items:
                    if item.text.lower() in removed_set:
                        item.enabled = False
                
                logger.info(f"Loaded {len(removed_set)} previously removed keywords")
            except Exception as e:
                logger.warning(f"Could not load removed keywords: {e}")
    
    def _save_removed_keywords(self):
        """Save removed keywords for future runs."""
        if self.removed_keywords_file:
            removed = [item.text.lower() for item in self.items if not item.enabled]
            try:
                self.removed_keywords_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.removed_keywords_file, 'w') as f:
                    json.dump({
                        'removed': removed,
                        'updated': datetime.now().isoformat()
                    }, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save removed keywords: {e}")
    
    def review_interactive(self) -> List[str]:
        """
        Run interactive review interface.
        
        Returns:
            List of enabled keywords
        """
        print("\n" + "=" * 70)
        print("  KEYWORD REVIEW")
        print("=" * 70)
        print(f"  Topic: {self.topic}")
        print(f"  Total: {len(self.items)} keywords/entities")
        print("\n  Controls:")
        print("    ↑/↓ or j/k : Navigate")
        print("    SPACE/ENTER : Toggle keyword on/off")
        print("    a           : Add custom keyword")
        print("    d           : Delete selected (same as toggle off)")
        print("    r           : Reset all to enabled")
        print("    s           : Save and continue")
        print("    q           : Quit (use current selections)")
        print("=" * 70)
        
        try:
            self._display_list()
            
            while True:
                key = getch()
                
                if key in ('q', 'Q', '\x1b'):  # q or Escape
                    break
                elif key in ('s', 'S'):
                    self._save_removed_keywords()
                    break
                elif key in ('j', 'J') or key == '\x1b[B':  # Down
                    self.cursor = min(self.cursor + 1, len(self.items) - 1)
                elif key in ('k', 'K') or key == '\x1b[A':  # Up
                    self.cursor = max(self.cursor - 1, 0)
                elif key in (' ', '\r', '\n'):  # Space or Enter
                    if self.items:
                        self.items[self.cursor].enabled = not self.items[self.cursor].enabled
                elif key in ('d', 'D'):
                    if self.items:
                        self.items[self.cursor].enabled = False
                elif key in ('a', 'A'):
                    self._add_custom_keyword()
                elif key in ('r', 'R'):
                    for item in self.items:
                        item.enabled = True
                
                self._display_list()
        
        except Exception as e:
            logger.warning(f"Interactive mode error: {e}")
            # Fall back to simple mode
            return self.review_simple()
        
        # Return enabled keywords
        enabled = [item.text for item in self.items if item.enabled]
        print(f"\n  ✓ {len(enabled)} keywords enabled")
        return enabled
    
    def _display_list(self):
        """Display the keyword list with cursor."""
        # Simple display (works in all terminals)
        print("\n" + "─" * 60)
        
        # Group by category
        entities = [i for i in self.items if i.category == "entity"]
        general = [i for i in self.items if i.category == "general"]
        custom = [i for i in self.items if i.category == "custom"]
        
        row = 0
        
        if entities:
            print("  🏷️  ENTITIES (names, places, dates)")
            for item in entities:
                marker = "→" if row == self.cursor else " "
                status = "✓" if item.enabled else "✗"
                color_start = "" if item.enabled else "\033[90m"
                color_end = "\033[0m" if not item.enabled else ""
                print(f"  {marker} [{status}] {color_start}{item.text} ({item.entity_type}){color_end}")
                row += 1
            print()
        
        if general:
            print("  🔑 KEYWORDS")
            for item in general:
                marker = "→" if row == self.cursor else " "
                status = "✓" if item.enabled else "✗"
                color_start = "" if item.enabled else "\033[90m"
                color_end = "\033[0m" if not item.enabled else ""
                print(f"  {marker} [{status}] {color_start}{item.text}{color_end}")
                row += 1
            print()
        
        if custom:
            print("  ➕ CUSTOM")
            for item in custom:
                marker = "→" if row == self.cursor else " "
                status = "✓" if item.enabled else "✗"
                print(f"  {marker} [{status}] {item.text}")
                row += 1
        
        enabled_count = sum(1 for i in self.items if i.enabled)
        print(f"\n  Enabled: {enabled_count}/{len(self.items)}")
        print("  [SPACE] toggle | [a] add | [s] save | [q] quit")
    
    def _add_custom_keyword(self):
        """Add a custom keyword."""
        print("\n  Enter custom keyword (or press Enter to cancel):")
        try:
            custom = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        
        if custom:
            self.items.append(KeywordItem(
                text=custom,
                category="custom",
                enabled=True
            ))
            self.cursor = len(self.items) - 1
            print(f"  ✓ Added: {custom}")
    
    def review_simple(self) -> List[str]:
        """
        Simple review mode (fallback for non-interactive terminals).
        
        Returns:
            List of enabled keywords
        """
        print("\n" + "=" * 60)
        print("  KEYWORD REVIEW (Simple Mode)")
        print("=" * 60)
        
        # Show all keywords
        print("\n  Current keywords:")
        for i, item in enumerate(self.items, 1):
            category = f"[{item.entity_type}]" if item.entity_type else ""
            print(f"  {i:3}. {item.text} {category}")
        
        print(f"\n  Total: {len(self.items)} keywords")
        print("\n  Options:")
        print("    - Type numbers to remove (e.g., '3,7,12')")
        print("    - Type 'add:keyword' to add new keywords")
        print("    - Press Enter to continue with all keywords")
        
        try:
            response = input("\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            response = ""
        
        if response:
            # Handle removals
            remove_parts = [p.strip() for p in response.split(',') if not p.strip().startswith('add:')]
            for part in remove_parts:
                try:
                    idx = int(part) - 1
                    if 0 <= idx < len(self.items):
                        self.items[idx].enabled = False
                        print(f"  ✗ Removed: {self.items[idx].text}")
                except ValueError:
                    pass
            
            # Handle additions
            for part in response.split(','):
                part = part.strip()
                if part.lower().startswith('add:'):
                    new_kw = part[4:].strip()
                    if new_kw:
                        self.items.append(KeywordItem(text=new_kw, category="custom", enabled=True))
                        print(f"  ✓ Added: {new_kw}")
        
        self._save_removed_keywords()
        
        enabled = [item.text for item in self.items if item.enabled]
        return enabled


def review_keywords(
    keywords: List[str],
    entities: List[Dict] = None,
    topic: str = "",
    project_dir: Path = None
) -> List[str]:
    """
    Review and edit keywords before download.
    
    Args:
        keywords: List of extracted keywords
        entities: List of entity dicts
        topic: Main topic
        project_dir: Project directory for saving removed keywords
        
    Returns:
        List of enabled keywords
    """
    reviewer = KeywordReviewer(keywords, entities, topic)
    
    # Set up removed keywords file
    if project_dir:
        reviewer.set_removed_keywords_file(str(project_dir / ".cache" / "removed_keywords.json"))
    
    # Try interactive mode first
    try:
        # Check if we're in an interactive terminal
        if sys.stdin.isatty():
            return reviewer.review_interactive()
    except:
        pass
    
    # Fall back to simple mode
    return reviewer.review_simple()


# =============================================================================
# FACE DETECTION PREFERENCE
# =============================================================================

def prompt_face_preference() -> str:
    """
    Prompt user for face detection preference.
    
    Returns:
        One of: "no_faces", "few_faces", "any", "prefer_faces"
    """
    print("\n" + "─" * 60)
    print("  FACE DETECTION PREFERENCE")
    print("─" * 60)
    print("\n  How should footage with faces be handled?")
    print()
    print("  1. 🏔️  No faces      - Prefer scenery, B-roll, landscapes")
    print("  2. 👥 Few faces     - Some people OK (rescue workers, crowds from distance)")
    print("  3. 🎲 Any           - No preference (use whatever matches best)")
    print("  4. 🧑 Prefer faces  - Human interest, reactions, interviews")
    print()
    
    while True:
        try:
            choice = input("  Select [1-4, default=2]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return "few_faces"
        
        if not choice or choice == "2":
            print("  ✓ Selected: Few faces (some people OK)")
            return "few_faces"
        elif choice == "1":
            print("  ✓ Selected: No faces (scenery/B-roll)")
            return "no_faces"
        elif choice == "3":
            print("  ✓ Selected: Any (no preference)")
            return "any"
        elif choice == "4":
            print("  ✓ Selected: Prefer faces (human interest)")
            return "prefer_faces"
        else:
            print("  Invalid choice. Enter 1-4.")


# =============================================================================
# CLIP GRADING
# =============================================================================

@dataclass
class ClipGrade:
    """Grade for a single clip."""
    clip_path: str
    clip_name: str
    grade: int  # 1-5 stars
    segment_index: int
    voiceover_text: str
    graded_at: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""


class ClipGrader:
    """Interactive clip grading interface."""
    
    def __init__(self, grades_file: str = "clip_grades.json"):
        self.grades_file = Path(grades_file)
        self.grades: Dict[str, ClipGrade] = {}
        self._load_grades()
    
    def _load_grades(self):
        """Load existing grades from file."""
        if self.grades_file.exists():
            try:
                with open(self.grades_file, 'r') as f:
                    data = json.load(f)
                
                for path, grade_data in data.get('grades', {}).items():
                    self.grades[path] = ClipGrade(**grade_data)
                
                logger.info(f"Loaded {len(self.grades)} existing clip grades")
            except Exception as e:
                logger.warning(f"Could not load grades: {e}")
    
    def _save_grades(self):
        """Save grades to file."""
        try:
            self.grades_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'grades': {path: asdict(grade) for path, grade in self.grades.items()},
                'updated': datetime.now().isoformat()
            }
            
            with open(self.grades_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save grades: {e}")
    
    def get_grade(self, clip_path: str) -> Optional[int]:
        """Get existing grade for a clip."""
        if clip_path in self.grades:
            return self.grades[clip_path].grade
        return None
    
    def get_adjustment(self, clip_path: str, adjustments: Dict[int, float]) -> float:
        """
        Get confidence adjustment based on clip grade.
        
        Args:
            clip_path: Path to clip
            adjustments: Dict mapping grade (1-5) to adjustment
            
        Returns:
            Confidence adjustment (positive or negative)
        """
        grade = self.get_grade(clip_path)
        if grade and grade in adjustments:
            return adjustments[grade]
        return 0.0
    
    def grade_matches(self, matches: List[Any], allow_skip: bool = True) -> bool:
        """
        Grade matched clips interactively.
        
        Args:
            matches: List of MatchResult objects
            allow_skip: Allow user to skip grading
            
        Returns:
            True if grading completed, False if skipped
        """
        print("\n" + "=" * 70)
        print("  CLIP GRADING")
        print("=" * 70)
        print("  Rate the matched clips to improve future selections.")
        print("  Scale: 1=Bad, 2=Poor, 3=Average, 4=Good, 5=Excellent")
        print()
        
        if allow_skip:
            print("  Press 's' to skip grading, or any key to continue...")
            try:
                key = getch()
                if key.lower() == 's':
                    print("  Skipping grading.")
                    return False
            except:
                pass
        
        graded_count = 0
        total = len(matches)
        
        for i, match in enumerate(matches):
            if not hasattr(match, 'primary_match') or not match.primary_match:
                continue
            
            pm = match.primary_match
            if not hasattr(pm, 'video_segment') or not pm.video_segment:
                continue
            
            vid_seg = pm.video_segment
            clip_path = vid_seg.source_file
            clip_name = Path(clip_path).stem if clip_path else "Unknown"
            
            # Get voiceover text
            vo_text = ""
            if hasattr(pm, 'voiceover_segment') and pm.voiceover_segment:
                vo_text = pm.voiceover_segment.text[:80] + "..." if len(pm.voiceover_segment.text) > 80 else pm.voiceover_segment.text
            
            print(f"\n  [{i+1}/{total}] {clip_name}")
            print(f"  Voiceover: \"{vo_text}\"")
            print(f"  Confidence: {pm.confidence:.2f}")
            
            # Check existing grade
            existing = self.get_grade(clip_path)
            if existing:
                print(f"  Current grade: {'⭐' * existing}")
            
            print("\n  Rate [1-5], Enter to skip, 'q' to quit grading:")
            
            try:
                key = getch()
            except:
                key = input("  > ").strip()
            
            if key.lower() == 'q':
                break
            
            try:
                grade = int(key)
                if 1 <= grade <= 5:
                    self.grades[clip_path] = ClipGrade(
                        clip_path=clip_path,
                        clip_name=clip_name,
                        grade=grade,
                        segment_index=i,
                        voiceover_text=vo_text
                    )
                    graded_count += 1
                    print(f"  ✓ Graded: {'⭐' * grade}")
            except ValueError:
                print("  Skipped.")
        
        self._save_grades()
        print(f"\n  ✓ Graded {graded_count} clips")
        return True


def grade_clips_after_match(
    matches: List[Any],
    project_dir: Path = None,
    allow_skip: bool = True
) -> ClipGrader:
    """
    Prompt user to grade clips after matching.
    
    Args:
        matches: List of MatchResult objects
        project_dir: Project directory for grades file
        allow_skip: Allow skipping
        
    Returns:
        ClipGrader instance with grades
    """
    grades_file = "clip_grades.json"
    if project_dir:
        grades_file = str(project_dir / ".cache" / "clip_grades.json")
    
    grader = ClipGrader(grades_file)
    grader.grade_matches(matches, allow_skip)
    return grader


# =============================================================================
# GLOBAL CACHE UTILITIES
# =============================================================================

class GlobalCache:
    """
    Manages cross-project cache for transcriptions, embeddings, and grades.
    """
    
    def __init__(self, global_cache_dir: str = None, install_dir: str = None):
        """
        Initialize global cache.
        
        Args:
            global_cache_dir: Explicit path, or None to use install_dir/.global_cache
            install_dir: Installation directory
        """
        if global_cache_dir:
            self.cache_dir = Path(global_cache_dir)
        elif install_dir:
            self.cache_dir = Path(install_dir) / ".global_cache"
        else:
            self.cache_dir = Path.home() / ".voiceover-matcher-cache"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Sub-directories
        self.transcripts_dir = self.cache_dir / "transcripts"
        self.embeddings_dir = self.cache_dir / "embeddings"
        self.grades_file = self.cache_dir / "global_grades.json"
        self.scenes_dir = self.cache_dir / "scenes"
        
        for d in [self.transcripts_dir, self.embeddings_dir, self.scenes_dir]:
            d.mkdir(exist_ok=True)
        
        self._grades: Dict[str, int] = {}
        self._load_global_grades()
    
    def _load_global_grades(self):
        """Load global clip grades."""
        if self.grades_file.exists():
            try:
                with open(self.grades_file, 'r') as f:
                    data = json.load(f)
                self._grades = data.get('grades', {})
            except:
                pass
    
    def _save_global_grades(self):
        """Save global clip grades."""
        try:
            with open(self.grades_file, 'w') as f:
                json.dump({
                    'grades': self._grades,
                    'updated': datetime.now().isoformat()
                }, f, indent=2)
        except:
            pass
    
    def get_video_hash(self, video_path: str) -> str:
        """Generate hash for video file (based on path + size + mtime)."""
        import hashlib
        path = Path(video_path)
        
        try:
            stat = path.stat()
            key = f"{path.name}:{stat.st_size}:{stat.st_mtime}"
            return hashlib.md5(key.encode()).hexdigest()[:16]
        except:
            return hashlib.md5(str(path).encode()).hexdigest()[:16]
    
    def get_transcript(self, video_hash: str) -> Optional[List[Dict]]:
        """Get cached transcript."""
        cache_file = self.transcripts_dir / f"{video_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return None
    
    def save_transcript(self, video_hash: str, segments: List[Any]):
        """Save transcript to global cache."""
        cache_file = self.transcripts_dir / f"{video_hash}.json"
        try:
            data = [s.to_dict() if hasattr(s, 'to_dict') else s for s in segments]
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except:
            pass
    
    def get_grade(self, clip_path: str) -> Optional[int]:
        """Get global grade for clip."""
        # Normalize path for lookup
        key = Path(clip_path).name.lower()
        return self._grades.get(key)
    
    def set_grade(self, clip_path: str, grade: int):
        """Set global grade for clip."""
        key = Path(clip_path).name.lower()
        self._grades[key] = grade
        self._save_global_grades()
    
    def merge_project_grades(self, project_grades_file: str):
        """Merge project grades into global cache."""
        try:
            with open(project_grades_file, 'r') as f:
                data = json.load(f)
            
            for path, grade_info in data.get('grades', {}).items():
                grade = grade_info.get('grade') if isinstance(grade_info, dict) else grade_info
                if grade:
                    self.set_grade(path, grade)
        except:
            pass
