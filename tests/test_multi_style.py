"""
Comprehensive tests for multi-style OTIO generation module.

Covers:
- OTIOStyle dataclass
- Preset styles (default, strict, stock_heavy, fast_paced, cinematic)
- MultiStyleOTIOGenerator initialization
- Style management (add, create custom)
- Config conversion for matching and output
- Track definitions for stock footage
- Source type detection

Created: 2026-01-09 (Phase 8.2)
"""

from unittest.mock import Mock, MagicMock, patch
import pytest
import tempfile
from pathlib import Path

from src.multi_style import (
    OTIOStyle,
    STYLE_DEFAULT,
    STYLE_STRICT,
    STYLE_STOCK_HEAVY,
    STYLE_FAST_PACED,
    STYLE_CINEMATIC,
    PRESET_STYLES,
    MultiStyleOTIOGenerator,
    get_track_for_source,
    is_stock_footage,
    STOCK_FOOTAGE_TRACK,
    IMAGE_TRACK
)


# ============================================================================
# Test OTIOStyle Dataclass
# ============================================================================

class TestOTIOStyle:
    """Test OTIOStyle dataclass"""

    def test_init_default_values(self):
        """Test initialization with default values"""
        style = OTIOStyle(name="test", description="Test style")

        assert style.name == "test"
        assert style.description == "Test style"
        assert style.confidence_threshold == 0.5
        assert style.num_alternatives == 2
        assert style.include_strategy_tracks is True

    def test_init_custom_values(self):
        """Test initialization with custom values"""
        style = OTIOStyle(
            name="custom",
            description="Custom style",
            confidence_threshold=0.7,
            num_alternatives=3,
            prefer_longer_clips=True
        )

        assert style.confidence_threshold == 0.7
        assert style.num_alternatives == 3
        assert style.prefer_longer_clips is True

    def test_to_dict(self):
        """Test conversion to dict"""
        style = OTIOStyle(
            name="test",
            description="Test style",
            confidence_threshold=0.6,
            num_alternatives=3
        )

        result = style.to_dict()

        assert result['name'] == "test"
        assert result['description'] == "Test style"
        assert result['confidence_threshold'] == 0.6
        assert result['num_alternatives'] == 3
        assert 'strategy_tracks' in result


# ============================================================================
# Test Preset Styles
# ============================================================================

class TestPresetStyles:
    """Test preset style configurations"""

    def test_style_default(self):
        """Test default style preset"""
        assert STYLE_DEFAULT.name == "default"
        assert STYLE_DEFAULT.confidence_threshold == 0.5
        assert STYLE_DEFAULT.num_alternatives == 2
        assert STYLE_DEFAULT.include_strategy_tracks is True

    def test_style_strict(self):
        """Test strict style preset"""
        assert STYLE_STRICT.name == "strict"
        assert STYLE_STRICT.confidence_threshold == 0.7
        assert STYLE_STRICT.num_alternatives == 1
        assert STYLE_STRICT.include_strategy_tracks is False

    def test_style_stock_heavy(self):
        """Test stock_heavy style preset"""
        assert STYLE_STOCK_HEAVY.name == "stock_heavy"
        assert STYLE_STOCK_HEAVY.prefer_stock_footage is True
        assert STYLE_STOCK_HEAVY.prefer_youtube is False
        assert STYLE_STOCK_HEAVY.num_alternatives == 3

    def test_style_fast_paced(self):
        """Test fast_paced style preset"""
        assert STYLE_FAST_PACED.name == "fast_paced"
        assert STYLE_FAST_PACED.prefer_shorter_clips is True
        assert STYLE_FAST_PACED.ideal_speed_range == (0.7, 1.0)

    def test_style_cinematic(self):
        """Test cinematic style preset"""
        assert STYLE_CINEMATIC.name == "cinematic"
        assert STYLE_CINEMATIC.prefer_longer_clips is True
        assert STYLE_CINEMATIC.ideal_speed_range == (1.0, 1.5)
        assert STYLE_CINEMATIC.confidence_threshold == 0.6

    def test_preset_styles_dict(self):
        """Test PRESET_STYLES dictionary"""
        assert len(PRESET_STYLES) == 5
        assert "default" in PRESET_STYLES
        assert "strict" in PRESET_STYLES
        assert "stock_heavy" in PRESET_STYLES
        assert "fast_paced" in PRESET_STYLES
        assert "cinematic" in PRESET_STYLES


# ============================================================================
# Test MultiStyleOTIOGenerator Initialization
# ============================================================================

class TestMultiStyleOTIOGeneratorInit:
    """Test MultiStyleOTIOGenerator initialization"""

    def test_init_creates_output_dir(self):
        """Test initialization creates output directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"

            generator = MultiStyleOTIOGenerator(str(output_dir))

            assert output_dir.exists()
            assert generator.output_dir == output_dir

    def test_init_empty_styles(self):
        """Test initialization with no styles"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = MultiStyleOTIOGenerator(temp_dir)

            assert len(generator.styles) == 0
            assert len(generator.generated_files) == 0


# ============================================================================
# Test Style Management
# ============================================================================

class TestStyleManagement:
    """Test style management functionality"""

    def test_add_style(self):
        """Test adding a style"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = MultiStyleOTIOGenerator(temp_dir)
            style = OTIOStyle(name="test", description="Test style")

            generator.add_style(style)

            assert len(generator.styles) == 1
            assert generator.styles[0] == style

    def test_add_multiple_styles(self):
        """Test adding multiple styles"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = MultiStyleOTIOGenerator(temp_dir)

            generator.add_style(STYLE_DEFAULT)
            generator.add_style(STYLE_STRICT)

            assert len(generator.styles) == 2

    def test_add_preset(self):
        """Test adding a preset style by name"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = MultiStyleOTIOGenerator(temp_dir)

            generator.add_preset("strict")

            assert len(generator.styles) == 1
            assert generator.styles[0].name == "strict"

    def test_add_preset_unknown(self):
        """Test adding unknown preset (logs warning)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = MultiStyleOTIOGenerator(temp_dir)

            generator.add_preset("unknown_preset")

            # Should not add anything
            assert len(generator.styles) == 0

    def test_create_custom_style(self):
        """Test creating a custom style"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = MultiStyleOTIOGenerator(temp_dir)

            style = generator.create_custom_style(
                name="my_style",
                description="My custom style",
                confidence_threshold=0.8,
                num_alternatives=4,
                prefer_longer_clips=True
            )

            assert style.name == "my_style"
            assert style.description == "My custom style"
            assert style.confidence_threshold == 0.8
            assert style.num_alternatives == 4
            assert style.prefer_longer_clips is True

    def test_create_custom_style_ignores_invalid_attrs(self):
        """Test creating custom style ignores invalid attributes"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = MultiStyleOTIOGenerator(temp_dir)

            # Should not raise error
            style = generator.create_custom_style(
                name="test",
                description="Test",
                invalid_attr="value"
            )

            assert not hasattr(style, 'invalid_attr')


# ============================================================================
# Test Config Conversion
# ============================================================================

class TestConfigConversion:
    """Test style to config conversion"""

    def test_get_style_config_for_matching(self):
        """Test converting style to matching config"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = MultiStyleOTIOGenerator(temp_dir)
            style = OTIOStyle(
                name="test",
                description="Test",
                confidence_threshold=0.7,
                prefer_longer_clips=True,
                prefer_stock_footage=True
            )

            config = generator.get_style_config_for_matching(style)

            assert config['confidence_threshold'] == 0.7
            assert config['prefer_longer_clips'] is True
            assert config['prefer_stock_footage'] is True
            assert 'num_alternatives' not in config  # Output config only

    def test_get_style_config_for_output(self):
        """Test converting style to output config"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = MultiStyleOTIOGenerator(temp_dir)
            style = OTIOStyle(
                name="test",
                description="Test",
                num_alternatives=3,
                include_strategy_tracks=False,
                strategy_tracks=["visual_first"]
            )

            config = generator.get_style_config_for_output(style)

            assert config['num_alternatives'] == 3
            assert config['include_strategy_tracks'] is False
            assert config['strategy_tracks'] == ["visual_first"]
            assert 'confidence_threshold' not in config  # Matching config only

    def test_config_separation(self):
        """Test matching and output configs are separate"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = MultiStyleOTIOGenerator(temp_dir)
            style = STYLE_STRICT

            matching_config = generator.get_style_config_for_matching(style)
            output_config = generator.get_style_config_for_output(style)

            # Matching config should have confidence settings
            assert 'confidence_threshold' in matching_config
            # Output config should have track settings
            assert 'num_alternatives' in output_config
            # No overlap
            assert 'num_alternatives' not in matching_config
            assert 'confidence_threshold' not in output_config


# ============================================================================
# Test Track Definitions
# ============================================================================

class TestTrackDefinitions:
    """Test track definitions for different source types"""

    def test_get_track_for_youtube(self):
        """Test track assignment for YouTube source"""
        track = get_track_for_source("youtube")

        assert track == "V1"  # Primary track

    def test_get_track_for_pexels(self):
        """Test track assignment for Pexels stock footage"""
        track = get_track_for_source("pexels")

        assert track == STOCK_FOOTAGE_TRACK

    def test_get_track_for_pixabay(self):
        """Test track assignment for Pixabay stock footage"""
        track = get_track_for_source("pixabay")

        assert track == STOCK_FOOTAGE_TRACK

    def test_get_track_for_image(self):
        """Test track assignment for image sources"""
        track = get_track_for_source("image")

        assert track == IMAGE_TRACK

    def test_get_track_for_unknown_source(self):
        """Test track assignment for unknown source"""
        track = get_track_for_source("unknown")

        assert track == "V1"  # Default


# ============================================================================
# Test Source Type Detection
# ============================================================================

class TestSourceTypeDetection:
    """Test stock footage detection"""

    def test_is_stock_footage_pexels(self):
        """Test detection of Pexels stock footage"""
        metadata = {"source": "pexels"}

        assert is_stock_footage(metadata) is True

    def test_is_stock_footage_pixabay(self):
        """Test detection of Pixabay stock footage"""
        metadata = {"source": "pixabay"}

        assert is_stock_footage(metadata) is True

    def test_is_stock_footage_youtube(self):
        """Test YouTube is not stock footage"""
        metadata = {"source": "youtube"}

        assert is_stock_footage(metadata) is False

    def test_is_stock_footage_tag(self):
        """Test detection via tags"""
        metadata = {"source": "other", "tags": ["stock_footage"]}

        assert is_stock_footage(metadata) is True

    def test_is_stock_footage_stock_tag(self):
        """Test detection via 'stock' tag"""
        metadata = {"source": "other", "tags": ["stock"]}

        assert is_stock_footage(metadata) is True

    def test_is_stock_footage_flag(self):
        """Test detection via is_stock_footage flag"""
        metadata = {"is_stock_footage": True}

        assert is_stock_footage(metadata) is True

    def test_is_stock_footage_empty_metadata(self):
        """Test with empty metadata"""
        assert is_stock_footage({}) is False

    def test_is_stock_footage_none_metadata(self):
        """Test with None metadata"""
        assert is_stock_footage(None) is False

    def test_is_stock_footage_case_insensitive(self):
        """Test case-insensitive source matching"""
        metadata = {"source": "Pexels"}

        assert is_stock_footage(metadata) is True


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases"""

    def test_style_with_empty_strategy_tracks(self):
        """Test style with empty strategy tracks list"""
        style = OTIOStyle(
            name="test",
            description="Test",
            strategy_tracks=[]
        )

        assert style.strategy_tracks == []

    def test_to_dict_with_tuple(self):
        """Test to_dict handles tuple values"""
        style = OTIOStyle(
            name="test",
            description="Test",
            ideal_speed_range=(0.9, 1.1)
        )

        result = style.to_dict()

        assert result['ideal_speed_range'] == (0.9, 1.1)

    def test_generator_output_dir_already_exists(self):
        """Test initialization when output dir already exists"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir()

            # Should not raise error
            generator = MultiStyleOTIOGenerator(str(output_dir))

            assert generator.output_dir == output_dir

    def test_add_preset_all_presets(self):
        """Test adding all preset styles"""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = MultiStyleOTIOGenerator(temp_dir)

            for preset_name in PRESET_STYLES.keys():
                generator.add_preset(preset_name)

            assert len(generator.styles) == len(PRESET_STYLES)

    def test_style_default_factory_for_strategy_tracks(self):
        """Test default factory creates new list for each instance"""
        style1 = OTIOStyle(name="test1", description="Test 1")
        style2 = OTIOStyle(name="test2", description="Test 2")

        # Should be different list instances
        assert style1.strategy_tracks is not style2.strategy_tracks

        # But with same content
        assert style1.strategy_tracks == style2.strategy_tracks


# ============================================================================
# Test Interactive Prompts
# ============================================================================

class TestPromptForSecondStyle:
    """Test prompt_for_second_style function"""

    def test_select_preset_1_strict(self):
        """Test selecting preset 1 (strict)"""
        from src.multi_style import prompt_for_second_style

        with patch('builtins.input', return_value="1"):
            style = prompt_for_second_style()

        assert style.name == "strict"

    def test_select_preset_2_stock_heavy(self):
        """Test selecting preset 2 (stock_heavy)"""
        from src.multi_style import prompt_for_second_style

        with patch('builtins.input', return_value="2"):
            style = prompt_for_second_style()

        assert style.name == "stock_heavy"

    def test_select_preset_3_fast_paced(self):
        """Test selecting preset 3 (fast_paced)"""
        from src.multi_style import prompt_for_second_style

        with patch('builtins.input', return_value="3"):
            style = prompt_for_second_style()

        assert style.name == "fast_paced"

    def test_select_preset_4_cinematic(self):
        """Test selecting preset 4 (cinematic)"""
        from src.multi_style import prompt_for_second_style

        with patch('builtins.input', return_value="4"):
            style = prompt_for_second_style()

        assert style.name == "cinematic"

    def test_select_preset_by_name_strict(self):
        """Test selecting preset by name 'strict'"""
        from src.multi_style import prompt_for_second_style

        with patch('builtins.input', return_value="strict"):
            style = prompt_for_second_style()

        assert style.name == "strict"

    def test_select_preset_by_name_stock_heavy(self):
        """Test selecting preset by name 'stock_heavy'"""
        from src.multi_style import prompt_for_second_style

        with patch('builtins.input', return_value="stock_heavy"):
            style = prompt_for_second_style()

        assert style.name == "stock_heavy"

    def test_select_preset_by_name_fast_paced(self):
        """Test selecting preset by name 'fast_paced'"""
        from src.multi_style import prompt_for_second_style

        with patch('builtins.input', return_value="fast_paced"):
            style = prompt_for_second_style()

        assert style.name == "fast_paced"

    def test_select_preset_by_name_cinematic(self):
        """Test selecting preset by name 'cinematic'"""
        from src.multi_style import prompt_for_second_style

        with patch('builtins.input', return_value="cinematic"):
            style = prompt_for_second_style()

        assert style.name == "cinematic"

    def test_default_selection_empty_input(self):
        """Test default selection with empty input"""
        from src.multi_style import prompt_for_second_style

        with patch('builtins.input', return_value=""):
            style = prompt_for_second_style()

        assert style.name == "strict"

    def test_invalid_selection_defaults_to_strict(self):
        """Test invalid selection defaults to strict"""
        from src.multi_style import prompt_for_second_style

        with patch('builtins.input', return_value="invalid_choice"):
            style = prompt_for_second_style()

        assert style.name == "strict"

    def test_eof_error_handling(self):
        """Test EOFError returns strict style"""
        from src.multi_style import prompt_for_second_style

        with patch('builtins.input', side_effect=EOFError):
            style = prompt_for_second_style()

        assert style.name == "strict"

    def test_keyboard_interrupt_handling(self):
        """Test KeyboardInterrupt returns strict style"""
        from src.multi_style import prompt_for_second_style

        with patch('builtins.input', side_effect=KeyboardInterrupt):
            style = prompt_for_second_style()

        assert style.name == "strict"

    def test_select_custom_option_5(self):
        """Test selecting custom option 5"""
        from src.multi_style import prompt_for_second_style

        # Mock sequence for custom style
        inputs = iter(["5", "0.65", "2", "1", "2", "y", "mycustom", "My custom desc"])

        with patch('builtins.input', lambda _: next(inputs)):
            style = prompt_for_second_style()

        assert style.name == "mycustom"
        assert style.confidence_threshold == 0.65

    def test_select_custom_by_name(self):
        """Test selecting 'custom' by name"""
        from src.multi_style import prompt_for_second_style

        inputs = iter(["custom", "", "", "", "", "", "", ""])

        with patch('builtins.input', lambda _: next(inputs)):
            style = prompt_for_second_style()

        assert style.name == "custom"


class TestPromptCustomStyle:
    """Test _prompt_custom_style function"""

    def test_all_defaults(self):
        """Test custom style with all defaults"""
        from src.multi_style import _prompt_custom_style

        inputs = iter(["", "", "", "", "", "", ""])

        with patch('builtins.input', lambda _: next(inputs)):
            style = _prompt_custom_style()

        assert style.name == "custom"
        assert style.confidence_threshold == 0.5
        assert style.num_alternatives == 2
        assert style.prefer_shorter_clips is False
        assert style.prefer_longer_clips is False
        assert style.include_strategy_tracks is True

    def test_custom_confidence(self):
        """Test custom confidence threshold"""
        from src.multi_style import _prompt_custom_style

        inputs = iter(["0.75", "", "", "", "", "", ""])

        with patch('builtins.input', lambda _: next(inputs)):
            style = _prompt_custom_style()

        assert style.confidence_threshold == 0.75

    def test_custom_alternatives(self):
        """Test custom number of alternatives"""
        from src.multi_style import _prompt_custom_style

        inputs = iter(["", "5", "", "", "", "", ""])

        with patch('builtins.input', lambda _: next(inputs)):
            style = _prompt_custom_style()

        assert style.num_alternatives == 5

    def test_prefer_shorter_clips(self):
        """Test selecting shorter clips preference"""
        from src.multi_style import _prompt_custom_style

        inputs = iter(["", "", "2", "", "", "", ""])

        with patch('builtins.input', lambda _: next(inputs)):
            style = _prompt_custom_style()

        assert style.prefer_shorter_clips is True
        assert style.prefer_longer_clips is False

    def test_prefer_longer_clips(self):
        """Test selecting longer clips preference"""
        from src.multi_style import _prompt_custom_style

        inputs = iter(["", "", "3", "", "", "", ""])

        with patch('builtins.input', lambda _: next(inputs)):
            style = _prompt_custom_style()

        assert style.prefer_longer_clips is True
        assert style.prefer_shorter_clips is False

    def test_prefer_stock_footage(self):
        """Test selecting stock footage preference"""
        from src.multi_style import _prompt_custom_style

        inputs = iter(["", "", "", "2", "", "", ""])

        with patch('builtins.input', lambda _: next(inputs)):
            style = _prompt_custom_style()

        assert style.prefer_stock_footage is True
        assert style.prefer_youtube is False

    def test_prefer_youtube(self):
        """Test selecting YouTube preference"""
        from src.multi_style import _prompt_custom_style

        inputs = iter(["", "", "", "3", "", "", ""])

        with patch('builtins.input', lambda _: next(inputs)):
            style = _prompt_custom_style()

        assert style.prefer_stock_footage is False
        assert style.prefer_youtube is True

    def test_disable_strategy_tracks(self):
        """Test disabling strategy tracks"""
        from src.multi_style import _prompt_custom_style

        inputs = iter(["", "", "", "", "n", "", ""])

        with patch('builtins.input', lambda _: next(inputs)):
            style = _prompt_custom_style()

        assert style.include_strategy_tracks is False

    def test_custom_name(self):
        """Test custom style name"""
        from src.multi_style import _prompt_custom_style

        inputs = iter(["", "", "", "", "", "my_named_style", ""])

        with patch('builtins.input', lambda _: next(inputs)):
            style = _prompt_custom_style()

        assert style.name == "my_named_style"

    def test_custom_description(self):
        """Test custom style description"""
        from src.multi_style import _prompt_custom_style

        inputs = iter(["", "", "", "", "", "", "My custom description"])

        with patch('builtins.input', lambda _: next(inputs)):
            style = _prompt_custom_style()

        assert style.description == "My custom description"

    def test_eof_error_returns_partial_style(self):
        """Test EOFError returns partial custom style"""
        from src.multi_style import _prompt_custom_style

        with patch('builtins.input', side_effect=EOFError):
            style = _prompt_custom_style()

        assert style.name == "custom"

    def test_keyboard_interrupt_returns_partial_style(self):
        """Test KeyboardInterrupt returns partial custom style"""
        from src.multi_style import _prompt_custom_style

        with patch('builtins.input', side_effect=KeyboardInterrupt):
            style = _prompt_custom_style()

        assert style.name == "custom"

    def test_value_error_returns_partial_style(self):
        """Test invalid number input returns partial style"""
        from src.multi_style import _prompt_custom_style

        inputs = iter(["not_a_number"])  # Invalid confidence value

        with patch('builtins.input', lambda _: next(inputs)):
            style = _prompt_custom_style()

        # Should return style with default confidence
        assert style.name == "custom"


class TestPromptMultiStyleEnabled:
    """Test prompt_multi_style_enabled function"""

    def test_yes_lowercase(self):
        """Test 'y' returns True"""
        from src.multi_style import prompt_multi_style_enabled

        with patch('builtins.input', return_value="y"):
            result = prompt_multi_style_enabled()

        assert result is True

    def test_yes_full_word(self):
        """Test 'yes' returns True"""
        from src.multi_style import prompt_multi_style_enabled

        with patch('builtins.input', return_value="yes"):
            result = prompt_multi_style_enabled()

        assert result is True

    def test_no_returns_false(self):
        """Test 'n' returns False"""
        from src.multi_style import prompt_multi_style_enabled

        with patch('builtins.input', return_value="n"):
            result = prompt_multi_style_enabled()

        assert result is False

    def test_empty_returns_false(self):
        """Test empty input returns False (default)"""
        from src.multi_style import prompt_multi_style_enabled

        with patch('builtins.input', return_value=""):
            result = prompt_multi_style_enabled()

        assert result is False

    def test_random_input_returns_false(self):
        """Test random input returns False"""
        from src.multi_style import prompt_multi_style_enabled

        with patch('builtins.input', return_value="maybe"):
            result = prompt_multi_style_enabled()

        assert result is False

    def test_eof_error_returns_false(self):
        """Test EOFError returns False"""
        from src.multi_style import prompt_multi_style_enabled

        with patch('builtins.input', side_effect=EOFError):
            result = prompt_multi_style_enabled()

        assert result is False

    def test_keyboard_interrupt_returns_false(self):
        """Test KeyboardInterrupt returns False"""
        from src.multi_style import prompt_multi_style_enabled

        with patch('builtins.input', side_effect=KeyboardInterrupt):
            result = prompt_multi_style_enabled()

        assert result is False
