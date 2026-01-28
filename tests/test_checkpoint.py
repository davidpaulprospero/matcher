import pytest
#!/usr/bin/env python3
"""
Test script for Checkpoint and Saved Keywords features.

Run: python tests/test_checkpoint.py
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.checkpoint import (
    CheckpointManager, KeywordManager, SavedKeywords,
    CheckpointData, STAGE_ORDER, format_resume_prompt, format_keyword_prompt
)


@pytest.mark.integration
def test_checkpoint_manager():
    """Test CheckpointManager functionality"""
    print("\n" + "=" * 60)
    print("  TEST: CheckpointManager")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Test 1: Create checkpoint manager
        print("\n  [1] Creating CheckpointManager...")
        cm = CheckpointManager(tmpdir, config_hash="test123")
        assert not cm.exists(), "Checkpoint should not exist yet"
        print("      ✓ Checkpoint manager created")
        
        # Test 2: Save checkpoint for ANALYZE stage
        print("\n  [2] Saving ANALYZE checkpoint...")
        cm.set_voiceover(str(tmpdir / "test.srt"))
        cm.save("ANALYZE", {
            'keywords': ['keyword1', 'keyword2'],
            'segment_count': 5,
            'topic_context': 'Test topic'
        })
        assert cm.exists(), "Checkpoint file should exist"
        assert (tmpdir / "checkpoint.json").exists()
        print("      ✓ Checkpoint saved")
        
        # Test 3: Load checkpoint
        print("\n  [3] Loading checkpoint...")
        cm2 = CheckpointManager(tmpdir)
        data = cm2.load()
        assert data is not None, "Should load checkpoint data"
        assert data.last_completed_stage == "ANALYZE"
        assert data.analyze.get('keywords') == ['keyword1', 'keyword2']
        print(f"      ✓ Loaded: stage={data.last_completed_stage}, keywords={data.analyze.get('keywords')}")
        
        # Test 4: should_skip_stage
        print("\n  [4] Testing should_skip_stage...")
        assert cm2.should_skip_stage("ANALYZE") == True, "ANALYZE should be skipped"
        assert cm2.should_skip_stage("DOWNLOAD") == False, "DOWNLOAD should not be skipped"
        print("      ✓ Stage skip logic works")
        
        # Test 5: Validate checkpoint
        print("\n  [5] Validating checkpoint...")
        validation = cm2.validate()
        assert validation['valid'] == True
        assert validation['resume_from'] == "ENTITY_IMAGES"
        assert "ANALYZE" in validation['completed_stages']
        print(f"      ✓ Valid checkpoint, resume from: {validation['resume_from']}")
        
        # Test 6: Save more stages
        print("\n  [6] Saving multiple stages...")
        for stage in ["ENTITY_IMAGES", "ENTITY_VIDEOS", "DOWNLOAD"]:
            cm2.save(stage, {'count': 10})
        
        assert cm2.data.last_completed_stage == "DOWNLOAD"
        assert cm2.should_skip_stage("DOWNLOAD") == True
        assert cm2.should_skip_stage("REMIX") == False
        print(f"      ✓ Last stage: {cm2.data.last_completed_stage}")
        
        # Test 7: Get summary
        print("\n  [7] Getting summary...")
        summary = cm2.get_summary()
        assert "DOWNLOAD" in summary
        print(f"      ✓ Summary:\n{summary}")
        
        # Test 8: Clear checkpoint
        print("\n  [8] Clearing checkpoint...")
        cm2.clear()
        assert not cm2.exists(), "Checkpoint should be cleared"
        print("      ✓ Checkpoint cleared")
    
    print("\n  ✅ CheckpointManager tests PASSED")


@pytest.mark.integration
def test_keyword_manager():
    """Test KeywordManager functionality"""
    print("\n" + "=" * 60)
    print("  TEST: KeywordManager")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Test 1: Create keyword manager
        print("\n  [1] Creating KeywordManager...")
        km = KeywordManager(tmpdir)
        assert not km.has_presets(), "Should have no presets"
        print("      ✓ Keyword manager created")
        
        # Test 2: Save keywords
        print("\n  [2] Saving keywords...")
        name1 = km.save_keywords(
            keywords=['Elon Musk footage', 'Tesla factory'],
            topic_context='Elon Musk documentary',
            entities=[{'name': 'Elon Musk', 'type': 'PERSON'}],
            name='test_preset'
        )
        assert name1 == 'test_preset'
        assert km.has_presets()
        assert (tmpdir / "saved_keywords.json").exists()
        print(f"      ✓ Saved preset: {name1}")
        
        # Test 3: Save another preset (auto-name)
        print("\n  [3] Saving another preset (auto-name)...")
        name2 = km.save_keywords(
            keywords=['SpaceX launch', 'rocket footage'],
            topic_context='SpaceX missions'
        )
        assert name2 != 'test_preset'  # Should be auto-generated
        print(f"      ✓ Saved preset: {name2}")
        
        # Test 4: Get preset by name
        print("\n  [4] Getting preset by name...")
        preset = km.get_preset('test_preset')
        assert preset is not None
        assert preset.keywords == ['Elon Musk footage', 'Tesla factory']
        assert preset.topic_context == 'Elon Musk documentary'
        print(f"      ✓ Got preset: {preset.keywords}")
        
        # Test 5: Get latest preset
        print("\n  [5] Getting latest preset...")
        latest = km.get_latest()
        assert latest is not None
        # Latest should be one of the two saved presets (order may vary if saved in same second)
        assert latest.name in [name1, name2], f"Latest preset {latest.name} not in [{name1}, {name2}]"
        print(f"      ✓ Latest preset: {latest.name}")
        
        # Test 6: List presets
        print("\n  [6] Listing presets...")
        presets = km.list_presets()
        assert len(presets) == 2
        print(f"      ✓ Found {len(presets)} presets")
        
        # Test 7: Get summary
        print("\n  [7] Getting summary...")
        summary = km.get_summary()
        assert 'test_preset' in summary
        print(f"      ✓ Summary:\n{summary}")
        
        # Test 8: Delete preset
        print("\n  [8] Deleting preset...")
        deleted = km.delete_preset('test_preset')
        assert deleted == True
        assert km.get_preset('test_preset') is None
        assert len(km.list_presets()) == 1
        print("      ✓ Preset deleted")
        
        # Test 9: Reload from disk
        print("\n  [9] Reloading from disk...")
        km2 = KeywordManager(tmpdir)
        assert km2.has_presets()
        assert len(km2.list_presets()) == 1
        print("      ✓ Reloaded successfully")
    
    print("\n  ✅ KeywordManager tests PASSED")


@pytest.mark.fast
def test_stage_order():
    """Test stage ordering logic"""
    print("\n" + "=" * 60)
    print("  TEST: Stage Order Logic")
    print("=" * 60)
    
    print(f"\n  Stage order: {STAGE_ORDER}")
    
    # Test stage indices
    assert STAGE_ORDER.index("ANALYZE") < STAGE_ORDER.index("DOWNLOAD")
    assert STAGE_ORDER.index("DOWNLOAD") < STAGE_ORDER.index("TRANSCRIBE")
    assert STAGE_ORDER.index("TRANSCRIBE") < STAGE_ORDER.index("MATCH")
    assert STAGE_ORDER.index("MATCH") < STAGE_ORDER.index("OUTPUT")
    
    print("  ✓ Stage order is correct")
    
    print("\n  ✅ Stage order tests PASSED")


@pytest.mark.integration
def test_format_prompts():
    """Test prompt formatting functions"""
    print("\n" + "=" * 60)
    print("  TEST: Format Prompts")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Test resume prompt
        print("\n  [1] Testing resume prompt...")
        cm = CheckpointManager(tmpdir)
        cm.save("DOWNLOAD", {'video_count': 5})
        
        prompt = format_resume_prompt(cm)
        assert "CHECKPOINT FOUND" in prompt
        assert "[R] Resume" in prompt
        assert "[F] Fresh" in prompt
        print("      ✓ Resume prompt formatted")
        
        # Test keyword prompt
        print("\n  [2] Testing keyword prompt...")
        km = KeywordManager(tmpdir)
        km.save_keywords(['test'], topic_context='test topic')
        
        prompt = format_keyword_prompt(km)
        assert "SAVED KEYWORDS FOUND" in prompt
        assert "[U] Use saved" in prompt
        assert "[N] Generate new" in prompt
        print("      ✓ Keyword prompt formatted")
    
    print("\n  ✅ Format prompt tests PASSED")


@pytest.mark.integration
def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "=" * 60)
    print("  TEST: Edge Cases")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Test 1: Empty checkpoint load
        print("\n  [1] Loading non-existent checkpoint...")
        cm = CheckpointManager(tmpdir)
        data = cm.load()
        assert data is None
        print("      ✓ Returns None for non-existent checkpoint")
        
        # Test 2: Get stage data before save
        print("\n  [2] Getting stage data before save...")
        stage_data = cm.get_stage_data("ANALYZE")
        assert stage_data == {}
        print("      ✓ Returns empty dict for missing stage")
        
        # Test 3: should_skip with no data
        print("\n  [3] should_skip with no checkpoint...")
        assert cm.should_skip_stage("ANALYZE") == False
        print("      ✓ Returns False when no checkpoint")
        
        # Test 4: Validate empty checkpoint
        print("\n  [4] Validating empty checkpoint...")
        validation = cm.validate()
        assert validation['valid'] == False
        print("      ✓ Empty checkpoint is invalid")
        
        # Test 5: Get non-existent keyword preset
        print("\n  [5] Getting non-existent keyword preset...")
        km = KeywordManager(tmpdir)
        preset = km.get_preset('nonexistent')
        assert preset is None
        print("      ✓ Returns None for missing preset")
        
        # Test 6: Delete non-existent preset
        print("\n  [6] Deleting non-existent preset...")
        deleted = km.delete_preset('nonexistent')
        assert deleted == False
        print("      ✓ Returns False for missing preset")
    
    print("\n  ✅ Edge case tests PASSED")


@pytest.mark.integration
def test_json_structure():
    """Test that JSON files have correct structure"""
    print("\n" + "=" * 60)
    print("  TEST: JSON File Structure")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Test checkpoint.json structure
        print("\n  [1] Testing checkpoint.json structure...")
        cm = CheckpointManager(tmpdir, config_hash="abc123")
        cm.set_voiceover("/path/to/voiceover.srt")
        cm.save("ANALYZE", {'keywords': ['test'], 'segment_count': 5})
        
        with open(tmpdir / "checkpoint.json") as f:
            data = json.load(f)
        
        assert 'version' in data
        assert 'created_at' in data
        assert 'updated_at' in data
        assert 'last_completed_stage' in data
        assert 'config_hash' in data
        assert 'voiceover_path' in data
        assert 'analyze' in data
        print("      ✓ checkpoint.json has correct structure")
        print(f"      Keys: {list(data.keys())}")
        
        # Test saved_keywords.json structure
        print("\n  [2] Testing saved_keywords.json structure...")
        km = KeywordManager(tmpdir)
        km.save_keywords(
            keywords=['kw1', 'kw2'],
            topic_context='topic',
            entities=[{'name': 'Entity', 'type': 'ORG'}],
            name='preset1'
        )
        
        with open(tmpdir / "saved_keywords.json") as f:
            data = json.load(f)
        
        assert 'version' in data
        assert 'presets' in data
        assert 'preset1' in data['presets']
        
        preset = data['presets']['preset1']
        assert 'name' in preset
        assert 'created_at' in preset
        assert 'keywords' in preset
        assert 'topic_context' in preset
        assert 'entities' in preset
        print("      ✓ saved_keywords.json has correct structure")
        print(f"      Preset keys: {list(preset.keys())}")
    
    print("\n  ✅ JSON structure tests PASSED")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("  CHECKPOINT & SAVED KEYWORDS TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("CheckpointManager", test_checkpoint_manager),
        ("KeywordManager", test_keyword_manager),
        ("Stage Order", test_stage_order),
        ("Format Prompts", test_format_prompts),
        ("Edge Cases", test_edge_cases),
        ("JSON Structure", test_json_structure),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed, None))
        except Exception as e:
            results.append((name, False, str(e)))
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("  TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for _, p, _ in results if p)
    failed = len(results) - passed
    
    for name, p, error in results:
        status = "✅ PASS" if p else "❌ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"         Error: {error}")
    
    print(f"\n  Total: {passed}/{len(results)} passed")
    
    if failed > 0:
        print("\n  ❌ SOME TESTS FAILED")
        return False
    else:
        print("\n  ✅ ALL TESTS PASSED")
        return True


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
