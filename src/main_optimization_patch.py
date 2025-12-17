#!/usr/bin/env python3
"""
MAIN.PY INTEGRATION PATCH - v2.4 Performance Optimizations

This file contains the exact changes needed to integrate the three
performance optimizations into your existing main.py.

=============================================================================
SUMMARY OF CHANGES
=============================================================================

CHANGE 1: Import the optimized modules (Line ~37)
CHANGE 2: Replace stage_transcribe_index() method
CHANGE 3: Modify stage_confidence_enforcement() to use delta-aware indexing
CHANGE 4: Add performance timing/logging

=============================================================================
"""

# =============================================================================
# CHANGE 1: IMPORTS (Add after existing imports, around line 37)
# =============================================================================

IMPORTS_TO_ADD = '''
# Performance-optimized modules (v2.4)
try:
    from src.transcription_optimized import (
        transcribe_videos_parallel,
        DeltaAwareIndex,
        transcribe_voiceover_media
    )
    OPTIMIZED_TRANSCRIPTION = True
except ImportError:
    from src.transcription import transcribe_videos_parallel
    OPTIMIZED_TRANSCRIPTION = False
    
try:
    from src.embeddings_optimized import (
        compute_embeddings,
        get_embedding_provider,
        build_embedding_index,
        EmbeddingCache
    )
    OPTIMIZED_EMBEDDINGS = True
except ImportError:
    from src.embeddings import compute_embeddings, get_embedding_provider, build_embedding_index
    OPTIMIZED_EMBEDDINGS = False

try:
    from src.vision_optimized import process_video_vision
    OPTIMIZED_VISION = True
except ImportError:
    from src.vision import process_video_vision
    OPTIMIZED_VISION = False
'''


# =============================================================================
# CHANGE 2: REPLACE stage_transcribe_index() METHOD
# =============================================================================

NEW_STAGE_TRANSCRIBE_INDEX = '''
    def stage_transcribe_index(self, force_reprocess: bool = False) -> dict:
        """
        Stage 3: Transcribe and index videos (OPTIMIZED v2.4)
        
        OPTIMIZATIONS:
        1. Delta-aware indexing - only processes NEW videos
        2. Parallel transcription with ThreadPoolExecutor
        3. Batch embedding API calls (100 texts per call)
        4. Smart caching at multiple levels
        """
        self._print_stage(3, "TRANSCRIBE & INDEX")
        
        try:
            if OPTIMIZED_TRANSCRIPTION:
                from src.transcription_optimized import transcribe_videos_parallel, DeltaAwareIndex
            else:
                from src.transcription import transcribe_videos_parallel
            
            if OPTIMIZED_EMBEDDINGS:
                from src.embeddings_optimized import compute_embeddings, build_embedding_index, get_embedding_provider
            else:
                from src.embeddings import compute_embeddings, build_embedding_index, get_embedding_provider
            
            if OPTIMIZED_VISION:
                from src.vision_optimized import process_video_vision
            else:
                from src.vision import process_video_vision
            
            from src.utils import CacheManager, SRTSegment
        except ImportError as e:
            logger.warning(f"Could not import modules: {e}")
            return {'videos_indexed': 0}
        
        import time
        stage_start = time.time()
        
        video_dir = Path(self.config.downloaded_videos_dir)
        
        # Find all video files
        video_extensions = {'.mp4', '.mkv', '.webm', '.avi', '.mov', '.mxf'}
        videos = []
        for ext in video_extensions:
            videos.extend(video_dir.rglob(f'*{ext}'))
        
        video_paths = [str(v) for v in videos]
        print(f"  Found {len(video_paths)} videos")
        
        # Initialize cache
        cache = CacheManager(self.config.cache_dir)
        self.cache = cache
        
        # =====================================================================
        # OPTIMIZATION 1: Delta-aware transcription
        # Only transcribe NEW videos, load existing from cache
        # =====================================================================
        transcribe_start = time.time()
        
        if OPTIMIZED_TRANSCRIPTION:
            print(f"  Using optimized parallel transcription...")
            transcripts = transcribe_videos_parallel(
                video_paths, 
                cache, 
                self.config,
                max_workers=4,
                force_reprocess=force_reprocess
            )
        else:
            # Fallback to original
            transcripts = transcribe_videos_parallel(video_paths, cache, self.config)
        
        transcribe_time = time.time() - transcribe_start
        print(f"  ✓ Transcribed {len(transcripts)} videos in {transcribe_time:.1f}s")
        
        # =====================================================================
        # OPTIMIZATION 2: Selective vision processing
        # Only process scenes where transcript is sparse
        # =====================================================================
        if self.config.vision.enabled:
            vision_start = time.time()
            print(f"  Analyzing scenes with selective vision...")
            
            all_scenes = {}
            total_api_calls = 0
            skipped_videos = 0
            
            for video_path in videos:
                video_str = str(video_path)
                video_transcripts = transcripts.get(video_str, [])
                
                # Convert to SRTSegment if needed
                segments = []
                for t in video_transcripts:
                    if hasattr(t, 'start_time'):
                        segments.append(t)
                    elif isinstance(t, dict):
                        segments.append(SRTSegment(
                            index=t.get('index', 0),
                            start_time=t.get('start_time', 0),
                            end_time=t.get('end_time', 0),
                            text=t.get('text', '')
                        ))
                
                # Selective vision processing
                scenes = process_video_vision(
                    video_str, 
                    segments, 
                    cache, 
                    self.config,
                    max_scenes_per_video=3  # Cap per video
                )
                
                if scenes:
                    all_scenes[video_str] = scenes
                else:
                    skipped_videos += 1
            
            vision_time = time.time() - vision_start
            total_scenes = sum(len(s) for s in all_scenes.values())
            print(f"  ✓ Vision: {total_scenes} scenes, {skipped_videos} videos skipped (good transcript)")
            print(f"  ✓ Vision time: {vision_time:.1f}s")
        
        # =====================================================================
        # OPTIMIZATION 3: Batch embedding computation
        # Groups into batches of 100 for API efficiency
        # =====================================================================
        embed_start = time.time()
        
        # Extract texts for embedding
        all_texts = []
        text_metadata = []
        for video_path, segments in transcripts.items():
            for seg in segments:
                if isinstance(seg, dict):
                    text = seg.get('text', '')
                else:
                    text = getattr(seg, 'text', '')
                if text and text.strip():
                    all_texts.append(text)
                    text_metadata.append({'video': video_path, 'segment': seg})
        
        print(f"  Computing embeddings ({self.config.embedding.provider})...")
        print(f"    {len(all_texts)} text segments")
        
        embedding_provider = get_embedding_provider(self.config)
        
        if OPTIMIZED_EMBEDDINGS:
            # Batch embedding with caching
            embeddings = compute_embeddings(
                texts=all_texts,
                provider=embedding_provider,
                cache=cache,
                cache_key="video_segments",
                show_progress=True
            )
        else:
            embeddings = compute_embeddings(
                texts=all_texts,
                provider=embedding_provider,
                cache=cache,
                cache_key="video_segments"
            )
        
        embed_time = time.time() - embed_start
        print(f"  ✓ Computed {len(embeddings)} embeddings in {embed_time:.1f}s")
        
        # Build FAISS index
        if self.config.indexing.use_faiss:
            print(f"  Building FAISS index...")
            self.embedding_index = build_embedding_index(embeddings, self.config)
        
        # Store for matching
        self.transcripts = transcripts
        self.embeddings = embeddings
        self.text_metadata = text_metadata
        
        # Performance summary
        total_time = time.time() - stage_start
        print(f"\\n  ═══ Stage 3 Performance ═══")
        print(f"    Transcription: {transcribe_time:.1f}s")
        if self.config.vision.enabled:
            print(f"    Vision: {vision_time:.1f}s")
        print(f"    Embeddings: {embed_time:.1f}s")
        print(f"    Total: {total_time:.1f}s")
        
        return {
            'videos_indexed': len(videos),
            'embeddings': len(embeddings),
            'transcribe_time': transcribe_time,
            'embed_time': embed_time,
            'total_time': total_time
        }
'''


# =============================================================================
# CHANGE 3: UPDATE stage_confidence_enforcement() 
# =============================================================================

NEW_CONFIDENCE_ENFORCEMENT_SNIPPET = '''
            # ═══════════════════════════════════════════════════════════════
            # OPTIMIZATION: Delta-aware re-indexing
            # Only process NEWLY downloaded videos, not all videos
            # ═══════════════════════════════════════════════════════════════
            print(f"    Re-indexing new footage only (delta-aware)...")
            try:
                # Force reprocess=False means delta indexing kicks in
                # Only new stock footage will be transcribed
                self.stage_transcribe_index(force_reprocess=False)
                self.stage_match()
            except Exception as e:
                print(f"    ⚠ Re-match error: {e}")
                break
'''


# =============================================================================
# CHANGE 4: ADD INIT ATTRIBUTES (in Pipeline.__init__)
# =============================================================================

NEW_INIT_ATTRIBUTES = '''
        # Performance tracking
        self.stage_timings = {}
        
        # Optimization flags
        self.use_delta_indexing = True
        self.use_batch_embeddings = True
        self.use_selective_vision = True
'''


# =============================================================================
# INSTALLATION INSTRUCTIONS
# =============================================================================

INSTALLATION = '''
================================================================================
INSTALLATION INSTRUCTIONS
================================================================================

1. COPY OPTIMIZED MODULES to your src/ directory:
   
   cp transcription_optimized.py D:\\matcher-alt\\src\\
   cp embeddings_optimized.py D:\\matcher-alt\\src\\
   cp vision_optimized.py D:\\matcher-alt\\src\\

2. EDIT main.py:

   a) Add imports after line ~37:
      - Copy the IMPORTS_TO_ADD block
   
   b) Replace stage_transcribe_index() method:
      - Find the existing method (search for "def stage_transcribe_index")
      - Replace entire method with NEW_STAGE_TRANSCRIBE_INDEX
   
   c) Update stage_confidence_enforcement():
      - Find the re-indexing section (search for "Re-indexing new footage")
      - Add force_reprocess=False parameter

3. TEST:
   
   python main.py --voiceover test.srt --match-only
   
   You should see:
   - "Using optimized parallel transcription..."
   - "Video index: X cached, Y new"
   - Batch progress: "Batch 1/N (100 texts)"

================================================================================
EXPECTED PERFORMANCE IMPROVEMENTS
================================================================================

BEFORE (from your log):
  Stage 3: 89 minutes
  - Transcription: ~60 min (sequential)
  - Embeddings: ~20 min (one-by-one API calls)
  - Vision: ~9 min (all scenes processed)

AFTER (estimated):
  Stage 3: 25-35 minutes
  - Transcription: ~15-20 min (parallel, delta-aware)
  - Embeddings: ~5-8 min (batched API calls)
  - Vision: ~3-5 min (selective processing)

During confidence enforcement retries:
  BEFORE: 8+ min per retry (full re-index)
  AFTER: <30 sec per retry (delta-aware, only new files)

Total pipeline improvement: ~55-65% faster

================================================================================
'''

if __name__ == '__main__':
    print(INSTALLATION)
