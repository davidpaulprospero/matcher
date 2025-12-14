"""
Keyword and entity extraction module
With TF-IDF based auto keyword weight detection (topic-agnostic)
"""

import logging
import re
import math
from typing import List, Set, Tuple, Dict
from collections import Counter

from .config import Config
from .utils import SRTSegment, Topic, ProgressBar

logger = logging.getLogger(__name__)


# Common stop words to filter out
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
    'she', 'we', 'they', 'what', 'which', 'who', 'whom', 'when', 'where',
    'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there',
    'then', 'once', 'if', 'because', 'until', 'while', 'about', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'between',
    'under', 'again', 'further', 'then', 'once', 'up', 'down', 'out', 'off',
    'over', 'any', 'much', 'many', 'still', 'get', 'got', 'getting', 'being',
    'like', 'really', 'right', 'going', 'know', 'think', 'see', 'look',
    'come', 'make', 'way', 'take', 'want', 'say', 'said', 'says'
}


# =============================================================================
# TF-IDF KEYWORD WEIGHT DETECTION (Topic-Agnostic)
# =============================================================================

class KeywordWeightExtractor:
    """
    Extracts weighted keywords from text using TF-IDF.
    Topic-agnostic: works for any content (disasters, cooking, tech, etc.)
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.kw_config = config.keyword_weights
        self._boost_terms: Set[str] = set()
        self._penalty_terms: Set[str] = set()
        self._tfidf_scores: Dict[str, float] = {}
    
    def extract_weighted_terms(self, texts: List[str]) -> Tuple[Set[str], Set[str], Dict[str, float]]:
        """
        Extract weighted terms from texts using TF-IDF.
        
        Args:
            texts: List of text segments (voiceover segments)
        
        Returns:
            Tuple of (boost_terms, penalty_terms, tfidf_scores)
        """
        if not self.kw_config.enabled:
            return set(), set(), {}
        
        # Start with custom terms from config
        boost_terms = set(t.lower() for t in self.kw_config.custom_boost_terms)
        penalty_terms = set(t.lower() for t in self.kw_config.custom_penalty_terms)
        
        if not self.kw_config.auto_detect or not texts:
            return boost_terms, penalty_terms, {}
        
        # Compute TF-IDF scores
        tfidf_scores = self._compute_tfidf(texts)
        
        # Get top N terms as boost terms
        top_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        auto_boost_count = self.kw_config.auto_detect_count
        
        for term, score in top_terms[:auto_boost_count]:
            if term not in penalty_terms:  # Don't add if manually penalized
                boost_terms.add(term)
        
        logger.info(f"Auto-detected {len(boost_terms)} boost keywords using TF-IDF")
        if boost_terms:
            logger.debug(f"Top boost terms: {list(boost_terms)[:10]}")
        
        self._boost_terms = boost_terms
        self._penalty_terms = penalty_terms
        self._tfidf_scores = tfidf_scores
        
        return boost_terms, penalty_terms, tfidf_scores
    
    def _compute_tfidf(self, texts: List[str]) -> Dict[str, float]:
        """
        Compute TF-IDF scores for all terms across texts.
        """
        # Tokenize all documents
        documents = []
        for text in texts:
            words = self._tokenize(text)
            documents.append(words)
        
        if not documents:
            return {}
        
        # Compute document frequency (how many docs contain each term)
        df = Counter()
        for doc in documents:
            unique_terms = set(doc)
            for term in unique_terms:
                df[term] += 1
        
        # Compute TF-IDF
        num_docs = len(documents)
        tfidf_scores = {}
        
        # Global term frequency (sum across all documents)
        global_tf = Counter()
        for doc in documents:
            global_tf.update(doc)
        
        for term, freq in global_tf.items():
            if len(term) < 3:  # Skip very short terms
                continue
            if term in STOP_WORDS:
                continue
            
            # TF: log-normalized frequency
            tf = 1 + math.log(freq) if freq > 0 else 0
            
            # IDF: inverse document frequency (rarer = higher weight)
            idf = math.log(num_docs / (1 + df[term]))
            
            tfidf_scores[term] = tf * idf
        
        return tfidf_scores
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Extract words, including hyphenated terms
        words = re.findall(r'\b[a-zA-Z][a-zA-Z-]*[a-zA-Z]\b|\b[a-zA-Z]{2,}\b', text.lower())
        # Filter stop words
        return [w for w in words if w not in STOP_WORDS and len(w) >= 3]
    
    def compute_keyword_boost(
        self,
        voiceover_text: str,
        video_text: str,
        visual_keywords: List[str] = None
    ) -> float:
        """
        Compute confidence boost based on keyword matches.
        
        Args:
            voiceover_text: Voiceover segment text
            video_text: Video segment text or keywords
            visual_keywords: Optional visual description keywords
        
        Returns:
            Boost value (positive) or penalty (negative)
        """
        if not self.kw_config.enabled:
            return 0.0
        
        vo_terms = set(self._tokenize(voiceover_text))
        vid_terms = set(self._tokenize(video_text))
        vis_terms = set(k.lower() for k in (visual_keywords or []))
        
        all_video_terms = vid_terms | vis_terms
        
        boost = 0.0
        boost_factor = self.kw_config.boost_factor
        
        # Boost for matching boost terms
        for term in vo_terms & self._boost_terms:
            if term in all_video_terms:
                boost += boost_factor
        
        # Penalty for matching penalty terms (negative boost)
        for term in vo_terms & self._penalty_terms:
            if term in all_video_terms:
                boost -= boost_factor * 0.5  # Half penalty
        
        # Cap the boost
        return max(-0.3, min(0.3, boost))
    
    @property
    def boost_terms(self) -> Set[str]:
        """Get current boost terms"""
        return self._boost_terms
    
    @property
    def penalty_terms(self) -> Set[str]:
        """Get current penalty terms"""
        return self._penalty_terms


# Global keyword weight extractor instance
_keyword_extractor: KeywordWeightExtractor = None


def get_keyword_extractor() -> KeywordWeightExtractor:
    """Get the global keyword extractor"""
    return _keyword_extractor


def init_keyword_extractor(config: Config, voiceover_texts: List[str]) -> KeywordWeightExtractor:
    """
    Initialize the keyword weight extractor with voiceover texts.
    
    Args:
        config: Configuration
        voiceover_texts: List of voiceover segment texts
    
    Returns:
        Initialized KeywordWeightExtractor
    """
    global _keyword_extractor
    
    extractor = KeywordWeightExtractor(config)
    extractor.extract_weighted_terms(voiceover_texts)
    
    _keyword_extractor = extractor
    return extractor


def extract_keywords_simple(text: str, min_length: int = 3) -> List[str]:
    """
    Simple keyword extraction using word frequency.
    """
    # Tokenize
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    # Filter
    words = [w for w in words if len(w) >= min_length and w not in STOP_WORDS]
    
    # Count frequency
    word_counts = Counter(words)
    
    # Return top keywords
    return [word for word, count in word_counts.most_common(10)]


def extract_keywords_with_llm(
    text: str,
    config: Config
) -> Tuple[List[str], List[str]]:
    """
    Extract keywords and entities using LLM.
    Returns (keywords, entities)
    """
    prompt = f"""Extract key information from this text:

"{text}"

Respond in JSON format:
{{
    "keywords": ["keyword1", "keyword2", ...],  // Main topics/concepts (5-10)
    "entities": ["entity1", "entity2", ...]     // Named entities (people, places, organizations)
}}"""

    try:
        if config.gemini_api_key:
            import google.generativeai as genai
            genai.configure(api_key=config.gemini_api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            text_response = response.text
        elif config.anthropic_api_key:
            import anthropic
            client = anthropic.Anthropic(api_key=config.anthropic_api_key)
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            text_response = response.content[0].text
        else:
            # Fallback to simple extraction
            keywords = extract_keywords_simple(text)
            return keywords, []
        
        # Parse JSON response
        import json
        json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return data.get('keywords', []), data.get('entities', [])
    
    except Exception as e:
        logger.warning(f"LLM keyword extraction failed: {e}")
    
    # Fallback
    keywords = extract_keywords_simple(text)
    return keywords, []


def extract_keywords_batch(
    segments: List[SRTSegment],
    config: Config
) -> List[SRTSegment]:
    """
    Extract keywords for multiple segments.
    Updates segments in place and returns them.
    """
    if not config.keywords.enabled:
        return segments
    
    logger.info(f"Extracting keywords from {len(segments)} segments...")
    progress = ProgressBar(len(segments), "Extracting keywords")
    
    # Batch segments for efficiency
    batch_size = 20
    
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i+batch_size]
        
        # Combine texts for batch processing
        combined_text = "\n---\n".join([f"[{j}] {seg.text}" for j, seg in enumerate(batch)])
        
        if config.keywords.extract_entities:
            # Use LLM for keyword + entity extraction
            prompt = f"""Extract keywords and entities from each numbered segment:

{combined_text}

Respond in JSON format:
{{
    "segments": [
        {{"id": 0, "keywords": [...], "entities": [...]}},
        {{"id": 1, "keywords": [...], "entities": [...]}},
        ...
    ]
}}"""
            
            try:
                if config.gemini_api_key:
                    import google.generativeai as genai
                    genai.configure(api_key=config.gemini_api_key)
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    response = model.generate_content(prompt)
                    
                    import json
                    json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group())
                        for seg_data in data.get('segments', []):
                            idx = seg_data.get('id', 0)
                            if 0 <= idx < len(batch):
                                batch[idx].keywords = seg_data.get('keywords', [])
                                batch[idx].entities = seg_data.get('entities', [])
            except Exception as e:
                logger.debug(f"Batch keyword extraction failed: {e}")
                # Fallback to simple extraction
                for seg in batch:
                    seg.keywords = extract_keywords_simple(seg.text)
        else:
            # Simple keyword extraction only
            for seg in batch:
                seg.keywords = extract_keywords_simple(seg.text)
        
        progress.update(len(batch))
    
    progress.close()
    return segments


def segment_into_topics(
    segments: List[SRTSegment],
    config: Config
) -> List[Topic]:
    """
    Segment voiceover into topics based on content similarity.
    """
    if not config.topics.enabled or len(segments) < config.topics.min_topic_segments:
        # Return single topic with all segments
        return [Topic(
            topic_id=0,
            name="Main",
            segments=segments,
            keywords=list(set(kw for seg in segments for kw in seg.keywords)),
            start_time=segments[0].start_time if segments else 0,
            end_time=segments[-1].end_time if segments else 0
        )]
    
    logger.info("Segmenting voiceover into topics...")
    
    # Use keyword overlap to detect topic boundaries
    topics = []
    current_topic_segments = [segments[0]]
    current_keywords = set(segments[0].keywords)
    
    for seg in segments[1:]:
        seg_keywords = set(seg.keywords)
        
        # Calculate keyword overlap
        if current_keywords and seg_keywords:
            overlap = len(current_keywords & seg_keywords) / len(current_keywords | seg_keywords)
        else:
            overlap = 0
        
        if overlap >= config.topics.similarity_threshold:
            # Same topic
            current_topic_segments.append(seg)
            current_keywords.update(seg_keywords)
        else:
            # New topic
            if len(current_topic_segments) >= config.topics.min_topic_segments:
                topics.append(Topic(
                    topic_id=len(topics),
                    name=f"Topic {len(topics) + 1}",
                    segments=current_topic_segments,
                    keywords=list(current_keywords),
                    start_time=current_topic_segments[0].start_time,
                    end_time=current_topic_segments[-1].end_time
                ))
                current_topic_segments = [seg]
                current_keywords = seg_keywords
            else:
                # Not enough segments, merge with current
                current_topic_segments.append(seg)
                current_keywords.update(seg_keywords)
    
    # Add final topic
    if current_topic_segments:
        topics.append(Topic(
            topic_id=len(topics),
            name=f"Topic {len(topics) + 1}",
            segments=current_topic_segments,
            keywords=list(current_keywords),
            start_time=current_topic_segments[0].start_time,
            end_time=current_topic_segments[-1].end_time
        ))
    
    # Name topics based on keywords
    for topic in topics:
        if topic.keywords:
            topic.name = ", ".join(topic.keywords[:3]).title()
    
    logger.info(f"Created {len(topics)} topics")
    for topic in topics:
        logger.info(f"  - {topic.name} ({len(topic.segments)} segments)")
    
    return topics


def find_keyword_matches(
    voiceover_keywords: List[str],
    video_keywords: List[str],
    visual_keywords: List[str] = None
) -> Tuple[float, bool, bool]:
    """
    Find keyword overlap between voiceover and video.
    Returns (boost_score, is_keyword_match, is_visual_match)
    """
    vo_set = set(k.lower() for k in voiceover_keywords)
    vid_set = set(k.lower() for k in video_keywords)
    vis_set = set(k.lower() for k in (visual_keywords or []))
    
    # Text keyword match
    text_overlap = len(vo_set & vid_set)
    is_keyword_match = text_overlap > 0
    
    # Visual keyword match
    visual_overlap = len(vo_set & vis_set)
    is_visual_match = visual_overlap > 0
    
    # Calculate boost score
    boost = 0.0
    if is_keyword_match:
        boost += 0.05 * text_overlap  # 5% boost per matching keyword
    if is_visual_match:
        boost += 0.03 * visual_overlap  # 3% boost per visual match
    
    return min(boost, 0.2), is_keyword_match, is_visual_match  # Cap at 20% boost
