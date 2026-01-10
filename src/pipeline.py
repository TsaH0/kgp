"""
Main Pipeline Orchestrator.

This module ties together all phases of the system:
- Phase 1: Narrative Ingestion
- Phase 2: Character Timeline Extraction
- Phase 3: Training
- Phase 4: Inference

It provides the main entry points for running the complete pipeline.
"""

import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from .config import config, SystemConfig, DATA_DIR, OUTPUT_DIR, CACHE_DIR
from .models import (
    BackstoryEntry, BookNarrativeStructure, CharacterTimeline,
    ConsistencyResult, ConsistencyVerdict
)
from .narrative_ingestion import PathwayNarrativeIngestion, create_ingestion_pipeline
from .character_extraction import CharacterTimelineExtractor, extract_all_character_timelines
from .consistency_verifier import HybridConsistencyVerifier, ConsistencyTrainer

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and parses CSV data files."""
    
    @staticmethod
    def load_train_csv(file_path: Path) -> List[BackstoryEntry]:
        """Load training data from CSV."""
        entries = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    entry = BackstoryEntry(
                        id=int(row['id']),
                        book_name=row['book_name'],
                        character=row['char'],
                        caption=row.get('caption', ''),
                        content=row['content'],
                        label=row.get('label', None)
                    )
                    entries.append(entry)
                except Exception as e:
                    logger.warning(f"Failed to parse row: {e}")
        
        logger.info(f"Loaded {len(entries)} training entries")
        return entries
    
    @staticmethod
    def load_test_csv(file_path: Path) -> List[BackstoryEntry]:
        """Load test data from CSV."""
        entries = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    entry = BackstoryEntry(
                        id=int(row['id']),
                        book_name=row['book_name'],
                        character=row['char'],
                        caption=row.get('caption', ''),
                        content=row['content'],
                        label=None  # Test data has no labels
                    )
                    entries.append(entry)
                except Exception as e:
                    logger.warning(f"Failed to parse row: {e}")
        
        logger.info(f"Loaded {len(entries)} test entries")
        return entries
    
    @staticmethod
    def save_results_csv(
        results: List[ConsistencyResult],
        file_path: Path
    ):
        """Save results to CSV."""
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'label'])
            
            for result in sorted(results, key=lambda r: r.backstory_id):
                label = 'consistent' if result.prediction == 1 else 'contradict'
                writer.writerow([result.backstory_id, label])
        
        logger.info(f"Saved {len(results)} results to {file_path}")


class CacheManager:
    """Manages caching of intermediate results."""
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_structure_cache_path(self, book_name: str) -> Path:
        """Get cache path for a book structure."""
        safe_name = book_name.replace(' ', '_').replace('/', '_')
        return self.cache_dir / f"structure_{safe_name}.json"
    
    def get_timeline_cache_path(self, book_name: str, character: str) -> Path:
        """Get cache path for a character timeline."""
        safe_book = book_name.replace(' ', '_').replace('/', '_')
        safe_char = character.replace(' ', '_').replace('/', '_')
        return self.cache_dir / f"timeline_{safe_book}_{safe_char}.json"
    
    def save_structure(self, structure: BookNarrativeStructure):
        """Save a structure to cache."""
        path = self.get_structure_cache_path(structure.book_name)
        
        # Convert to serializable format
        data = {
            'book_name': structure.book_name,
            'total_lines': structure.total_lines,
            'total_chapters': structure.total_chapters,
            'chunk_count': len(structure.chunks),
            'chapters': {k: list(v) for k, v in structure.chapter_boundaries.items()}
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def save_timeline(self, timeline: CharacterTimeline):
        """Save a timeline to cache."""
        path = self.get_timeline_cache_path(timeline.book_name, timeline.character_name)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(timeline.to_dict(), f, indent=2, default=str)


class NarrativeConsistencyPipeline:
    """
    Main pipeline orchestrator for the Narrative Consistency Verification System.
    
    This class coordinates all phases of the system and provides the main
    entry point for running the complete workflow.
    """
    
    def __init__(
        self,
        system_config: Optional[SystemConfig] = None,
        use_llm: bool = True,
        use_cache: bool = True
    ):
        self.config = system_config or config
        self.use_llm = use_llm
        self.use_cache = use_cache
        
        # Initialize components
        self.ingestion = create_ingestion_pipeline(self.config.chunking)
        self.timeline_extractor = CharacterTimelineExtractor(
            self.config.extraction,
            use_llm=use_llm
        )
        self.verifier = HybridConsistencyVerifier(
            self.config.consistency,
            use_llm=use_llm
        )
        self.trainer = ConsistencyTrainer(self.verifier)
        self.cache = CacheManager(CACHE_DIR)
        
        # State
        self.narrative_structures: Dict[str, BookNarrativeStructure] = {}
        self.is_trained = False
    
    def run_full_pipeline(
        self,
        data_dir: Path = DATA_DIR,
        output_path: Optional[Path] = None
    ) -> List[ConsistencyResult]:
        """
        Run the complete pipeline end-to-end.
        
        Steps:
        1. Ingest novels
        2. Build character timelines
        3. Train on training data
        4. Run inference on test data
        5. Save results
        """
        logger.info("=" * 60)
        logger.info("KGPH Pathway - Narrative Consistency Verification System")
        logger.info("=" * 60)
        
        # Phase 1: Narrative Ingestion
        logger.info("\n📚 PHASE 1: Narrative Ingestion")
        self.ingest_narratives(data_dir)
        
        # Phase 2: Character Timeline Extraction
        logger.info("\n👤 PHASE 2: Character Timeline Extraction")
        train_data = DataLoader.load_train_csv(self.config.train_csv)
        test_data = DataLoader.load_test_csv(self.config.test_csv)
        
        # Extract unique characters
        all_characters = self._get_unique_characters(train_data + test_data)
        self.extract_character_timelines(all_characters)
        
        # Phase 3: Training
        logger.info("\n🎯 PHASE 3: Training")
        self.train(train_data)
        
        # Phase 4: Inference
        logger.info("\n🔍 PHASE 4: Inference")
        results = self.predict(test_data)
        
        # Save results
        output_path = output_path or self.config.results_csv
        DataLoader.save_results_csv(results, output_path)
        
        logger.info("\n✅ Pipeline complete!")
        logger.info(f"Results saved to: {output_path}")
        
        return results
    
    def ingest_narratives(self, data_dir: Path = DATA_DIR):
        """
        Phase 1: Ingest all novels from data directory.
        
        Processes entire novels without truncation or summarization.
        """
        logger.info(f"Ingesting novels from: {data_dir}")
        
        self.narrative_structures = self.ingestion.ingest_all_novels(data_dir)
        
        for name, structure in self.narrative_structures.items():
            logger.info(f"  ✓ {name}: {structure.total_lines} lines, "
                       f"{len(structure.chunks)} chunks")
            
            if self.use_cache:
                self.cache.save_structure(structure)
    
    def extract_character_timelines(
        self,
        characters_by_book: Dict[str, List[str]]
    ):
        """
        Phase 2: Extract character timelines from narratives.
        
        Builds CharacterStateTimeline for each character.
        """
        logger.info("Extracting character timelines...")
        
        for book_name, characters in characters_by_book.items():
            structure = self._get_structure_for_book(book_name)
            
            if structure is None:
                logger.warning(f"No structure found for book: {book_name}")
                continue
            
            logger.info(f"  Processing {book_name}: {len(characters)} characters")
            
            for character in characters:
                try:
                    timeline = self.timeline_extractor.extract_timeline(
                        structure, character
                    )
                    structure.add_character_timeline(timeline)
                    
                    if self.use_cache:
                        self.cache.save_timeline(timeline)
                    
                    logger.info(f"    ✓ {character}: {len(timeline.states)} states")
                except Exception as e:
                    logger.error(f"    ✗ {character}: {e}")
    
    def train(self, training_entries: List[BackstoryEntry]) -> Dict[str, float]:
        """
        Phase 3: Train the consistency verifier.
        
        Uses labeled training data to tune thresholds and weights.
        Does NOT modify narrative representations.
        """
        logger.info(f"Training on {len(training_entries)} examples...")
        
        metrics = self.trainer.train(training_entries, self.narrative_structures)
        self.is_trained = True
        
        logger.info(f"Training metrics:")
        logger.info(f"  Accuracy:  {metrics.get('accuracy', 0):.2%}")
        logger.info(f"  Precision: {metrics.get('precision', 0):.2%}")
        logger.info(f"  Recall:    {metrics.get('recall', 0):.2%}")
        logger.info(f"  F1 Score:  {metrics.get('f1', 0):.2%}")
        
        return metrics
    
    def predict(
        self,
        test_entries: List[BackstoryEntry],
        verbose: bool = False
    ) -> List[ConsistencyResult]:
        """
        Phase 4: Run inference on test data.
        
        Compares each backstory against the frozen narrative structure.
        """
        logger.info(f"Running inference on {len(test_entries)} test entries...")
        
        results = []
        
        for i, entry in enumerate(test_entries):
            structure = self._get_structure_for_book(entry.book_name)
            
            if structure is None:
                logger.warning(f"No structure for: {entry.book_name}")
                # Create uncertain result
                result = ConsistencyResult(
                    backstory_id=entry.id,
                    verdict=ConsistencyVerdict.UNCERTAIN,
                    confidence=0.0,
                    reasoning="Book not found in narrative structures"
                )
            else:
                result = self.verifier.verify_consistency(entry, structure)
            
            results.append(result)
            
            if verbose or (i + 1) % 10 == 0:
                logger.info(f"  [{i+1}/{len(test_entries)}] ID {entry.id}: "
                           f"{result.verdict.value} ({result.confidence:.2f})")
        
        # Summary
        consistent = sum(1 for r in results if r.verdict == ConsistencyVerdict.CONSISTENT)
        contradictory = sum(1 for r in results if r.verdict == ConsistencyVerdict.CONTRADICTORY)
        uncertain = sum(1 for r in results if r.verdict == ConsistencyVerdict.UNCERTAIN)
        
        logger.info(f"Inference complete:")
        logger.info(f"  Consistent:   {consistent}")
        logger.info(f"  Contradictory: {contradictory}")
        logger.info(f"  Uncertain:    {uncertain}")
        
        return results
    
    def _get_unique_characters(
        self,
        entries: List[BackstoryEntry]
    ) -> Dict[str, List[str]]:
        """Get unique characters grouped by book."""
        characters_by_book: Dict[str, set] = {}
        
        for entry in entries:
            book = entry.book_name
            if book not in characters_by_book:
                characters_by_book[book] = set()
            characters_by_book[book].add(entry.character)
        
        return {book: list(chars) for book, chars in characters_by_book.items()}
    
    def _get_structure_for_book(
        self,
        book_name: str
    ) -> Optional[BookNarrativeStructure]:
        """Get narrative structure for a book, handling name variations."""
        # Direct match
        if book_name in self.narrative_structures:
            return self.narrative_structures[book_name]
        
        # Normalized match
        book_lower = book_name.lower().replace('_', ' ')
        for name, structure in self.narrative_structures.items():
            name_lower = name.lower()
            if book_lower in name_lower or name_lower in book_lower:
                return structure
        
        # Special cases
        mappings = {
            'in search of the castaways': 'In search of the castaways',
            'the count of monte cristo': 'The Count of Monte Cristo'
        }
        
        for pattern, actual in mappings.items():
            if pattern in book_lower:
                return self.narrative_structures.get(actual)
        
        return None


def create_pipeline(
    use_llm: bool = True,
    use_cache: bool = True
) -> NarrativeConsistencyPipeline:
    """Factory function to create a configured pipeline."""
    return NarrativeConsistencyPipeline(
        system_config=config,
        use_llm=use_llm,
        use_cache=use_cache
    )


def run_pipeline(
    data_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
    use_llm: bool = True
) -> List[ConsistencyResult]:
    """
    Convenience function to run the complete pipeline.
    
    Args:
        data_dir: Directory containing novels and CSVs
        output_path: Path for results.csv output
        use_llm: Whether to use Ollama LLM
    
    Returns:
        List of consistency results
    """
    pipeline = create_pipeline(use_llm=use_llm)
    
    return pipeline.run_full_pipeline(
        data_dir=data_dir or DATA_DIR,
        output_path=output_path
    )
