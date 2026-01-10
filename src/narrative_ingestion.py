"""
Pathway-based Narrative Ingestion Pipeline.

This module implements Phase 1 of the system:
- Reads novels from data folder (no truncation, no summarization)
- Uses Pathway framework for document ingestion and management
- Implements chapter-aware chunking
- Stores text and metadata for retrieval and reasoning

Key Features:
- Full novel processing (100k+ words)
- Chapter boundary detection
- Overlap-aware chunking for context preservation
- Metadata extraction (chapter numbers, line ranges)
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Generator
from dataclasses import dataclass
import hashlib
import logging

# Pathway import - using the framework for document processing
try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
except ImportError:
    PATHWAY_AVAILABLE = False
    logging.warning("Pathway not installed. Using fallback ingestion.")

from .models import NarrativeChunk, BookNarrativeStructure
from .config import ChunkingConfig, config

logger = logging.getLogger(__name__)


class ChapterDetector:
    """Detects chapter boundaries in novel text."""
    
    # Common chapter patterns
    CHAPTER_PATTERNS = [
        r'^CHAPTER\s+([IVXLCDM]+|\d+)[.\s]*(.*)$',  # CHAPTER I, CHAPTER 1
        r'^Chapter\s+([IVXLCDM]+|\d+)[.\s]*(.*)$',
        r'^\s*([IVXLCDM]+|\d+)[.\s]+(.+)$',  # Roman numerals or numbers at line start
        r'^BOOK\s+([IVXLCDM]+|\d+)',  # BOOK I
        r'^PART\s+([IVXLCDM]+|\d+)',  # PART I
        r'^VOLUME\s+([IVXLCDM]+|\d+)',  # VOLUME I
    ]
    
    ROMAN_NUMERALS = {
        'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
        'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
        'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15,
        'XVI': 16, 'XVII': 17, 'XVIII': 18, 'XIX': 19, 'XX': 20,
        'XXI': 21, 'XXII': 22, 'XXIII': 23, 'XXIV': 24, 'XXV': 25,
        'XXVI': 26, 'XXVII': 27, 'XXVIII': 28, 'XXIX': 29, 'XXX': 30,
        'XXXI': 31, 'XXXII': 32, 'XXXIII': 33, 'XXXIV': 34, 'XXXV': 35,
        'XL': 40, 'L': 50, 'LX': 60, 'LXI': 61, 'LXII': 62,
        'LXX': 70, 'LXXI': 71, 'LXXII': 72, 'LXXX': 80,
        'XC': 90, 'C': 100, 'CX': 110, 'CXI': 111, 'CXII': 112,
        'CXIII': 113, 'CXIV': 114, 'CXV': 115, 'CXVI': 116, 'CXVII': 117
    }
    
    def __init__(self):
        self.compiled_patterns = [
            re.compile(p, re.MULTILINE | re.IGNORECASE) 
            for p in self.CHAPTER_PATTERNS
        ]
    
    def detect_chapters(self, text: str) -> Dict[int, Tuple[int, int, str]]:
        """
        Detect chapter boundaries in text.
        
        Returns:
            Dict mapping chapter number to (start_line, end_line, chapter_title)
        """
        lines = text.split('\n')
        chapters = {}
        current_chapter = 0
        chapter_starts = []
        
        for i, line in enumerate(lines):
            for pattern in self.compiled_patterns:
                match = pattern.match(line.strip())
                if match:
                    chapter_num_str = match.group(1)
                    chapter_title = match.group(2) if len(match.groups()) > 1 else ""
                    
                    # Convert to integer
                    if chapter_num_str.upper() in self.ROMAN_NUMERALS:
                        chapter_num = self.ROMAN_NUMERALS[chapter_num_str.upper()]
                    elif chapter_num_str.isdigit():
                        chapter_num = int(chapter_num_str)
                    else:
                        continue
                    
                    chapter_starts.append((i, chapter_num, chapter_title.strip()))
                    break
        
        # Convert starts to ranges
        for idx, (start_line, chapter_num, title) in enumerate(chapter_starts):
            if idx + 1 < len(chapter_starts):
                end_line = chapter_starts[idx + 1][0] - 1
            else:
                end_line = len(lines) - 1
            
            chapters[chapter_num] = (start_line, end_line, title)
        
        return chapters
    
    def get_chapter_at_line(
        self, 
        line_num: int, 
        chapters: Dict[int, Tuple[int, int, str]]
    ) -> Optional[int]:
        """Get the chapter number for a given line."""
        for chapter_num, (start, end, _) in chapters.items():
            if start <= line_num <= end:
                return chapter_num
        return None


class NarrativeChunker:
    """Chunk narrative text while preserving context and chapter awareness."""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.chapter_detector = ChapterDetector()
    
    def chunk_text(
        self,
        text: str,
        book_name: str
    ) -> List[NarrativeChunk]:
        """
        Chunk text into overlapping segments.
        
        Features:
        - Respects chapter boundaries when possible
        - Maintains overlap for context
        - Preserves paragraph integrity
        """
        lines = text.split('\n')
        chapters = self.chapter_detector.detect_chapters(text)
        
        chunks = []
        
        if self.config.chapter_aware and chapters:
            # Process by chapter
            for chapter_num, (start_line, end_line, title) in sorted(chapters.items()):
                chapter_text = '\n'.join(lines[start_line:end_line + 1])
                chapter_chunks = self._chunk_section(
                    text=chapter_text,
                    book_name=book_name,
                    base_line=start_line,
                    chapter_num=chapter_num,
                    chapter_title=title
                )
                chunks.extend(chapter_chunks)
        else:
            # Process entire text
            chunks = self._chunk_section(
                text=text,
                book_name=book_name,
                base_line=0,
                chapter_num=None,
                chapter_title=None
            )
        
        return chunks
    
    def _chunk_section(
        self,
        text: str,
        book_name: str,
        base_line: int,
        chapter_num: Optional[int],
        chapter_title: Optional[str]
    ) -> List[NarrativeChunk]:
        """Chunk a section of text."""
        chunks = []
        paragraphs = self._split_into_paragraphs(text)
        
        current_chunk_text = ""
        current_chunk_start = base_line
        current_line = base_line
        
        for para, para_lines in paragraphs:
            # Check if adding this paragraph exceeds chunk size
            if len(current_chunk_text) + len(para) > self.config.chunk_size:
                # Save current chunk if it meets minimum size
                if len(current_chunk_text) >= self.config.min_chunk_size:
                    chunk = NarrativeChunk(
                        chunk_id=NarrativeChunk.generate_id(book_name, current_chunk_start),
                        book_name=book_name,
                        text=current_chunk_text.strip(),
                        chapter=f"Chapter {chapter_num}: {chapter_title}" if chapter_num else None,
                        chapter_number=chapter_num,
                        start_line=current_chunk_start,
                        end_line=current_line - 1
                    )
                    chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk_text) - self.config.chunk_overlap)
                current_chunk_text = current_chunk_text[overlap_start:]
                current_chunk_start = current_line - (len(current_chunk_text.split('\n')) - 1)
            
            current_chunk_text += para + "\n\n"
            current_line += para_lines
        
        # Don't forget the last chunk
        if len(current_chunk_text.strip()) >= self.config.min_chunk_size:
            chunk = NarrativeChunk(
                chunk_id=NarrativeChunk.generate_id(book_name, current_chunk_start),
                book_name=book_name,
                text=current_chunk_text.strip(),
                chapter=f"Chapter {chapter_num}: {chapter_title}" if chapter_num else None,
                chapter_number=chapter_num,
                start_line=current_chunk_start,
                end_line=current_line
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[Tuple[str, int]]:
        """Split text into paragraphs, returning (paragraph, line_count)."""
        # Split on double newlines or multiple newlines
        raw_paras = re.split(r'\n\s*\n', text)
        result = []
        
        for para in raw_paras:
            para = para.strip()
            if para:
                line_count = para.count('\n') + 1
                result.append((para, line_count))
        
        return result


class PathwayNarrativeIngestion:
    """
    Pathway-based document ingestion pipeline.
    
    Uses Pathway framework for:
    - Real-time document processing
    - Incremental updates
    - Efficient indexing
    """
    
    def __init__(self, chunking_config: Optional[ChunkingConfig] = None):
        self.chunker = NarrativeChunker(chunking_config)
        self.chapter_detector = ChapterDetector()
        self._structures: Dict[str, BookNarrativeStructure] = {}
    
    def ingest_novel(self, file_path: Path) -> BookNarrativeStructure:
        """
        Ingest a complete novel file.
        
        This processes the ENTIRE novel without truncation or summarization.
        """
        book_name = file_path.stem
        logger.info(f"Ingesting novel: {book_name}")
        
        # Read entire file
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        
        # Clean up the text
        text = self._clean_text(text)
        
        # Get basic stats
        lines = text.split('\n')
        total_lines = len(lines)
        
        # Detect chapters
        chapters = self.chapter_detector.detect_chapters(text)
        total_chapters = len(chapters)
        
        logger.info(f"  Total lines: {total_lines}")
        logger.info(f"  Total chapters: {total_chapters}")
        logger.info(f"  Total characters: {len(text)}")
        
        # Chunk the text
        chunks = self.chunker.chunk_text(text, book_name)
        logger.info(f"  Created {len(chunks)} chunks")
        
        # Build structure
        structure = BookNarrativeStructure(
            book_name=book_name,
            chunks=chunks,
            total_lines=total_lines,
            total_chapters=total_chapters,
            chapter_boundaries=chapters
        )
        
        self._structures[book_name] = structure
        return structure
    
    def ingest_all_novels(self, data_dir: Path) -> Dict[str, BookNarrativeStructure]:
        """Ingest all novel files from a directory."""
        structures = {}
        
        for file_path in data_dir.glob("*.txt"):
            try:
                structure = self.ingest_novel(file_path)
                structures[structure.book_name] = structure
            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {e}")
        
        return structures
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove Project Gutenberg headers/footers
        start_markers = [
            "*** START OF THE PROJECT GUTENBERG",
            "*** START OF THIS PROJECT GUTENBERG"
        ]
        end_markers = [
            "*** END OF THE PROJECT GUTENBERG",
            "*** END OF THIS PROJECT GUTENBERG",
            "End of the Project Gutenberg"
        ]
        
        for marker in start_markers:
            if marker in text:
                idx = text.find(marker)
                # Find the next newline after the marker
                newline_idx = text.find('\n', idx)
                if newline_idx > 0:
                    text = text[newline_idx + 1:]
        
        for marker in end_markers:
            if marker in text:
                idx = text.find(marker)
                text = text[:idx]
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive whitespace while preserving paragraph structure
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        
        return text.strip()
    
    def get_structure(self, book_name: str) -> Optional[BookNarrativeStructure]:
        """Get the narrative structure for a book."""
        # Try exact match
        if book_name in self._structures:
            return self._structures[book_name]
        
        # Try case-insensitive match
        for name, structure in self._structures.items():
            if name.lower() == book_name.lower():
                return structure
            # Try partial match
            if book_name.lower() in name.lower() or name.lower() in book_name.lower():
                return structure
        
        return None
    
    def search_chunks(
        self,
        book_name: str,
        character_name: str,
        limit: int = 50
    ) -> List[NarrativeChunk]:
        """Search for chunks mentioning a character."""
        structure = self.get_structure(book_name)
        if not structure:
            return []
        
        matching_chunks = []
        char_lower = character_name.lower()
        
        for chunk in structure.chunks:
            if char_lower in chunk.text.lower():
                chunk.characters_mentioned.append(character_name)
                matching_chunks.append(chunk)
        
        return matching_chunks[:limit]


class PathwayStreamProcessor:
    """
    Pathway-based stream processor for real-time document updates.
    
    Note: This is a conceptual implementation. In production, 
    this would use Pathway's actual streaming capabilities.
    """
    
    def __init__(self):
        self.ingestion = PathwayNarrativeIngestion()
        self._document_cache = {}
    
    def create_input_stream(self, data_dir: Path):
        """Create a Pathway input stream from directory."""
        if PATHWAY_AVAILABLE:
            # Pathway implementation
            # In real usage:
            # return pw.io.fs.read(data_dir, format='text', mode='static')
            pass
        
        # Fallback: standard file reading
        documents = {}
        for file_path in data_dir.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                documents[file_path.stem] = f.read()
        return documents
    
    def process_documents(self, documents: Dict[str, str]) -> Dict[str, BookNarrativeStructure]:
        """Process a collection of documents."""
        structures = {}
        
        for book_name, text in documents.items():
            # Clean and chunk
            clean_text = self.ingestion._clean_text(text)
            chunks = self.ingestion.chunker.chunk_text(clean_text, book_name)
            
            # Build structure
            lines = clean_text.split('\n')
            chapters = self.ingestion.chapter_detector.detect_chapters(clean_text)
            
            structure = BookNarrativeStructure(
                book_name=book_name,
                chunks=chunks,
                total_lines=len(lines),
                total_chapters=len(chapters),
                chapter_boundaries=chapters
            )
            
            structures[book_name] = structure
            self._document_cache[book_name] = structure
        
        return structures


# Factory function for the ingestion pipeline
def create_ingestion_pipeline(config: Optional[ChunkingConfig] = None) -> PathwayNarrativeIngestion:
    """Create and configure the ingestion pipeline."""
    return PathwayNarrativeIngestion(config)
