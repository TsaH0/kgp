"""
Configuration settings for the Narrative Consistency Verification System.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = PROJECT_ROOT / ".cache"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


@dataclass
class OllamaConfig:
    """Configuration for Ollama LLM endpoint."""
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"  # Default model, can be changed to mistral, qwen, etc.
    timeout: int = 120
    max_retries: int = 3
    temperature: float = 0.1  # Low temperature for consistent extraction
    context_length: int = 8192


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    chunk_size: int = 2000  # Characters per chunk
    chunk_overlap: int = 400  # Overlap between chunks
    min_chunk_size: int = 500  # Minimum chunk size
    chapter_aware: bool = True  # Try to respect chapter boundaries


@dataclass
class CharacterExtractionConfig:
    """Configuration for character state extraction."""
    extract_beliefs: bool = True
    extract_motivations: bool = True
    extract_fears: bool = True
    extract_actions: bool = True
    extract_relationships: bool = True
    extract_constraints: bool = True
    extract_temporal_markers: bool = True
    max_evidence_spans: int = 5
    confidence_threshold: float = 0.6


@dataclass
class ConsistencyConfig:
    """Configuration for consistency verification."""
    # Scoring weights
    belief_weight: float = 0.20
    motivation_weight: float = 0.15
    action_weight: float = 0.20
    relationship_weight: float = 0.15
    constraint_weight: float = 0.20
    temporal_weight: float = 0.10
    
    # Thresholds (will be tuned during training)
    consistency_threshold: float = 0.5
    contradiction_penalty: float = 0.3
    
    # Verification settings
    min_evidence_matches: int = 2
    require_temporal_consistency: bool = True
    allow_implicit_matches: bool = True


@dataclass
class SystemConfig:
    """Main system configuration."""
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    extraction: CharacterExtractionConfig = field(default_factory=CharacterExtractionConfig)
    consistency: ConsistencyConfig = field(default_factory=ConsistencyConfig)
    
    # File paths
    train_csv: Path = DATA_DIR / "train.csv"
    test_csv: Path = DATA_DIR / "test.csv"
    results_csv: Path = OUTPUT_DIR / "results.csv"
    
    # Processing settings
    batch_size: int = 10
    use_cache: bool = True
    verbose: bool = True
    
    # Novel files
    @property
    def novel_files(self) -> List[Path]:
        """Get all novel files from data directory."""
        return list(DATA_DIR.glob("*.txt"))


# Default configuration instance
config = SystemConfig()
