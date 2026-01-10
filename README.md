# KGPH Pathway - Narrative Consistency Verification System

## Track A: Systems Reasoning with NLP and Generative AI
### Kharagpur Data Science Hackathon 2026

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Phase Details](#phase-details)
4. [Technical Implementation](#technical-implementation)
5. [Usage](#usage)
6. [Limitations and Failure Cases](#limitations-and-failure-cases)
7. [Track A Compliance](#track-a-compliance)

---

## 🎯 System Overview

This system verifies whether a hypothetical character backstory is **globally consistent** with a full-length novel (100k+ words). It is a **structured causal consistency classification problem**, not a text-generation task.

### Key Features

- ✅ **Full novel processing** - No truncation, no summarization
- ✅ **Pathway framework integration** for document ingestion
- ✅ **Local Ollama LLM** for constrained extraction tasks
- ✅ **Temporal consistency tracking** across narrative
- ✅ **Evidence aggregation** from multiple text spans
- ✅ **Binary classification output** (consistent/contradictory)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NARRATIVE CONSISTENCY VERIFICATION SYSTEM                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐   │
│  │   PHASE 1        │     │    PHASE 2        │     │    PHASE 3       │   │
│  │   Narrative      │────▶│    Character      │────▶│    Training      │   │
│  │   Ingestion      │     │    Extraction     │     │    Phase         │   │
│  └──────────────────┘     └───────────────────┘     └──────────────────┘   │
│         │                        │                         │               │
│         ▼                        ▼                         ▼               │
│  ┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐   │
│  │ BookNarrative    │     │ CharacterTimeline │     │ Trained          │   │
│  │ Structure        │     │ (per character)   │     │ Thresholds       │   │
│  │ - Chunks         │     │ - States          │     │ - Weights        │   │
│  │ - Chapters       │     │ - Constraints     │     │ - Rules          │   │
│  │ - Metadata       │     │ - Evidence        │     │                  │   │
│  └──────────────────┘     └───────────────────┘     └──────────────────┘   │
│                                                             │               │
│                           ┌───────────────────┐             │               │
│                           │    PHASE 4        │◀────────────┘               │
│                           │    Inference      │                             │
│                           └───────────────────┘                             │
│                                    │                                        │
│                                    ▼                                        │
│                           ┌───────────────────┐                             │
│                           │   results.csv     │                             │
│                           │   1 = consistent  │                             │
│                           │   0 = contradict  │                             │
│                           └───────────────────┘                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Interaction

```
┌─────────────────┐
│     Novels      │
│   (.txt files)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  PathwayNarrativeIngestion  │
│  ├── ChapterDetector    │
│  ├── NarrativeChunker   │
│  └── StreamProcessor    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  CharacterTimelineExtractor │
│  ├── NameResolver       │
│  ├── TemporalParser     │
│  ├── StateBuilder       │
│  └── LLM Extractor      │◀──── Ollama API
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  HybridConsistencyVerifier  │
│  ├── RuleBasedVerifier  │
│  ├── LLM Checker        │◀──── Ollama API
│  └── ScoreCombiner      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│    ConsistencyResult    │
│    (binary output)      │
└─────────────────────────┘
```

---

## 📖 Phase Details

### Phase 1: Narrative Ingestion

**Purpose**: Read and process complete novels without truncation.

**Process**:
1. Read all `.txt` files from data directory
2. Clean Gutenberg headers/footers
3. Detect chapter boundaries using regex patterns
4. Chunk text with overlap for context preservation
5. Store chunks with metadata (chapter, line range, book name)

**Key Classes**:
- `PathwayNarrativeIngestion` - Main ingestion pipeline
- `ChapterDetector` - Identifies chapter boundaries
- `NarrativeChunker` - Creates overlapping chunks

**Chunking Strategy**:
```python
chunk_size = 2000 characters
overlap = 400 characters
chapter_aware = True  # Respects chapter boundaries
```

---

### Phase 2: Character Timeline Extraction

**Purpose**: Build structured representation of each character's evolution.

**Schema (CharacterStateTimeline)**:
```python
CharacterTimeline:
    character_name: str
    book_name: str
    states: List[CharacterState]  # Ordered by narrative progression
        - time_segment: str
        - chapter_range: (start, end)
        - beliefs: List[str]
        - motivations: List[str]
        - fears: List[str]
        - goals: List[str]
        - actions_taken: List[str]
        - constraints_established: List[Constraint]
        - relationships: List[Relationship]
        - supporting_evidence: List[EvidenceSpan]
    permanent_constraints: List[Constraint]
    irreversible_events: List[str]
    core_traits: List[str]
```

**LLM Usage (STRICTLY LIMITED)**:
- Extract beliefs, motivations, constraints from text chunks
- Normalize natural-language claims into structured fields
- **NOT** for: end-to-end reasoning, summarization, answer generation

---

### Phase 3: Training Phase

**Purpose**: Learn consistency scoring from labeled examples.

**Process**:
1. Load `train.csv` with ground-truth labels
2. For each training example:
   - Compare backstory against prebuilt timeline
   - Record true/false positives and negatives
3. Optimize:
   - Classification threshold
   - Component weights (belief, action, temporal, etc.)
   - Contradiction penalty

**Important**: Training does NOT modify narrative structures.

---

### Phase 4: Inference Phase

**Purpose**: Classify test backstories as consistent/contradictory.

**Decision Logic**:
```
For each backstory:
    1. Normalize claim into structured components
    2. Check against character evolution
    3. Verify established beliefs
    4. Confirm against irreversible events
    5. Validate temporal constraints
    6. Combine scores with trained weights
    7. Apply threshold → binary output
```

**Output**:
- `1` → Consistent
- `0` → Contradictory

---

## 💻 Technical Implementation

### Project Structure

```
kgph_pathway/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration classes
│   ├── models.py             # Data models (CharacterState, etc.)
│   ├── llm_client.py         # Ollama integration
│   ├── narrative_ingestion.py # Phase 1 implementation
│   ├── character_extraction.py # Phase 2 implementation
│   ├── consistency_verifier.py # Phases 3-4 implementation
│   └── pipeline.py           # Orchestration
├── data/
│   ├── *.txt                 # Novel files
│   ├── train.csv             # Training data
│   └── test.csv              # Test data
├── output/
│   └── results.csv           # Predictions
├── main.py                   # Entry point
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

### Ollama Integration

```python
# Configuration
OllamaConfig:
    base_url: "http://localhost:11434"
    model: "llama3.2"  # or mistral, qwen, etc.
    temperature: 0.1   # Low for consistent extraction
    context_length: 8192

# Usage is LIMITED to:
1. CharacterExtractor - Extract structured info from chunks
2. ClaimNormalizer - Parse backstory into components
3. ConsistencyChecker - Compare claim to evidence
```

### Consistency Scoring

```python
# Component weights (tuned during training)
weights = {
    'belief': 0.20,
    'motivation': 0.15,
    'action': 0.20,
    'relationship': 0.15,
    'constraint': 0.20,
    'temporal': 0.10
}

# Final score = weighted combination + rule penalties
final_score = (
    rule_score * 0.4 +
    component_weighted_avg * 0.4 +
    llm_score * 0.2  # if available
)

# Verdict
if contradictions_found and (count >= 2 or score < 0.4):
    verdict = CONTRADICTORY
elif score >= threshold:
    verdict = CONSISTENT
else:
    verdict = CONTRADICTORY
```

---

## 🚀 Usage

### Basic Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama (if using LLM)
ollama serve

# Run full pipeline
python main.py

# Run without LLM (faster, less accurate)
python main.py --no-llm

# Training only
python main.py --train-only

# Inference only
python main.py --infer-only
```

### Programmatic Usage

```python
from src.pipeline import NarrativeConsistencyPipeline, DataLoader
from src.config import DATA_DIR

# Create pipeline
pipeline = NarrativeConsistencyPipeline(use_llm=True)

# Run full pipeline
results = pipeline.run_full_pipeline(data_dir=DATA_DIR)

# Or run phases separately
pipeline.ingest_narratives(DATA_DIR)
pipeline.extract_character_timelines(characters_by_book)
pipeline.train(training_entries)
results = pipeline.predict(test_entries)
```

---

## ⚠️ Limitations and Failure Cases

### Long-Context Limitations

1. **Chunk boundary issues**: Information split across chunks may be missed
2. **Cross-chapter references**: Foreshadowing/callbacks may not be linked
3. **Implicit information**: Unstated implications harder to capture

### LLM Limitations

1. **Extraction errors**: LLM may miss or hallucinate character info
2. **Semantic similarity**: Simple overlap checks may miss paraphrases
3. **Temporal reasoning**: Complex timeline logic may be imprecise

### Rule-Based Limitations

1. **Keyword matching**: May miss semantic equivalents
2. **Negation handling**: Complex negations may be misinterpreted
3. **Cultural references**: Historical context may be missing

### Known Edge Cases

| Case | Issue | Mitigation |
|------|-------|------------|
| Character aliases | "Monte Cristo" vs "Dantès" | Name resolver with known aliases |
| Unreliable narrators | Stated ≠ actual | N/A - assumes narrator reliability |
| Flashbacks | Non-linear timeline | Temporal marker parsing |
| Translation variants | Different wordings | Semantic similarity (limited) |

---

## ✅ Track A Compliance

### Requirements Met

| Requirement | Implementation |
|-------------|----------------|
| Pathway framework | `PathwayNarrativeIngestion`, stream processing |
| Long-context handling | Full novel ingestion, no truncation |
| Temporal consistency | `TemporalMarkerParser`, state ordering |
| Evidence aggregation | `EvidenceSpan`, multi-source scoring |
| Local Ollama | `OllamaClient`, HTTP API integration |
| Not naive RAG | Multi-phase pipeline, rule-based checks |
| Binary classification | `ConsistencyResult.prediction` (0/1) |

### LLM Usage Constraints

✅ **Allowed**:
- Extracting beliefs/motivations from text
- Normalizing claims to structured format
- Comparing claims to evidence

❌ **NOT Used For**:
- End-to-end answer generation
- Novel summarization
- Replacing symbolic checks

---

## 📊 Performance Notes

- **Novel ingestion**: ~5 seconds per novel
- **Character extraction**: ~10-30 seconds per character (with LLM)
- **Training**: ~2-5 minutes for 80 examples
- **Inference**: ~1-3 seconds per backstory (with LLM)

Without LLM, all operations are ~10x faster but less accurate.

---

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# Chunking
chunk_size = 2000
chunk_overlap = 400

# LLM
model = "llama3.2"
temperature = 0.1

# Consistency
consistency_threshold = 0.5
belief_weight = 0.20
# ... etc
```

---

## 📄 License

MIT License - Kharagpur Data Science Hackathon 2026
