# KGPH Pathway - Narrative Consistency Verification System
# Track A: Systems Reasoning with NLP and Generative AI
# Kharagpur Data Science Hackathon 2026

"""
This package implements a causal consistency classification system that verifies
whether hypothetical character backstories are globally consistent with full-length novels.

Architecture Overview:
- Phase 1: Narrative Ingestion using Pathway framework
- Phase 2: Character State Timeline extraction
- Phase 3: Training phase with train.csv
- Phase 4: Inference phase with test.csv

Key Components:
- PathwayNarrativeIngestion: Document ingestion and chunking
- CharacterStateExtractor: LLM-based character information extraction
- ConsistencyVerifier: Rule-based + LLM hybrid consistency checking
- TemporalConstraintGraph: Tracks temporal ordering of events
"""

__version__ = "1.0.0"
__author__ = "KGPH Team"
