"""
Data models for the Narrative Consistency Verification System.

This module defines the core data structures for representing:
- Character states and timelines
- Narrative chunks and evidence
- Constraints and relationships
- Consistency verification results
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any, Tuple
from enum import Enum
from datetime import datetime
import json
import hashlib


class ConstraintType(Enum):
    """Types of narrative constraints."""
    BELIEF = "belief"
    MOTIVATION = "motivation"
    FEAR = "fear"
    ACTION = "action"
    RELATIONSHIP = "relationship"
    TEMPORAL = "temporal"
    PHYSICAL = "physical"  # Physical impossibilities
    LOGICAL = "logical"  # Logical contradictions


class ConsistencyVerdict(Enum):
    """Possible consistency verdicts."""
    CONSISTENT = "consistent"
    CONTRADICTORY = "contradictory"
    UNCERTAIN = "uncertain"


class RelationshipType(Enum):
    """Types of character relationships."""
    FAMILY = "family"
    FRIEND = "friend" 
    ENEMY = "enemy"
    MENTOR = "mentor"
    COLLEAGUE = "colleague"
    ROMANTIC = "romantic"
    UNKNOWN = "unknown"


@dataclass
class EvidenceSpan:
    """A span of text that serves as evidence for a claim."""
    text: str
    chunk_id: str
    chapter: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "chapter": self.chapter,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "confidence": self.confidence
        }


@dataclass
class Constraint:
    """A narrative constraint extracted from the text."""
    constraint_type: ConstraintType
    description: str
    is_positive: bool = True  # True = must be true, False = must be false
    temporal_scope: Optional[str] = None  # e.g., "before chapter 10", "childhood"
    evidence: List[EvidenceSpan] = field(default_factory=list)
    confidence: float = 1.0
    
    def conflicts_with(self, other: 'Constraint') -> bool:
        """Check if this constraint conflicts with another."""
        if self.constraint_type != other.constraint_type:
            return False
        # Simple conflict: same type, opposite polarity, similar description
        if self.is_positive != other.is_positive:
            # Check for semantic similarity (simplified)
            return self._descriptions_similar(other.description)
        return False
    
    def _descriptions_similar(self, other_desc: str) -> bool:
        """Check if two descriptions are semantically similar (simplified)."""
        # This is a placeholder - in production, use embeddings
        self_words = set(self.description.lower().split())
        other_words = set(other_desc.lower().split())
        overlap = len(self_words & other_words)
        union = len(self_words | other_words)
        return overlap / union > 0.3 if union > 0 else False


@dataclass
class Relationship:
    """A relationship between characters."""
    target_character: str
    relationship_type: RelationshipType
    description: str
    temporal_scope: Optional[str] = None
    evidence: List[EvidenceSpan] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "target": self.target_character,
            "type": self.relationship_type.value,
            "description": self.description,
            "temporal_scope": self.temporal_scope
        }


@dataclass
class CharacterState:
    """The state of a character at a specific point in the narrative."""
    time_segment: str  # e.g., "chapter 1-3", "childhood", "before imprisonment"
    chapter_range: Optional[Tuple[int, int]] = None
    
    # Core attributes
    beliefs: List[str] = field(default_factory=list)
    motivations: List[str] = field(default_factory=list)
    fears: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    
    # Actions and events
    actions_taken: List[str] = field(default_factory=list)
    events_experienced: List[str] = field(default_factory=list)
    
    # Constraints
    constraints_established: List[Constraint] = field(default_factory=list)
    contradictions_ruled_out: List[str] = field(default_factory=list)
    
    # Relationships
    relationships: List[Relationship] = field(default_factory=list)
    
    # Evidence
    supporting_evidence: List[EvidenceSpan] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "time_segment": self.time_segment,
            "chapter_range": self.chapter_range,
            "beliefs": self.beliefs,
            "motivations": self.motivations,
            "fears": self.fears,
            "goals": self.goals,
            "actions_taken": self.actions_taken,
            "events_experienced": self.events_experienced,
            "constraints": [c.description for c in self.constraints_established],
            "relationships": [r.to_dict() for r in self.relationships]
        }


@dataclass
class CharacterTimeline:
    """Complete timeline of a character's evolution through the narrative."""
    character_name: str
    book_name: str
    states: List[CharacterState] = field(default_factory=list)
    
    # Global constraints that apply throughout
    permanent_constraints: List[Constraint] = field(default_factory=list)
    irreversible_events: List[str] = field(default_factory=list)
    
    # Character summary
    core_traits: List[str] = field(default_factory=list)
    character_arc: Optional[str] = None
    
    def add_state(self, state: CharacterState):
        """Add a character state to the timeline."""
        self.states.append(state)
        self.states.sort(key=lambda s: s.chapter_range[0] if s.chapter_range else 0)
    
    def get_state_at_chapter(self, chapter: int) -> Optional[CharacterState]:
        """Get the character state at a specific chapter."""
        for state in self.states:
            if state.chapter_range:
                if state.chapter_range[0] <= chapter <= state.chapter_range[1]:
                    return state
        return None
    
    def get_all_beliefs(self) -> List[str]:
        """Get all beliefs across all states."""
        beliefs = []
        for state in self.states:
            beliefs.extend(state.beliefs)
        return beliefs
    
    def get_all_actions(self) -> List[str]:
        """Get all actions across all states."""
        actions = []
        for state in self.states:
            actions.extend(state.actions_taken)
        return actions
    
    def get_all_relationships(self) -> List[Relationship]:
        """Get all relationships across all states."""
        relationships = []
        for state in self.states:
            relationships.extend(state.relationships)
        return relationships
    
    def to_dict(self) -> Dict:
        return {
            "character_name": self.character_name,
            "book_name": self.book_name,
            "states": [s.to_dict() for s in self.states],
            "permanent_constraints": [c.description for c in self.permanent_constraints],
            "irreversible_events": self.irreversible_events,
            "core_traits": self.core_traits,
            "character_arc": self.character_arc
        }


@dataclass
class NarrativeChunk:
    """A chunk of narrative text with metadata."""
    chunk_id: str
    book_name: str
    text: str
    chapter: Optional[str] = None
    chapter_number: Optional[int] = None
    start_line: int = 0
    end_line: int = 0
    
    # Extracted information
    characters_mentioned: List[str] = field(default_factory=list)
    temporal_markers: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash(self.chunk_id)
    
    @staticmethod
    def generate_id(book_name: str, start_line: int) -> str:
        """Generate a unique chunk ID."""
        content = f"{book_name}_{start_line}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass 
class BackstoryEntry:
    """A backstory entry from train/test CSV."""
    id: int
    book_name: str
    character: str
    caption: Optional[str]
    content: str
    label: Optional[str] = None  # 'consistent' or 'contradict', None for test
    
    @property
    def is_consistent(self) -> Optional[bool]:
        """Return True if consistent, False if contradictory, None if unknown."""
        if self.label is None:
            return None
        return self.label.lower() == "consistent"


@dataclass
class ConsistencyResult:
    """Result of consistency verification."""
    backstory_id: int
    verdict: ConsistencyVerdict
    confidence: float
    
    # Detailed scores
    belief_score: float = 0.0
    motivation_score: float = 0.0
    action_score: float = 0.0
    relationship_score: float = 0.0
    constraint_score: float = 0.0
    temporal_score: float = 0.0
    
    # Evidence and reasoning
    supporting_evidence: List[EvidenceSpan] = field(default_factory=list)
    contradicting_evidence: List[EvidenceSpan] = field(default_factory=list)
    reasoning: str = ""
    
    # Detected issues
    contradictions: List[str] = field(default_factory=list)
    temporal_violations: List[str] = field(default_factory=list)
    constraint_violations: List[str] = field(default_factory=list)
    
    @property
    def final_score(self) -> float:
        """Calculate weighted final score."""
        # Weights from config (hardcoded here for simplicity)
        return (
            self.belief_score * 0.20 +
            self.motivation_score * 0.15 +
            self.action_score * 0.20 +
            self.relationship_score * 0.15 +
            self.constraint_score * 0.20 +
            self.temporal_score * 0.10
        )
    
    @property
    def prediction(self) -> int:
        """Return 1 for consistent, 0 for contradictory."""
        return 1 if self.verdict == ConsistencyVerdict.CONSISTENT else 0
    
    def to_dict(self) -> Dict:
        return {
            "id": self.backstory_id,
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "final_score": self.final_score,
            "prediction": self.prediction,
            "contradictions": self.contradictions,
            "reasoning": self.reasoning
        }


@dataclass
class BookNarrativeStructure:
    """Complete narrative structure for a book."""
    book_name: str
    chunks: List[NarrativeChunk] = field(default_factory=list)
    character_timelines: Dict[str, CharacterTimeline] = field(default_factory=dict)
    
    # Book metadata
    total_lines: int = 0
    total_chapters: int = 0
    chapter_boundaries: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    
    def get_character_timeline(self, character_name: str) -> Optional[CharacterTimeline]:
        """Get timeline for a specific character."""
        # Try exact match first
        if character_name in self.character_timelines:
            return self.character_timelines[character_name]
        # Try case-insensitive match
        for name, timeline in self.character_timelines.items():
            if name.lower() == character_name.lower():
                return timeline
        # Try partial match
        for name, timeline in self.character_timelines.items():
            if character_name.lower() in name.lower() or name.lower() in character_name.lower():
                return timeline
        return None
    
    def add_character_timeline(self, timeline: CharacterTimeline):
        """Add a character timeline."""
        self.character_timelines[timeline.character_name] = timeline
