"""
Character State Extraction and Timeline Building.

This module implements Phase 2 of the system:
- Builds CharacterStateTimeline from ingested novels
- Tracks character evolution over narrative progression
- Extracts beliefs, motivations, fears, actions, relationships
- Maintains temporal ordering and evidence spans

The structure is PERSISTENT and tracks how constraints evolve over time.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
import json

from .models import (
    CharacterState, CharacterTimeline, BookNarrativeStructure,
    NarrativeChunk, Constraint, ConstraintType, Relationship,
    RelationshipType, EvidenceSpan
)
from .llm_client import (
    CharacterExtractor, FallbackExtractor, 
    get_character_extractor, OllamaClient
)
from .config import CharacterExtractionConfig, config

logger = logging.getLogger(__name__)


class CharacterNameResolver:
    """Resolves character name variations and aliases."""
    
    # Known character aliases for the novels
    KNOWN_ALIASES = {
        # In Search of the Castaways
        "Jacques Paganel": ["Paganel", "Jacques", "the geographer"],
        "Lord Glenarvan": ["Glenarvan", "Lord Edward", "Edward Glenarvan"],
        "Lady Helena": ["Helena", "Lady Glenarvan", "Helena Glenarvan"],
        "Major McNabbs": ["McNabbs", "MacNabbs", "Major MacNabb", "the major"],
        "Captain Mangles": ["Mangles", "John Mangles", "the captain"],
        "Tom Ayrton/Ben Joyce": ["Ayrton", "Tom Ayrton", "Ben Joyce", "Joyce"],
        "Thalcave": ["the Patagonian", "the guide"],
        "Kai-Koumou": ["the Maori chief", "the chief"],
        "Captain Grant": ["Harry Grant", "Grant"],
        
        # The Count of Monte Cristo
        "Edmond Dantès": ["Dantès", "Edmond", "Count of Monte Cristo", "the Count", "Monte Cristo", "Sinbad the Sailor"],
        "Abbé Faria": ["Faria", "the abbé", "the old priest"],
        "Fernand Mondego": ["Fernand", "Count de Morcerf", "Morcerf"],
        "Mercedes": ["Mercédès", "Countess de Morcerf"],
        "Danglars": ["Baron Danglars", "the baron"],
        "Villefort": ["Gérard de Villefort", "the prosecutor", "the procureur"],
        "Noirtier": ["Noirtier de Villefort", "M. Noirtier", "the old man"],
        "Caderousse": ["Gaspard Caderousse"],
        "Maximilian Morrel": ["Maximilian", "Max", "Morrel"],
        "Valentine": ["Valentine de Villefort", "Mademoiselle de Villefort"],
        "Haydée": ["the Greek slave", "the princess"],
        "Albert de Morcerf": ["Albert", "the Viscount"],
    }
    
    def __init__(self):
        # Build reverse lookup
        self._alias_to_canonical = {}
        for canonical, aliases in self.KNOWN_ALIASES.items():
            self._alias_to_canonical[canonical.lower()] = canonical
            for alias in aliases:
                self._alias_to_canonical[alias.lower()] = canonical
    
    def resolve(self, name: str) -> str:
        """Resolve a character name to its canonical form."""
        name_lower = name.lower().strip()
        if name_lower in self._alias_to_canonical:
            return self._alias_to_canonical[name_lower]
        return name.strip()
    
    def get_search_names(self, canonical_name: str) -> List[str]:
        """Get all names/aliases to search for a character."""
        names = [canonical_name]
        if canonical_name in self.KNOWN_ALIASES:
            names.extend(self.KNOWN_ALIASES[canonical_name])
        return names


class TemporalMarkerParser:
    """Parse temporal markers from text to enable ordering."""
    
    # Temporal pattern categories
    ABSOLUTE_PATTERNS = [
        (r'\b(\d{4})\b', 'year'),  # Years like 1815
        (r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),?\s*(\d{4})?\b', 'date'),
    ]
    
    RELATIVE_PATTERNS = [
        (r'\b(at age|aged?|when .+ was)\s*(\d+)\b', 'age'),
        (r'\b(\d+)\s*years?\s*(old|of age)\b', 'age'),
        (r'\b(childhood|youth|boyhood|girlhood|infancy)\b', 'life_stage'),
        (r'\b(adulthood|maturity|middle age|old age)\b', 'life_stage'),
    ]
    
    SEQUENCE_PATTERNS = [
        (r'\b(before|prior to|previously)\b', 'before'),
        (r'\b(after|following|subsequently|later)\b', 'after'),
        (r'\b(during|while|when)\b', 'during'),
        (r'\b(first|initially|originally)\b', 'start'),
        (r'\b(finally|eventually|ultimately)\b', 'end'),
    ]
    
    def __init__(self):
        self.absolute = [(re.compile(p, re.IGNORECASE), t) for p, t in self.ABSOLUTE_PATTERNS]
        self.relative = [(re.compile(p, re.IGNORECASE), t) for p, t in self.RELATIVE_PATTERNS]
        self.sequence = [(re.compile(p, re.IGNORECASE), t) for p, t in self.SEQUENCE_PATTERNS]
    
    def extract_markers(self, text: str) -> List[Dict]:
        """Extract all temporal markers from text."""
        markers = []
        
        for pattern, marker_type in self.absolute:
            for match in pattern.finditer(text):
                markers.append({
                    'type': 'absolute',
                    'subtype': marker_type,
                    'value': match.group(),
                    'position': match.start()
                })
        
        for pattern, marker_type in self.relative:
            for match in pattern.finditer(text):
                markers.append({
                    'type': 'relative',
                    'subtype': marker_type,
                    'value': match.group(),
                    'position': match.start()
                })
        
        for pattern, marker_type in self.sequence:
            for match in pattern.finditer(text):
                markers.append({
                    'type': 'sequence',
                    'subtype': marker_type,
                    'value': match.group(),
                    'position': match.start()
                })
        
        return sorted(markers, key=lambda m: m['position'])
    
    def estimate_temporal_order(self, markers: List[Dict]) -> int:
        """Estimate relative temporal position based on markers."""
        # Return a score where lower = earlier in narrative
        score = 50  # Default middle
        
        for marker in markers:
            if marker['subtype'] == 'year':
                try:
                    year = int(marker['value'])
                    score = year - 1800  # Normalize around 1800s
                except:
                    pass
            elif marker['subtype'] == 'age':
                # Extract age number
                match = re.search(r'\d+', marker['value'])
                if match:
                    age = int(match.group())
                    score = age  # Younger = earlier
            elif marker['subtype'] in ['childhood', 'boyhood', 'girlhood', 'infancy']:
                score = 10
            elif marker['subtype'] in ['youth']:
                score = 20
            elif marker['subtype'] in ['adulthood', 'maturity']:
                score = 40
            elif marker['subtype'] in ['old age']:
                score = 70
            elif marker['subtype'] == 'start':
                score = min(score, 15)
            elif marker['subtype'] == 'end':
                score = max(score, 85)
        
        return score


class CharacterStateBuilder:
    """Builds CharacterState objects from extracted information."""
    
    def __init__(self, extraction_config: Optional[CharacterExtractionConfig] = None):
        self.config = extraction_config or config.extraction
        self.name_resolver = CharacterNameResolver()
        self.temporal_parser = TemporalMarkerParser()
    
    def build_state(
        self,
        chunk: NarrativeChunk,
        character_name: str,
        extracted_info: Dict
    ) -> CharacterState:
        """Build a CharacterState from extracted information."""
        # Determine time segment
        time_segment = self._determine_time_segment(chunk, extracted_info)
        
        # Parse temporal markers
        temporal_markers = extracted_info.get('temporal_markers', [])
        
        # Build constraints from extracted info
        constraints = self._build_constraints(extracted_info, chunk)
        
        # Build relationships
        relationships = self._build_relationships(
            extracted_info.get('relationships', []),
            chunk
        )
        
        # Create evidence span for this extraction
        evidence = EvidenceSpan(
            text=chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text,
            chunk_id=chunk.chunk_id,
            chapter=chunk.chapter,
            start_line=chunk.start_line,
            end_line=chunk.end_line
        )
        
        return CharacterState(
            time_segment=time_segment,
            chapter_range=(chunk.start_line, chunk.end_line),
            beliefs=extracted_info.get('beliefs', []),
            motivations=extracted_info.get('motivations', []),
            fears=extracted_info.get('fears', []),
            goals=extracted_info.get('goals', []),
            actions_taken=extracted_info.get('actions', []),
            events_experienced=[],  # Filled from context
            constraints_established=constraints,
            relationships=relationships,
            supporting_evidence=[evidence]
        )
    
    def _determine_time_segment(
        self, 
        chunk: NarrativeChunk,
        extracted_info: Dict
    ) -> str:
        """Determine the time segment for a state."""
        # Use chapter if available
        if chunk.chapter:
            return chunk.chapter
        
        # Use temporal markers if available
        markers = extracted_info.get('temporal_markers', [])
        if markers:
            # Find the most specific marker
            for marker in markers:
                if isinstance(marker, str):
                    return marker
        
        # Fall back to line range
        return f"Lines {chunk.start_line}-{chunk.end_line}"
    
    def _build_constraints(
        self,
        extracted_info: Dict,
        chunk: NarrativeChunk
    ) -> List[Constraint]:
        """Build constraints from extracted information."""
        constraints = []
        
        # Constraints from explicit constraint field
        for constraint_text in extracted_info.get('constraints', []):
            constraints.append(Constraint(
                constraint_type=ConstraintType.LOGICAL,
                description=constraint_text,
                evidence=[EvidenceSpan(
                    text=constraint_text,
                    chunk_id=chunk.chunk_id
                )]
            ))
        
        # Actions become action constraints
        for action in extracted_info.get('actions', []):
            constraints.append(Constraint(
                constraint_type=ConstraintType.ACTION,
                description=action,
                evidence=[EvidenceSpan(
                    text=action,
                    chunk_id=chunk.chunk_id
                )]
            ))
        
        # Beliefs become belief constraints
        for belief in extracted_info.get('beliefs', []):
            constraints.append(Constraint(
                constraint_type=ConstraintType.BELIEF,
                description=belief,
                evidence=[EvidenceSpan(
                    text=belief,
                    chunk_id=chunk.chunk_id
                )]
            ))
        
        return constraints
    
    def _build_relationships(
        self,
        relationship_data: List[Dict],
        chunk: NarrativeChunk
    ) -> List[Relationship]:
        """Build Relationship objects from extracted data."""
        relationships = []
        
        for rel_data in relationship_data:
            if isinstance(rel_data, dict):
                target = rel_data.get('character', 'unknown')
                rel_type_str = rel_data.get('relationship', 'unknown')
                description = rel_data.get('description', '')
            else:
                continue
            
            # Map string to enum
            try:
                rel_type = RelationshipType(rel_type_str.lower())
            except ValueError:
                rel_type = RelationshipType.UNKNOWN
            
            relationships.append(Relationship(
                target_character=target,
                relationship_type=rel_type,
                description=description,
                evidence=[EvidenceSpan(
                    text=description,
                    chunk_id=chunk.chunk_id
                )]
            ))
        
        return relationships


class CharacterTimelineExtractor:
    """
    Extracts and builds CharacterTimeline objects from narrative structures.
    
    This class implements the core of Phase 2:
    - Incrementally processes chunks
    - Builds evolving character state over time
    - Tracks constraints and their temporal scope
    """
    
    def __init__(
        self,
        extraction_config: Optional[CharacterExtractionConfig] = None,
        use_llm: bool = True
    ):
        self.config = extraction_config or config.extraction
        self.use_llm = use_llm
        self.name_resolver = CharacterNameResolver()
        self.state_builder = CharacterStateBuilder(self.config)
        self.temporal_parser = TemporalMarkerParser()
        
        # LLM-based extractor
        self.llm_extractor = get_character_extractor() if use_llm else None
        self.fallback_extractor = FallbackExtractor()
    
    def extract_timeline(
        self,
        narrative_structure: BookNarrativeStructure,
        character_name: str
    ) -> CharacterTimeline:
        """
        Extract a complete timeline for a character from a book.
        
        This processes chunks in order, building up the character's
        state incrementally as they evolve through the narrative.
        """
        canonical_name = self.name_resolver.resolve(character_name)
        search_names = self.name_resolver.get_search_names(canonical_name)
        
        logger.info(f"Extracting timeline for {canonical_name} from {narrative_structure.book_name}")
        
        # Find relevant chunks
        relevant_chunks = self._find_relevant_chunks(
            narrative_structure.chunks,
            search_names
        )
        
        logger.info(f"  Found {len(relevant_chunks)} relevant chunks")
        
        # Extract states from each chunk
        states = []
        all_constraints = []
        irreversible_events = []
        
        for chunk in relevant_chunks:
            # Extract information from chunk
            extracted_info = self._extract_from_chunk(chunk, canonical_name)
            
            if extracted_info:
                # Build state
                state = self.state_builder.build_state(
                    chunk, canonical_name, extracted_info
                )
                states.append(state)
                
                # Collect constraints
                all_constraints.extend(state.constraints_established)
                
                # Check for irreversible events
                irreversible = self._detect_irreversible_events(state)
                irreversible_events.extend(irreversible)
        
        # Merge overlapping states
        merged_states = self._merge_states(states)
        
        # Build permanent constraints (appear in multiple states)
        permanent_constraints = self._identify_permanent_constraints(all_constraints)
        
        # Infer core traits
        core_traits = self._infer_core_traits(merged_states)
        
        # Build character arc summary
        character_arc = self._summarize_arc(merged_states)
        
        timeline = CharacterTimeline(
            character_name=canonical_name,
            book_name=narrative_structure.book_name,
            states=merged_states,
            permanent_constraints=permanent_constraints,
            irreversible_events=list(set(irreversible_events)),
            core_traits=core_traits,
            character_arc=character_arc
        )
        
        logger.info(f"  Built timeline with {len(merged_states)} states")
        return timeline
    
    def _find_relevant_chunks(
        self,
        chunks: List[NarrativeChunk],
        search_names: List[str]
    ) -> List[NarrativeChunk]:
        """Find chunks that mention the character."""
        relevant = []
        
        for chunk in chunks:
            text_lower = chunk.text.lower()
            for name in search_names:
                if name.lower() in text_lower:
                    relevant.append(chunk)
                    break
        
        return relevant
    
    def _extract_from_chunk(
        self,
        chunk: NarrativeChunk,
        character_name: str
    ) -> Optional[Dict]:
        """Extract character information from a chunk."""
        # Try LLM extraction first
        if self.use_llm and self.llm_extractor:
            try:
                result = self.llm_extractor.extract_character_info(
                    chunk.text,
                    character_name
                )
                if result:
                    return result
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}")
        
        # Fall back to rule-based extraction
        return self.fallback_extractor.extract_basic_info(
            chunk.text,
            character_name
        )
    
    def _detect_irreversible_events(self, state: CharacterState) -> List[str]:
        """Detect irreversible events (death, imprisonment, etc.)."""
        irreversible_keywords = [
            'died', 'killed', 'executed', 'murdered',
            'imprisoned', 'incarcerated', 'jailed',
            'married', 'divorced', 'widowed',
            'born', 'lost', 'destroyed'
        ]
        
        events = []
        for action in state.actions_taken:
            action_lower = action.lower()
            for keyword in irreversible_keywords:
                if keyword in action_lower:
                    events.append(action)
                    break
        
        return events
    
    def _merge_states(self, states: List[CharacterState]) -> List[CharacterState]:
        """Merge states that are from the same narrative segment."""
        if not states:
            return []
        
        # Sort by chapter range
        sorted_states = sorted(
            states,
            key=lambda s: s.chapter_range[0] if s.chapter_range else 0
        )
        
        # Simple merge: combine states with same time segment
        merged = {}
        for state in sorted_states:
            key = state.time_segment
            if key in merged:
                # Merge into existing state
                existing = merged[key]
                existing.beliefs.extend(state.beliefs)
                existing.motivations.extend(state.motivations)
                existing.fears.extend(state.fears)
                existing.goals.extend(state.goals)
                existing.actions_taken.extend(state.actions_taken)
                existing.constraints_established.extend(state.constraints_established)
                existing.relationships.extend(state.relationships)
                existing.supporting_evidence.extend(state.supporting_evidence)
            else:
                merged[key] = state
        
        # Deduplicate within each state
        for state in merged.values():
            state.beliefs = list(set(state.beliefs))
            state.motivations = list(set(state.motivations))
            state.fears = list(set(state.fears))
            state.goals = list(set(state.goals))
            state.actions_taken = list(set(state.actions_taken))
        
        return list(merged.values())
    
    def _identify_permanent_constraints(
        self,
        constraints: List[Constraint]
    ) -> List[Constraint]:
        """Identify constraints that are permanent throughout narrative."""
        # Count constraint descriptions
        constraint_counts = defaultdict(int)
        constraint_objects = {}
        
        for c in constraints:
            key = c.description.lower().strip()
            constraint_counts[key] += 1
            constraint_objects[key] = c
        
        # Constraints appearing multiple times are likely permanent
        permanent = []
        for key, count in constraint_counts.items():
            if count >= 2:
                permanent.append(constraint_objects[key])
        
        return permanent
    
    def _infer_core_traits(self, states: List[CharacterState]) -> List[str]:
        """Infer core character traits from states."""
        # Collect all motivations and beliefs
        all_motivations = []
        all_beliefs = []
        
        for state in states:
            all_motivations.extend(state.motivations)
            all_beliefs.extend(state.beliefs)
        
        # Count frequencies
        motivation_counts = defaultdict(int)
        belief_counts = defaultdict(int)
        
        for m in all_motivations:
            motivation_counts[m.lower()] += 1
        for b in all_beliefs:
            belief_counts[b.lower()] += 1
        
        # Top traits
        traits = []
        for m, count in sorted(motivation_counts.items(), key=lambda x: -x[1])[:3]:
            traits.append(f"Motivation: {m}")
        for b, count in sorted(belief_counts.items(), key=lambda x: -x[1])[:3]:
            traits.append(f"Belief: {b}")
        
        return traits
    
    def _summarize_arc(self, states: List[CharacterState]) -> str:
        """Summarize the character's arc."""
        if not states:
            return "No character arc data available."
        
        first_state = states[0]
        last_state = states[-1]
        
        arc_parts = []
        
        if first_state.beliefs:
            arc_parts.append(f"Initially believed: {first_state.beliefs[0]}")
        
        if last_state.beliefs and last_state.beliefs != first_state.beliefs:
            arc_parts.append(f"Later came to believe: {last_state.beliefs[0]}")
        
        if len(states) > 1:
            all_actions = []
            for s in states:
                all_actions.extend(s.actions_taken)
            if all_actions:
                arc_parts.append(f"Key actions: {', '.join(all_actions[:3])}")
        
        return " → ".join(arc_parts) if arc_parts else "Character arc not determined."


def extract_all_character_timelines(
    narrative_structure: BookNarrativeStructure,
    character_names: List[str],
    use_llm: bool = True
) -> Dict[str, CharacterTimeline]:
    """
    Extract timelines for multiple characters from a book.
    
    Args:
        narrative_structure: The ingested narrative structure
        character_names: List of character names to extract
        use_llm: Whether to use LLM for extraction
    
    Returns:
        Dict mapping character names to their timelines
    """
    extractor = CharacterTimelineExtractor(use_llm=use_llm)
    timelines = {}
    
    for name in character_names:
        try:
            timeline = extractor.extract_timeline(narrative_structure, name)
            timelines[timeline.character_name] = timeline
            narrative_structure.add_character_timeline(timeline)
        except Exception as e:
            logger.error(f"Failed to extract timeline for {name}: {e}")
    
    return timelines
