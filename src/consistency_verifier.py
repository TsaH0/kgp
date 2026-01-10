"""
Consistency Verification System.

This module implements the core consistency verification logic:
- Compares backstory claims against character timelines
- Uses rule-based + LLM hybrid approach
- Detects contradictions, temporal violations, and constraint violations
- Outputs binary classification (consistent/contradictory)

Decision Logic:
- Check compatibility with character evolution
- Verify established beliefs
- Confirm against irreversible events
- Validate long-term constraints
- Require causal and temporal consistency (surface plausibility is NOT enough)
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass

from .models import (
    CharacterTimeline, CharacterState, BookNarrativeStructure,
    BackstoryEntry, ConsistencyResult, ConsistencyVerdict,
    Constraint, ConstraintType, EvidenceSpan
)
from .llm_client import (
    ConsistencyChecker, ClaimNormalizer,
    get_consistency_checker, get_claim_normalizer
)
from .character_extraction import CharacterNameResolver, TemporalMarkerParser
from .config import ConsistencyConfig, config

logger = logging.getLogger(__name__)


@dataclass
class ClaimAnalysis:
    """Analysis of a backstory claim."""
    factual_claims: List[str]
    temporal_claims: List[Dict]
    relationship_claims: List[Dict]
    psychological_claims: List[str]
    constraints_implied: List[str]
    preconditions: List[str]


class RuleBasedVerifier:
    """Rule-based consistency verification."""
    
    # Contradiction patterns
    CONTRADICTION_PATTERNS = [
        # Death contradictions
        (r'died.*(\d{4})', r'alive.*after.*(\d{4})', 'death_timing'),
        (r'killed', r'survived', 'death_contradiction'),
        
        # Location contradictions
        (r'born in (\w+)', r'never .* (\w+)', 'birthplace'),
        
        # Relationship contradictions
        (r'father.*died.*before', r'father.*taught.*later', 'father_death_timing'),
        (r'no (?:siblings?|brothers?|sisters?)', r'(?:brother|sister)', 'sibling_contradiction'),
        
        # Temporal impossibilities
        (r'at age (\d+).*(\d{4})', None, 'age_year_check'),  # Needs custom logic
    ]
    
    # Keywords indicating irreversible events
    IRREVERSIBLE_EVENTS = {
        'died', 'killed', 'executed', 'murdered', 'death',
        'lost', 'destroyed', 'burned', 'drowned',
        'married', 'divorced', 'widowed',
        'born', 'imprisoned for life'
    }
    
    def __init__(self):
        self.name_resolver = CharacterNameResolver()
        self.temporal_parser = TemporalMarkerParser()
    
    def check_basic_consistency(
        self,
        backstory: str,
        timeline: CharacterTimeline
    ) -> Tuple[float, List[str], List[str]]:
        """
        Perform basic rule-based consistency checking.
        
        Returns:
            Tuple of (score, contradictions, supporting_matches)
        """
        contradictions = []
        supporting_matches = []
        
        backstory_lower = backstory.lower()
        
        # Check against irreversible events
        for event in timeline.irreversible_events:
            event_lower = event.lower()
            # Check if backstory contradicts the irreversible event
            if self._contradicts_event(backstory_lower, event_lower):
                contradictions.append(f"Contradicts irreversible event: {event}")
        
        # Check against permanent constraints
        for constraint in timeline.permanent_constraints:
            is_consistent, reason = self._check_constraint(
                backstory_lower, constraint
            )
            if not is_consistent:
                contradictions.append(f"Violates constraint: {reason}")
        
        # Check temporal consistency
        temporal_issues = self._check_temporal_consistency(
            backstory_lower, timeline
        )
        contradictions.extend(temporal_issues)
        
        # Check for supporting evidence
        for state in timeline.states:
            matches = self._find_supporting_matches(backstory_lower, state)
            supporting_matches.extend(matches)
        
        # Calculate score
        if contradictions:
            base_score = 0.3  # Start low if contradictions found
            penalty = len(contradictions) * 0.1
            score = max(0.0, base_score - penalty)
        else:
            base_score = 0.5  # Neutral if no contradictions
            bonus = min(0.4, len(supporting_matches) * 0.1)
            score = min(1.0, base_score + bonus)
        
        return score, contradictions, supporting_matches
    
    def _contradicts_event(self, backstory: str, event: str) -> bool:
        """Check if backstory contradicts an irreversible event."""
        # Extract key elements from event
        for irreversible in self.IRREVERSIBLE_EVENTS:
            if irreversible in event:
                # Check for contradicting claims in backstory
                if self._has_contradicting_claim(backstory, event, irreversible):
                    return True
        return False
    
    def _has_contradicting_claim(
        self, 
        backstory: str, 
        event: str, 
        event_type: str
    ) -> bool:
        """Check for specific contradictions based on event type."""
        if event_type in ['died', 'killed', 'executed', 'murdered']:
            # If event says someone died, check if backstory claims they did something after
            # Extract year from event if present
            year_match = re.search(r'\b(18\d{2}|19\d{2})\b', event)
            if year_match:
                event_year = int(year_match.group())
                # Check backstory for actions after this year
                backstory_years = re.findall(r'\b(18\d{2}|19\d{2})\b', backstory)
                for bs_year in backstory_years:
                    if int(bs_year) > event_year:
                        # Activity after death year
                        if 'was still' in backstory or 'continued' in backstory:
                            return True
        
        return False
    
    def _check_constraint(
        self,
        backstory: str,
        constraint: Constraint
    ) -> Tuple[bool, str]:
        """Check if backstory violates a constraint."""
        constraint_lower = constraint.description.lower()
        
        # Simple keyword-based checking
        # This is a simplified version - in production use semantic similarity
        
        if constraint.is_positive:
            # Constraint says something IS true
            # Check if backstory says it's NOT true
            negation_patterns = [
                f"never {constraint_lower}",
                f"not {constraint_lower}",
                f"didn't {constraint_lower}",
                f"no {constraint_lower}"
            ]
            for pattern in negation_patterns:
                if pattern[:20] in backstory:  # Check first 20 chars
                    return False, constraint.description
        else:
            # Constraint says something is NOT true
            # Check if backstory claims it IS true
            if constraint_lower[:30] in backstory:
                return False, constraint.description
        
        return True, ""
    
    def _check_temporal_consistency(
        self,
        backstory: str,
        timeline: CharacterTimeline
    ) -> List[str]:
        """Check for temporal consistency issues."""
        issues = []
        
        # Extract years and ages from backstory
        backstory_years = [int(y) for y in re.findall(r'\b(18\d{2}|19\d{2})\b', backstory)]
        backstory_ages = [int(a) for a in re.findall(r'\bat age (\d+)\b', backstory)]
        
        # Check for impossible age-year combinations
        for state in timeline.states:
            for action in state.actions_taken:
                action_years = [int(y) for y in re.findall(r'\b(18\d{2}|19\d{2})\b', action)]
                action_ages = [int(a) for a in re.findall(r'\bat age (\d+)\b', action)]
                
                # Cross-check
                for bs_year in backstory_years:
                    for action_year in action_years:
                        # Events in timeline should be temporally ordered
                        if abs(bs_year - action_year) > 100:
                            issues.append(f"Temporal span issue: {bs_year} vs {action_year}")
        
        return issues
    
    def _find_supporting_matches(
        self,
        backstory: str,
        state: CharacterState
    ) -> List[str]:
        """Find elements in backstory that match the character state."""
        matches = []
        
        # Check beliefs
        for belief in state.beliefs:
            if self._semantic_overlap(backstory, belief):
                matches.append(f"Matches belief: {belief[:50]}")
        
        # Check actions
        for action in state.actions_taken:
            if self._semantic_overlap(backstory, action):
                matches.append(f"Matches action: {action[:50]}")
        
        # Check relationships
        for rel in state.relationships:
            if rel.target_character.lower() in backstory:
                matches.append(f"References relationship with: {rel.target_character}")
        
        return matches
    
    def _semantic_overlap(self, text1: str, text2: str, threshold: float = 0.3) -> bool:
        """Check for semantic overlap between two texts (simplified)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'is', 'was', 'were', 'are', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'could', 'should', 'may', 'might', 'must', 'shall', 'can',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'into', 'through', 'during', 'before', 'after',
                     'he', 'she', 'it', 'they', 'his', 'her', 'its', 'their',
                     'and', 'but', 'or', 'nor', 'so', 'yet'}
        
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2)
        min_len = min(len(words1), len(words2))
        
        return overlap / min_len > threshold if min_len > 0 else False


class HybridConsistencyVerifier:
    """
    Hybrid consistency verifier combining rule-based and LLM approaches.
    
    This is the main entry point for consistency verification.
    """
    
    def __init__(
        self,
        consistency_config: Optional[ConsistencyConfig] = None,
        use_llm: bool = True
    ):
        self.config = consistency_config or config.consistency
        self.use_llm = use_llm
        
        self.rule_verifier = RuleBasedVerifier()
        self.name_resolver = CharacterNameResolver()
        
        if use_llm:
            self.llm_checker = get_consistency_checker()
            self.claim_normalizer = get_claim_normalizer()
        else:
            self.llm_checker = None
            self.claim_normalizer = None
        
        # Trained thresholds (will be updated by training)
        self.consistency_threshold = self.config.consistency_threshold
        self.weights = {
            'belief': self.config.belief_weight,
            'motivation': self.config.motivation_weight,
            'action': self.config.action_weight,
            'relationship': self.config.relationship_weight,
            'constraint': self.config.constraint_weight,
            'temporal': self.config.temporal_weight
        }
    
    def verify_consistency(
        self,
        backstory: BackstoryEntry,
        narrative_structure: BookNarrativeStructure
    ) -> ConsistencyResult:
        """
        Verify if a backstory is consistent with the narrative.
        
        This implements the full verification pipeline:
        1. Normalize the backstory claim
        2. Find the relevant character timeline
        3. Apply rule-based checks
        4. Apply LLM-based checks (if available)
        5. Combine scores and make verdict
        """
        # Get character timeline
        timeline = narrative_structure.get_character_timeline(backstory.character)
        
        if timeline is None:
            # No timeline available - cannot verify
            logger.warning(f"No timeline for character: {backstory.character}")
            return self._create_uncertain_result(backstory.id)
        
        # Step 1: Normalize the claim
        normalized_claim = self._normalize_claim(backstory)
        
        # Step 2: Rule-based verification
        rule_score, contradictions, supports = self.rule_verifier.check_basic_consistency(
            backstory.content,
            timeline
        )
        
        # Step 3: LLM verification (if available)
        llm_result = None
        if self.use_llm and self.llm_checker:
            llm_result = self._llm_verify(backstory, timeline)
        
        # Step 4: Calculate component scores
        scores = self._calculate_component_scores(
            backstory, timeline, normalized_claim, llm_result
        )
        
        # Step 5: Combine and make verdict
        final_score = self._combine_scores(scores, rule_score, llm_result)
        
        # Collect all contradictions
        all_contradictions = contradictions.copy()
        if llm_result and 'contradictions' in llm_result:
            all_contradictions.extend(llm_result.get('contradictions', []))
        
        # Determine verdict
        if all_contradictions:
            # If any contradictions found, lean toward contradictory
            if len(all_contradictions) >= 2 or final_score < 0.4:
                verdict = ConsistencyVerdict.CONTRADICTORY
            elif final_score > 0.6:
                verdict = ConsistencyVerdict.CONSISTENT
            else:
                verdict = ConsistencyVerdict.CONTRADICTORY
        else:
            if final_score >= self.consistency_threshold:
                verdict = ConsistencyVerdict.CONSISTENT
            else:
                verdict = ConsistencyVerdict.CONTRADICTORY
        
        # Build result
        result = ConsistencyResult(
            backstory_id=backstory.id,
            verdict=verdict,
            confidence=abs(final_score - 0.5) * 2,  # Distance from uncertainty
            belief_score=scores.get('belief', 0.5),
            motivation_score=scores.get('motivation', 0.5),
            action_score=scores.get('action', 0.5),
            relationship_score=scores.get('relationship', 0.5),
            constraint_score=scores.get('constraint', 0.5),
            temporal_score=scores.get('temporal', 0.5),
            contradictions=all_contradictions,
            reasoning=self._generate_reasoning(
                verdict, final_score, contradictions, supports, llm_result
            )
        )
        
        return result
    
    def _normalize_claim(self, backstory: BackstoryEntry) -> Optional[ClaimAnalysis]:
        """Normalize the backstory claim into structured format."""
        if not self.use_llm or not self.claim_normalizer:
            return None
        
        try:
            result = self.claim_normalizer.normalize_claim(
                backstory.content,
                backstory.character
            )
            
            if result:
                return ClaimAnalysis(
                    factual_claims=result.get('factual_claims', []),
                    temporal_claims=result.get('temporal_claims', []),
                    relationship_claims=result.get('relationship_claims', []),
                    psychological_claims=result.get('psychological_claims', []),
                    constraints_implied=result.get('constraints', []),
                    preconditions=result.get('required_preconditions', [])
                )
        except Exception as e:
            logger.warning(f"Claim normalization failed: {e}")
        
        return None
    
    def _llm_verify(
        self,
        backstory: BackstoryEntry,
        timeline: CharacterTimeline
    ) -> Optional[Dict]:
        """Use LLM to verify consistency."""
        # Prepare evidence from timeline
        evidence = []
        
        # Add beliefs
        evidence.extend([f"Belief: {b}" for b in timeline.get_all_beliefs()[:10]])
        
        # Add actions
        evidence.extend([f"Action: {a}" for a in timeline.get_all_actions()[:10]])
        
        # Add relationships
        for rel in timeline.get_all_relationships()[:5]:
            evidence.append(f"Relationship: {rel.description}")
        
        # Add irreversible events
        evidence.extend([f"Irreversible: {e}" for e in timeline.irreversible_events[:5]])
        
        # Add permanent constraints
        evidence.extend([f"Constraint: {c.description}" 
                        for c in timeline.permanent_constraints[:5]])
        
        try:
            return self.llm_checker.check_consistency(
                backstory.content,
                backstory.character,
                evidence
            )
        except Exception as e:
            logger.warning(f"LLM verification failed: {e}")
            return None
    
    def _calculate_component_scores(
        self,
        backstory: BackstoryEntry,
        timeline: CharacterTimeline,
        normalized_claim: Optional[ClaimAnalysis],
        llm_result: Optional[Dict]
    ) -> Dict[str, float]:
        """Calculate individual component scores."""
        scores = {
            'belief': 0.5,
            'motivation': 0.5,
            'action': 0.5,
            'relationship': 0.5,
            'constraint': 0.5,
            'temporal': 0.5
        }
        
        backstory_lower = backstory.content.lower()
        
        # Belief score
        all_beliefs = timeline.get_all_beliefs()
        if all_beliefs:
            matches = sum(1 for b in all_beliefs if self._text_overlap(backstory_lower, b.lower()))
            scores['belief'] = min(1.0, 0.3 + matches * 0.15)
        
        # Action score  
        all_actions = timeline.get_all_actions()
        if all_actions:
            matches = sum(1 for a in all_actions if self._text_overlap(backstory_lower, a.lower()))
            scores['action'] = min(1.0, 0.3 + matches * 0.15)
        
        # Relationship score
        all_rels = timeline.get_all_relationships()
        if all_rels:
            matches = sum(1 for r in all_rels if r.target_character.lower() in backstory_lower)
            scores['relationship'] = min(1.0, 0.3 + matches * 0.2)
        
        # Constraint score
        constraint_violations = 0
        for c in timeline.permanent_constraints:
            if self._violates_constraint(backstory_lower, c):
                constraint_violations += 1
        scores['constraint'] = max(0.0, 1.0 - constraint_violations * 0.3)
        
        # Temporal score
        temporal_issues = self._count_temporal_issues(backstory_lower, timeline)
        scores['temporal'] = max(0.0, 1.0 - temporal_issues * 0.25)
        
        return scores
    
    def _text_overlap(self, text1: str, text2: str) -> bool:
        """Check for significant text overlap."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        overlap = len(words1 & words2)
        return overlap >= 3
    
    def _violates_constraint(self, backstory: str, constraint: Constraint) -> bool:
        """Check if backstory violates a constraint."""
        constraint_lower = constraint.description.lower()
        
        # Extract key terms from constraint
        key_terms = [w for w in constraint_lower.split() 
                    if len(w) > 4 and w not in ['about', 'their', 'would', 'could', 'should']]
        
        # Check for direct contradiction patterns
        for term in key_terms[:3]:
            if f"never {term}" in backstory or f"not {term}" in backstory:
                if term in constraint_lower:
                    return True
        
        return False
    
    def _count_temporal_issues(self, backstory: str, timeline: CharacterTimeline) -> int:
        """Count temporal consistency issues."""
        issues = 0
        
        # Extract years from backstory
        backstory_years = set(int(y) for y in re.findall(r'\b(1[78]\d{2})\b', backstory))
        
        # Check against timeline events
        for event in timeline.irreversible_events:
            event_years = set(int(y) for y in re.findall(r'\b(1[78]\d{2})\b', event.lower()))
            
            # If backstory mentions activity after a death event
            if 'died' in event.lower() or 'killed' in event.lower():
                for bs_year in backstory_years:
                    for ev_year in event_years:
                        if bs_year > ev_year:
                            # Check if backstory implies activity after death
                            if 'continued' in backstory or 'later' in backstory:
                                issues += 1
        
        return issues
    
    def _combine_scores(
        self,
        component_scores: Dict[str, float],
        rule_score: float,
        llm_result: Optional[Dict]
    ) -> float:
        """Combine all scores into final score."""
        # Weighted component score
        weighted_sum = sum(
            component_scores.get(key, 0.5) * weight
            for key, weight in self.weights.items()
        )
        
        # Combine with rule score
        combined = 0.4 * rule_score + 0.4 * weighted_sum
        
        # Add LLM score if available
        if llm_result:
            llm_confidence = llm_result.get('confidence', 0.5)
            llm_verdict = llm_result.get('verdict', 'UNCERTAIN')
            
            if llm_verdict == 'CONSISTENT':
                llm_score = 0.5 + llm_confidence * 0.5
            elif llm_verdict == 'CONTRADICTORY':
                llm_score = 0.5 - llm_confidence * 0.5
            else:
                llm_score = 0.5
            
            combined = combined * 0.7 + llm_score * 0.3
        else:
            combined = combined / 0.8  # Normalize if no LLM
        
        return max(0.0, min(1.0, combined))
    
    def _generate_reasoning(
        self,
        verdict: ConsistencyVerdict,
        score: float,
        contradictions: List[str],
        supports: List[str],
        llm_result: Optional[Dict]
    ) -> str:
        """Generate human-readable reasoning for the verdict."""
        parts = []
        
        parts.append(f"Verdict: {verdict.value} (score: {score:.2f})")
        
        if contradictions:
            parts.append(f"Contradictions found ({len(contradictions)}):")
            for c in contradictions[:3]:
                parts.append(f"  - {c}")
        
        if supports:
            parts.append(f"Supporting evidence ({len(supports)}):")
            for s in supports[:3]:
                parts.append(f"  - {s}")
        
        if llm_result and 'reasoning' in llm_result:
            parts.append(f"LLM analysis: {llm_result['reasoning']}")
        
        return "\n".join(parts)
    
    def _create_uncertain_result(self, backstory_id: int) -> ConsistencyResult:
        """Create an uncertain result when verification cannot be performed."""
        return ConsistencyResult(
            backstory_id=backstory_id,
            verdict=ConsistencyVerdict.UNCERTAIN,
            confidence=0.0,
            reasoning="Could not verify: character timeline not available"
        )
    
    def update_thresholds(
        self,
        new_threshold: float,
        new_weights: Optional[Dict[str, float]] = None
    ):
        """Update thresholds based on training."""
        self.consistency_threshold = new_threshold
        if new_weights:
            self.weights.update(new_weights)


class ConsistencyTrainer:
    """
    Trains the consistency verifier on labeled data.
    
    This implements Phase 3: Training Phase
    - Uses train.csv to tune thresholds and weights
    - Does NOT modify the narrative structure
    - Only adjusts how consistency is judged
    """
    
    def __init__(self, verifier: HybridConsistencyVerifier):
        self.verifier = verifier
        self.training_results: List[Tuple[ConsistencyResult, bool]] = []
    
    def train(
        self,
        training_entries: List[BackstoryEntry],
        narrative_structures: Dict[str, BookNarrativeStructure]
    ) -> Dict[str, float]:
        """
        Train the verifier on labeled data.
        
        Returns optimized parameters.
        """
        logger.info(f"Training on {len(training_entries)} examples")
        
        self.training_results = []
        
        # Run verification on all training examples
        for entry in training_entries:
            structure = self._get_structure_for_entry(entry, narrative_structures)
            if structure:
                result = self.verifier.verify_consistency(entry, structure)
                ground_truth = entry.is_consistent
                self.training_results.append((result, ground_truth))
        
        # Analyze results and optimize thresholds
        optimal_threshold = self._optimize_threshold()
        optimal_weights = self._optimize_weights()
        
        # Update verifier
        self.verifier.update_thresholds(optimal_threshold, optimal_weights)
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        logger.info(f"Training complete. Accuracy: {metrics['accuracy']:.2%}")
        logger.info(f"Optimal threshold: {optimal_threshold:.3f}")
        
        return {
            'threshold': optimal_threshold,
            **metrics
        }
    
    def _get_structure_for_entry(
        self,
        entry: BackstoryEntry,
        structures: Dict[str, BookNarrativeStructure]
    ) -> Optional[BookNarrativeStructure]:
        """Get the narrative structure for a backstory entry."""
        book_name = entry.book_name
        
        # Try exact match
        if book_name in structures:
            return structures[book_name]
        
        # Try normalized match
        for name, structure in structures.items():
            if self._books_match(book_name, name):
                return structure
        
        return None
    
    def _books_match(self, name1: str, name2: str) -> bool:
        """Check if two book names refer to the same book."""
        # Normalize names
        n1 = name1.lower().replace('_', ' ').replace('-', ' ')
        n2 = name2.lower().replace('_', ' ').replace('-', ' ')
        
        # Check for containment
        return n1 in n2 or n2 in n1 or n1 == n2
    
    def _optimize_threshold(self) -> float:
        """Find optimal classification threshold."""
        best_threshold = 0.5
        best_accuracy = 0.0
        
        for threshold in [i/20 for i in range(1, 20)]:
            correct = 0
            for result, ground_truth in self.training_results:
                if ground_truth is not None:
                    predicted = result.final_score >= threshold
                    if predicted == ground_truth:
                        correct += 1
            
            accuracy = correct / len(self.training_results) if self.training_results else 0
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        return best_threshold
    
    def _optimize_weights(self) -> Dict[str, float]:
        """Optimize component weights (simplified grid search)."""
        # For now, return default weights
        # In production, use proper optimization
        return {
            'belief': 0.20,
            'motivation': 0.15,
            'action': 0.20,
            'relationship': 0.15,
            'constraint': 0.20,
            'temporal': 0.10
        }
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate training metrics."""
        if not self.training_results:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        tp = fp = tn = fn = 0
        
        for result, ground_truth in self.training_results:
            if ground_truth is None:
                continue
            
            predicted = result.verdict == ConsistencyVerdict.CONSISTENT
            
            if predicted and ground_truth:
                tp += 1
            elif predicted and not ground_truth:
                fp += 1
            elif not predicted and ground_truth:
                fn += 1
            else:
                tn += 1
        
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }
