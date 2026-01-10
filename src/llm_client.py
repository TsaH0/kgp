"""
Ollama LLM Client for local inference.

This module provides a wrapper around the Ollama API for:
- Extracting character information from text chunks
- Normalizing natural-language claims
- Comparing backstory claims to stored constraints

LLM usage is STRICTLY LIMITED to these tasks - no end-to-end reasoning.
"""

import json
import requests
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from .config import OllamaConfig, config

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with local Ollama LLM endpoint."""
    
    def __init__(self, ollama_config: Optional[OllamaConfig] = None):
        self.config = ollama_config or config.ollama
        self.base_url = self.config.base_url
        self.model = self.config.model
        self.timeout = self.config.timeout
        self._is_available = None
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        if self._is_available is not None:
            return self._is_available
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            self._is_available = response.status_code == 200
            return self._is_available
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            self._is_available = False
            return False
    
    def list_models(self) -> List[str]:
        """List available models."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                return [m['name'] for m in data.get('models', [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        return []
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 2048,
        json_mode: bool = False
    ) -> Optional[str]:
        """Generate text using the LLM."""
        if not self.is_available():
            logger.warning("Ollama not available, using fallback")
            return None
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature or self.config.temperature,
                    "num_predict": max_tokens
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            if json_mode:
                payload["format"] = "json"
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.error(f"Ollama error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return None
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            return None
    
    def extract_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        retries: int = 3
    ) -> Optional[Dict]:
        """Generate and parse JSON response."""
        for attempt in range(retries):
            response = self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                json_mode=True
            )
            
            if response:
                try:
                    # Clean up response
                    response = response.strip()
                    # Handle markdown code blocks
                    if response.startswith("```"):
                        lines = response.split("\n")
                        response = "\n".join(lines[1:-1])
                    
                    return json.loads(response)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse error (attempt {attempt+1}): {e}")
                    # Try to extract JSON from response
                    try:
                        start = response.find("{")
                        end = response.rfind("}") + 1
                        if start >= 0 and end > start:
                            return json.loads(response[start:end])
                    except:
                        pass
            
            if attempt < retries - 1:
                time.sleep(1)
        
        return None


class CharacterExtractor:
    """Extract character information from text using LLM."""
    
    EXTRACTION_SYSTEM_PROMPT = """You are a precise literary analysis assistant. Your task is to extract structured character information from narrative text.

RULES:
1. Only extract information EXPLICITLY stated or DIRECTLY implied in the text
2. Do NOT invent or hallucinate information
3. Be specific and cite evidence from the text
4. If information is not present, leave the field empty
5. Distinguish between beliefs, motivations, fears, and actions

Output must be valid JSON."""

    EXTRACTION_PROMPT_TEMPLATE = """Analyze the following text passage and extract information about the character "{character_name}".

TEXT:
{text}

Extract the following information about {character_name} (if present in the text):
- beliefs: What {character_name} believes to be true
- motivations: What drives {character_name}'s actions
- fears: What {character_name} is afraid of
- goals: What {character_name} is trying to achieve
- actions: Specific actions {character_name} takes in this passage
- relationships: Other characters mentioned and their relationship to {character_name}
- temporal_markers: Any time references (dates, ages, "before X", "after Y")
- constraints: Facts that constrain what could be true about {character_name}

Return a JSON object with these fields. Use empty arrays [] for missing information.

OUTPUT FORMAT:
{{
    "beliefs": ["belief1", "belief2"],
    "motivations": ["motivation1"],
    "fears": ["fear1"],
    "goals": ["goal1"],
    "actions": ["action1", "action2"],
    "relationships": [{{"character": "name", "relationship": "type", "description": "details"}}],
    "temporal_markers": ["marker1"],
    "constraints": ["constraint1"]
}}"""

    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or OllamaClient()
    
    def extract_character_info(
        self,
        text: str,
        character_name: str
    ) -> Optional[Dict]:
        """Extract character information from a text chunk."""
        prompt = self.EXTRACTION_PROMPT_TEMPLATE.format(
            character_name=character_name,
            text=text[:4000]  # Limit text length
        )
        
        result = self.client.extract_json(
            prompt=prompt,
            system_prompt=self.EXTRACTION_SYSTEM_PROMPT
        )
        
        if result:
            # Validate and normalize the result
            return self._normalize_extraction(result)
        
        return None
    
    def _normalize_extraction(self, data: Dict) -> Dict:
        """Normalize extracted data to expected format."""
        normalized = {
            "beliefs": [],
            "motivations": [],
            "fears": [],
            "goals": [],
            "actions": [],
            "relationships": [],
            "temporal_markers": [],
            "constraints": []
        }
        
        for key in normalized:
            if key in data:
                value = data[key]
                if isinstance(value, list):
                    normalized[key] = [str(v) if not isinstance(v, dict) else v for v in value]
                elif isinstance(value, str):
                    normalized[key] = [value] if value else []
        
        return normalized


class ClaimNormalizer:
    """Normalize natural language claims into structured format."""
    
    NORMALIZATION_PROMPT = """Analyze the following backstory claim and break it down into structured components.

BACKSTORY CLAIM:
{claim}

CHARACTER: {character_name}

Extract and categorize the key assertions in this backstory:
1. Factual claims (events, states, actions)
2. Temporal claims (when things happened)
3. Relationship claims (connections to other characters)
4. Psychological claims (beliefs, motivations, fears)
5. Constraint implications (what must be true/false for this to hold)

Return as JSON:
{{
    "factual_claims": ["claim1", "claim2"],
    "temporal_claims": [{{"event": "description", "time_reference": "when"}}],
    "relationship_claims": [{{"character": "name", "relationship": "type"}}],
    "psychological_claims": ["claim1"],
    "constraints": ["constraint1"],
    "required_preconditions": ["what must be true for this backstory to be possible"]
}}"""

    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or OllamaClient()
    
    def normalize_claim(
        self,
        claim: str,
        character_name: str
    ) -> Optional[Dict]:
        """Normalize a backstory claim into structured format."""
        prompt = self.NORMALIZATION_PROMPT.format(
            claim=claim,
            character_name=character_name
        )
        
        return self.client.extract_json(
            prompt=prompt,
            system_prompt="You are a precise claim analysis assistant. Extract structured information from text."
        )


class ConsistencyChecker:
    """Use LLM to check consistency between claims and evidence."""
    
    CONSISTENCY_CHECK_PROMPT = """You are a logical consistency checker. Determine if a backstory claim is CONSISTENT or CONTRADICTORY with the established narrative evidence.

BACKSTORY CLAIM about {character_name}:
{claim}

ESTABLISHED NARRATIVE EVIDENCE:
{evidence}

INSTRUCTIONS:
1. Check each assertion in the claim against the evidence
2. Look for direct contradictions (X happened vs X did not happen)
3. Check temporal consistency (events must happen in possible order)
4. Check logical consistency (no impossible situations)
5. Surface plausibility is NOT enough - require causal/temporal consistency

Return JSON:
{{
    "verdict": "CONSISTENT" or "CONTRADICTORY" or "UNCERTAIN",
    "confidence": 0.0 to 1.0,
    "supporting_matches": ["evidence that supports the claim"],
    "contradictions": ["specific contradictions found"],
    "temporal_issues": ["timeline problems"],
    "reasoning": "brief explanation of your verdict"
}}"""

    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or OllamaClient()
    
    def check_consistency(
        self,
        claim: str,
        character_name: str,
        evidence: List[str]
    ) -> Optional[Dict]:
        """Check if a claim is consistent with evidence."""
        evidence_text = "\n".join(f"- {e}" for e in evidence[:20])  # Limit evidence
        
        prompt = self.CONSISTENCY_CHECK_PROMPT.format(
            character_name=character_name,
            claim=claim,
            evidence=evidence_text
        )
        
        return self.client.extract_json(
            prompt=prompt,
            system_prompt="You are a precise logical consistency checker. Focus on factual and temporal contradictions."
        )


class FallbackExtractor:
    """Rule-based fallback when LLM is not available."""
    
    # Common relationship indicators
    RELATIONSHIP_WORDS = {
        'father': 'family',
        'mother': 'family', 
        'son': 'family',
        'daughter': 'family',
        'brother': 'family',
        'sister': 'family',
        'wife': 'family',
        'husband': 'family',
        'friend': 'friend',
        'enemy': 'enemy',
        'mentor': 'mentor',
        'teacher': 'mentor',
        'colleague': 'colleague',
        'companion': 'friend',
        'love': 'romantic',
        'beloved': 'romantic'
    }
    
    # Temporal markers
    TEMPORAL_PATTERNS = [
        r'\b(\d{4})\b',  # Years
        r'\b(at age \d+)\b',
        r'\b(when .+ was \d+)\b',
        r'\b(before|after|during)\b',
        r'\b(childhood|youth|adulthood)\b',
        r'\bchapter\s+(\d+)\b'
    ]
    
    # Action verbs
    ACTION_VERBS = [
        'killed', 'saved', 'escaped', 'discovered', 
        'married', 'betrayed', 'imprisoned', 'freed',
        'traveled', 'fought', 'died', 'survived'
    ]
    
    def extract_basic_info(self, text: str, character_name: str) -> Dict:
        """Extract basic information using rules."""
        import re
        
        result = {
            "beliefs": [],
            "motivations": [],
            "fears": [],
            "goals": [],
            "actions": [],
            "relationships": [],
            "temporal_markers": [],
            "constraints": []
        }
        
        text_lower = text.lower()
        char_lower = character_name.lower()
        
        # Only process if character is mentioned
        if char_lower not in text_lower:
            return result
        
        # Extract temporal markers
        for pattern in self.TEMPORAL_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            result["temporal_markers"].extend(matches)
        
        # Extract actions
        for verb in self.ACTION_VERBS:
            if verb in text_lower:
                # Find sentences containing both character and action
                sentences = text.split('.')
                for sent in sentences:
                    if char_lower in sent.lower() and verb in sent.lower():
                        result["actions"].append(sent.strip())
        
        # Extract relationships
        for rel_word, rel_type in self.RELATIONSHIP_WORDS.items():
            if rel_word in text_lower:
                result["relationships"].append({
                    "character": "unknown",
                    "relationship": rel_type,
                    "description": f"Contains '{rel_word}'"
                })
        
        return result


# Singleton instances
_ollama_client = None
_character_extractor = None
_claim_normalizer = None
_consistency_checker = None


def get_ollama_client() -> OllamaClient:
    """Get or create Ollama client singleton."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client


def get_character_extractor() -> CharacterExtractor:
    """Get or create character extractor singleton."""
    global _character_extractor
    if _character_extractor is None:
        _character_extractor = CharacterExtractor(get_ollama_client())
    return _character_extractor


def get_claim_normalizer() -> ClaimNormalizer:
    """Get or create claim normalizer singleton."""
    global _claim_normalizer
    if _claim_normalizer is None:
        _claim_normalizer = ClaimNormalizer(get_ollama_client())
    return _claim_normalizer


def get_consistency_checker() -> ConsistencyChecker:
    """Get or create consistency checker singleton."""
    global _consistency_checker
    if _consistency_checker is None:
        _consistency_checker = ConsistencyChecker(get_ollama_client())
    return _consistency_checker
