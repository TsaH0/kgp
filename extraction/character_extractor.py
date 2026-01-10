import json
import re
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ExtractionResult:
    meta: dict
    character: dict
    extracted_sentences: list
    claims: list
    contradictions: list
    debug: dict


def create_extraction_prompt(
    chunk_id: str, 
    character_names: list[str], 
    chunk_text: str,
    backstory: Optional[dict] = None
) -> str:
    """Generate the extraction prompt with populated variables."""
    character_list = "\n".join(f"- {name}" for name in character_names)
    backstory_section = json.dumps(backstory, indent=2) if backstory else "null"
    
    return f"""You are a structured knowledge extraction and contradiction detection engine.

You will be given:
1. A text chunk extracted from a book (.txt files ingested from a data folder).
2. A list of known character names from train.csv.
3. An optional structured backstory for a character (may be null).

Your tasks:
1. Identify whether any character from the provided character list appears in the text chunk.
2. If a character appears, extract ONLY sentences relevant to that character.
3. Convert the extracted information into atomic, contradiction-ready factual claims.
4. If a backstory is provided, compare the new claims against it and identify direct contradictions.
5. Produce detailed debug information explaining each decision.

Rules:
- Output MUST be valid JSON only.
- Do NOT include markdown.
- Do NOT include explanations outside JSON.
- Do NOT invent characters or facts.
- If no known character appears, set character fields to null and claims to empty.
- Claims must be atomic, minimal, and directly grounded in the text.
- A contradiction exists ONLY if the same subject + predicate + scope has incompatible objects.
- Temporal differences are NOT contradictions unless explicitly conflicting.
- Prefer fewer, high-confidence claims over speculative ones.

STRICT OUTPUT SCHEMA:

{{
  "meta": {{
    "chunk_id": string,
    "matched_character": string | null,
    "match_confidence": number
  }},
  "character": {{
    "name": string | null,
    "role": string | null,
    "core_traits": [string],
    "domain": string | null
  }},
  "extracted_sentences": [string],
  "claims": [
    {{
      "subject": string,
      "predicate": string,
      "object": string,
      "polarity": "affirm" | "deny",
      "scope": "permanent" | "temporal",
      "time_ref": string | null
    }}
  ],
  "contradictions": [
    {{
      "subject": string,
      "predicate": string,
      "backstory_value": string,
      "new_value": string,
      "scope": string,
      "severity": "low" | "medium" | "high"
    }}
  ],
  "debug": {{
    "character_matching_reasoning": string,
    "sentence_selection_reasoning": string,
    "claim_construction_reasoning": string,
    "contradiction_reasoning": string,
    "ignored_information": string,
    "overall_confidence": number
  }}
}}

CHUNK_ID: {chunk_id}

KNOWN CHARACTERS (from train.csv):
{character_list}

CHARACTER BACKSTORY:
{backstory_section}

BOOK TEXT CHUNK:
{chunk_text}

Identify the character, extract structured facts, and detect contradictions strictly following the schema."""


def validate_extraction_output(output: str) -> tuple[bool, Optional[dict], Optional[str]]:
    """Validate and parse the JSON output from extraction."""
    try:
        cleaned = re.sub(r'^```json?\s*', '', output.strip())
        cleaned = re.sub(r'\s*```$', '', cleaned)
        
        data = json.loads(cleaned)
        
        required_keys = ["meta", "character", "extracted_sentences", "claims", "contradictions", "debug"]
        for key in required_keys:
            if key not in data:
                return False, None, f"Missing required key: {key}"
        
        meta_keys = ["chunk_id", "matched_character", "match_confidence"]
        for key in meta_keys:
            if key not in data["meta"]:
                return False, None, f"Missing meta key: {key}"
        
        # Validate claims structure
        for i, claim in enumerate(data.get("claims", [])):
            if claim.get("polarity") not in ["affirm", "deny"]:
                return False, None, f"Invalid polarity in claim {i}"
            if claim.get("scope") not in ["permanent", "temporal"]:
                return False, None, f"Invalid scope in claim {i}"
        
        # Validate contradictions structure
        for i, contradiction in enumerate(data.get("contradictions", [])):
            if contradiction.get("severity") not in ["low", "medium", "high"]:
                return False, None, f"Invalid severity in contradiction {i}"
        
        return True, data, None
        
    except json.JSONDecodeError as e:
        return False, None, f"JSON parse error: {str(e)}"


def create_empty_result(chunk_id: str) -> dict:
    """Create an empty result when no character is found."""
    return {
        "meta": {
            "chunk_id": chunk_id,
            "matched_character": None,
            "match_confidence": 0.0
        },
        "character": {
            "name": None,
            "role": None,
            "core_traits": [],
            "domain": None
        },
        "extracted_sentences": [],
        "claims": [],
        "contradictions": [],
        "debug": {
            "character_matching_reasoning": "No known characters found in text",
            "sentence_selection_reasoning": "N/A",
            "claim_construction_reasoning": "N/A",
            "contradiction_reasoning": "N/A",
            "ignored_information": "Entire chunk",
            "overall_confidence": 0.0
        }
    }
