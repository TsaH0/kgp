#!/usr/bin/env python3
"""
Character Knowledge Extraction with Pathway Ingestion + Background Verification:
1. Use Pathway to ingest book files from data folder
2. Load characters from train.csv
3. BUILD KNOWLEDGE DATA STRUCTURE per character from books
4. VERIFY each backstory statement against knowledge structure
5. If ANY statement contradicts -> mark as "contradict"
6. Groq provides detailed reasoning for the final verdict
"""

import os
import sys
import json
import re
import csv
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import pathway as pw

# Load environment variables from .env
load_dotenv()

# Also load from .env.example for any additional keys not in .env
env_example = Path(__file__).parent / ".env.example"
if env_example.exists():
    load_dotenv(env_example, override=False)  # Don't override existing values


###############################################################################
# KNOWLEDGE DATA STRUCTURE
###############################################################################

class CharacterKnowledge:
    """
    Data structure to store extracted knowledge about a character.
    This accumulates facts, relationships, timeline events, and traits
    from all book sources.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.facts = []           # List of atomic facts {"subject", "predicate", "object", "source", "certainty"}
        self.relationships = []    # List of {"target", "type", "description", "source"}
        self.timeline = []        # List of {"event", "time_reference", "source", "certainty"}
        self.traits = []          # List of personality traits
        self.locations = []       # Places associated with character
        self.quotes = []          # Direct quotes mentioning character
        self.sources = set()      # Book sources
        self.role = ""            # Character's role/occupation
        self.summary = ""         # LLM-generated summary
    
    def add_from_extraction(self, extraction: dict, source: str):
        """Add extracted data from LLM response to the knowledge structure."""
        self.sources.add(source)
        
        if isinstance(extraction, dict):
            # Add facts
            for fact in extraction.get("facts", []):
                self.facts.append({
                    **fact,
                    "source": source
                })
            
            # Add timeline events
            for event in extraction.get("timeline", []):
                self.timeline.append({
                    **event,
                    "source": source
                })
            
            # Add character info
            char_info = extraction.get("character", {})
            if isinstance(char_info, dict):
                if char_info.get("role"):
                    self.role = char_info.get("role", self.role)
                
                for trait in char_info.get("traits", []):
                    if trait and trait not in self.traits:
                        self.traits.append(trait)
                
                for rel in char_info.get("relationships", []):
                    if rel:
                        self.relationships.append({
                            "description": rel,
                            "source": source
                        })
            
            # Add summary
            if extraction.get("summary"):
                self.summary = extraction.get("summary", self.summary)
    
    def add_sentences(self, sentences: list[str]):
        """Add relevant quotes from the book."""
        for s in sentences[:10]:  # Keep top 10 most relevant
            if s not in self.quotes:
                self.quotes.append(s)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization and LLM prompts."""
        return {
            "name": self.name,
            "role": self.role,
            "facts": self.facts,
            "relationships": self.relationships,
            "timeline": self.timeline,
            "traits": self.traits,
            "locations": self.locations,
            "summary": self.summary,
            "sources": list(self.sources),
            "sample_quotes": self.quotes[:5]
        }
    
    def get_knowledge_summary(self) -> str:
        """Get a text summary of stored knowledge for verification."""
        lines = [f"KNOWLEDGE BASE FOR: {self.name}", "="*50]
        
        if self.role:
            lines.append(f"\nRole: {self.role}")
        
        if self.traits:
            lines.append(f"\nTraits: {', '.join(self.traits)}")
        
        if self.facts:
            lines.append(f"\nFacts ({len(self.facts)}):")
            for fact in self.facts[:15]:
                lines.append(f"  - {fact.get('subject', '')} {fact.get('predicate', '')} {fact.get('object', '')}")
        
        if self.timeline:
            lines.append(f"\nTimeline ({len(self.timeline)}):")
            for event in self.timeline[:10]:
                lines.append(f"  - {event.get('event', '')} [{event.get('time_reference', 'N/A')}]")
        
        if self.relationships:
            lines.append(f"\nRelationships ({len(self.relationships)}):")
            for rel in self.relationships[:10]:
                lines.append(f"  - {rel.get('description', '')}")
        
        if self.summary:
            lines.append(f"\nSummary: {self.summary}")
        
        lines.append(f"\nSources: {', '.join(self.sources)}")
        
        return "\n".join(lines)


###############################################################################
# DATA LOADING
###############################################################################

def load_characters_from_csv(csv_path: str) -> list[dict]:
    """Load character data from train.csv including backstory statements."""
    characters = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                char_data = {
                    "id": row.get('id', ''),
                    "name": row.get('char') or row.get('character') or row.get('name') or row.get('Character', ''),
                    "backstory": row.get('content') or row.get('backstory') or row.get('description') or row.get('background', ''),
                    "book": row.get('book_name') or row.get('book') or row.get('source', ''),
                    "label": row.get('label', ''),
                    "caption": row.get('caption', ''),
                    "raw_row": row
                }
                if char_data["name"].strip():
                    characters.append(char_data)
        print(f"✓ Loaded {len(characters)} character statements from {csv_path}")
        return characters
    except FileNotFoundError:
        print(f"⚠️  train.csv not found at {csv_path}")
        return []
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return []


def get_unique_characters(character_data: list[dict]) -> list[str]:
    """Get unique character names from loaded data."""
    seen = set()
    unique = []
    for c in character_data:
        name = c.get("name", "").strip()
        if name and name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


###############################################################################
# BOOK INGESTION
###############################################################################

def ingest_books_directly(data_folder: Path) -> list[dict]:
    """
    Primary book ingestion using direct file reading.
    This is the most reliable method for loading book files.
    """
    print("\n📚 BOOK INGESTION (Direct File Reading)")
    print("="*60)
    
    books = []
    txt_files = list(data_folder.glob("*.txt"))
    
    print(f"Found {len(txt_files)} .txt file(s) in {data_folder}")
    
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                books.append({
                    "filename": txt_file.stem,
                    "content": content,
                    "size": len(content)
                })
                print(f"  ✓ {txt_file.name}: {len(content):,} chars")
        except Exception as e:
            print(f"  ⚠️ Failed to read {txt_file}: {e}")
    
    print(f"\n✓ Loaded {len(books)} book(s)")
    print("="*60)
    return books


class PathwayBookIngestion:
    """
    Use Pathway framework to ingest book files (optional enhancement).
    Falls back to direct reading if Pathway has issues.
    """
    
    def __init__(self, data_folder: str):
        self.data_folder = Path(data_folder)
        if not self.data_folder.exists():
            raise ValueError(f"Data folder not found: {data_folder}")
    
    def ingest_books(self) -> list[dict]:
        """Ingest all .txt files - uses direct reading as primary method."""
        # Use direct file reading as primary method (more reliable)
        books = ingest_books_directly(self.data_folder)
        
        if books:
            return books
        
        # Try Pathway as fallback
        print("\n📚 Trying Pathway ingestion as fallback...")
        try:
            return self._pathway_ingest()
        except Exception as e:
            print(f"⚠️ Pathway fallback also failed: {e}")
            return []
    
    def _pathway_ingest(self) -> list[dict]:
        """Internal Pathway-based ingestion."""
        book_table = pw.io.fs.read(
            path=str(self.data_folder),
            format="plaintext",
            mode="static",
            with_metadata=True
        )
        
        @pw.udf
        def extract_book_info(data, metadata) -> dict:
            content = str(data)
            path_str = str(metadata)
            if "path" in path_str:
                import re
                match = re.search(r"'path':\s*'([^']+)'", path_str)
                if match:
                    filename = Path(match.group(1)).stem
                else:
                    filename = "unknown"
            else:
                filename = "unknown"
            return {"filename": filename, "content": content}
        
        processed = book_table.select(
            book_info=extract_book_info(pw.this.data, pw.this._metadata)
        )
        
        results = []
        
        def on_change(key, row, time, is_addition):
            if is_addition:
                book_info = row.get('book_info', {})
                if isinstance(book_info, dict):
                    results.append(book_info)
        
        pw.io.subscribe(processed, on_change=on_change)
        pw.run()
        
        return results


###############################################################################
# SENTENCE EXTRACTION
###############################################################################

def extract_character_sentences(text: str, characters: list[str]) -> dict[str, list[str]]:
    """Extract sentences mentioning each character."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    character_sentences = {char: [] for char in characters}
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue
        
        for char in characters:
            # Match character name (handles multi-word names like "Tom Ayrton/Ben Joyce")
            char_parts = re.split(r'[/\s]+', char)
            for part in char_parts:
                if len(part) > 2 and re.search(rf'\b{re.escape(part)}\b', sentence, re.IGNORECASE):
                    if sentence not in character_sentences[char]:
                        character_sentences[char].append(sentence)
                    break
    
    return character_sentences


###############################################################################
# GROQ API WITH MULTI-KEY ROTATION
###############################################################################

class GroqAPIClient:
    """
    Groq API client with support for multiple API keys and automatic rotation
    when hitting rate limits.
    """
    
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        
        if not self.api_keys:
            raise ValueError("No Groq API keys found! Set GROQ_API_KEY or NEW_API_KEY in .env")
        
        print(f"✓ Loaded {len(self.api_keys)} API key(s)")
    
    def _load_api_keys(self) -> list[str]:
        """Load all available API keys from environment."""
        keys = []
        # Check various key names
        key_names = ["GROQ_API_KEY", "NEW_API_KEY", "SECOND_API_KEY", "THIRD_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"]
        for name in key_names:
            key = os.getenv(name)
            if key and key.strip() and key not in keys:
                keys.append(key.strip())
        return keys
    
    def _get_current_key(self) -> str:
        """Get current API key."""
        return self.api_keys[self.current_key_index]
    
    def _rotate_key(self) -> bool:
        """Rotate to next API key. Returns True if rotation happened."""
        if len(self.api_keys) > 1:
            old_index = self.current_key_index
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            print(f"  🔄 Rotating API key: {old_index + 1} → {self.current_key_index + 1}")
            return True
        return False
    
    def send(self, prompt: str, quiet: bool = False, max_retries: int = 3) -> dict:
        """Send request to Groq API with automatic key rotation on rate limit."""
        
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a knowledge extraction and verification engine. Output only valid JSON, no markdown."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 8192
        }
        
        if not quiet:
            print("\n" + "="*60)
            print("📤 GROQ API REQUEST")
            print("="*60)
            print(f"Model: {self.model}")
            print(f"Prompt length: {len(prompt)} chars")
            print(f"Prompt preview:\n{prompt[:500]}...")
            print("="*60)
        
        attempts = 0
        keys_tried = set()
        
        while attempts < max_retries:
            current_key = self._get_current_key()
            key_preview = f"{current_key[:8]}...{current_key[-4:]}"
            
            if current_key in keys_tried and len(keys_tried) >= len(self.api_keys):
                # All keys have been tried
                return {"success": False, "error": "All API keys exhausted (rate limited)", "status": 429}
            
            keys_tried.add(current_key)
            headers["Authorization"] = f"Bearer {current_key}"
            
            try:
                response = requests.post(self.url, headers=headers, json=payload, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    if not quiet:
                        print("\n" + "="*60)
                        print("📥 GROQ API RESPONSE")
                        print("="*60)
                        print(f"Status: {response.status_code}")
                        print(f"Tokens used: {result.get('usage', {}).get('total_tokens', 'N/A')}")
                        print(f"\nResponse:\n{content[:500]}...")
                        print("="*60)
                    return {"success": True, "content": content, "raw": result}
                
                elif response.status_code == 429:
                    # Rate limited - try rotating to next key
                    print(f"  ⚠️ Rate limit hit on key {key_preview}")
                    if self._rotate_key():
                        attempts += 1
                        continue
                    else:
                        return {"success": False, "error": "Rate limited and no other keys available", "status": 429}
                
                else:
                    error_text = response.text
                    if not quiet:
                        print(f"Error: {error_text}")
                    return {"success": False, "error": error_text, "status": response.status_code}
                    
            except Exception as e:
                print(f"❌ Request failed: {e}")
                attempts += 1
                if attempts < max_retries:
                    self._rotate_key()
                    continue
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Max retries exceeded"}


# Global API client instance
_groq_client = None

def get_groq_client() -> GroqAPIClient:
    """Get or create the global Groq API client."""
    global _groq_client
    if _groq_client is None:
        _groq_client = GroqAPIClient()
    return _groq_client


def send_to_groq(prompt: str, api_key: str = None, quiet: bool = False) -> dict:
    """Send request to Groq API (wrapper for compatibility)."""
    client = get_groq_client()
    return client.send(prompt, quiet=quiet)


def parse_json_response(content: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks and truncation."""
    try:
        # Remove markdown code blocks
        content = re.sub(r'^```json?\s*', '', content.strip())
        content = re.sub(r'\s*```$', '', content)
        
        # Try direct parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try to repair truncated JSON by closing brackets
        repaired = content.rstrip()
        
        # Count open brackets and close them
        open_braces = repaired.count('{') - repaired.count('}')
        open_brackets = repaired.count('[') - repaired.count(']')
        
        # Remove trailing incomplete string/value
        if repaired and repaired[-1] not in '}]"0123456789nulltruefalse':
            # Find last complete value
            last_complete = max(
                repaired.rfind('"}'),
                repaired.rfind('"\n'),
                repaired.rfind('],'),
                repaired.rfind('},'),
                repaired.rfind('null'),
                repaired.rfind('true'),
                repaired.rfind('false')
            )
            if last_complete > 0:
                repaired = repaired[:last_complete+1]
        
        # Close arrays and objects
        repaired += ']' * open_brackets
        repaired += '}' * open_braces
        
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            # Last resort: extract what we can
            return _extract_partial_json(content)
            
    except Exception as e:
        return {"error": f"JSON parse failed: {e}", "raw": content[:500]}


def _extract_partial_json(content: str) -> dict:
    """Extract partial data from truncated JSON."""
    result = {"partial": True, "facts": [], "timeline": [], "character": {}}
    
    # Try to extract character name
    name_match = re.search(r'"name":\s*"([^"]+)"', content)
    if name_match:
        result["character"]["name"] = name_match.group(1)
    
    # Try to extract role
    role_match = re.search(r'"role":\s*"([^"]+)"', content)
    if role_match:
        result["character"]["role"] = role_match.group(1)
    
    # Try to extract traits
    traits_match = re.search(r'"traits":\s*\[([^\]]+)', content)
    if traits_match:
        traits_str = traits_match.group(1)
        traits = re.findall(r'"([^"]+)"', traits_str)
        result["character"]["traits"] = traits[:5]
    
    # Try to extract summary
    summary_match = re.search(r'"summary":\s*"([^"]+)"', content)
    if summary_match:
        result["summary"] = summary_match.group(1)
    
    return result


###############################################################################
# PROMPTS
###############################################################################

def create_extraction_prompt(character: str, sentences: list[str]) -> str:
    """Create prompt for character knowledge extraction."""
    # Limit sentences to reduce output size
    limited_sentences = sentences[:25]
    sentences_text = "\n".join(f"- {s[:200]}" for s in limited_sentences)
    
    return f"""Extract KEY knowledge about "{character}" from these sentences. Be CONCISE.

SENTENCES:
{sentences_text}

Output a JSON object with this EXACT structure (keep arrays SHORT, max 5 items each):
{{
  "character": {{
    "name": "{character}",
    "role": "brief role description",
    "traits": ["max 5 traits"],
    "relationships": ["max 5 key relationships"]
  }},
  "timeline": [
    {{"event": "brief event", "time_reference": "when", "certainty": "high/medium/low"}}
  ],
  "facts": [
    {{"subject": "{character}", "predicate": "action", "object": "target", "type": "permanent/temporal"}}
  ],
  "summary": "One sentence summary"
}}

IMPORTANT: Keep response under 1000 tokens. Max 5 items per array. Return ONLY valid JSON."""


def create_statement_verification_prompt(character: str, knowledge: CharacterKnowledge, backstory_statement: str) -> str:
    """
    Create prompt for verifying a SINGLE backstory statement against extracted knowledge.
    Returns a truth percentage - NO "unverifiable" allowed.
    """
    knowledge_text = knowledge.get_knowledge_summary()
    
    return f"""You are a fact-checking expert. Analyze if this STATEMENT about a character is TRUE based on EXTRACTED KNOWLEDGE from books.

CHARACTER: {character}

EXTRACTED KNOWLEDGE FROM BOOKS:
{knowledge_text}

STATEMENT TO VERIFY:
"{backstory_statement}"

INSTRUCTIONS:
1. Carefully analyze the extracted knowledge
2. Determine what percentage of this statement is TRUE based on known facts
3. A statement CONTRADICTS (Truth < 50%) if AND ONLY IF:
   - It DIRECTLY conflicts with a specific known fact (e.g. "born in 1800" vs "born in 1900")
   - It claims actions impossible given the timeline (e.g. "died in 1815" vs "imprisoned in 1820")
   - It attributes traits that differ fundamentally (e.g. "coward" vs "known for extreme bravery")
4. A statement is TRUE/CONSISTENT (Truth >= 50%) if:
   - It is supported by known facts
   - It adds new backstory that is PLAUSIBLE and NOT contradicted by anything
   - IMPORTANT: "Not mentioned in text" does NOT mean contradiction! New details are Consistent.
   - Example: If text says "he is a sailor", and statement says "he learned sailing from his father" -> This is CONSISTENT (80%), not contradictory.

5. YOU MUST give a truth_percentage between 0 and 100:
   - 0-30%: Clear Contradiction
   - 31-49%: Likely Contradiction (strong tension)
   - 50-69%: Plausible / Neutral (backstory not mentioned but fits) -> output "is_contradiction": false
   - 70-100%: Supported / Highly Consistent -> output "is_contradiction": false

Output a JSON object with this exact structure:
{{
  "statement": "{backstory_statement[:100]}...",
  "truth_percentage": 0 to 100,
  "is_contradiction": true or false,
  "reasoning": "Detailed explanation of why this percentage",
  "supporting_facts": ["facts that support the statement"],
  "contradicting_facts": ["facts that contradict the statement"]
}}

Return ONLY the JSON object, no other text."""


def create_final_verdict_prompt(character: str, statement_verifications: list[dict], original_statement: str) -> str:
    """
    Create prompt for generating final verdict.
    Uses truth_percentage: if ALL >= 50% -> no contradiction, if ANY < 50% -> contradiction
    """
    verifications_text = json.dumps(statement_verifications, indent=2)
    
    return f"""You are a final verdict generator. Based on the verification below, provide a FINAL VERDICT.

CHARACTER: {character}

ORIGINAL BACKSTORY STATEMENT:
"{original_statement}"

VERIFICATION RESULT:
{verifications_text}

RULES FOR FINAL VERDICT:
1. Look at the truth_percentage in the verification
2. If truth_percentage >= 50 -> verdict is "no contradiction"
3. If truth_percentage < 50 -> verdict is "contradiction"
4. If is_contradiction is true -> verdict is "contradiction"

Output a JSON object:
{{
  "character": "{character}",
  "truth_percentage": the truth percentage from verification,
  "final_verdict": "contradiction" or "no contradiction",
  "reason": "One sentence explanation"
}}

Return ONLY the JSON object, no other text."""


###############################################################################
# VERIFICATION PIPELINE
###############################################################################

def verify_statement(character: str, knowledge: CharacterKnowledge, backstory: str, api_key: str) -> dict:
    """
    Verify a single backstory statement against the knowledge structure.
    Returns verification result with truth_percentage.
    """
    prompt = create_statement_verification_prompt(character, knowledge, backstory)
    response = send_to_groq(prompt, api_key, quiet=True)
    
    if response["success"]:
        result = parse_json_response(response["content"])
        if "error" not in result:
            # Ensure we have truth_percentage
            if "truth_percentage" not in result:
                # Try to infer from old format
                if result.get("is_contradiction"):
                    result["truth_percentage"] = 25
                else:
                    result["truth_percentage"] = 75
            return result
        else:
            # Parse error - default to 50% (uncertain)
            return {
                "statement": backstory[:100],
                "truth_percentage": 50,
                "is_contradiction": False,
                "reasoning": "Failed to parse LLM response - defaulting to uncertain",
                "supporting_facts": [],
                "contradicting_facts": []
            }
    else:
        # API error - default to 50% (uncertain)
        return {
            "statement": backstory[:100],
            "truth_percentage": 50,
            "is_contradiction": False,
            "reasoning": f"API error: {response.get('error', 'Unknown')} - defaulting to uncertain",
            "supporting_facts": [],
            "contradicting_facts": []
        }


def get_final_verdict(character: str, verifications: list[dict], original_statement: str, api_key: str) -> dict:
    """
    Generate final verdict based on truth_percentage.
    If truth_percentage >= 50 -> no contradiction
    If truth_percentage < 50 OR is_contradiction=true -> contradiction
    """
    # Get truth percentage from verification
    verification = verifications[0] if verifications else {}
    truth_pct = verification.get("truth_percentage", 50)
    is_contradiction = verification.get("is_contradiction", False)
    
    # Determine verdict based on percentage
    if is_contradiction or truth_pct < 50:
        final_verdict = "contradiction"
    else:
        final_verdict = "no contradiction"
    
    prompt = create_final_verdict_prompt(character, verifications, original_statement)
    response = send_to_groq(prompt, api_key, quiet=False)
    
    if response["success"]:
        result = parse_json_response(response["content"])
        if "error" not in result:
            # Ensure consistent format
            result["truth_percentage"] = result.get("truth_percentage", truth_pct)
            # Normalize verdict to expected format
            verdict = result.get("final_verdict", "").lower()
            if "contradiction" in verdict and "no" not in verdict:
                result["final_verdict"] = "contradiction"
            else:
                result["final_verdict"] = "no contradiction"
            return result
    
    # Fallback if API fails - use computed verdict
    return {
        "character": character,
        "truth_percentage": truth_pct,
        "final_verdict": final_verdict,
        "reason": verification.get("reasoning", f"Truth percentage: {truth_pct}%")
    }


###############################################################################
# OUTPUT
###############################################################################

def save_predictions_csv(predictions: list[dict], output_path: str):
    """Save predictions to predict.csv - simplified format with just id and verdict."""
    filepath = Path(output_path) / "predict.csv"
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Simple header - just id and label
        writer.writerow(["id", "label"])
        
        # Data rows - just id and "contradiction" or "no contradiction"
        for pred in predictions:
            verdict = pred.get("final_verdict", "no contradiction")
            # Normalize to expected format
            if "contradiction" in verdict.lower() and "no" not in verdict.lower():
                label = "contradiction"
            else:
                label = "no contradiction"
            
            writer.writerow([
                pred.get("id", ""),
                label
            ])
    
    print(f"\n✓ Predictions saved to: {filepath}")
    return filepath


def save_output(data: dict, output_dir: str, filename: str):
    """Save output to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved: {filepath}")


###############################################################################
# MAIN PIPELINE
###############################################################################

def main():
    """Main execution with Pathway ingestion and background verification."""
    print("\n" + "="*60)
    print("CHARACTER KNOWLEDGE EXTRACTION + VERIFICATION PIPELINE")
    print("="*60 + "\n")
    
    # Configuration
    base_path = Path("/home/Tejesh/Documents/kgph_pathway")
    data_folder = base_path / "data"
    output_folder = base_path / "output"
    csv_path = data_folder / "train.csv"
    
    # Process all characters (set to None for all, or a number to limit)
    MAX_CHARACTERS = None  # Process all
    
    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not set!")
        sys.exit(1)
    
    print(f"✓ API Key: {api_key[:10]}...{api_key[-5:]}")
    
    # Load character statements from CSV
    all_character_data = load_characters_from_csv(str(csv_path))
    if MAX_CHARACTERS:
        all_character_data = all_character_data[:MAX_CHARACTERS]
    
    unique_character_names = get_unique_characters(all_character_data)
    
    if not unique_character_names:
        print("\n⚠️  No characters found in train.csv!")
        sys.exit(1)
    
    print(f"\n👥 Unique characters ({len(unique_character_names)}): {unique_character_names}")
    print(f"📝 Total statements to verify: {len(all_character_data)}")
    
    # =========================================================================
    # PHASE 1: INGEST BOOKS
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 1: BOOK INGESTION")
    print("="*60)
    
    try:
        print("\n🚀 Starting book ingestion...")
        ingestion = PathwayBookIngestion(str(data_folder))
        books = ingestion.ingest_books()
    except Exception as e:
        print(f"⚠️  Ingestion failed: {e}")
        print("Falling back to direct file reading...")
        books = ingest_books_directly(data_folder)
    
    if not books:
        print("\n❌ No books found in data folder!")
        sys.exit(1)
    
    # =========================================================================
    # PHASE 2: BUILD KNOWLEDGE DATA STRUCTURE
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 2: BUILD KNOWLEDGE DATA STRUCTURE")
    print("="*60)
    
    knowledge_map: dict[str, CharacterKnowledge] = {}
    
    # Initialize knowledge structures for each unique character, loading if exists
    for character_name in unique_character_names:
        safe_name = re.sub(r'[^\w]', '_', character_name)
        knowledge_file = output_folder / "knowledge_base" / f"{safe_name}_knowledge.json"
        
        if knowledge_file.exists():
            try:
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    k = CharacterKnowledge(character_name)
                    k.facts = data.get("facts", [])
                    k.timeline = data.get("timeline", [])
                    k.relationships = data.get("relationships", [])
                    k.traits = data.get("traits", [])
                    k.role = data.get("role", "")
                    k.summary = data.get("summary", "")
                    k.quotes = data.get("quotes", [])
                    k.sentences = data.get("sentences", []) # Add sentences for completeness
                    k.sources = data.get("sources", [])
                    knowledge_map[character_name] = k
                    print(f"  ✓ Loaded existing knowledge for {character_name} from {knowledge_file.name}")
            except Exception as e:
                print(f"  ⚠️ Failed to load existing knowledge for {character_name} from {knowledge_file.name}, creating new: {e}")
                knowledge_map[character_name] = CharacterKnowledge(character_name)
        else:
            knowledge_map[character_name] = CharacterKnowledge(character_name)

    # Extract and accumulate knowledge from all books for characters not fully loaded
    for book in books:
        book_name = book["filename"] # Use 'filename' as in original code
        print(f"\n📖 Processing: {book_name}")
        
        # Extract sentences for all unique characters from the current book
        char_sentences_in_book = extract_character_sentences(book['content'], unique_character_names)
        
        for character_name in unique_character_names:
            knowledge = knowledge_map[character_name]
            
            # If knowledge was loaded and has content, assume it's complete and skip extraction for this character
            # This logic assumes that if a knowledge file exists and has facts, it's already processed all relevant books.
            # For a more granular approach, one would need to track which books contributed to the knowledge.
            if knowledge.facts or knowledge.timeline or knowledge.quotes:
                # print(f"  Skipping extraction for {character_name} (knowledge already loaded).")
                continue

            sentences = char_sentences_in_book.get(character_name, [])
            
            if not sentences:
                continue
            
            print(f"\n  🔍 {character_name}: Found {len(sentences)} sentences in {book_name}")
            knowledge.add_sentences(sentences)
            
            # Send to Groq for structured extraction
            prompt = create_extraction_prompt(character_name, sentences)
            response = send_to_groq(prompt, api_key, quiet=True)
            
            if response["success"]:
                extracted = parse_json_response(response["content"])
                if "error" not in extracted:
                    knowledge_map[character_name].add_from_extraction(extracted, book['filename'])
                    print(f"    ✓ Extracted: {len(extracted.get('facts', []))} facts, {len(extracted.get('timeline', []))} events")
                else:
                    print(f"    ⚠️ Parse error: {extracted.get('raw', '')[:100]}")
            else:
                print(f"    ⚠️ API error: {response.get('error', 'Unknown')}")
    
    # Save knowledge base
    print("\n📁 Saving knowledge base...")
    for char_name, knowledge in knowledge_map.items():
        if knowledge.facts or knowledge.timeline or knowledge.quotes:
            safe_char = re.sub(r'[^\w]', '_', char_name)
            save_output(
                knowledge.to_dict(),
                str(output_folder / "knowledge_base"),
                f"{safe_char}_knowledge.json"
            )
    
    # Print knowledge summary
    print("\n" + "-"*60)
    print("KNOWLEDGE BASE SUMMARY")
    print("-"*60)
    for char_name, knowledge in knowledge_map.items():
        print(f"\n{char_name}:")
        print(f"  Facts: {len(knowledge.facts)}")
        print(f"  Timeline: {len(knowledge.timeline)}")
        print(f"  Relationships: {len(knowledge.relationships)}")
        print(f"  Traits: {len(knowledge.traits)}")
        print(f"  Sources: {', '.join(knowledge.sources)}")
    
    # =========================================================================
    # PHASE 3: VERIFY EACH BACKSTORY STATEMENT
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 3: VERIFY BACKSTORY STATEMENTS")
    print("="*60)
    
    all_predictions = []
    
    for idx, char_info in enumerate(all_character_data):
        char_name = char_info["name"]
        backstory = char_info.get("backstory", "")
        char_id = char_info.get("id", str(idx))
        
        print(f"\n{'='*60}")
        print(f"📋 [{idx+1}/{len(all_character_data)}] Verifying: {char_name} (ID: {char_id})")
        print(f"Statement: {backstory[:100]}...")
        print("="*60)
        
        knowledge = knowledge_map.get(char_name)
        
        if not knowledge or (not knowledge.facts and not knowledge.timeline and not knowledge.quotes):
            print(f"  ⚠️ No knowledge available for {char_name}")
            prediction = {
                "id": char_id,
                "character": char_name,
                "final_verdict": "unverifiable",
                "confidence": 0.0,
                "summary_reason": "No knowledge extracted from books for this character",
                "detailed_reasoning": "Character not found or no relevant information in the book corpus",
                "contradiction_details": [],
                "key_evidence": ""
            }
            all_predictions.append(prediction)
            continue
        
        # Verify the backstory statement
        print(f"\n  🔍 Verifying statement...")
        verification = verify_statement(char_name, knowledge, backstory, api_key)
        
        print(f"  📊 Verification result: {verification.get('verification_result', 'unknown')}")
        print(f"  💭 Reasoning: {verification.get('reasoning', 'N/A')[:200]}...")
        
        # Generate final verdict with Groq reasoning
        print(f"\n  🎯 Generating final verdict with reasoning...")
        final_verdict = get_final_verdict(char_name, [verification], backstory, api_key)
        final_verdict["id"] = char_id
        
        print(f"\n  ✅ FINAL VERDICT: {final_verdict.get('final_verdict', 'unknown').upper()}")
        print(f"  📝 Reason: {final_verdict.get('summary_reason', 'N/A')}")
        
        all_predictions.append(final_verdict)
        
        # Save individual verification
        safe_char = re.sub(r'[^\w]', '_', char_name)
        save_output(
            {
                "id": char_id,
                "character": char_name,
                "backstory": backstory,
                "knowledge_summary": knowledge.get_knowledge_summary(),
                "verification": verification,
                "final_verdict": final_verdict
            },
            str(output_folder / "verifications"),
            f"{char_id}_{safe_char}_verification.json"
        )
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save predict.csv
    save_predictions_csv(all_predictions, str(output_folder))
    
    # Save all results
    save_output(
        {
            "knowledge_base": {name: k.to_dict() for name, k in knowledge_map.items()},
            "predictions": all_predictions,
            "timestamp": datetime.now().isoformat(),
            "books_processed": [b.get("filename") for b in books],
            "total_statements": len(all_character_data)
        },
        str(output_folder),
        f"full_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY")
    print("="*60)
    
    consistent_count = sum(1 for p in all_predictions if p.get("final_verdict") == "consistent")
    contradict_count = sum(1 for p in all_predictions if p.get("final_verdict") == "contradict")
    unverifiable_count = sum(1 for p in all_predictions if p.get("final_verdict") == "unverifiable")
    
    print(f"\nBooks processed: {len(books)}")
    print(f"Characters in knowledge base: {len(knowledge_map)}")
    print(f"Statements verified: {len(all_predictions)}")
    print(f"\nVERDICT BREAKDOWN:")
    print(f"  ✅ Consistent: {consistent_count}")
    print(f"  ❌ Contradict: {contradict_count}")
    print(f"  ❓ Unverifiable: {unverifiable_count}")
    
    print(f"\n📁 Output directory: {output_folder}")
    print(f"📄 Predictions: {output_folder}/predict.csv")
    print(f"📚 Knowledge base: {output_folder}/knowledge_base/")
    print(f"🔍 Verifications: {output_folder}/verifications/")
    print("="*60 + "\n")
    
    return all_predictions


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
