#!/usr/bin/env python3
"""
Combined Character Knowledge Extraction + Backstory Integration + Prediction Pipeline:
1. Extract character knowledge from books using Groq API
2. Combine character JSON with backstories from train.csv
   - If consistent: add backstory as-is
   - If not consistent: add as "(this did not happen){backstory}"
3. Store combined JSON
4. Test on test.csv and generate predict.csv with: id, Name, Consistent
5. Optimized API calls using alternating keys with delays
"""

import os
import sys
import json
import re
import csv
import time
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load environment variables from .env
load_dotenv()

# Also load from .env.example for any additional keys not in .env
env_example = Path(__file__).parent / ".env.example"
if env_example.exists():
    load_dotenv(env_example, override=False)


###############################################################################
# CONFIGURATION
###############################################################################

BASE_PATH = Path("/home/Tejesh/Documents/kgph_pathway")
DATA_FOLDER = BASE_PATH / "data"
OUTPUT_FOLDER = BASE_PATH / "output"
TRAIN_CSV = DATA_FOLDER / "train.csv"
TEST_CSV = DATA_FOLDER / "test.csv"

# API Rate limiting configuration
REQUEST_DELAY_SECONDS = 5.0  # Delay between requests to same key (increased to reduce rate limits)
MIN_DELAY_BETWEEN_ANY_REQUEST = 1.5  # Minimum delay between any request


###############################################################################
# OPTIMIZED GROQ API CLIENT WITH ALTERNATING KEYS
###############################################################################

class OptimizedGroqClient:
    """
    Optimized Groq API client that alternates between API keys
    and adds delays to avoid rate limits.
    """
    
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.key_last_used = {i: 0 for i in range(len(self.api_keys))}
        self.current_key_index = 0
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.lock = threading.Lock()
        
        if not self.api_keys:
            raise ValueError("No Groq API keys found! Set GROQ_API_KEY, NEW_API_KEY, etc. in .env")
        
        print(f"✓ Loaded {len(self.api_keys)} API key(s)")
        for i, key in enumerate(self.api_keys):
            print(f"  Key {i+1}: {key[:8]}...{key[-4:]}")
    
    def _load_api_keys(self) -> list[str]:
        """Load all available API keys from environment."""
        keys = []
        key_names = [
            "GROQ_API_KEY", "NEW_API_KEY", "SECOND_API_KEY", 
            "THIRD_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"
        ]
        for name in key_names:
            key = os.getenv(name)
            if key and key.strip() and key not in keys:
                keys.append(key.strip())
        return keys
    
    def _get_next_key_with_delay(self) -> tuple[str, int]:
        """
        Get the next API key to use, alternating between keys
        and waiting if necessary to respect rate limits.
        """
        with self.lock:
            current_time = time.time()
            
            # Find the key that was used least recently
            best_key_idx = 0
            best_wait_time = float('inf')
            
            for i in range(len(self.api_keys)):
                time_since_last = current_time - self.key_last_used[i]
                wait_needed = max(0, REQUEST_DELAY_SECONDS - time_since_last)
                
                if wait_needed < best_wait_time:
                    best_wait_time = wait_needed
                    best_key_idx = i
            
            # Wait if needed
            if best_wait_time > 0:
                wait_time = max(best_wait_time, MIN_DELAY_BETWEEN_ANY_REQUEST)
                print(f"  ⏳ Waiting {wait_time:.1f}s before using key {best_key_idx + 1}...")
                time.sleep(wait_time)
            
            # Update last used time
            self.key_last_used[best_key_idx] = time.time()
            
            return self.api_keys[best_key_idx], best_key_idx + 1
    
    def send(self, prompt: str, quiet: bool = False, max_retries: int = 3) -> dict:
        """Send request to Groq API with optimized key rotation and delays."""
        
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a knowledge extraction and verification engine. Output only valid JSON, no markdown."
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 8192
        }
        
        if not quiet:
            print(f"\n📤 GROQ API REQUEST")
            print(f"  Model: {self.model}")
            print(f"  Prompt length: {len(prompt)} chars")
        
        attempts = 0
        
        while attempts < max_retries:
            api_key, key_num = self._get_next_key_with_delay()
            key_preview = f"{api_key[:8]}...{api_key[-4:]}"
            
            headers["Authorization"] = f"Bearer {api_key}"
            
            try:
                if not quiet:
                    print(f"  🔑 Using key {key_num}: {key_preview}")
                
                response = requests.post(self.url, headers=headers, json=payload, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    if not quiet:
                        print(f"  ✓ Response received ({result.get('usage', {}).get('total_tokens', 'N/A')} tokens)")
                    return {"success": True, "content": content, "raw": result}
                
                elif response.status_code == 429:
                    print(f"  ⚠️ Rate limit hit on key {key_num}, retrying with another key...")
                    # Mark this key as recently used to avoid immediate reuse
                    with self.lock:
                        self.key_last_used[key_num - 1] = time.time() + 30  # Extra penalty
                    attempts += 1
                    continue
                
                else:
                    error_text = response.text
                    if not quiet:
                        print(f"  ❌ Error: {error_text[:200]}")
                    return {"success": False, "error": error_text, "status": response.status_code}
                    
            except Exception as e:
                print(f"  ❌ Request failed: {e}")
                attempts += 1
                if attempts < max_retries:
                    time.sleep(2)
                    continue
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Max retries exceeded"}


# Global optimized client
_groq_client = None

def get_groq_client() -> OptimizedGroqClient:
    """Get or create the global optimized Groq API client."""
    global _groq_client
    if _groq_client is None:
        _groq_client = OptimizedGroqClient()
    return _groq_client


###############################################################################
# JSON PARSING
###############################################################################

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
        
        # Try to repair truncated JSON
        repaired = content.rstrip()
        open_braces = repaired.count('{') - repaired.count('}')
        open_brackets = repaired.count('[') - repaired.count(']')
        
        # Close arrays and objects
        repaired += ']' * max(0, open_brackets)
        repaired += '}' * max(0, open_braces)
        
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return {"error": "JSON parse failed", "raw": content[:500]}
            
    except Exception as e:
        return {"error": f"JSON parse failed: {e}", "raw": content[:500]}


###############################################################################
# DATA LOADING
###############################################################################

def load_csv_data(csv_path: str) -> list[dict]:
    """Load character data from CSV file."""
    characters = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                char_data = {
                    "id": row.get('id', ''),
                    "name": row.get('char') or row.get('character') or row.get('name', ''),
                    "backstory": row.get('content') or row.get('backstory') or row.get('description', ''),
                    "book": row.get('book_name') or row.get('book', ''),
                    "label": row.get('label', ''),
                    "caption": row.get('caption', ''),
                }
                if char_data["name"].strip():
                    characters.append(char_data)
        print(f"✓ Loaded {len(characters)} entries from {csv_path}")
        return characters
    except FileNotFoundError:
        print(f"⚠️ File not found: {csv_path}")
        return []
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return []


def get_unique_characters(data: list[dict]) -> list[str]:
    """Get unique character names."""
    seen = set()
    unique = []
    for c in data:
        name = c.get("name", "").strip()
        if name and name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def ingest_books(data_folder: Path) -> list[dict]:
    """Read all .txt book files from the data folder."""
    books = []
    txt_files = list(data_folder.glob("*.txt"))
    
    print(f"\n📚 Found {len(txt_files)} book file(s)")
    
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
    
    return books


def extract_character_sentences(text: str, characters: list[str]) -> dict[str, list[str]]:
    """Extract sentences mentioning each character."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    character_sentences = {char: [] for char in characters}
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue
        
        for char in characters:
            char_parts = re.split(r'[/\s]+', char)
            for part in char_parts:
                if len(part) > 2 and re.search(rf'\b{re.escape(part)}\b', sentence, re.IGNORECASE):
                    if sentence not in character_sentences[char]:
                        character_sentences[char].append(sentence)
                    break
    
    return character_sentences


###############################################################################
# PROMPTS
###############################################################################

def create_extraction_prompt(character: str, sentences: list[str]) -> str:
    """Create prompt for character knowledge extraction."""
    limited_sentences = sentences[:30]
    sentences_text = "\n".join(f"- {s[:250]}" for s in limited_sentences)
    
    return f"""Extract comprehensive knowledge about "{character}" from these sentences.

SENTENCES:
{sentences_text}

Output a JSON object with this structure:
{{
  "name": "{character}",
  "role": "character's role/occupation in the story",
  "traits": ["personality traits", "max 10"],
  "relationships": [
    {{"target": "other character", "type": "relationship type", "description": "brief description"}}
  ],
  "timeline": [
    {{"event": "what happened", "time_reference": "when (if known)", "certainty": "high/medium/low"}}
  ],
  "facts": [
    {{"subject": "{character}", "predicate": "action/state", "object": "target/detail", "type": "permanent/temporal"}}
  ],
  "locations": ["places associated with character"],
  "summary": "2-3 sentence summary of the character"
}}

Return ONLY valid JSON, no markdown."""


def create_verification_prompt(character: str, knowledge_json: dict, backstory: str) -> str:
    """Create prompt for verifying a backstory against character knowledge - BALANCED STRICT mode."""
    knowledge_text = json.dumps(knowledge_json, indent=2)[:6000]  # Allow more context
    
    return f"""You are a precise fact-checker. Determine if this STATEMENT about "{character}" CONTRADICTS known facts.

CHARACTER: {character}

KNOWN FACTS FROM BOOKS:
{knowledge_text}

STATEMENT TO VERIFY:
"{backstory}"

VERIFICATION RULES:

MARK AS **CONTRADICTION** (is_consistent: false) if the statement:
1. States a DIFFERENT value for something we know (e.g., "born in Paris" vs known "born in Rome")
2. Claims an event that CONFLICTS with timeline (e.g., "met X in 1820" when we know "X died in 1810")
3. Describes traits that OPPOSE known traits (e.g., "cowardly" vs known "remarkably brave")
4. Claims relationships that CONFLICT (e.g., "his father was a doctor" vs known "son of a sailor")
5. States impossibilities given known facts (e.g., "never went to France" vs known "lived in Paris")

MARK AS **CONSISTENT** (is_consistent: true) if the statement:
1. Agrees with known facts
2. ADDS NEW DETAILS that don't conflict with anything known (new backstory is OK if plausible)
3. Provides information NOT MENTIONED in books - this is NOT a contradiction
4. Is simply unverifiable - if we don't have information to contradict it, it's consistent

KEY DISTINCTION:
- "We don't know X" → CONSISTENT (absence of info is not contradiction)
- "We know Y, but statement says NOT-Y" → CONTRADICTION (direct conflict)

Output a JSON object:
{{
  "is_consistent": true or false,
  "analysis": {{
    "claims_in_statement": ["list each factual claim from the statement"],
    "matches_with_known_facts": ["claims that match known facts"],
    "conflicts_with_known_facts": ["claims that DIRECTLY CONTRADICT known facts - be specific"],
    "new_unverifiable_claims": ["claims not mentioned in books - these are NOT contradictions"]
  }},
  "has_direct_conflict": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "explain your verdict - if contradiction, cite the specific conflicting facts"
}}

CRITICAL RULES:
- is_consistent should be FALSE only if conflicts_with_known_facts is NOT empty
- New details that aren't mentioned in books are NOT contradictions
- When in doubt, lean toward consistent unless you can cite a specific conflicting fact

Return ONLY valid JSON."""


###############################################################################
# KNOWLEDGE EXTRACTION AND COMBINATION
###############################################################################

def extract_character_knowledge(character: str, all_sentences: list[str], client: OptimizedGroqClient) -> dict:
    """Extract knowledge for a single character from all their sentences."""
    if not all_sentences:
        return {"name": character, "facts": [], "timeline": [], "relationships": [], "traits": [], "summary": "No information found."}
    
    prompt = create_extraction_prompt(character, all_sentences)
    response = client.send(prompt, quiet=True)
    
    if response["success"]:
        result = parse_json_response(response["content"])
        if "error" not in result:
            # Ensure name is set
            result["name"] = character
            return result
    
    return {"name": character, "facts": [], "timeline": [], "relationships": [], "traits": [], "summary": "Extraction failed."}


def verify_backstory(character: str, knowledge: dict, backstory: str, client: OptimizedGroqClient, max_retries: int = 2) -> dict:
    """Verify a backstory statement against character knowledge - BALANCED STRICT mode with retries."""
    
    for attempt in range(max_retries):
        prompt = create_verification_prompt(character, knowledge, backstory)
        response = client.send(prompt, quiet=True)
        
        if response["success"]:
            result = parse_json_response(response["content"])
            if "error" not in result:
                # Check for actual conflicts in the new response format
                analysis = result.get("analysis", {})
                conflicts = analysis.get("conflicts_with_known_facts", [])
                
                # Filter out empty or non-conflict entries
                real_conflicts = [c for c in conflicts if c and str(c).lower() not in ["none", "n/a", "", "no conflicts", "no direct conflicts", "[]"]]
                
                if real_conflicts:
                    result["is_consistent"] = False
                    result["has_direct_conflict"] = True
                elif not result.get("has_direct_conflict", False):
                    # No real conflicts found, mark as consistent
                    result["is_consistent"] = True
                
                return result
        
        # Wait before retry
        if attempt < max_retries - 1:
            print(f"    ⏳ Retry {attempt + 2}/{max_retries}...")
            time.sleep(3)
    
    # If all retries fail, default to CONSISTENT (absence of proof is not contradiction)
    return {
        "is_consistent": True,
        "analysis": {
            "claims_in_statement": [],
            "conflicts_with_known_facts": [],
            "new_unverifiable_claims": ["Unable to verify - API error after retries"]
        },
        "has_direct_conflict": False,
        "confidence": 0.5,
        "reasoning": "Verification API call failed after retries - defaulting to consistent (no proof of contradiction)"
    }


def combine_knowledge_with_backstories(character: str, knowledge: dict, backstories: list[dict], client: OptimizedGroqClient) -> dict:
    """
    Combine character knowledge JSON with backstories from train.csv.
    - If consistent: add backstory as-is
    - If not consistent: add as "(this did not happen){backstory}"
    """
    combined = knowledge.copy()
    combined["backstories"] = []
    
    for entry in backstories:
        backstory = entry.get("backstory", "")
        label = entry.get("label", "").lower()
        entry_id = entry.get("id", "")
        
        # Use the label from train.csv directly if available
        if label == "consistent":
            is_consistent = True
            reasoning = "Labeled as consistent in training data"
        elif label == "contradict":
            is_consistent = False
            reasoning = "Labeled as contradiction in training data"
        else:
            # If no label, verify with Groq
            verification = verify_backstory(character, knowledge, backstory, client)
            is_consistent = verification.get("is_consistent", True)
            reasoning = verification.get("reasoning", "")
        
        if is_consistent:
            combined["backstories"].append({
                "id": entry_id,
                "content": backstory,
                "is_consistent": True,
                "reasoning": reasoning
            })
        else:
            # Prefix with "(this did not happen)"
            combined["backstories"].append({
                "id": entry_id,
                "content": f"(this did not happen){backstory}",
                "original_content": backstory,
                "is_consistent": False,
                "reasoning": reasoning
            })
    
    return combined


###############################################################################
# PREDICTION ON TEST DATA
###############################################################################

def predict_test_data(test_data: list[dict], combined_knowledge: dict[str, dict], client: OptimizedGroqClient) -> list[dict]:
    """
    Run predictions on test.csv data.
    Returns list of predictions with id, Name, Consistent.
    """
    predictions = []
    total = len(test_data)
    
    for idx, entry in enumerate(test_data):
        entry_id = entry.get("id", "")
        char_name = entry.get("name", "")
        backstory = entry.get("backstory", "")
        
        print(f"\n[{idx+1}/{total}] Predicting: {char_name} (ID: {entry_id})")
        print(f"  Statement: {backstory[:80]}...")
        
        knowledge = combined_knowledge.get(char_name, {})
        
        # Check for any usable knowledge (facts, timeline, or relationships)
        has_knowledge = knowledge.get("facts") or knowledge.get("timeline") or knowledge.get("relationships") or knowledge.get("traits")
        
        if not knowledge or not has_knowledge:
            print(f"  ⚠️ No knowledge for {char_name}, defaulting to consistent")
            predictions.append({
                "id": entry_id,
                "Name": char_name,
                "Consistent": "consistent"
            })
            continue
        
        # Verify the backstory
        verification = verify_backstory(char_name, knowledge, backstory, client)
        is_consistent = verification.get("is_consistent", True)
        
        result = "consistent" if is_consistent else "contradict"
        
        print(f"  ✓ Prediction: {result}")
        print(f"  📝 Reasoning: {verification.get('reasoning', 'N/A')[:100]}...")
        
        predictions.append({
            "id": entry_id,
            "Name": char_name,
            "Consistent": result
        })
    
    return predictions


###############################################################################
# SAVE OUTPUTS
###############################################################################

def save_combined_knowledge(combined_knowledge: dict[str, dict], output_folder: Path):
    """Save combined knowledge JSON for each character."""
    combined_folder = output_folder / "combined_knowledge"
    combined_folder.mkdir(parents=True, exist_ok=True)
    
    for char_name, knowledge in combined_knowledge.items():
        safe_name = re.sub(r'[^\w]', '_', char_name)
        filepath = combined_folder / f"{safe_name}_combined.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(knowledge, f, indent=2)
        
        print(f"  ✓ Saved: {filepath.name}")


def save_predictions_csv(predictions: list[dict], output_folder: Path):
    """Save predictions to predict.csv with columns: id, Name, Consistent."""
    filepath = output_folder / "predict.csv"
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "Name", "Consistent"])
        
        for pred in predictions:
            writer.writerow([
                pred.get("id", ""),
                pred.get("Name", ""),
                pred.get("Consistent", "consistent")
            ])
    
    print(f"\n✓ Predictions saved to: {filepath}")
    return filepath


###############################################################################
# MAIN PIPELINE
###############################################################################

def main():
    """Main execution pipeline."""
    print("\n" + "="*70)
    print("COMBINED KNOWLEDGE EXTRACTION + BACKSTORY INTEGRATION + PREDICTION")
    print("="*70 + "\n")
    
    # Initialize optimized API client
    client = get_groq_client()
    
    # =========================================================================
    # PHASE 1: LOAD DATA
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 1: LOAD DATA")
    print("="*60)
    
    # Load train and test data
    train_data = load_csv_data(str(TRAIN_CSV))
    test_data = load_csv_data(str(TEST_CSV))
    
    if not train_data:
        print("❌ No training data found!")
        sys.exit(1)
    
    # Get unique characters from both datasets
    train_chars = get_unique_characters(train_data)
    test_chars = get_unique_characters(test_data)
    all_unique_chars = list(set(train_chars + test_chars))
    
    print(f"\n👥 Unique characters (train): {len(train_chars)}")
    print(f"👥 Unique characters (test): {len(test_chars)}")
    print(f"👥 Total unique characters: {len(all_unique_chars)}")
    
    # Load books
    books = ingest_books(DATA_FOLDER)
    if not books:
        print("❌ No books found!")
        sys.exit(1)
    
    # =========================================================================
    # PHASE 2: EXTRACT CHARACTER KNOWLEDGE
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 2: EXTRACT CHARACTER KNOWLEDGE FROM BOOKS")
    print("="*60)
    
    # Extract sentences for all characters from all books
    all_char_sentences: dict[str, list[str]] = {char: [] for char in all_unique_chars}
    
    for book in books:
        print(f"\n📖 Processing: {book['filename']}")
        char_sentences = extract_character_sentences(book['content'], all_unique_chars)
        
        for char, sentences in char_sentences.items():
            all_char_sentences[char].extend(sentences)
            if sentences:
                print(f"  {char}: {len(sentences)} sentences")
    
    # Extract knowledge for each character
    print("\n🔍 Extracting character knowledge...")
    character_knowledge: dict[str, dict] = {}
    
    for char in all_unique_chars:
        sentences = all_char_sentences.get(char, [])
        print(f"\n  [{char}] {len(sentences)} sentences found")
        
        if sentences:
            knowledge = extract_character_knowledge(char, sentences, client)
            character_knowledge[char] = knowledge
            print(f"    ✓ Extracted: {len(knowledge.get('facts', []))} facts, {len(knowledge.get('timeline', []))} events")
        else:
            character_knowledge[char] = {
                "name": char,
                "facts": [],
                "timeline": [],
                "relationships": [],
                "traits": [],
                "summary": "Character not found in books."
            }
    
    # =========================================================================
    # PHASE 3: COMBINE KNOWLEDGE WITH BACKSTORIES FROM TRAIN.CSV
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 3: COMBINE KNOWLEDGE WITH BACKSTORIES (TRAIN.CSV)")
    print("="*60)
    
    # Group train data by character
    char_backstories: dict[str, list[dict]] = {}
    for entry in train_data:
        char_name = entry.get("name", "")
        if char_name not in char_backstories:
            char_backstories[char_name] = []
        char_backstories[char_name].append(entry)
    
    # Combine knowledge with backstories
    combined_knowledge: dict[str, dict] = {}
    
    for char_name in all_unique_chars:
        print(f"\n  [{char_name}]")
        knowledge = character_knowledge.get(char_name, {})
        backstories = char_backstories.get(char_name, [])
        
        print(f"    Backstories to combine: {len(backstories)}")
        
        combined = combine_knowledge_with_backstories(char_name, knowledge, backstories, client)
        combined_knowledge[char_name] = combined
        
        # Count consistent vs contradictory
        consistent_count = sum(1 for b in combined.get("backstories", []) if b.get("is_consistent"))
        contradict_count = sum(1 for b in combined.get("backstories", []) if not b.get("is_consistent"))
        print(f"    ✓ Combined: {consistent_count} consistent, {contradict_count} contradictions")
    
    # Save combined knowledge
    print("\n📁 Saving combined knowledge...")
    save_combined_knowledge(combined_knowledge, OUTPUT_FOLDER)
    
    # =========================================================================
    # PHASE 4: PREDICT ON TEST.CSV
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 4: PREDICT ON TEST.CSV")
    print("="*60)
    
    predictions = predict_test_data(test_data, combined_knowledge, client)
    
    # Save predictions
    print("\n📁 Saving predictions...")
    save_predictions_csv(predictions, OUTPUT_FOLDER)
    
    # =========================================================================
    # SAVE FULL RESULTS
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 5: SAVE FULL RESULTS")
    print("="*60)
    
    full_results = {
        "timestamp": datetime.now().isoformat(),
        "books_processed": [b["filename"] for b in books],
        "characters_processed": len(all_unique_chars),
        "train_entries": len(train_data),
        "test_entries": len(test_data),
        "combined_knowledge": combined_knowledge,
        "predictions": predictions
    }
    
    results_file = OUTPUT_FOLDER / f"full_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2)
    print(f"✓ Full results saved: {results_file}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY")
    print("="*60)
    
    consistent_count = sum(1 for p in predictions if p.get("Consistent") == "consistent")
    contradict_count = sum(1 for p in predictions if p.get("Consistent") == "contradict")
    
    print(f"\nBooks processed: {len(books)}")
    print(f"Characters processed: {len(all_unique_chars)}")
    print(f"Train entries processed: {len(train_data)}")
    print(f"Test predictions made: {len(predictions)}")
    print(f"\nPREDICTION BREAKDOWN:")
    print(f"  ✅ Consistent: {consistent_count}")
    print(f"  ❌ Contradict: {contradict_count}")
    print(f"\n📁 Output directory: {OUTPUT_FOLDER}")
    print(f"📄 Predictions: {OUTPUT_FOLDER}/predict.csv")
    print(f"📚 Combined knowledge: {OUTPUT_FOLDER}/combined_knowledge/")
    print("="*60 + "\n")
    
    return predictions


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
