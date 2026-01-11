#!/usr/bin/env python3
"""
Prediction-Only Script:
Uses pre-built combined knowledge JSON files to predict contradictions in test.csv.
NO new extraction requests - only uses existing data structure files.
"""

import os
import json
import time
import csv
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Paths
BASE_PATH = Path("/home/Tejesh/Documents/kgph_pathway")
DATA_FOLDER = BASE_PATH / "data"
OUTPUT_FOLDER = BASE_PATH / "output"
COMBINED_KNOWLEDGE_FOLDER = OUTPUT_FOLDER / "combined_knowledge"
TEST_CSV = DATA_FOLDER / "test.csv"

# Rate limiting
REQUEST_DELAY_SECONDS = 2.0

###############################################################################
# GROQ API CLIENT - SIMPLE VERSION
###############################################################################

class SimpleGroqClient:
    """Simple Groq client with key rotation."""
    
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.current_key_idx = 0
        self.last_request_time = 0
        
    def _load_api_keys(self):
        keys = []
        for key_name in ["GROQ_API_KEY", "SECOND_API_KEY", "THIRD_API_KEY", "FOURTH_API_KEY", "FIFTH_API_KEY"]:
            key = os.getenv(key_name)
            if key:
                keys.append(key)
                print(f"  ✓ Loaded {key_name}")
        if not keys:
            raise ValueError("No API keys found!")
        return keys
    
    def _get_next_key(self):
        # Rotate keys
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        return self.api_keys[self.current_key_idx]
    
    def send(self, prompt: str, max_retries: int = 3):
        """Send request to Groq API."""
        
        for attempt in range(max_retries):
            # Rate limiting
            elapsed = time.time() - self.last_request_time
            if elapsed < REQUEST_DELAY_SECONDS:
                time.sleep(REQUEST_DELAY_SECONDS - elapsed)
            
            api_key = self._get_next_key()
            self.last_request_time = time.time()
            
            try:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,  # Low temp for consistency
                        "max_tokens": 100    # Small response needed - just one word
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    content = response.json()["choices"][0]["message"]["content"]
                    return {"success": True, "content": content}
                elif response.status_code == 429:
                    print(f"    ⏳ Rate limited, waiting...")
                    time.sleep(10)
                else:
                    print(f"    ⚠️ API error: {response.status_code}")
                    
            except Exception as e:
                print(f"    ⚠️ Request error: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(3)
        
        return {"success": False, "content": ""}


###############################################################################
# IMPROVED COMPARISON PROMPT - SIMPLE AND FOCUSED
###############################################################################

def create_simple_comparison_prompt(character: str, knowledge: dict, statement: str) -> str:
    """
    Create a balanced comparison prompt that uses training examples to guide classification.
    
    Key design: Use labeled examples to help the model understand the PATTERNS
    of what makes something a contradiction vs consistent.
    """
    
    # Extract the key facts as a simple bulleted list
    facts_list = []
    
    # Handle facts - could be list or dict
    facts_data = knowledge.get("facts", [])
    if isinstance(facts_data, list):
        for fact in facts_data:
            if isinstance(fact, dict):
                subj = fact.get("subject", "")
                pred = fact.get("predicate", "")
                obj = fact.get("object", "")
                if pred and obj:
                    facts_list.append(f"- {subj} {pred}: {obj}")
            elif isinstance(fact, str) and fact:
                facts_list.append(f"- {fact}")
    elif isinstance(facts_data, dict):
        for key, value in facts_data.items():
            if value:
                facts_list.append(f"- {key}: {value}")
    
    # Add timeline events
    for event in knowledge.get("timeline", []):
        if isinstance(event, dict):
            time_ref = event.get("time_reference", event.get("year", ""))
            desc = event.get("event", event.get("description", ""))
            if desc:
                facts_list.append(f"- {time_ref}: {desc}" if time_ref else f"- {desc}")
        elif isinstance(event, str):
            facts_list.append(f"- {event}")
    
    # Add relationships
    for rel in knowledge.get("relationships", []):
        if isinstance(rel, dict):
            target = rel.get("target", rel.get("person", rel.get("name", "")))
            rel_type = rel.get("type", rel.get("relationship", rel.get("relation", "")))
            if target and rel_type:
                facts_list.append(f"- Relationship with {target}: {rel_type}")
        elif isinstance(rel, str):
            facts_list.append(f"- {rel}")
    
    # Add traits
    traits = knowledge.get("traits", [])
    if traits:
        facts_list.append(f"- Character traits: {', '.join(traits[:6])}")
    
    # Add role/summary if available
    if knowledge.get("role"):
        facts_list.append(f"- Role: {knowledge['role']}")
    
    # Add locations
    locations = knowledge.get("locations", [])
    if locations:
        facts_list.append(f"- Known locations: {', '.join(locations[:5])}")
    
    facts_text = "\n".join(facts_list[:25]) if facts_list else "Limited facts from book extraction."
    
    # Extract MORE examples from backstories (labeled training data)
    consistent_examples = []
    contradiction_examples = []
    
    for bs in knowledge.get("backstories", []):
        content = bs.get("content", "")
        original = bs.get("original_content", content)
        is_consistent = bs.get("is_consistent", True)
        
        # Use original content for contradictions (without the prefix)
        clean_content = original.replace("(this did not happen)", "").strip()
        
        if is_consistent and len(consistent_examples) < 3:
            consistent_examples.append(clean_content[:180])
        elif not is_consistent and len(contradiction_examples) < 3:
            contradiction_examples.append(clean_content[:180])
    
    # Build examples section - SHOW BOTH types prominently
    examples_text = ""
    
    if contradiction_examples:
        examples_text += "STATEMENTS THAT ARE **CONTRADICTIONS** (these conflict with established canon):\n"
        for i, ex in enumerate(contradiction_examples, 1):
            examples_text += f"  {i}. \"{ex}...\"\n"
        examples_text += "\n"
    
    if consistent_examples:
        examples_text += "STATEMENTS THAT ARE **CONSISTENT** (these do NOT conflict with canon):\n"
        for i, ex in enumerate(consistent_examples, 1):
            examples_text += f"  {i}. \"{ex}...\"\n"
        examples_text += "\n"
    
    return f"""You are classifying whether a statement about the character "{character}" is a CONTRADICTION or CONSISTENT with the established story.

CHARACTER: {character}

FACTS EXTRACTED FROM THE BOOK:
{facts_text}

TRAINING EXAMPLES FROM THIS CHARACTER:
{examples_text}
NEW STATEMENT TO CLASSIFY:
"{statement}"

CLASSIFICATION GUIDELINES:
A statement is a CONTRADICTION if it:
- States events that CONFLICT with known timeline (wrong dates, impossible sequences)
- Claims meetings/interactions with people the character couldn't have met
- Describes the character in a location they never visited according to canon
- Gives the character relationships/roles that conflict with established facts
- Attributes actions that are OPPOSITE to the character's known personality/values
- Mentions historical events incorrectly placed in the character's timeline

A statement is CONSISTENT if it:
- Adds plausible backstory that doesn't conflict with known facts
- Describes events that COULD have happened (not contradicted by canon)
- Matches the character's established personality and circumstances
- Is simply unverifiable (no facts to contradict it)

Look at the CONTRADICTION examples above - does the new statement follow similar patterns?
Look at the CONSISTENT examples above - does the new statement follow similar patterns?

Your classification (exactly one word - CONTRADICTION or CONSISTENT):"""


###############################################################################
# LOAD EXISTING KNOWLEDGE
###############################################################################

def load_combined_knowledge():
    """Load all pre-built combined knowledge files."""
    knowledge = {}
    
    if not COMBINED_KNOWLEDGE_FOLDER.exists():
        print(f"⚠️ Combined knowledge folder not found: {COMBINED_KNOWLEDGE_FOLDER}")
        return knowledge
    
    for json_file in COMBINED_KNOWLEDGE_FOLDER.glob("*_combined.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract character name from filename (e.g., "Thalcave_combined.json" -> "Thalcave")
            char_name = json_file.stem.replace("_combined", "")
            
            # Handle different name formats
            # Some test.csv entries use "Tom Ayrton/Ben Joyce" 
            if "Tom_Ayrton" in char_name:
                knowledge["Tom Ayrton/Ben Joyce"] = data
                knowledge["Tom Ayrton"] = data
                knowledge["Ayrton"] = data
            elif "Kai_Koumou" in char_name:
                knowledge["Kai-Koumou"] = data
                knowledge["Kai Koumou"] = data
            else:
                # Replace underscores with spaces for matching
                knowledge[char_name.replace("_", " ")] = data
                knowledge[char_name] = data
            
            print(f"  ✓ Loaded knowledge for: {char_name}")
            
        except Exception as e:
            print(f"  ⚠️ Error loading {json_file}: {e}")
    
    return knowledge


def load_test_data():
    """Load test.csv data."""
    data = []
    
    with open(TEST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Map column names
            entry = {
                "id": row.get("id", ""),
                "book": row.get("book_name", row.get("book", "")),
                "char": row.get("char", row.get("character", "")),
                "caption": row.get("caption", ""),
                "content": row.get("content", row.get("backstory", ""))
            }
            data.append(entry)
    
    print(f"  ✓ Loaded {len(data)} test entries")
    return data


###############################################################################
# PREDICTION
###############################################################################

def predict_all(test_data: list, knowledge: dict, client: SimpleGroqClient):
    """Run predictions on all test data."""
    
    predictions = []
    total = len(test_data)
    
    for idx, entry in enumerate(test_data):
        entry_id = entry["id"]
        char_name = entry["char"]
        statement = entry["content"]
        
        print(f"\n[{idx+1}/{total}] {char_name} (ID: {entry_id})")
        print(f"  Statement: {statement[:80]}...")
        
        # Find matching knowledge
        char_knowledge = None
        for key in knowledge:
            if key.lower() == char_name.lower() or char_name.lower() in key.lower() or key.lower() in char_name.lower():
                char_knowledge = knowledge[key]
                break
        
        if not char_knowledge:
            print(f"  ⚠️ No knowledge found for '{char_name}', defaulting to consistent")
            predictions.append({
                "id": entry_id,
                "Name": char_name,
                "Consistent": "consistent"
            })
            continue
        
        # Create prompt and get response
        prompt = create_simple_comparison_prompt(char_name, char_knowledge, statement)
        response = client.send(prompt)
        
        if response["success"]:
            answer = response["content"].strip().upper()
            
            # Parse the response - look for CONTRADICTION or CONSISTENT
            if "CONTRADICTION" in answer:
                result = "contradict"
                print(f"  ❌ CONTRADICTION")
            else:
                result = "consistent"
                print(f"  ✓ CONSISTENT")
        else:
            # Default to consistent on API failure
            result = "consistent"
            print(f"  ⚠️ API failed, defaulting to consistent")
        
        predictions.append({
            "id": entry_id,
            "Name": char_name,
            "Consistent": result
        })
    
    return predictions


def save_predictions(predictions: list):
    """Save predictions to CSV."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_FOLDER / f"predictions_{timestamp}.csv"
    
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "Name", "Consistent"])
        writer.writeheader()
        writer.writerows(predictions)
    
    print(f"\n✓ Saved predictions to: {output_file}")
    
    # Also save to predict.csv (overwrite)
    predict_file = OUTPUT_FOLDER / "predict.csv"
    with open(predict_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "Name", "Consistent"])
        writer.writeheader()
        writer.writerows(predictions)
    
    print(f"✓ Updated: {predict_file}")
    
    return output_file


###############################################################################
# MAIN
###############################################################################

def main():
    print("=" * 60)
    print("PREDICTION-ONLY PIPELINE")
    print("Uses existing combined knowledge - NO new extraction requests")
    print("=" * 60)
    
    # Load existing knowledge
    print("\n📚 Loading pre-built combined knowledge...")
    knowledge = load_combined_knowledge()
    
    if not knowledge:
        print("❌ No combined knowledge found! Run extraction first.")
        return
    
    print(f"\n📊 Loaded knowledge for {len(knowledge)} character variants")
    
    # Load test data
    print("\n📋 Loading test data...")
    test_data = load_test_data()
    
    # Create client
    print("\n🔑 Initializing Groq client...")
    client = SimpleGroqClient()
    
    # Run predictions
    print("\n🔮 Running predictions...")
    predictions = predict_all(test_data, knowledge, client)
    
    # Save results
    print("\n💾 Saving predictions...")
    save_predictions(predictions)
    
    # Summary
    consistent_count = sum(1 for p in predictions if p["Consistent"] == "consistent")
    contradict_count = sum(1 for p in predictions if p["Consistent"] == "contradict")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total predictions: {len(predictions)}")
    print(f"  Consistent: {consistent_count}")
    print(f"  Contradict: {contradict_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
