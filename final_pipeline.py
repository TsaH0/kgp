#!/usr/bin/env python3
"""
Final Pipeline: Book Facts (from combined_knowledge) + Train1.csv -> New KB -> Test1.csv
- Reuses extracted book knowledge (facts, timeline, relationships, traits)
- Removes old backstories from train.csv
- Replaces with train1.csv labeled facts
- Predicts test1.csv using Groq
"""

import os
import json
import time
import csv
import re
from pathlib import Path
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Paths
BASE_PATH = Path("/home/Tejesh/Documents/kgph_pathway")
DATA_FOLDER = BASE_PATH / "data"
OUTPUT_FOLDER = BASE_PATH / "output"
OLD_KB_FOLDER = OUTPUT_FOLDER / "combined_knowledge"
NEW_KB_FOLDER = OUTPUT_FOLDER / "new_knowledge_base"
RESPONSES_FOLDER = OUTPUT_FOLDER / "responses_analysis"
TRAIN_CSV = DATA_FOLDER / "train1.csv"
TEST_CSV = DATA_FOLDER / "test1.csv"
PREDICTION_FILE = OUTPUT_FOLDER / "predictions_final.csv"

# Ensure responses folder exists
RESPONSES_FOLDER.mkdir(parents=True, exist_ok=True)

# Ensure output folder exists
NEW_KB_FOLDER.mkdir(parents=True, exist_ok=True)

# Rate limiting
REQUEST_DELAY_SECONDS = 1.0

###############################################################################
# TEXT PREPROCESSING
###############################################################################

def clean_text(text):
    """Clean text by fixing common encoding issues (mojibake)."""
    if not text:
        return ""
    
    # Convert to string if needed
    text = str(text)
    
    # Use regex to remove common mojibake patterns
    # Pattern: â followed by special chars (common UTF-8 misread as Latin-1)
    import re
    
    # Smart quotes and apostrophes
    text = re.sub(r'â€™', "'", text)  # right single quote
    text = re.sub(r'â€˜', "'", text)  # left single quote
    text = re.sub(r'â€œ', '"', text)  # left double quote
    text = re.sub(r'â€[^a-zA-Z0-9\s]', '"', text)  # right double quote and similar
    
    # Dashes
    text = re.sub(r'â€"', '-', text)  # em-dash/en-dash
    
    # Ellipsis
    text = re.sub(r'â€¦', '...', text)
    
    # Fix double-encoded UTF-8 (Ã patterns)
    replacements = [
        ('Ã©', 'é'), ('Ã¨', 'è'), ('Ãª', 'ê'), ('Ã«', 'ë'),
        ('Ã ', 'à'), ('Ã¢', 'â'), ('Ã¤', 'ä'),
        ('Ã®', 'î'), ('Ã¯', 'ï'), ('Ã´', 'ô'), ('Ã¶', 'ö'),
        ('Ã¹', 'ù'), ('Ã»', 'û'), ('Ã¼', 'ü'),
        ('Ã§', 'ç'), ('Ã±', 'ñ'),
        ('Ã‰', 'É'), ('Ãº', 'ú'), ('Ã³', 'ó'), ('Ã­', 'í'), ('Ã¡', 'á'),
    ]
    for bad, good in replacements:
        text = text.replace(bad, good)
    
    # Remove any remaining weird characters
    text = re.sub(r'[^\x00-\x7F]+', lambda m: m.group(0) if all(ord(c) < 256 for c in m.group(0)) else '', text)
    
    # Strip whitespace
    text = text.strip()
    
    return text

###############################################################################
# GROQ CLIENT
###############################################################################

class SimpleGroqClient:
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.idx = 0
        self.last_req = 0
        
    def _load_api_keys(self):
        keys = []
        for k in ["GROQ_API_KEY", "SECOND_API_KEY", "THIRD_API_KEY", "FOURTH_API_KEY", "FIFTH_API_KEY"]:
            val = os.getenv(k)
            if val:
                keys.append(val)
        return keys if keys else ["err"]
    
    def send(self, prompt):
        for attempt in range(3):
            elapsed = time.time() - self.last_req
            if elapsed < REQUEST_DELAY_SECONDS:
                time.sleep(REQUEST_DELAY_SECONDS - elapsed)
            
            key = self.api_keys[self.idx]
            self.idx = (self.idx + 1) % len(self.api_keys)
            self.last_req = time.time()
            
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "max_tokens": 150  # Increased for reasoning
                    },
                    timeout=30
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                if resp.status_code == 429:
                    print(f"    ⏳ Rate limited, waiting...")
                    time.sleep(5)
            except Exception as e:
                print(f"    ⚠️ Error: {e}")
            time.sleep(2)
        return "consistent"

###############################################################################
# STEP 1: BUILD NEW KNOWLEDGE BASE
###############################################################################

def build_new_knowledge_base():
    """
    Build new KB by:
    1. Loading book facts from combined_knowledge (facts, timeline, relationships, traits)
    2. Removing old backstories
    3. Adding train1.csv labeled facts
    """
    print("=" * 60)
    print("BUILDING NEW KNOWLEDGE BASE")
    print("=" * 60)
    
    # 1. Load train1.csv facts
    print("\n📚 Loading train1.csv...")
    train_data = {}  # char -> {"verified": [], "falsehoods": []}
    
    with open(TRAIN_CSV, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            char = clean_text(row["char"])
            content = clean_text(row["content"])
            label = row["label"].strip().lower()
            
            if char not in train_data:
                train_data[char] = {"verified": [], "falsehoods": []}
            
            if label == "consistent":
                train_data[char]["verified"].append(content)
            else:
                train_data[char]["falsehoods"].append(content)
    
    print(f"   Found {len(train_data)} characters in train1.csv")
    
    # 2. Load book facts from combined_knowledge
    print("\n📖 Loading book facts from combined_knowledge...")
    book_data = {}  # name -> {facts, timeline, relationships, traits}
    
    for f in OLD_KB_FOLDER.glob("*_combined.json"):
        try:
            with open(f, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            
            name = clean_text(data.get("name", ""))
            if not name:
                continue
            
            # Extract ONLY book knowledge (NOT backstories)
            book_info = {
                "role": clean_text(data.get("role", "")),
                "facts": [],
                "timeline": [],
                "relationships": [],
                "traits": []
            }
            
            # Clean and extract facts
            raw_facts = data.get("facts", [])
            if isinstance(raw_facts, list):
                for rf in raw_facts[:15]:  # Limit to 15
                    if isinstance(rf, dict) and rf.get("predicate"):
                        s = clean_text(str(rf.get("subject", "")))
                        p = clean_text(str(rf.get("predicate", "")))
                        o = clean_text(str(rf.get("object", "")))
                        book_info["facts"].append(f"{s} {p} {o}")
                    elif isinstance(rf, str):
                        book_info["facts"].append(clean_text(rf))
            
            # Clean timeline
            timeline = data.get("timeline", [])
            for evt in timeline[:10]:  # Limit to 10
                if isinstance(evt, dict):
                    yr = clean_text(str(evt.get("year", evt.get("time_reference", ""))))
                    desc = clean_text(evt.get("event", ""))
                    if desc:
                        book_info["timeline"].append(f"{yr}: {desc}" if yr else desc)
            
            # Clean relationships
            rels = data.get("relationships", [])
            for rel in rels[:10]:  # Limit to 10
                if isinstance(rel, dict):
                    person = clean_text(rel.get("person", ""))
                    reltype = clean_text(rel.get("relationship", ""))
                    if person and reltype:
                        book_info["relationships"].append(f"{reltype}: {person}")
            
            # Clean traits
            traits = data.get("traits", [])
            if isinstance(traits, list):
                for t in traits[:5]:  # Limit to 5
                    book_info["traits"].append(clean_text(str(t)))
            
            book_data[name] = book_info
            
            # Add name variants
            if "Tom" in name or "Ayrton" in name:
                book_data["Tom Ayrton/Ben Joyce"] = book_info
            if "Kai" in name:
                book_data["Kai-Koumou"] = book_info
                
        except Exception as e:
            print(f"   ⚠️ Error loading {f.name}: {e}")
    
    print(f"   Found {len(book_data)} characters in combined_knowledge")
    
    # 3. Merge and create new KB files
    print("\n🔨 Creating new knowledge base files...")
    
    for char, tdata in train_data.items():
        # Find book data for this character
        bdata = None
        for bname in book_data:
            if bname in char or char in bname:
                bdata = book_data[bname]
                break
        
        # Build new KB structure with EXPLICIT labels in the facts
        verified_with_labels = [f"(THIS IS TRUE) {fact}" for fact in tdata["verified"]]
        falsehoods_with_labels = [f"(THIS IS NOT TRUE) {fact}" for fact in tdata["falsehoods"]]
        
        new_kb = {
            "character": char,
            "book_knowledge": {
                "role": bdata["role"] if bdata else "",
                "facts": bdata["facts"] if bdata else [],
                "timeline": bdata["timeline"] if bdata else [],
                "relationships": bdata["relationships"] if bdata else [],
                "traits": bdata["traits"] if bdata else []
            },
            "train1_verified_facts": verified_with_labels,
            "train1_known_falsehoods": falsehoods_with_labels
        }
        
        # Save
        safe_name = char.replace("/", "_").replace("\\", "_")
        out_path = NEW_KB_FOLDER / f"{safe_name}.json"
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(new_kb, f, indent=2, ensure_ascii=False)
        
        book_count = len(new_kb["book_knowledge"]["facts"]) + len(new_kb["book_knowledge"]["timeline"])
        print(f"   ✓ {safe_name}.json (Book: {book_count}, Verified: {len(tdata['verified'])}, Falsehoods: {len(tdata['falsehoods'])})")
    
    print("\n✅ New knowledge base created successfully!")

###############################################################################
# STEP 2: PREDICT TEST FILES
###############################################################################

def predict_test(test_file, output_file):
    """Predict a test file using the new knowledge base and Groq."""
    print("\n" + "=" * 60)
    print(f"PREDICTING {test_file.name}")
    print("=" * 60)
    
    client = SimpleGroqClient()
    predictions = []
    
    # Load new KB
    print("\n📂 Loading new knowledge base...")
    kb = {}
    for f in NEW_KB_FOLDER.glob("*.json"):
        with open(f, "r", encoding="utf-8") as jf:
            data = json.load(jf)
        name = data["character"]
        kb[name] = data
        # Add variants
        if "Tom Ayrton" in name:
            kb["Ayrton"] = data
        if "Kai" in name:
            kb["Kai-Koumou"] = data
    
    print(f"   Loaded {len(kb)} character knowledge bases")
    
    # Load and predict test data
    print(f"\n🔮 Running predictions on {test_file.name}...\n")
    
    with open(test_file, "r", encoding="utf-8", errors="replace") as f:
        reader = list(csv.DictReader(f))
        total = len(reader)
        
        for i, row in enumerate(reader):
            char = clean_text(row.get("char", ""))
            stmt = clean_text(row.get("content", ""))
            
            # Find KB
            char_kb = None
            for k in kb:
                if k in char or char in k:
                    char_kb = kb[k]
                    break
            
            if not char_kb:
                res = "consistent"  # No info = cannot contradict
            else:
                # Build context from KB
                bk = char_kb["book_knowledge"]
                
                # Book facts
                book_lines = []
                if bk["role"]:
                    book_lines.append(f"Role: {bk['role']}")
                book_lines.extend(bk["facts"][:10])
                book_lines.extend(bk["timeline"][:5])
                book_text = "\n".join([f"- {x}" for x in book_lines]) if book_lines else "No book facts."
                
                # Verified history
                verified = char_kb["train1_verified_facts"]
                verified_text = "\n".join([f"- {x[:200]}" for x in verified[:10]]) if verified else "None."
                
                # Known falsehoods
                falsehoods = char_kb["train1_known_falsehoods"]
                false_text = "\n".join([f"- {x[:200]}" for x in falsehoods[:10]]) if falsehoods else "None."
                
                # IMPROVED Prompt (same as benchmark)
                prompt = f"""You are a fact-checker. Determine if the NEW STATEMENT about {char} is CONSISTENT or a CONTRADICTION.

BOOK KNOWLEDGE:
{book_text}

VERIFIED TRUE FACTS:
{verified_text}

KNOWN FALSE CLAIMS:
{false_text}

NEW STATEMENT:
"{stmt[:400]}"

IMPORTANT RULES:
1. ONLY mark as CONTRADICTION if you can cite SPECIFIC conflicting evidence from the knowledge base.
2. New information that is NOT mentioned in the knowledge base is NOT a contradiction - it's just new info.
3. Vague similarities to false claims are NOT enough - there must be a DIRECT match or conflict.
4. When in doubt, mark as CONSISTENT.

Respond in this format:
VERDICT: [CONSISTENT or CONTRADICTION]
EVIDENCE: [Quote the specific fact from the knowledge base that proves your verdict.]"""

                resp = client.send(prompt).strip()
                
                # Parse verdict from response
                if "VERDICT:" in resp.upper():
                    verdict_line = resp.upper().split("VERDICT:")[1].split("\n")[0].strip()
                    res = "contradict" if "CONTRADICTION" in verdict_line else "consistent"
                else:
                    res = "contradict" if "CONTRADICTION" in resp.upper() else "consistent"
            
            print(f"[{i+1}/{total}] {char}: {res}")
            predictions.append({"id": row["id"], "Name": char, "Consistent": res})
    
    # Save predictions
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "Name", "Consistent"])
        writer.writeheader()
        writer.writerows(predictions)
    
    # Summary
    consistent_count = sum(1 for p in predictions if p["Consistent"] == "consistent")
    contradict_count = sum(1 for p in predictions if p["Consistent"] == "contradict")
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Predictions: {len(predictions)}")
    print(f"Consistent: {consistent_count}")
    print(f"Contradict: {contradict_count}")
    print(f"\n✅ Saved to {PREDICTION_FILE}")

###############################################################################
# STEP 3: BENCHMARK ON TRAIN1.CSV
###############################################################################

def benchmark_train():
    """Run predictions on train1.csv and compare with actual labels."""
    print("\n" + "=" * 60)
    print("BENCHMARKING ON TRAIN1.CSV")
    print("=" * 60)
    
    client = SimpleGroqClient()
    
    # Load new KB
    print("\n📂 Loading new knowledge base...")
    kb = {}
    for f in NEW_KB_FOLDER.glob("*.json"):
        with open(f, "r", encoding="utf-8") as jf:
            data = json.load(jf)
        name = data["character"]
        kb[name] = data
        if "Tom Ayrton" in name:
            kb["Ayrton"] = data
        if "Kai" in name:
            kb["Kai-Koumou"] = data
    
    print(f"   Loaded {len(kb)} character knowledge bases")
    
    # Run predictions on train1.csv
    print("\n🔮 Running benchmark predictions...\n")
    
    y_true = []
    y_pred = []
    response_log = []  # Log all responses for analysis
    
    with open(TRAIN_CSV, "r", encoding="utf-8", errors="replace") as f:
        reader = list(csv.DictReader(f))
        total = len(reader)
        
        for i, row in enumerate(reader):
            char = clean_text(row.get("char", ""))
            stmt = clean_text(row.get("content", ""))
            true_label = row.get("label", "consistent").strip().lower()
            row_id = row.get("id", str(i))
            
            # Find KB
            char_kb = None
            for k in kb:
                if k in char or char in k:
                    char_kb = kb[k]
                    break
            
            if not char_kb:
                pred = "consistent"
                reasoning = "No knowledge base found for this character."
            else:
                # Build context from KB
                bk = char_kb["book_knowledge"]
                
                book_lines = []
                if bk["role"]:
                    book_lines.append(f"Role: {bk['role']}")
                book_lines.extend(bk["facts"][:10])
                book_lines.extend(bk["timeline"][:5])
                book_text = "\n".join([f"- {x}" for x in book_lines]) if book_lines else "No book facts."
                
                verified = char_kb["train1_verified_facts"]
                verified_text = "\n".join([f"- {x[:200]}" for x in verified[:10]]) if verified else "None."
                
                falsehoods = char_kb["train1_known_falsehoods"]
                false_text = "\n".join([f"- {x[:200]}" for x in falsehoods[:10]]) if falsehoods else "None."
                
                # IMPROVED PROMPT - More conservative, requires concrete evidence
                prompt = f"""You are a fact-checker. Determine if the NEW STATEMENT about {char} is CONSISTENT or a CONTRADICTION.

BOOK KNOWLEDGE:
{book_text}

VERIFIED TRUE FACTS:
{verified_text}

KNOWN FALSE CLAIMS:
{false_text}

NEW STATEMENT:
"{stmt[:400]}"

IMPORTANT RULES:
1. ONLY mark as CONTRADICTION if you can cite SPECIFIC conflicting evidence from the knowledge base.
2. New information that is NOT mentioned in the knowledge base is NOT a contradiction - it's just new info.
3. Vague similarities to false claims are NOT enough - there must be a DIRECT match or conflict.
4. When in doubt, mark as CONSISTENT.

Respond in this format:
VERDICT: [CONSISTENT or CONTRADICTION]
EVIDENCE: [Quote the specific fact from the knowledge base that proves your verdict. If CONSISTENT, say "No conflicting evidence found" or cite supporting fact.]"""

                resp = client.send(prompt).strip()
                reasoning = resp
                
                # Parse verdict from response
                if "VERDICT:" in resp.upper():
                    verdict_line = resp.upper().split("VERDICT:")[1].split("\n")[0].strip()
                    pred = "contradict" if "CONTRADICTION" in verdict_line else "consistent"
                else:
                    pred = "contradict" if "CONTRADICTION" in resp.upper() else "consistent"
            
            y_true.append(true_label)
            y_pred.append(pred)
            
            # Log response for analysis
            response_log.append({
                "id": row_id,
                "character": char,
                "statement": stmt[:200],
                "actual_label": true_label,
                "predicted": pred,
                "correct": pred == true_label,
                "reasoning": reasoning
            })
            
            match = "✓" if pred == true_label else "❌"
            print(f"[{i+1}/{total}] {char}: Pred={pred}, Actual={true_label} {match}")
    
    # Calculate metrics
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "contradict" and p == "contradict")
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == "consistent" and p == "consistent")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == "consistent" and p == "contradict")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "contradict" and p == "consistent")
    
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total Samples: {total}")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 Score:  {f1:.2%}")
    print("-" * 30)
    print("Confusion Matrix:")
    print(f"  True Positives (Contradictions caught):  {tp}")
    print(f"  False Negatives (Missed contradictions): {fn}")
    print(f"  True Negatives (Correctly consistent):   {tn}")
    print(f"  False Positives (False alarms):          {fp}")
    
    # Save response log for analysis
    log_file = RESPONSES_FOLDER / f"benchmark_log_{int(time.time())}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(response_log, f, indent=2, ensure_ascii=False)
    print(f"\n📝 Response log saved to {log_file}")
    
    # Print wrong predictions for analysis
    wrong_predictions = [r for r in response_log if not r["correct"]]
    if wrong_predictions:
        print(f"\n⚠️ WRONG PREDICTIONS ({len(wrong_predictions)}):")
        for wp in wrong_predictions[:5]:  # Show first 5
            print(f"\n  ID: {wp['id']} | {wp['character']}")
            print(f"  Statement: {wp['statement'][:100]}...")
            print(f"  Actual: {wp['actual_label']} | Predicted: {wp['predicted']}")
            print(f"  Reasoning: {wp['reasoning'][:150]}...")

###############################################################################
# MAIN
###############################################################################

def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        # Just run benchmark, skip KB rebuild
        benchmark_train()
    elif len(sys.argv) > 1 and sys.argv[1] == "predict":
        # Just run predictions (skip KB rebuild)
        test1_file = DATA_FOLDER / "test1.csv"
        test_file = DATA_FOLDER / "test.csv"
        prediction1_file = OUTPUT_FOLDER / "prediction1.csv"
        prediction_file = OUTPUT_FOLDER / "prediction.csv"
        
        print("\n" + "#" * 60)
        print("RUNNING PREDICTIONS ON BOTH TEST FILES")
        print("#" * 60)
        
        # Predict test1.csv -> prediction1.csv
        predict_test(test1_file, prediction1_file)
        
        # Predict test.csv -> prediction.csv
        predict_test(test_file, prediction_file)
        
        print("\n" + "#" * 60)
        print("ALL PREDICTIONS COMPLETE")
        print("#" * 60)
        print(f"\n📁 prediction1.csv: {prediction1_file}")
        print(f"📁 prediction.csv:  {prediction_file}")
    else:
        build_new_knowledge_base()
        test1_file = DATA_FOLDER / "test1.csv"
        prediction1_file = OUTPUT_FOLDER / "prediction1.csv"
        predict_test(test1_file, prediction1_file)

if __name__ == "__main__":
    main()
