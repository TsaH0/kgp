import os
from extraction import ExtractionProcessor

# Set your Groq API key
os.environ["GROQ_API_KEY"] = "your_api_key_here"

# Initialize processor
processor = ExtractionProcessor(
    groq_model="mixtral-8x7b-32768",
    output_dir="/home/Tejesh/Documents/kgph_pathway/output"
)

# Example: Process a single chunk
chunk_text = """
Harry Potter was a young wizard who lived with his aunt and uncle. 
He had messy black hair and wore round glasses. His parents had died 
when he was very young, killed by the dark wizard Voldemort.
"""

character_names = ["Harry Potter", "Hermione Granger", "Ron Weasley"]

backstory = {
    "name": "Harry Potter",
    "role": "student",
    "traits": ["brave", "loyal"],
    "claims": [
        {
            "subject": "Harry Potter",
            "predicate": "has_hair_color",
            "object": "brown",
            "scope": "permanent"
        }
    ]
}

result = processor.process_chunk(
    chunk_id="hp_chapter1_001",
    chunk_text=chunk_text,
    character_names=character_names,
    backstory=backstory
)

print("\n" + "="*80)
print("FINAL RESULT")
print("="*80)
print(f"Success: {result['success']}")
print(f"Files saved to: {result['output_paths']}")

if result['extraction']:
    print(f"\nMatched Character: {result['extraction']['meta']['matched_character']}")
    print(f"Claims Extracted: {len(result['extraction']['claims'])}")
    print(f"Contradictions Found: {len(result['extraction']['contradictions'])}")
