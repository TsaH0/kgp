# Knowledge Graph Extraction System

Structured knowledge extraction and contradiction detection using Groq API and Pathway framework.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. API Key is Already Configured

✅ Your API key is already set in the `.env` file.

No additional setup needed!

### 3. Prepare Book Data

Create a `data` folder and add your book text files:

```bash
mkdir -p data
cp /path/to/your/book.txt data/
```

Supported format: `.txt` files with plain text

### 4. Run Extraction Pipeline

**Full Pathway Pipeline** (ingests all books in data folder):

```bash
python run_extraction.py
```

**Single Example Mode** (test with sample text):

```bash
python run_extraction.py --example
```

## ⚠️ Security Note

Your API key is stored in `.env` file which is:

- ✅ Automatically loaded by the script
- ✅ Excluded from git (via .gitignore)
- ⚠️ Keep this file secure and don't share it

## How It Works

### Pathway Ingestion Pipeline

1. **File Discovery**: Automatically discovers `.txt` files in `data/` folder
2. **Text Chunking**: Splits books into overlapping chunks (1000 chars, 200 overlap)
3. **Smart Boundaries**: Breaks at sentence/paragraph boundaries
4. **Streaming**: Processes chunks as they're created

### Extraction Pipeline

1. **Character Matching**: Identifies known characters in each chunk
2. **Sentence Extraction**: Extracts relevant sentences
3. **Fact Generation**: Creates atomic claims
4. **Contradiction Detection**: Compares against backstory
5. **Output Storage**: Saves all artifacts

## Project Structure

```
kgph_pathway/
├── data/                    # Input: Book .txt files
├── output/                  # Output: All extraction results
│   ├── extractions/        # Parsed JSON results
│   ├── requests/           # Prompts sent to API
│   ├── responses/          # Raw API responses
│   └── logs/               # Processing logs
├── extraction/             # Extraction logic
│   ├── groq_client.py     # Groq API client
│   ├── character_extractor.py  # Prompt generation
│   ├── output_manager.py  # File output handler
│   └── processor.py       # Main processor
├── ingestion/             # Pathway ingestion
│   └── pathway_connector.py  # Pathway pipeline
├── config/                # Configuration files
└── run_extraction.py      # Main entry point
```

## Configuration

### Chunk Settings

Edit `run_extraction.py`:

```python
ingestion = BookIngestionPipeline(
    data_folder="data",
    chunk_size=1000,      # Characters per chunk
    overlap=200           # Overlap between chunks
)
```

### Character Names

Update character list in `run_extraction.py`:

```python
character_names = [
    "Harry Potter",
    "Hermione Granger",
    # Add more characters...
]
```

### Groq Model

Change model in `run_extraction.py`:

```python
processor = ExtractionProcessor(
    groq_model="mixtral-8x7b-32768"  # or llama2-70b-4096, etc.
)
```

## Example Output

```
PATHWAY BOOK INGESTION + KNOWLEDGE EXTRACTION
================================================

📚 Found 2 book file(s):
  - harry_potter_ch1.txt
  - harry_potter_ch2.txt

🔧 Initializing Pathway ingestion pipeline...
🔧 Initializing Groq extraction processor...
👥 Tracking 7 characters

🚀 Running Pathway pipeline...

================================================
Processing: harry_potter_ch1_chunk_0001
================================================

GROQ API REQUEST
================================================
Model: mixtral-8x7b-32768
Temperature: 0.1
...

GROQ API RESPONSE
================================================
Status Code: 200
Total Tokens: 2456
...

📁 Saving outputs for chunk: harry_potter_ch1_chunk_0001
✓ Request saved: output/requests/...
✓ Response saved: output/responses/...
✓ Extraction saved: output/extractions/...
✓ Log saved: output/logs/...

✓ PATHWAY PIPELINE COMPLETE
================================================
📁 Check output folder: /home/Tejesh/Documents/kgph_pathway/output
```

## Pathway Features Used

- **File Connectors**: Automatic text file discovery
- **UDFs**: Custom chunking and extraction logic
- **Streaming**: Real-time processing as files are read
- **Table Operations**: Structured data transformation

## Troubleshooting

### No files found

```
⚠️  No .txt files found in data/
```

**Solution**: Add `.txt` book files to the `data/` folder

### Pathway import error

```
ModuleNotFoundError: No module named 'pathway'
```

**Solution**: `pip install pathway>=0.7.0`

### Memory issues with large books

**Solution**: Reduce `chunk_size` or process files one at a time
