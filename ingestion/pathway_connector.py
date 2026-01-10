import pathway as pw
from typing import Optional, List
import os
from pathlib import Path


class BookIngestionPipeline:
    """Pathway-based pipeline for ingesting book text files."""
    
    def __init__(self, data_folder: str, chunk_size: int = 1000, overlap: int = 200):
        self.data_folder = Path(data_folder)
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        if not self.data_folder.exists():
            raise ValueError(f"Data folder does not exist: {data_folder}")
    
    def create_text_source(self):
        """Create Pathway connector for text files."""
        return pw.io.fs.read(
            path=str(self.data_folder),
            format="plaintext",
            mode="static",
            with_metadata=True
        )
    
    def chunk_text(self, text: str, filename: str) -> List[dict]:
        """Split text into overlapping chunks."""
        chunks = []
        text_length = len(text)
        
        if text_length == 0:
            return chunks
        
        start = 0
        chunk_num = 0
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk_text = text[start:end]
            
            # Try to break at sentence boundaries
            if end < text_length:
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size * 0.7:
                    end = start + break_point + 1
                    chunk_text = text[start:end]
            
            chunks.append({
                "chunk_id": f"{filename}_chunk_{chunk_num:04d}",
                "text": chunk_text.strip(),
                "filename": filename,
                "start_pos": start,
                "end_pos": end,
                "chunk_num": chunk_num
            })
            
            chunk_num += 1
            start = end - self.overlap if end < text_length else text_length
        
        return chunks
    
    def process_books(self) -> pw.Table:
        """Create Pathway table with chunked book data."""
        
        # Read text files
        text_source = self.create_text_source()
        
        # Define chunking UDF
        @pw.udf
        def create_chunks(data, path) -> List[dict]:
            """UDF to chunk text from each file. Cast Json inputs to str."""
            text_content = str(data)
            filepath = str(path)
            filename = Path(filepath).stem
            return self.chunk_text(text_content, filename)
        
        # Apply chunking
        chunked = text_source.select(
            chunks=create_chunks(pw.this.data, pw.this._metadata["path"])
        )
        
        # Flatten chunks into rows
        result = chunked.flatten(pw.this.chunks).select(
            chunk_id=pw.this.chunks["chunk_id"],
            text=pw.this.chunks["text"],
            filename=pw.this.chunks["filename"],
            chunk_num=pw.this.chunks["chunk_num"],
            start_pos=pw.this.chunks["start_pos"],
            end_pos=pw.this.chunks["end_pos"]
        )
        
        return result


class PathwayExtractionConnector:
    """Connect Pathway ingestion with Groq extraction."""
    
    def __init__(self, ingestion_pipeline: BookIngestionPipeline, 
                 extraction_processor, character_names: List[str]):
        self.ingestion = ingestion_pipeline
        self.processor = extraction_processor
        self.character_names = character_names
    
    def run_pipeline(self) -> pw.Table:
        """Run complete ingestion + extraction pipeline."""
        
        # Get chunked books
        chunks_table = self.ingestion.process_books()
        
        # Define extraction UDF
        @pw.udf
        def extract_knowledge(chunk_id: str, text: str) -> dict:
            """UDF to extract knowledge from each chunk."""
            result = self.processor.process_chunk(
                chunk_id=chunk_id,
                chunk_text=text,
                character_names=self.character_names,
                backstory=None
            )
            return result
        
        # Apply extraction to each chunk
        extracted = chunks_table.select(
            chunk_id=pw.this.chunk_id,
            filename=pw.this.filename,
            extraction=extract_knowledge(pw.this.chunk_id, pw.this.text)
        )
        
        return extracted
