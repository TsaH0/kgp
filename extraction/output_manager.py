import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional


class OutputManager:
    def __init__(self, base_output_dir: str = "/home/Tejesh/Documents/kgph_pathway/output"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.extractions_dir = self.base_output_dir / "extractions"
        self.requests_dir = self.base_output_dir / "requests"
        self.responses_dir = self.base_output_dir / "responses"
        self.logs_dir = self.base_output_dir / "logs"
        
        for dir_path in [self.extractions_dir, self.requests_dir, 
                         self.responses_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def save_request(self, chunk_id: str, prompt: str, metadata: Optional[dict] = None) -> str:
        """Save the request prompt."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{chunk_id}_{timestamp}_request.txt"
        filepath = self.requests_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            if metadata:
                f.write("=== METADATA ===\n")
                f.write(json.dumps(metadata, indent=2))
                f.write("\n\n")
            
            f.write("=== PROMPT ===\n")
            f.write(prompt)
        
        print(f"✓ Request saved: {filepath}")
        return str(filepath)
    
    def save_response(self, chunk_id: str, response: dict) -> str:
        """Save the raw API response."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{chunk_id}_{timestamp}_response.json"
        filepath = self.responses_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(response, f, indent=2)
        
        print(f"✓ Response saved: {filepath}")
        return str(filepath)
    
    def save_extraction(self, chunk_id: str, extraction_data: dict) -> str:
        """Save the parsed extraction result."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{chunk_id}_{timestamp}_extraction.json"
        filepath = self.extractions_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(extraction_data, f, indent=2)
        
        print(f"✓ Extraction saved: {filepath}")
        return str(filepath)
    
    def save_log(self, chunk_id: str, log_data: dict) -> str:
        """Save processing log."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{chunk_id}_{timestamp}_log.json"
        filepath = self.logs_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"✓ Log saved: {filepath}")
        return str(filepath)
    
    def save_all(self, chunk_id: str, prompt: str, response: dict, 
                 extraction_data: Optional[dict] = None, 
                 metadata: Optional[dict] = None) -> dict:
        """Save all artifacts for a processing run."""
        print(f"\n📁 Saving outputs for chunk: {chunk_id}")
        print("-" * 60)
        
        paths = {
            "request": self.save_request(chunk_id, prompt, metadata),
            "response": self.save_response(chunk_id, response)
        }
        
        if extraction_data:
            paths["extraction"] = self.save_extraction(chunk_id, extraction_data)
        
        log_data = {
            "chunk_id": chunk_id,
            "timestamp": datetime.now().isoformat(),
            "success": response.get("success", False),
            "files_created": paths,
            "metadata": metadata
        }
        
        paths["log"] = self.save_log(chunk_id, log_data)
        
        print("-" * 60)
        print(f"✓ All outputs saved for {chunk_id}\n")
        
        return paths
