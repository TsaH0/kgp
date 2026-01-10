from typing import Optional, List
from .groq_client import GroqClient
from .output_manager import OutputManager
from .character_extractor import (
    create_extraction_prompt,
    validate_extraction_output,
    create_empty_result
)


class ExtractionProcessor:
    def __init__(self, 
                 groq_api_key: Optional[str] = None,
                 groq_model: str = "mixtral-8x7b-32768",
                 output_dir: str = "/home/Tejesh/Documents/kgph_pathway/output"):
        self.groq_client = GroqClient(api_key=groq_api_key, model=groq_model)
        self.output_manager = OutputManager(base_output_dir=output_dir)
    
    def process_chunk(self,
                     chunk_id: str,
                     chunk_text: str,
                     character_names: List[str],
                     backstory: Optional[dict] = None) -> dict:
        """Process a single text chunk through the extraction pipeline."""
        
        print(f"\n{'='*80}")
        print(f"PROCESSING CHUNK: {chunk_id}")
        print(f"{'='*80}\n")
        
        # Step 1: Create prompt
        prompt = create_extraction_prompt(
            chunk_id=chunk_id,
            character_names=character_names,
            chunk_text=chunk_text,
            backstory=backstory
        )
        
        # Step 2: Send to Groq
        response = self.groq_client.send_extraction_request(prompt)
        
        # Step 3: Validate and parse
        extraction_data = None
        validation_error = None
        
        if response["success"]:
            is_valid, extraction_data, error_msg = validate_extraction_output(
                response["content"]
            )
            
            if not is_valid:
                print(f"\n⚠️  Validation Error: {error_msg}")
                validation_error = error_msg
                extraction_data = create_empty_result(chunk_id)
        else:
            print(f"\n❌ API request failed")
            extraction_data = create_empty_result(chunk_id)
        
        # Step 4: Save all outputs
        metadata = {
            "character_names": character_names,
            "backstory_provided": backstory is not None,
            "validation_passed": validation_error is None,
            "validation_error": validation_error
        }
        
        output_paths = self.output_manager.save_all(
            chunk_id=chunk_id,
            prompt=prompt,
            response=response,
            extraction_data=extraction_data,
            metadata=metadata
        )
        
        return {
            "chunk_id": chunk_id,
            "success": response["success"] and validation_error is None,
            "extraction": extraction_data,
            "output_paths": output_paths,
            "validation_error": validation_error
        }
    
    def process_batch(self,
                     chunks: List[dict],
                     character_names: List[str],
                     backstories: Optional[dict] = None) -> List[dict]:
        """Process multiple chunks in batch."""
        results = []
        
        for i, chunk in enumerate(chunks, 1):
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            chunk_text = chunk.get("text", "")
            character = chunk.get("character")
            
            backstory = None
            if backstories and character:
                backstory = backstories.get(character)
            
            result = self.process_chunk(
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                character_names=character_names,
                backstory=backstory
            )
            
            results.append(result)
        
        return results
