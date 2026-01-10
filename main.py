#!/usr/bin/env python3
"""
KGPH Pathway - Narrative Consistency Verification System
Main entry point for running the system.

Track A: Systems Reasoning with NLP and Generative AI
Kharagpur Data Science Hackathon 2026

Usage:
    python main.py                  # Run full pipeline
    python main.py --no-llm         # Run without LLM (faster, less accurate)
    python main.py --train-only     # Only run training phase
    python main.py --infer-only     # Only run inference (requires prior training)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path if running as script
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import NarrativeConsistencyPipeline, DataLoader
from src.config import config, DATA_DIR, OUTPUT_DIR


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(OUTPUT_DIR / 'pipeline.log')
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description='KGPH Pathway - Narrative Consistency Verification System'
    )
    
    parser.add_argument(
        '--no-llm', 
        action='store_true',
        help='Run without LLM (uses rule-based fallback)'
    )
    
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Only run training phase'
    )
    
    parser.add_argument(
        '--infer-only',
        action='store_true', 
        help='Only run inference phase'
    )
    
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=DATA_DIR,
        help='Directory containing data files'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=OUTPUT_DIR / 'results.csv',
        help='Output path for results'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='llama3.2',
        help='Ollama model to use (e.g., llama3.2, mistral, qwen)'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Update config if needed
    if args.model:
        config.ollama.model = args.model
    
    # Create pipeline
    use_llm = not args.no_llm
    pipeline = NarrativeConsistencyPipeline(
        system_config=config,
        use_llm=use_llm,
        use_cache=True
    )
    
    try:
        if args.train_only:
            # Training only
            logger.info("Running training phase only...")
            
            pipeline.ingest_narratives(args.data_dir)
            
            train_data = DataLoader.load_train_csv(config.train_csv)
            characters = pipeline._get_unique_characters(train_data)
            pipeline.extract_character_timelines(characters)
            
            metrics = pipeline.train(train_data)
            
            logger.info(f"Training complete. Accuracy: {metrics['accuracy']:.2%}")
            
        elif args.infer_only:
            # Inference only
            logger.info("Running inference phase only...")
            
            if not pipeline.narrative_structures:
                logger.warning("No narrative structures loaded. Running ingestion first...")
                pipeline.ingest_narratives(args.data_dir)
                
                test_data = DataLoader.load_test_csv(config.test_csv)
                characters = pipeline._get_unique_characters(test_data)
                pipeline.extract_character_timelines(characters)
            
            test_data = DataLoader.load_test_csv(config.test_csv)
            results = pipeline.predict(test_data, verbose=args.verbose)
            
            DataLoader.save_results_csv(results, args.output)
            
        else:
            # Full pipeline
            results = pipeline.run_full_pipeline(
                data_dir=args.data_dir,
                output_path=args.output
            )
            
            # Print summary
            consistent = sum(1 for r in results if r.prediction == 1)
            contradictory = len(results) - consistent
            
            print(f"\n{'='*60}")
            print(f"RESULTS SUMMARY")
            print(f"{'='*60}")
            print(f"Total predictions: {len(results)}")
            print(f"  Consistent:    {consistent} ({consistent/len(results)*100:.1f}%)")
            print(f"  Contradictory: {contradictory} ({contradictory/len(results)*100:.1f}%)")
            print(f"\nResults saved to: {args.output}")
            
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
