#!/usr/bin/env python3
"""
Simple entry point for CANDLE/FCIS.
Usage: python run.py [config_path]
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import CANDLEPipeline


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    
    print("=" * 70)
    print("CANDLE/FCIS - Financial Causal Inference System")
    print("=" * 70)
    print(f"Config: {config_path}")
    print()
    
    try:
        pipeline = CANDLEPipeline(config_path)
        summary = pipeline.run_full_pipeline()
        
        print("\n" + "=" * 70)
        print("EXECUTION COMPLETE")
        print("=" * 70)
        print(f"\nResults saved in:")
        for name, path in summary['output_paths'].items():
            print(f"  • {name}: {path}")
        
        print(f"\nKey Metrics:")
        print(f"  • Data points: {summary['data_points']}")
        print(f"  • Variables: {summary['n_variables']}")
        print(f"  • Causal edges: {summary['n_causal_edges']}")
        print(f"  • Duration: {summary['duration_seconds']:.1f} seconds")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure config.yaml exists or provide a valid path:")
        print("  python run.py path/to/config.yaml")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
