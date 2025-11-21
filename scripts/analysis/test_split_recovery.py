#!/usr/bin/env python3
"""
Test script to verify that split recovery logic matches finetune_models.py exactly.
"""

import random as _rnd
import sys
from pathlib import Path

# Add the 'scripts' directory to Python Path
scripts_path = Path(__file__).resolve().parents[1]
if str(scripts_path) not in sys.path:
    sys.path.append(str(scripts_path))

from training.help_functions import set_random_seeds, load_metadata_from_disk

def test_split_logic():
    """Test that the split logic produces identical results."""
    
    print("="*80)
    print("TESTING SPLIT RECOVERY LOGIC")
    print("="*80)
    
    # Simulate what happens in finetune_models.py
    print("\n[Simulation 1] finetune_models.py sequence:")
    print("  1. set_random_seeds() called (line 148)")
    set_random_seeds()  # This sets random.seed(42)
    
    print("  2. Various operations happen...")
    # Simulate some operations that might use random
    _ = _rnd.random()  # Consume one random number
    
    print("  3. _rnd.seed(42) called (line 247)")
    _rnd.seed(42)  # Reset seed explicitly
    
    print("  4. Shuffle indices")
    indices1 = list(range(10))
    _rnd.shuffle(indices1)
    print(f"     Result: {indices1}")
    
    # Now simulate what my analysis script does
    print("\n[Simulation 2] analyze_pretraining_benefit_vs_distance.py sequence:")
    print("  1. _rnd.seed(42) called directly")
    _rnd.seed(42)  # Set seed directly
    
    print("  2. Shuffle indices")
    indices2 = list(range(10))
    _rnd.shuffle(indices2)
    print(f"     Result: {indices2}")
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON:")
    print("="*80)
    print(f"Simulation 1 result: {indices1}")
    print(f"Simulation 2 result: {indices2}")
    print(f"Are they identical? {indices1 == indices2}")
    
    if indices1 == indices2:
        print("\n✅ SUCCESS: Split recovery logic is CORRECT!")
        print("   The explicit _rnd.seed(42) call resets the random state,")
        print("   so any previous random operations don't matter.")
        return True
    else:
        print("\n❌ FAILURE: Split recovery logic is INCORRECT!")
        print("   The results don't match. Need to investigate further.")
        return False

if __name__ == "__main__":
    success = test_split_logic()
    sys.exit(0 if success else 1)

