#!/usr/bin/env python
"""Run all tests for the project."""

import subprocess
import sys

def run_tests():
    """Run all tests."""
    print("="*60)
    print("Running All Tests")
    print("="*60)
    
    # Quick test
    print("\n[1/2] Quick functionality test...")
    result1 = subprocess.run([sys.executable, 'scripts/quick_test.py'])
    
    # Unit tests
    print("\n[2/2] Unit tests...")
    result2 = subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-v'])
    
    print("\n" + "="*60)
    if result1.returncode == 0 and result2.returncode == 0:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*60)
    
    return result1.returncode == 0 and result2.returncode == 0

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
