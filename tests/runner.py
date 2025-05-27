#!/usr/bin/env python
import os
import subprocess
import sys

def run_python_tests():
    """Run all Python tests"""
    print("Running Python tests...")
    result = subprocess.run(["pytest", "-v"], cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0

def run_rust_tests():
    """Run all Rust tests"""
    print("Running Rust tests...")
    result = subprocess.run(
        ["cargo", "test", "--release"], 
        cwd=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "compiled/rust")
    )
    return result.returncode == 0

def run_benchmarks():
    """Run performance benchmarks"""
    print("Running benchmarks...")
    result = subprocess.run(
        ["python", "tests/benchmarks/benchmark_performance.py"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    return result.returncode == 0

if __name__ == "__main__":
    all_passed = True
    
    # Run Rust tests first
    if not run_rust_tests():
        print("❌ Rust tests failed")
        all_passed = False
    else:
        print("✅ Rust tests passed")
    
    # Run Python tests
    if not run_python_tests():
        print("❌ Python tests failed")
        all_passed = False
    else:
        print("✅ Python tests passed")
    
    # Run benchmarks if tests passed
    if all_passed and "--with-benchmarks" in sys.argv:
        if not run_benchmarks():
            print("❌ Benchmarks failed")
            all_passed = False
        else:
            print("✅ Benchmarks completed")
    
    if all_passed:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("⚠️ Some tests failed")
        sys.exit(1)