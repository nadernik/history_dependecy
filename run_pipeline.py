#!/usr/bin/env python3
"""
Simple runner script for the serial dependence analysis
Uses the new modular architecture.
"""

from main import main

if __name__ == "__main__":
    print("🧠 Running serial dependence analysis with modular architecture...")
    analyzer, results = main()
    print("✅ Analysis complete!")