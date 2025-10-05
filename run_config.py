#!/usr/bin/env python
"""Command-line entrypoint to run MLB Network Analysis with a JSON config.

Usage (PowerShell):
  python run_config.py --config config/example_minimal.json
"""
from config.loader import load_config
from pipeline import run_pipeline
import argparse

def main():
    ap = argparse.ArgumentParser(description='Run MLB Network Analysis pipeline via JSON config')
    ap.add_argument('--config','-c', required=True, help='Path to JSON config file')
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_pipeline(cfg)

if __name__ == '__main__':
    main()
