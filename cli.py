"""CLI entrypoint with subcommands for MLB-Network-Analysis pipeline.

Commands:
  scrape  - ensure raw statcast data present for configured years
  edges   - generate edge-only and unipartite edge lists
  rank    - compute rankings (assumes edges exist)
  full    - run full pipeline (equivalent to previous run_config)

Usage:
  python cli.py --config path/to/config.json full
"""
from __future__ import annotations
import argparse
from config.loader import load_config
from pipeline import run_pipeline, generate_edges, compute_rankings, ensure_scraped


def cmd_scrape(cfg):
    ensure_scraped(cfg['years'], cfg['paths']['raw_data_dir'], cfg['scrape']['force'], cfg['logging']['progress'])


def cmd_edges(cfg):
    generate_edges(cfg)


def cmd_rank(cfg):
    compute_rankings(cfg)


def cmd_full(cfg):
    run_pipeline(cfg)


def main():
    ap = argparse.ArgumentParser(description='MLB Network Analysis CLI')
    ap.add_argument('--config','-c', required=True, help='Path to JSON config')
    ap.add_argument('--force-edges', action='store_true', help='Force regeneration of edges and unipartite outputs (overrides config)')
    ap.add_argument('command', choices=['scrape','edges','rank','full'], help='Subcommand to execute')
    args = ap.parse_args()
    cfg = load_config(args.config)
    # Allow runtime override to force edge regeneration
    if args.force_edges:
        cfg.setdefault('edges', {})
        cfg['edges']['force'] = True
    if args.command == 'scrape':
        cmd_scrape(cfg)
    elif args.command == 'edges':
        cmd_edges(cfg)
    elif args.command == 'rank':
        cmd_rank(cfg)
    else:
        cmd_full(cfg)

if __name__ == '__main__':
    main()
