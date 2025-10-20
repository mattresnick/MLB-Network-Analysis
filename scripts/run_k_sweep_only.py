import json
import os
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_k_sweep_only.py <config.json>")
        sys.exit(2)
    cfg_path = sys.argv[1]
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg_raw = json.load(f)
    # The configs in this repo are nested under 'pipeline'
    cfg = cfg_raw.get('pipeline', cfg_raw)
    # Ensure k-sweep is enabled
    cfg.setdefault('analysis', {}).setdefault('k_sweep', {})['enabled'] = True
    # Import pipeline from repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import pipeline as P
    out_dir = cfg['paths']['output_dir']
    formats = cfg.get('output', {}).get('formats', ['csv'])
    P._run_k_sweep_section(cfg, out_dir, formats, True)
    print("k-sweep complete")


if __name__ == '__main__':
    main()
