"""Lightweight tests for JSON config loading.
Run: python tests_config.py
"""
from config.loader import load_config, ConfigError
import json, os, sys

def test_example_minimal():
    cfg = load_config('config/example_minimal.json')
    assert cfg['years'] == [2019]
    assert 'springrank' in cfg['algorithms'] and cfg['algorithms']['springrank']

def test_example_full():
    cfg = load_config('config/example_full.json')
    assert 2015 in cfg['years'] and 2019 in cfg['years']
    assert 'handmade' in cfg['score_types']
    assert cfg['ranking']['top_n'] == 25

if __name__ == '__main__':
    failures = 0
    for fn in [test_example_minimal, test_example_full]:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except Exception as e:
            failures += 1
            print(f"[FAIL] {fn.__name__}: {e}")
    if failures:
        sys.exit(1)
    print('All tests passed.')
