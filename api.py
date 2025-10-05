"""FastAPI service exposing ranking data.

Endpoints:
  GET /health -> basic health check
  GET /ranks?score_type=handmade&group=batter&year=2019[&pitch_type=FF][&inning=1] -> full ranking table
  GET /top?score_type=handmade&group=batter&year=2019&n=25 -> top N ranks
  GET /player/{name}?score_type=handmade&group=batter -> per-player all years rows
  GET /mobility -> mobility report (if exists)
  GET /anomalies -> anomalies report (if exists)

Run:
  uvicorn api:app --reload
"""
from __future__ import annotations
import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from typing import Optional

OUTPUT_ROOT = os.environ.get('MLB_OUTPUT_DIR','outputs')

app = FastAPI(title='MLB Network Analysis API', version='0.1.0')


def _load_any(path_base: str):
    # Try csv then parquet then json
    csv = path_base + '.csv'
    if os.path.isfile(csv):
        return pd.read_csv(csv)
    pq = path_base + '.parquet'
    if os.path.isfile(pq):
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    js = path_base + '.json'
    if os.path.isfile(js):
        return pd.read_json(js)
    return None

@app.get('/health')
async def health():
    return {'status':'ok'}

@app.get('/ranks')
async def get_ranks(score_type: str, group: str, year: int, pitch_type: Optional[str]=None, inning: Optional[int]=None):
    if score_type not in {'handmade','frequency','pitch_type','inning'}:
        raise HTTPException(400,'invalid score_type')
    if group not in {'batter','pitcher'}:
        raise HTTPException(400,'invalid group')
    base_dir = os.path.join(OUTPUT_ROOT, score_type, group)
    if score_type == 'pitch_type':
        if not pitch_type:
            raise HTTPException(400,'pitch_type required for score_type=pitch_type')
        base_dir = os.path.join(base_dir, pitch_type)
    if score_type == 'inning':
        if inning is None:
            raise HTTPException(400,'inning required for score_type=inning')
        base_dir = os.path.join(base_dir, str(inning))
    df = _load_any(os.path.join(base_dir, f"{year}_springrank"))
    if df is None:
        raise HTTPException(404,'ranking not found')
    return df.to_dict(orient='records')

@app.get('/top')
async def get_top(score_type: str, group: str, year: int, n: int=25, pitch_type: Optional[str]=None, inning: Optional[int]=None):
    df = await get_ranks(score_type, group, year, pitch_type, inning)  # type: ignore
    # df is already list of dicts; slice
    return df[:n]

@app.get('/player/{name}')
async def player_history(name: str, score_type: str, group: str):
    # Search across all year files for player occurrences
    base_dir = os.path.join(OUTPUT_ROOT, score_type, group)
    results = []
    if not os.path.isdir(base_dir):
        raise HTTPException(404,'no data for those parameters')
    for fname in os.listdir(base_dir):
        if fname.endswith('_springrank.csv') or fname.endswith('_springrank.parquet') or fname.endswith('_springrank.json'):
            year = fname.split('_')[0]
            df = _load_any(os.path.join(base_dir, fname.rsplit('.',1)[0]))
            if df is not None:
                row = df[df['Player'].str.lower()==name.lower()]
                if not row.empty:
                    r = row.iloc[0].to_dict()
                    r['Year'] = int(year)
                    results.append(r)
    if not results:
        raise HTTPException(404,'player not found')
    return results

@app.get('/mobility')
async def mobility():
    df = _load_any(os.path.join(OUTPUT_ROOT,'mobility_report'))
    if df is None:
        raise HTTPException(404,'mobility report not found')
    return df.to_dict(orient='records')

@app.get('/anomalies')
async def anomalies():
    df = _load_any(os.path.join(OUTPUT_ROOT,'anomalies_report'))
    if df is None:
        raise HTTPException(404,'anomalies report not found')
    return df.to_dict(orient='records')

*** End File