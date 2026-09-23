"""
NASA POWER bulk download for all (lat, lon) locations in Train/Test.
Downloads 2007-2022 in yearly chunks, caches each chunk as JSON.
"""

import json, time, sys
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from datetime import date

DATA  = Path('data')
CACHE = DATA / 'nasa_cache'
CACHE.mkdir(exist_ok=True)

PARAMS = 'T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,WS2M,ALLSKY_SFC_SW_DWN'

train = pd.read_csv(DATA / 'Train.csv', parse_dates=['deathdate'])
test  = pd.read_csv(DATA / 'Test.csv',  parse_dates=['deathdate'])

locs = (
    pd.concat([train[['latitude','longitude']], test[['latitude','longitude']]])
    .drop_duplicates().reset_index(drop=True).round(6)
)

all_dates  = pd.concat([train['deathdate'], test['deathdate']])
START_DATE = (all_dates.min() - pd.Timedelta(days=95)).date()
END_DATE   = all_dates.max().date()


def cache_path(lat, lon, year):
    return CACHE / f'{lat:.6f}_{lon:.6f}_{year}.json'


def fetch_nasa_power(lat, lon, start, end, max_retries=5):
    url = (
        f'https://power.larc.nasa.gov/api/temporal/daily/point'
        f'?parameters={PARAMS}&community=AG'
        f'&longitude={lon}&latitude={lat}'
        f'&start={start}&end={end}&format=JSON'
    )
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                return r.json()
            elif r.status_code in (429, 500, 502, 503):
                wait = 2 ** attempt + 2
                print(f'  HTTP {r.status_code} — retry in {wait}s', flush=True)
                time.sleep(wait)
            else:
                print(f'  HTTP {r.status_code} unexpected — skip', flush=True)
                return {}
        except requests.RequestException as e:
            wait = 2 ** attempt + 2
            print(f'  Exception: {e} — retry in {wait}s', flush=True)
            time.sleep(wait)
    return {}


total_locs   = len(locs)
total_years  = END_DATE.year - START_DATE.year + 1
total_calls  = total_locs * total_years
completed    = 0
cache_hits   = 0

print(f'Download plan: {total_locs} locations × {total_years} years = {total_calls} API calls max', flush=True)
print(f'Window: {START_DATE} → {END_DATE}', flush=True)
print('Starting...', flush=True)

for i, row in locs.iterrows():
    lat = round(row['latitude'],  6)
    lon = round(row['longitude'], 6)

    days_downloaded = 0
    for year in range(START_DATE.year, END_DATE.year + 1):
        cp = cache_path(lat, lon, year)
        if cp.exists():
            cache_hits += 1
            completed  += 1
            # Count cached days
            try:
                with open(cp) as f:
                    d = json.load(f)
                if d and 'properties' in d:
                    days_downloaded += len(list(d['properties']['parameter']['T2M'].keys()))
            except Exception:
                pass
            continue

        chunk_start = max(START_DATE, date(year, 1, 1))
        chunk_end   = min(END_DATE,   date(year, 12, 31))
        data = fetch_nasa_power(
            lat, lon,
            chunk_start.strftime('%Y%m%d'),
            chunk_end.strftime('%Y%m%d'),
        )
        with open(cp, 'w') as f:
            json.dump(data, f)

        if data and 'properties' in data:
            days_downloaded += len(list(data['properties']['parameter']['T2M'].keys()))

        completed += 1
        time.sleep(1.2)

    pct = 100 * (i + 1) / total_locs
    print(
        f'[{i+1:2d}/{total_locs}] ({lat:.4f}, {lon:.4f}) '
        f'→ {days_downloaded} days  |  {pct:.0f}% done  '
        f'(cache hits so far: {cache_hits})',
        flush=True
    )

print(f'\nDone. {cache_hits}/{completed} from cache.', flush=True)
