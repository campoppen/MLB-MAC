# MAC Clone

This is a lightweight Streamlit prototype that mimics the Matchup Analysis Calculator concept from the linked article and Streamlit app.

## What it does

- Loads TrackMan CSVs from a folder tree
- Cleans mixed-schema exports and skips incompatible files
- Clusters a selected pitcher's arsenal with Gaussian mixture models
- Labels each cluster by the dominant pitch type
- Finds historically similar pitches each hitter has seen
- Produces a weighted MAC-style matchup table for a lineup
- Shows cluster detail and pitch-location comp plots

## Run it

```bash
streamlit run app.py
```

By default the app reads from a local `data` folder next to `app.py`.

## Notes

- Lower `MAC` is more pitcher-friendly in this first pass.
- Similarity is based on standardized distance over velocity, IVB, HB, and location.
- If a hitter has too few close comps, the app falls back to the nearest pitches.
- One CSV in the current dataset has a different schema, so it is intentionally skipped and surfaced in the UI.

## Deploy

For Streamlit Community Cloud, keep these files in the repo root:

- `app.py`
- `mac_engine.py`
- `requirements.txt`
- `data`

The app is set up to run from the repo root without any machine-specific paths.
