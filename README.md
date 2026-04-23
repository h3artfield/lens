# Heartfield Geometry Explorer

Interactive Streamlit app that walks viewers through the Heartfield idea, rewrite dynamics, feature hierarchy, and current experimental results with 3D and statistical visualizations.

## Run locally

```bash
pip install -r requirements.txt
streamlit run main.py
```

## Data mode

Drop experiment CSV exports in `data/`:

- `lifecycle_by_level.csv`
- `predictive_accuracy.csv`
- `coarse_rich_divergence.csv`
- `embedding_points.csv`

If files are missing, the app runs in demo mode.

## Mathematica handoff (before trial ends)

1. Open your working notebook and evaluate the cells that produce your final variables.
2. Copy `mathematica_export_for_streamlit.wl` into the same folder as the notebook (or set notebook directory to this repo root).
3. Evaluate:

```wolfram
Get["mathematica_export_for_streamlit.wl"]
```

4. Confirm CSVs appear in `data/`.
5. Re-run Streamlit and visuals will switch from demo to exported values automatically.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. In Streamlit Community Cloud, click **New app**.
3. Select repository `h3artfield/lens`, branch `main`, file path `main.py`.
4. Deploy.
