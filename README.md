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

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. In Streamlit Community Cloud, click **New app**.
3. Select repository `h3artfield/lens`, branch `main`, file path `main.py`.
4. Deploy.
