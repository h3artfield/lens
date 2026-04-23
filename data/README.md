Place exported experiment CSV files here.

Expected files:

- `lifecycle_by_level.csv`
  - columns: `generation`, `level`, `mean_radius`, `state_count`
- `predictive_accuracy.csv`
  - columns: `feature_level`, `mean_accuracy`, `std`, `n_reps`, `is_control`
- `coarse_rich_divergence.csv`
  - columns: `layer`, `distinct_expr`, `distinct_coarse`, `distinct_rich`, `coarse_entropy_norm`, `rich_entropy_norm`
- `embedding_points.csv`
  - columns: `x`, `y`, `z`, `operator`

If these files are missing, the Streamlit app runs in demo mode.
