"""Utility functions for the Olist Analysis pipeline."""
import os
import json
import pandas as pd


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def save_dataframe(df, path, format='csv'):
    """Save a DataFrame to CSV or Excel."""
    ensure_dir(os.path.dirname(path))
    if format == 'excel':
        df.to_excel(path, index=False, engine='openpyxl')
    else:
        df.to_csv(path, index=False)
    print(f"    -> Saved: {path}")


def load_config(config_path):
    """Load JSON configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)
