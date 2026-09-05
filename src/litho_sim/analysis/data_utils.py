"""
Validation and persistence helpers for dose-focus CD DataFrames.

The tidy format used throughout analysis: columns ``dose``, ``defocus_nm``,
``cd_nm``, one row per measured or simulated point.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def validate_dataframe(
    df: pd.DataFrame,
    required_cols: Sequence[str] = ("dose", "defocus_nm", "cd_nm"),
) -> None:
    """Raise :class:`ValueError` if *df* is missing required columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    required_cols : list of str
        Column names that must be present.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}. "
            f"Present: {list(df.columns)}"
        )
    for col in required_cols:
        n_null = int(df[col].isnull().sum())
        if n_null > 0:
            raise ValueError(f"Column '{col}' contains {n_null} null value(s).")
    logger.debug("DataFrame validation passed (%d rows).", len(df))


def pivot_cd_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot a tidy dose-focus frame into a CD matrix (rows=defocus, cols=dose).

    Parameters
    ----------
    df : pd.DataFrame
        Tidy frame with columns ``dose``, ``defocus_nm``, ``cd_nm``.

    Returns
    -------
    pd.DataFrame
        Pivot table: index = ``defocus_nm``, columns = ``dose``, values = ``cd_nm``.
    """
    validate_dataframe(df)
    return df.pivot_table(
        index="defocus_nm",
        columns="dose",
        values="cd_nm",
        aggfunc="mean",
    ).sort_index()


def save_results_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a results DataFrame to CSV, creating parent directories as needed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : Path
        Destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Results saved → %s (%d rows)", path, len(df))

