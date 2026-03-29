"""SDDS file I/O utilities for Elegant simulation output."""

import logging
from pathlib import Path

import numpy as np
import sdds

logger = logging.getLogger(__name__)


def get_column(fpath: str | Path, col_name: str) -> np.ndarray | None:
    """Load a specific column from an SDDS file.

    Handles multi-page data by concatenating pages.

    Parameters
    ----------
    fpath : str or Path
        Path to the SDDS file.
    col_name : str
        Column name to extract.

    Returns
    -------
    np.ndarray or None
        Flattened column data, or None if column not found.
    """
    ds = sdds.SDDS(0)
    ds.load(str(fpath))
    try:
        col_index = ds.columnName.index(col_name)
        data = ds.columnData[col_index]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (list, np.ndarray)):
            return np.concatenate(data)
        return np.array(data).ravel()
    except Exception as e:
        logger.warning("Could not load '%s' from %s: %s", col_name, Path(fpath).name, e)
        return None


def get_parameter(fpath: str | Path, param_name: str) -> float | None:
    """Load a specific parameter from an SDDS file.

    Parameters
    ----------
    fpath : str or Path
        Path to the SDDS file.
    param_name : str
        Parameter name to extract.

    Returns
    -------
    float or None
        Parameter value, or None if not found.
    """
    ds = sdds.SDDS(0)
    ds.load(str(fpath))
    try:
        idx = ds.parameterName.index(param_name)
        return ds.parameterData[idx][0]
    except (ValueError, IndexError) as e:
        logger.warning("Could not load param '%s' from %s: %s", param_name, Path(fpath).name, e)
        return None
