"""Hardcoded radp (bubble radius in DPD units) lookup table.

The ``radp`` value is the equilibrated bubble radius read from the 8th column
of ``samples_all.dat`` training data files.  Those files (10–23 MB each) are
too large to commit to git, so the values are hardcoded here from the
``Hierarchical_UQ_compression_dev`` training runs.

Lookup table:  experiment → {diameter_um → radp}
"""
from __future__ import annotations

RADP_LOOKUP: dict[str, dict[float, float]] = {
    "indentation": {
        3.2: 6.38,
        3.4: 6.80,
        5.8: 11.52,
    },
}


def get_radp(experiment: str, diameter_um: float) -> float:
    """Return the radp value for *experiment* and *diameter_um*.

    Raises
    ------
    ValueError
        If *experiment* or *diameter_um* is not in the lookup table.
    """
    if experiment not in RADP_LOOKUP:
        raise ValueError(
            f"Unknown experiment {experiment!r}. Known: {sorted(RADP_LOOKUP)}"
        )
    table = RADP_LOOKUP[experiment]
    if diameter_um not in table:
        raise ValueError(
            f"No radp entry for {experiment} diameter {diameter_um} µm. "
            f"Known diameters: {sorted(table)}"
        )
    return table[diameter_um]


def infer_radp_for_diameter(experiment: str, diameter_um: float) -> float:
    """Return radp, falling back to ``diameter_um / 0.5`` if not in lookup.

    The fallback formula ``diameter_um / 0.5`` mirrors the heuristic used in
    the UQ_DPD evaluation scripts when ``samples_all.dat`` is unavailable.
    """
    try:
        return get_radp(experiment, diameter_um)
    except ValueError:
        return diameter_um / 0.5
