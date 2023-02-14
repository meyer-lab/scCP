"""
Test the data import.
"""
import pytest
import numpy as np
from ..imports.cytok import IL2_flowXA
from ..imports.CoH import CoH_xarray

IL2data_import, _ = IL2_flowXA()

def test_import_CoH():
    """Test the CoH import."""
    cohXA, celltypeXA = CoH_xarray(saveXA=False)
    assert np.isfinite(cohXA[:, :, :, :100].to_numpy()).all()


def test_finite_data():
    """Test that all values in tensor has no NaN"""
    assert np.isfinite(IL2data_import.to_numpy()).all()
