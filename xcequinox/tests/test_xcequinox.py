"""
Unit and regression test for the xcequinox package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import xcequinox


def test_xcequinox_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "xcequinox" in sys.modules
