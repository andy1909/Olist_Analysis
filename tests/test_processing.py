# tests/test_processing.py
# ---------------------------------------------------------------------------
# DEPRECATED: This file has been superseded by the structured test hierarchy.
# All processing tests now live in:
#   tests/unit/data/test_processing.py
#
# This shim re-exports the new test classes so that running
#   python -m unittest tests.test_processing
# continues to work during the migration period.
# ---------------------------------------------------------------------------
from tests.unit.data.test_processing import (   # noqa: F401
    TestCleanAndConvertTypes,
    TestHandleMissingData,
    TestCreateLogisticsFeatures,
    TestCreateAggregatedFeatures,
    TestDropUnnecessaryColumns,
)

if __name__ == '__main__':
    import unittest
    unittest.main()
