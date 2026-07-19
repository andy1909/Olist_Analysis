# tests/test_analytics.py
# ---------------------------------------------------------------------------
# DEPRECATED: This file has been superseded by the structured test hierarchy.
# All analytics tests now live in:
#   tests/unit/models/test_analytics.py
#
# This shim re-exports the new test class so that running
#   python -m unittest tests.test_analytics
# continues to work during the migration period.
# ---------------------------------------------------------------------------
from tests.unit.models.test_analytics import (   # noqa: F401
    TestAggregateKpisForDashboard,
    TestForecastOrders,
)

if __name__ == '__main__':
    import unittest
    unittest.main()
