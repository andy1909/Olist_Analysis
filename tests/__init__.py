# tests/__init__.py
# Root test package for the Olist Analysis project.
# Test layout mirrors the src/ package structure:
#
#   tests/
#   ├── unit/
#   │   ├── data/           → mirrors src/data/
#   │   ├── models/         → mirrors src/models/
#   │   └── utils/          → mirrors src/utils/
#   └── integration/        → end-to-end pipeline tests
#
# Run all tests:
#   python -m unittest discover -s tests -v
#
# Run a specific suite:
#   python -m unittest tests.unit.data.test_processing -v
