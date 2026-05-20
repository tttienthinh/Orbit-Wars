# conftest.py — root pytest configuration
# Pre-import kaggle_environments so its native open_spiel_env extension is
# loaded before pytest forks/instruments tests. Without this, the extension
# crashes with an access violation on Windows when imported inside a test.
try:
    import kaggle_environments  # noqa: F401
except Exception:
    pass
