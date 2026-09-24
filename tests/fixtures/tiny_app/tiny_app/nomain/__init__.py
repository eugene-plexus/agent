"""A package with no `__main__`, so `python -m tiny_app.nomain` cannot run.

The install's verify step must refuse it rather than let the first
spawn discover it.
"""
