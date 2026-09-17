"""Entry point: `python run.py`.

Thin wrapper so the documented run command works without installing the
project as a package. Runs the full pipeline with the default Config (CFG)
defined in macro_regime_classifier.py.
"""
from macro_regime_classifier import main, CFG

if __name__ == "__main__":
    main(CFG)
