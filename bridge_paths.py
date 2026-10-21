# -*- coding: utf-8 -*-
"""Common paths for Bridge AI. No external model or VMamba sources are bundled."""
from pathlib import Path

BASE = Path(__file__).resolve().parent
HOME = Path.home()

def default_checkpoint():
    candidates = [
        BASE / 'models' / 'best_model_epoch27.pth',
        HOME / 'Desktop' / 'best_model_epoch27.pth',
        HOME / 'Desktop' / 'best_model.pth',
    ]
    return next((p for p in candidates if p.is_file()), candidates[0])

def default_vmamba_models():
    candidates = [
        BASE / 'VMamba-main' / 'classification' / 'models',
        HOME / 'Desktop' / 'VMamba-main' / 'classification' / 'models',
    ]
    return next((p for p in candidates if (p / 'vmamba.py').is_file()), candidates[1])
