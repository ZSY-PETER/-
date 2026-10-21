# -*- coding: utf-8 -*-
"""Force Unicode-safe stdout/stderr in the Windows GUI's Python process tree."""
from __future__ import annotations
import os
import sys


def set_utf8_stdio():
    """On Windows, stdout redirected to pipes can otherwise default to GBK."""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')


def utf8_child_env():
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8:backslashreplace'
    env['PYTHONUTF8'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    return env
