"""Vercel Serverless Function Entry Point for NIVARAN.

Exports the WSGI app instance for @vercel/python runtime.
"""
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Mark serverless environment
os.environ["VERCEL"] = "1"

from app import create_app
from config import Config

# WSGI application callable for Vercel
app = create_app(Config)
