"""Config module (baseline imports from app, causing a cycle)."""

from app import APP_NAME

SETTINGS = {"name": APP_NAME}
