"""Error hierarchy for omnipex.

Purpose: one catchable base (``OmnipexError``) so entry points (CLI, future MCP
server / API) can convert any expected failure into a clean message and exit
code, while genuine bugs still surface as tracebacks.

Extension point: new error kinds subclass ``OmnipexError``; entry points need
no changes.
"""
from __future__ import annotations


class OmnipexError(Exception):
    """Base for all expected omnipex failures (bad input, unsupported usage)."""


class ManifestError(OmnipexError):
    """A pipeline manifest file is missing, unreadable, or malformed."""


class PipelineValidationError(OmnipexError):
    """A pipeline's structure is invalid (duplicate ids, dangling edges, cycles)."""


class UnsupportedChangeError(OmnipexError):
    """The diff contains changes this version cannot analyze (e.g. structural)."""
