"""On-host queue. Stdlib only: a broken venv must not be able to break the queue."""

USER_AGENT = "gpuc/0.1"
"""Cloudflare 403s urllib's default User-Agent as a banned browser signature."""
