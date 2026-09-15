"""On-host queue. Stdlib only: a broken venv must not be able to break the queue."""

from gpuc._version import user_agent

USER_AGENT = user_agent()
"""Also load-bearing for RunPod: Cloudflare 403s urllib's default User-Agent."""
