"""
Centralized secret loading — this is the ONLY place credentials get read from.

Priority order:
  1. Streamlit Cloud's built-in secrets manager (st.secrets) — used by the
     deployed app. Configured in the app's dashboard under Settings > Secrets,
     never committed to git.
  2. Environment variables — used for local development, populated from a
     `.env` file (via python-dotenv, see .env.example) or set directly in
     whatever host you deploy to.

No secret is ever written into source code. If something required is
missing, this raises a clear error instead of silently running with None
(which would otherwise fail confusingly deep inside praw).
"""

import os
from dotenv import load_dotenv

# No-op if no .env file exists (e.g. in production, where real env vars /
# st.secrets are used instead).
load_dotenv()


def get_secret(key: str) -> str:
    """Look up `key` in st.secrets first, then environment variables."""
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        # Either streamlit isn't installed/running, or there's no
        # secrets.toml configured locally — fall through to env vars.
        pass

    value = os.getenv(key)
    if not value:
        raise RuntimeError(
            f"Missing secret '{key}'.\n"
            f"  Local dev       -> add it to a .env file (see .env.example)\n"
            f"  Streamlit Cloud -> add it under App settings -> Secrets"
        )
    return value
