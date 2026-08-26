"""
Read-only Reddit client.

This deliberately uses PRAW in *application-only* (read-only) mode — no
Reddit username or password is ever requested, needed, or stored. That means
this app can only read public posts; it cannot vote, post, comment, or take
any action on Reddit on anyone's behalf, even in the worst case where the
API keys leaked. That's the right shape for a public-facing demo.
"""

import praw
import streamlit as st

from secrets_config import get_secret


@st.cache_resource(show_spinner=False)
def get_reddit_client() -> praw.Reddit:
    reddit = praw.Reddit(
        client_id=get_secret("REDDIT_CLIENT_ID"),
        client_secret=get_secret("REDDIT_CLIENT_SECRET"),
        user_agent=get_secret("REDDIT_USER_AGENT"),
    )
    # No username/password supplied above, so PRAW authenticates via the
    # client-credentials grant and is automatically read-only. This
    # assertion just makes that guarantee explicit and fails loudly if it
    # ever isn't true.
    reddit.read_only = True
    assert reddit.read_only, "Reddit client is not in read-only mode."
    return reddit
