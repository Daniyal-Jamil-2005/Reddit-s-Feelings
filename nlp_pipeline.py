"""
Text cleaning + VADER sentiment scoring + keyword extraction.
Reused/simplified from the original class project's Part 2 and Part 3.
"""

import math
import re
import string
from collections import Counter

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer


def _ensure_nltk_data() -> None:
    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4", "vader_lexicon"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass  # best-effort; a missing optional package shouldn't crash the app


_ensure_nltk_data()

_EXTRA_STOP = {
    "like", "also", "get", "got", "one", "would", "could", "know", "think",
    "really", "much", "even", "many", "use", "used", "using", "way", "go",
    "going", "make", "want", "need", "time", "new", "amp", "deleted",
    "removed", "reddit", "im", "dont", "just", "post", "comment", "people",
    "thing", "something", "anything", "anyone", "someone", "see", "still", "well",
}
STOPWORDS = set(stopwords.words("english")) | _EXTRA_STOP
LEMMATIZER = WordNetLemmatizer()
_SIA = SentimentIntensityAnalyzer()


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/markdown/punctuation/numbers, tokenize,
    remove stopwords + short tokens, lemmatize."""
    if not isinstance(text, str) or text.strip() in ("", "[No Body]", "[deleted]", "[removed]"):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[*_~`>#+\-=|{}]", " ", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= 3]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def sentiment_label_and_score(text: str):
    """Returns (label, compound_score) using VADER on the raw (uncleaned)
    text — VADER is tuned for natural language, not stripped tokens."""
    if not isinstance(text, str) or text.strip() == "":
        return "Neutral", 0.0
    compound = _SIA.polarity_scores(text)["compound"]
    if compound >= 0.05:
        return "Positive", compound
    elif compound <= -0.05:
        return "Negative", compound
    return "Neutral", compound


def top_keyphrases(cleaned_texts, exclude_terms=None, n: int = 14):
    """Extract unigrams, bigrams, and trigrams across a collection of cleaned texts,
    excluding search terms and stopwords to surface meaningful key phrases."""
    exclude_terms = {t.lower() for t in (exclude_terms or [])}
    phrase_counts = Counter()

    for text in cleaned_texts:
        if not text:
            continue
        tokens = [w for w in text.split() if w not in exclude_terms and len(w) >= 3]
        if not tokens:
            continue

        # Bigrams (higher weight)
        for bg in zip(tokens[:-1], tokens[1:]):
            if bg[0] not in STOPWORDS and bg[1] not in STOPWORDS:
                phrase_counts[f"{bg[0]} {bg[1]}"] += 2

        # Trigrams (highest weight)
        for tg in zip(tokens[:-2], tokens[1:-1], tokens[2:]):
            if tg[0] not in STOPWORDS and tg[2] not in STOPWORDS:
                phrase_counts[f"{tg[0]} {tg[1]} {tg[2]}"] += 3

        # Single words (unigrams)
        for w in tokens:
            if w not in STOPWORDS:
                phrase_counts[w] += 1

    # Filter out phrases that are just search terms
    results = []
    for phrase, score in phrase_counts.most_common(n * 3):
        if phrase in exclude_terms:
            continue
        # Avoid duplicate phrases where single word is already in a multi-word phrase
        results.append((phrase, score))
        if len(results) >= n:
            break
    return results


def compute_weighted_metrics(df: pd.DataFrame, query: str):
    """Compute engagement-weighted sentiment metrics (upvotes + comments)
    and return structured summary metrics for the executive banner."""
    if df.empty:
        return {
            "verdict": "No Data",
            "score": 0,
            "badge_color": "#6B6B62",
            "pos_pct": 0.0,
            "neu_pct": 0.0,
            "neg_pct": 0.0,
            "total_engagement": 0,
            "summary_text": "No posts found for this query.",
        }

    # Calculate engagement weight per post
    weights = df.apply(
        lambda r: 1.0 + math.log1p(max(0, float(r.get("score", 0))) + max(0, float(r.get("num_comments", 0)))),
        axis=1,
    )
    total_weight = weights.sum()

    pos_mask = df["sentiment"] == "Positive"
    neu_mask = df["sentiment"] == "Neutral"
    neg_mask = df["sentiment"] == "Negative"

    pos_pct = (weights[pos_mask].sum() / total_weight) if total_weight > 0 else 0.0
    neu_pct = (weights[neu_mask].sum() / total_weight) if total_weight > 0 else 0.0
    neg_pct = (weights[neg_mask].sum() / total_weight) if total_weight > 0 else 0.0

    # Weighted compound average
    weighted_compound = (df["compound"] * weights).sum() / total_weight if total_weight > 0 else 0.0

    # 0-100 Sentiment Index Score
    sentiment_index = int(round(((weighted_compound + 1.0) / 2.0) * 100))

    if sentiment_index >= 75:
        verdict = "Overwhelmingly Positive"
        badge_color = "#10B981"  # Emerald Green
    elif sentiment_index >= 58:
        verdict = "Mostly Positive"
        badge_color = "#059669"  # Medium Green
    elif sentiment_index >= 43:
        verdict = "Mixed / Balanced"
        badge_color = "#F59E0B"  # Amber
    elif sentiment_index >= 28:
        verdict = "Mostly Negative"
        badge_color = "#EF4444"  # Red
    else:
        verdict = "Overwhelmingly Negative"
        badge_color = "#B91C1C"  # Dark Red

    total_upvotes = int(df["score"].sum())
    total_comments = int(df["num_comments"].sum())
    total_engagement = total_upvotes + total_comments

    # Generate human-readable summary phrase
    if pos_pct >= 0.55:
        summary_text = (
            f"Reddit discussions around **{query}** are strongly optimistic ({pos_pct:.0%} positive sentiment). "
            f"Community members frequently share positive experiences, praise features, and recommend it."
        )
    elif neg_pct >= 0.40:
        summary_text = (
            f"Reddit sentiment for **{query}** leans negative ({neg_pct:.0%} negative sentiment). "
            f"Discussions focus heavily on user complaints, bugs, or pricing dissatisfaction."
        )
    elif neu_pct >= 0.50:
        summary_text = (
            f"Reddit discussions on **{query}** are largely neutral and informational ({neu_pct:.0%} neutral). "
            f"Users are primarily asking questions, sharing news, or seeking advice."
        )
    else:
        summary_text = (
            f"Reddit sentiment for **{query}** is mixed ({pos_pct:.0%} positive vs {neg_pct:.0%} negative). "
            f"Community members express divided opinions with distinct praise and criticisms."
        )

    return {
        "verdict": verdict,
        "score": sentiment_index,
        "badge_color": badge_color,
        "pos_pct": pos_pct,
        "neu_pct": neu_pct,
        "neg_pct": neg_pct,
        "weighted_compound": weighted_compound,
        "total_upvotes": total_upvotes,
        "total_comments": total_comments,
        "total_engagement": total_engagement,
        "summary_text": summary_text,
    }

