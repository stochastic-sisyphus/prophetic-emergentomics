"""
GDELT Integration Layer for Prophetic Emergentomics.

Provides real-time access to global event and sentiment data from
the GDELT Project (Global Database of Events, Language, and Tone).

Key capabilities:
- Real-time economic event tracking from 100,000+ global news sources
- Sentiment analysis and tone tracking across 100+ languages
- Geographic and temporal event clustering
- Economic theme filtering and extraction
"""

from emergentomics.gdelt.client import GDELTClient
from emergentomics.gdelt.parsers import GDELTEventParser, GDELTSentimentParser
from emergentomics.gdelt.collectors import (
    EconomicEventCollector,
    SentimentCollector,
    ThemeCollector,
)

__all__ = [
    "GDELTClient",
    "GDELTEventParser",
    "GDELTSentimentParser",
    "EconomicEventCollector",
    "SentimentCollector",
    "ThemeCollector",
]
