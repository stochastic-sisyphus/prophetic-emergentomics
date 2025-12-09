"""
Medallion Data Architecture for Prophetic Emergentomics.

Implements the Bronze → Silver → Gold data quality layers:

- Bronze: Raw, unprocessed data from GDELT and other sources
- Silver: Cleaned, consolidated, contextually enriched data
- Gold: Analytics-ready, LLM-synthesized economic intelligence
"""

from emergentomics.medallion.storage import DataStore, BronzeStore, SilverStore, GoldStore
from emergentomics.medallion.pipeline import MedallionPipeline

__all__ = [
    "DataStore",
    "BronzeStore",
    "SilverStore",
    "GoldStore",
    "MedallionPipeline",
]
