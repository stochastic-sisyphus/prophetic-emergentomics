"""
Prophetic Emergentomics: Event-Driven Economic Intelligence Platform

A Context Augmented Generation (CAG) framework enhanced with GDELT real-time
intelligence for emergent economic analysis.

Core Philosophy:
- Traditional econometrics give us the skeleton; LLMs add the nervous system
- GDP says growth while people feel recession - we bridge that gap
- Forecasting isn't broken, it's outdated for systems evolving faster than it can interpret

Architecture:
- Bronze Layer: Raw data ingestion (GDELT events, sentiment, economic indicators)
- Silver Layer: Cleaned, consolidated, contextually enriched data
- Gold Layer: Analytics-ready, LLM-synthesized economic intelligence

Key Components:
- CAG Framework: Context Augmented Generation for economic analysis
- GDELT Integration: Real-time global event and sentiment tracking
- LLM Synthesis: Bridge between statistical metrics and lived economic reality
- Pattern Discovery: Emergence detection and opportunity identification
"""

__version__ = "0.1.0"
__author__ = "Vanessa Beck"

from emergentomics.core.config import Settings, get_settings
from emergentomics.cag.engine import ContextAugmentedEconomicAnalyzer
from emergentomics.gdelt.client import GDELTClient
from emergentomics.synthesis.engine import EconomicSynthesisEngine
from emergentomics.intelligence.detector import EconomicOpportunityDetector

__all__ = [
    "Settings",
    "get_settings",
    "ContextAugmentedEconomicAnalyzer",
    "GDELTClient",
    "EconomicSynthesisEngine",
    "EconomicOpportunityDetector",
]
