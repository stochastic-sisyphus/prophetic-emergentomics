"""
Context Augmented Generation (CAG) Framework for Economic Analysis.

CAG bridges statistical metrics with lived reality through LLM synthesis.
"Traditional econometrics give us the skeleton; LLMs add the nervous system."

Core Philosophy:
- GDP says growth while people feel recession - we explain why
- Context is not just data, it's meaning-making
- Multi-layer reality mapping: statistical + contextual + experiential
"""

from emergentomics.cag.engine import ContextAugmentedEconomicAnalyzer
from emergentomics.cag.context_builder import EconomicContextBuilder
from emergentomics.cag.prompts import CAGPromptTemplates

__all__ = [
    "ContextAugmentedEconomicAnalyzer",
    "EconomicContextBuilder",
    "CAGPromptTemplates",
]
