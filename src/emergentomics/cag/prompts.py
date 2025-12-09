"""
CAG Prompt Templates for Economic Analysis.

These templates structure how context is presented to LLMs for
synthesis, ensuring consistent and high-quality economic intelligence.
"""

from datetime import datetime
from typing import Any, Optional

from emergentomics.core.models import (
    EconomicContext,
    EconomicEvent,
    EconomicSentiment,
    EmergenceSignal,
    TheoreticalPrediction,
)


class CAGPromptTemplates:
    """
    Templates for Context Augmented Generation prompts.

    Each template is designed to elicit specific types of analysis
    from the LLM while grounding responses in provided context.
    """

    SYSTEM_PROMPT = """You are an expert economic analyst specializing in emergent economic patterns and complexity economics. Your role is to synthesize statistical data, real-time events, and sentiment signals into coherent economic intelligence.

Core Principles:
1. Bridge the gap between what metrics say and what people experience
2. Identify emergence signals that traditional forecasting misses
3. Recognize narrative economics - how stories shape economic reality
4. Maintain epistemic humility - acknowledge uncertainty as information

Analytical Framework:
- Emergence Theory: How macro patterns arise from micro interactions
- Complexity Economics: Non-linear dynamics, feedback loops, network effects
- Narrative Economics: Collective beliefs and their economic impact
- Adaptive Markets: Evolution and regime changes in economic systems

Output Style:
- Be direct and analytical
- Distinguish between observed patterns and inferred implications
- Quantify confidence when possible
- Flag contradictions between metrics and sentiment
- Identify what traditional analysis would miss"""

    @classmethod
    def format_events_summary(cls, events: list[EconomicEvent], max_events: int = 20) -> str:
        """Format events for prompt inclusion."""
        if not events:
            return "No recent events available."

        lines = []
        for event in events[:max_events]:
            tone_str = f"tone={event.tone:.1f}" if event.tone else "tone=N/A"
            themes = ", ".join(event.themes[:3]) if event.themes else "no themes"
            lines.append(
                f"- [{event.timestamp.strftime('%Y-%m-%d %H:%M')}] "
                f"{event.event_description or 'No title'} "
                f"({tone_str}, themes: {themes})"
            )

        return "\n".join(lines)

    @classmethod
    def format_sentiment_summary(cls, sentiment: Optional[EconomicSentiment]) -> str:
        """Format sentiment snapshot for prompt inclusion."""
        if not sentiment:
            return "No sentiment data available."

        return f"""Sentiment Snapshot ({sentiment.aggregation_period}):
- Overall Tone: {sentiment.overall_tone:.1f} ({sentiment.polarity.value})
- Anxiety Score: {sentiment.anxiety_score:.2f}
- Optimism Score: {sentiment.optimism_score:.2f}
- Uncertainty Score: {sentiment.uncertainty_score:.2f}
- Articles Analyzed: {sentiment.article_count}
- Source Diversity: {sentiment.source_diversity}
- Top Themes: {', '.join(sentiment.top_themes[:5])}"""

    @classmethod
    def format_theoretical_context(
        cls,
        predictions: list[TheoreticalPrediction],
        frameworks: list[str],
    ) -> str:
        """Format theoretical predictions for prompt inclusion."""
        if not predictions and not frameworks:
            return "No theoretical context provided."

        lines = [f"Relevant Theoretical Frameworks: {', '.join(frameworks)}"]
        lines.append("\nTheoretical Predictions:")

        for pred in predictions[:5]:
            lines.append(
                f"- [{pred.theoretical_framework}] {pred.description}"
            )

        return "\n".join(lines)

    @classmethod
    def format_narratives(
        cls,
        dominant: list[str],
        emerging: list[str],
        conflicts: list[str],
    ) -> str:
        """Format narrative analysis for prompt inclusion."""
        lines = []

        if dominant:
            lines.append("Dominant Narratives:")
            for n in dominant:
                lines.append(f"  - {n}")

        if emerging:
            lines.append("\nEmerging Narratives:")
            for n in emerging:
                lines.append(f"  - {n}")

        if conflicts:
            lines.append("\nNarrative Conflicts:")
            for n in conflicts:
                lines.append(f"  - {n}")

        return "\n".join(lines) if lines else "No narrative patterns detected."

    @classmethod
    def build_analysis_prompt(
        cls,
        context: EconomicContext,
        query: str,
        analysis_depth: str = "moderate",
    ) -> str:
        """
        Build a complete analysis prompt from context.

        Args:
            context: The assembled economic context
            query: The specific question or focus
            analysis_depth: shallow, moderate, or deep

        Returns:
            Complete prompt string
        """
        events_summary = cls.format_events_summary(context.recent_events)
        sentiment_summary = cls.format_sentiment_summary(context.sentiment_snapshot)
        theoretical_context = cls.format_theoretical_context(
            context.theoretical_predictions,
            context.relevant_frameworks,
        )
        narrative_context = cls.format_narratives(
            context.dominant_narratives,
            context.emerging_narratives,
            context.narrative_conflicts,
        )

        depth_instructions = {
            "shallow": "Provide a concise 2-3 paragraph analysis focusing on key insights.",
            "moderate": "Provide a comprehensive analysis covering multiple dimensions with specific evidence.",
            "deep": "Provide an exhaustive analysis exploring all aspects, contradictions, and implications.",
        }

        return f"""# Economic Intelligence Analysis Request

## Focus Area
{context.focus_area}

## Specific Query
{query}

## Context Window: {context.data_freshness_hours:.1f} hours of data

---

## THEORETICAL LAYER
{theoretical_context}

---

## REAL-TIME EVENT LAYER
Recent Events ({len(context.recent_events)} total):
{events_summary}

Event Clusters:
{cls._format_clusters(context.event_clusters)}

---

## SENTIMENT LAYER
{sentiment_summary}

---

## NARRATIVE LAYER
{narrative_context}

---

## GEOGRAPHIC HOTSPOTS
{cls._format_hotspots(context.geographic_hotspots)}

---

## Context Quality
- Data Freshness: {context.data_freshness_hours:.1f} hours
- Source Diversity: {context.source_diversity_score:.2f}
- Context Completeness: {context.context_completeness:.2f}

---

## Analysis Instructions
{depth_instructions.get(analysis_depth, depth_instructions['moderate'])}

Key Questions to Address:
1. What do the statistics say vs. what are people likely experiencing?
2. What emergence signals are visible in this data?
3. What might traditional forecasting miss here?
4. What are the key uncertainties and how should they be interpreted?
5. What should be monitored going forward?

Provide your analysis now:"""

    @classmethod
    def _format_clusters(cls, clusters: list[dict[str, Any]]) -> str:
        """Format event clusters for prompt."""
        if not clusters:
            return "No significant event clusters detected."

        lines = []
        for cluster in clusters[:5]:
            lines.append(
                f"- {cluster['cluster_key']}: {cluster['event_count']} events, "
                f"avg tone={cluster['average_tone']:.1f}"
            )
        return "\n".join(lines)

    @classmethod
    def _format_hotspots(cls, hotspots: list) -> str:
        """Format geographic hotspots for prompt."""
        if not hotspots:
            return "No geographic concentration detected."

        return ", ".join(h.country_code or h.country_name or "Unknown" for h in hotspots[:10])

    @classmethod
    def build_emergence_detection_prompt(cls, context: EconomicContext) -> str:
        """Build a prompt specifically for emergence signal detection."""
        events_summary = cls.format_events_summary(context.recent_events, max_events=30)

        return f"""# Emergence Signal Detection

You are scanning for early signals of economic phase transitions, trend accelerations, and systemic shifts that traditional forecasting would miss.

## Detection Framework

Look for these emergence types:
1. **Phase Transitions**: Sudden regime changes, threshold crossings
2. **Trend Accelerations**: Non-linear momentum shifts
3. **Narrative Shifts**: Changes in dominant economic stories
4. **Policy Signals**: Early indicators of regulatory changes
5. **Sentiment Divergence**: Gaps between metrics and mood
6. **Geographic Clustering**: Spatial concentration of events
7. **Sector Contagion**: Cross-sector pattern propagation
8. **Technology Adoption**: Accelerating tech impact signals

## Current Data

Focus Area: {context.focus_area}

Events:
{events_summary}

Sentiment:
{cls.format_sentiment_summary(context.sentiment_snapshot)}

Narratives:
{cls.format_narratives(context.dominant_narratives, context.emerging_narratives, context.narrative_conflicts)}

## Detection Task

For each emergence signal you detect, provide:
1. Signal Type (from list above)
2. Description of the signal
3. Supporting evidence (specific events/data)
4. Strength (0.0-1.0)
5. Confidence (0.0-1.0)
6. Potential implications
7. Recommended monitoring

Output as structured analysis. Be conservative - only flag signals with real supporting evidence."""

    @classmethod
    def build_opportunity_detection_prompt(cls, context: EconomicContext) -> str:
        """Build a prompt for economic opportunity detection."""
        return f"""# Economic Opportunity Detection

Analyze the following economic context to identify potential opportunities that may not yet be widely recognized.

## Opportunity Types to Consider

1. **Market Timing**: Windows where sentiment/reality gaps create opportunities
2. **Sector Rotation**: Emerging sector strength before broad recognition
3. **Policy Arbitrage**: Early positioning based on policy trajectory
4. **Narrative Momentum**: Riding emerging narratives before peak
5. **Geographic Advantage**: Regional opportunities not fully priced
6. **Technology Wave**: Early technology adoption advantages
7. **Sentiment Reversal**: Extreme sentiment suggesting reversal potential

## Current Context

Focus Area: {context.focus_area}

Recent Events:
{cls.format_events_summary(context.recent_events, max_events=15)}

Sentiment Snapshot:
{cls.format_sentiment_summary(context.sentiment_snapshot)}

Dominant Narratives: {', '.join(context.dominant_narratives[:3]) if context.dominant_narratives else 'None identified'}

Emerging Narratives: {', '.join(context.emerging_narratives[:3]) if context.emerging_narratives else 'None identified'}

Theoretical Framework Insights:
{cls.format_theoretical_context(context.theoretical_predictions[:3], context.relevant_frameworks)}

## Analysis Instructions

For each opportunity identified:
1. Opportunity Type
2. Description
3. Supporting Evidence
4. Confidence Level (0.0-1.0)
5. Time Sensitivity (how quickly might this be widely recognized)
6. Risks to Consider
7. Recommended Actions

Be selective - only identify opportunities with genuine evidence support."""

    @classmethod
    def build_gap_analysis_prompt(
        cls,
        context: EconomicContext,
        statistical_indicators: dict[str, float],
    ) -> str:
        """
        Build a prompt to analyze gaps between statistics and reality.

        This is the core CAG insight: explaining why GDP says growth
        while people feel recession.
        """
        indicator_lines = "\n".join(
            f"- {name}: {value}"
            for name, value in statistical_indicators.items()
        )

        return f"""# Statistical Reality Gap Analysis

## The Core Question
Where do statistical measures diverge from lived economic experience?
Why might people feel differently than what the numbers suggest?

## Statistical Indicators
{indicator_lines if indicator_lines else "No indicators provided"}

## Sentiment Reality
{cls.format_sentiment_summary(context.sentiment_snapshot)}

Anxiety Level: {context.sentiment_snapshot.anxiety_score if context.sentiment_snapshot else 'N/A'}
Optimism Level: {context.sentiment_snapshot.optimism_score if context.sentiment_snapshot else 'N/A'}

## Real-Time Event Context
{cls.format_events_summary(context.recent_events, max_events=15)}

## Narrative Context
{cls.format_narratives(context.dominant_narratives, context.emerging_narratives, context.narrative_conflicts)}

## Analysis Task

1. **Identify Gaps**: Where do statistics and sentiment diverge?
2. **Explain the Gap**: Why might people experience the economy differently?
3. **Distribution Effects**: Who benefits vs. who struggles?
4. **Temporal Mismatch**: Are statistics lagging lived experience?
5. **Aggregation Artifacts**: What gets hidden in averages?
6. **Narrative Impact**: How are stories shaping perception?

Provide a nuanced analysis that helps explain why economic reality feels different from economic statistics."""
