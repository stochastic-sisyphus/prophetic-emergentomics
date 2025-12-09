"""
Economic Synthesis Engine.

Provides specialized synthesis capabilities for different types
of economic analysis, building on the core CAG framework.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional

import structlog

from emergentomics.core.config import get_settings
from emergentomics.core.models import (
    EconomicContext,
    EconomicIntelligence,
    EconomicSentiment,
    EmergenceSignal,
    EmergenceType,
    MedallionLayer,
)
from emergentomics.cag.engine import ContextAugmentedEconomicAnalyzer, LLMClient
from emergentomics.cag.context_builder import EconomicContextBuilder
from emergentomics.gdelt.collectors import SentimentCollector, EconomicEventCollector

logger = structlog.get_logger(__name__)


class EconomicSynthesisEngine:
    """
    High-level synthesis engine for economic intelligence.

    Provides domain-specific analysis methods that combine CAG
    with specialized synthesis strategies.

    Example:
        engine = EconomicSynthesisEngine()

        # Real-time market psychology
        psychology = await engine.synthesize_market_psychology(
            sectors=["technology", "finance"],
            lookback_hours=48,
        )

        # Policy impact analysis
        impact = await engine.analyze_policy_impact(
            policy_area="monetary_policy",
            event_description="Federal Reserve signals rate pause",
        )
    """

    def __init__(self):
        self.settings = get_settings()
        self.analyzer = ContextAugmentedEconomicAnalyzer()
        self.context_builder = EconomicContextBuilder()
        self.sentiment_collector = SentimentCollector()
        self.event_collector = EconomicEventCollector()
        self.llm_client = LLMClient()

    async def synthesize_market_psychology(
        self,
        sectors: Optional[list[str]] = None,
        countries: Optional[list[str]] = None,
        lookback_hours: int = 24,
    ) -> dict[str, Any]:
        """
        Synthesize current market psychology from multiple signals.

        Combines sentiment, narrative, and event data to produce
        a coherent picture of market mood and expectations.

        Args:
            sectors: Sectors to focus on
            countries: Countries to focus on
            lookback_hours: Data lookback period

        Returns:
            Market psychology synthesis
        """
        sectors = sectors or ["finance", "technology", "labor"]

        # Collect sector sentiments concurrently
        sentiment_tasks = [
            self.sentiment_collector.collect_sector_sentiment(
                sector=sector,
                lookback_hours=lookback_hours,
            )
            for sector in sectors
        ]

        sentiments = await asyncio.gather(*sentiment_tasks, return_exceptions=True)
        valid_sentiments = [s for s in sentiments if isinstance(s, EconomicSentiment)]

        # Build context for synthesis
        context = await self.context_builder.build_context(
            focus_area="Market Psychology Analysis",
            lookback_hours=lookback_hours,
            countries=countries,
        )

        # Calculate aggregate psychology metrics
        avg_anxiety = sum(s.anxiety_score for s in valid_sentiments) / len(valid_sentiments) if valid_sentiments else 0.5
        avg_optimism = sum(s.optimism_score for s in valid_sentiments) / len(valid_sentiments) if valid_sentiments else 0.5
        avg_uncertainty = sum(s.uncertainty_score for s in valid_sentiments) / len(valid_sentiments) if valid_sentiments else 0.5

        # Determine dominant mood
        if avg_anxiety > 0.6:
            dominant_mood = "anxious"
        elif avg_optimism > 0.6:
            dominant_mood = "optimistic"
        elif avg_uncertainty > 0.6:
            dominant_mood = "uncertain"
        else:
            dominant_mood = "neutral"

        # Generate synthesis
        synthesis_prompt = f"""Synthesize the current market psychology based on this data:

Sector Sentiments:
{self._format_sector_sentiments(sectors, valid_sentiments)}

Aggregate Metrics:
- Average Anxiety: {avg_anxiety:.2f}
- Average Optimism: {avg_optimism:.2f}
- Average Uncertainty: {avg_uncertainty:.2f}
- Dominant Mood: {dominant_mood}

Dominant Narratives: {', '.join(context.dominant_narratives[:3]) if context.dominant_narratives else 'None identified'}
Emerging Narratives: {', '.join(context.emerging_narratives[:3]) if context.emerging_narratives else 'None identified'}

Recent Event Context:
{self._summarize_events(context.recent_events[:10])}

Provide a cohesive synthesis of:
1. Current market mood and its drivers
2. Key psychological themes across sectors
3. Potential behavioral implications
4. What traditional sentiment measures might miss"""

        synthesis = await self.llm_client.generate(synthesis_prompt)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "lookback_hours": lookback_hours,
            "sectors_analyzed": sectors,
            "dominant_mood": dominant_mood,
            "aggregate_metrics": {
                "anxiety": avg_anxiety,
                "optimism": avg_optimism,
                "uncertainty": avg_uncertainty,
            },
            "sector_sentiments": {
                sector: {
                    "tone": s.overall_tone if isinstance(s, EconomicSentiment) else None,
                    "polarity": s.polarity.value if isinstance(s, EconomicSentiment) else None,
                }
                for sector, s in zip(sectors, sentiments)
            },
            "synthesis": synthesis,
            "context_id": context.id,
        }

    async def analyze_policy_impact(
        self,
        policy_area: str,
        event_description: Optional[str] = None,
        lookback_hours: int = 48,
    ) -> dict[str, Any]:
        """
        Analyze potential impact of policy developments.

        Uses CAG to synthesize theoretical expectations with
        real-time market/sentiment reactions.

        Args:
            policy_area: Area of policy (monetary_policy, fiscal_policy, etc.)
            event_description: Specific policy event to analyze
            lookback_hours: Data lookback period

        Returns:
            Policy impact analysis
        """
        # Collect events related to policy area
        events = await self.event_collector.collect_by_topic(
            topic=policy_area,
            lookback_hours=lookback_hours,
            max_events=100,
        )

        # Build context
        context = await self.context_builder.build_context(
            focus_area=f"Policy Impact: {policy_area}",
            query=event_description or policy_area,
            lookback_hours=lookback_hours,
        )

        # Generate analysis
        analysis_prompt = f"""Analyze the potential impact of developments in {policy_area}:

{"Specific Event: " + event_description if event_description else ""}

Theoretical Framework Context:
{self._format_frameworks(context.relevant_frameworks)}

Recent Policy-Related Events ({len(events)} total):
{self._summarize_events(events[:15])}

Current Sentiment:
- Overall Tone: {context.sentiment_snapshot.overall_tone if context.sentiment_snapshot else 'N/A'}
- Anxiety: {context.sentiment_snapshot.anxiety_score if context.sentiment_snapshot else 'N/A'}
- Uncertainty: {context.sentiment_snapshot.uncertainty_score if context.sentiment_snapshot else 'N/A'}

Analyze:
1. Expected direct impacts based on complexity economics
2. Potential second-order effects and feedback loops
3. Narrative implications and market psychology effects
4. Key uncertainties and scenario branches
5. What historical models might miss about this policy environment"""

        analysis = await self.llm_client.generate(analysis_prompt)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "policy_area": policy_area,
            "event_description": event_description,
            "events_analyzed": len(events),
            "analysis": analysis,
            "context_id": context.id,
            "relevant_frameworks": context.relevant_frameworks,
        }

    async def track_economic_stress(
        self,
        countries: Optional[list[str]] = None,
        lookback_hours: int = 24,
    ) -> dict[str, Any]:
        """
        Track economic stress indicators across regions.

        Synthesizes sentiment, event, and narrative data to
        produce stress indicators.

        Args:
            countries: Countries to track
            lookback_hours: Data lookback period

        Returns:
            Economic stress assessment
        """
        countries = countries or ["US", "UK", "DE", "CN", "JP"]

        # Collect country sentiments
        sentiments = await self.sentiment_collector.collect_multi_country_comparison(
            country_codes=countries,
            lookback_hours=lookback_hours,
        )

        # Collect stress-related events
        stress_events = await self.event_collector.collect_by_topic(
            topic="recession_risk",
            lookback_hours=lookback_hours,
            max_events=100,
        )

        # Calculate stress scores
        stress_scores = {}
        for country, sentiment in sentiments.items():
            if sentiment:
                # Stress = high anxiety + high uncertainty + negative tone
                stress = (
                    sentiment.anxiety_score * 0.4 +
                    sentiment.uncertainty_score * 0.3 +
                    max(0, -sentiment.overall_tone / 20) * 0.3
                )
                stress_scores[country] = min(stress, 1.0)
            else:
                stress_scores[country] = None

        # Identify highest stress regions
        sorted_stress = sorted(
            [(c, s) for c, s in stress_scores.items() if s is not None],
            key=lambda x: -x[1]
        )

        # Generate synthesis
        synthesis_prompt = f"""Analyze economic stress indicators:

Country Stress Scores:
{chr(10).join(f'- {c}: {s:.2f}' for c, s in sorted_stress)}

Recession-Risk Related Events ({len(stress_events)} total):
{self._summarize_events(stress_events[:10])}

Synthesize:
1. Overall assessment of economic stress levels
2. Regional patterns and divergences
3. Key stress drivers by region
4. Early warning signals to monitor
5. Comparison to historical stress episodes"""

        synthesis = await self.llm_client.generate(synthesis_prompt)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "lookback_hours": lookback_hours,
            "countries": countries,
            "stress_scores": stress_scores,
            "highest_stress": sorted_stress[:3] if sorted_stress else [],
            "stress_events_count": len(stress_events),
            "synthesis": synthesis,
        }

    async def synthesize_technology_disruption(
        self,
        technology_focus: str = "AI",
        sectors: Optional[list[str]] = None,
        lookback_hours: int = 48,
    ) -> dict[str, Any]:
        """
        Synthesize technology disruption signals.

        Tracks technology adoption momentum and its economic
        implications using CAG methodology.

        Args:
            technology_focus: Technology to track (AI, automation, etc.)
            sectors: Sectors to focus on
            lookback_hours: Data lookback period

        Returns:
            Technology disruption synthesis
        """
        sectors = sectors or ["technology", "labor_market", "markets"]

        # Collect technology-related events
        tech_events = await self.event_collector.collect_by_topic(
            topic="technology",
            lookback_hours=lookback_hours,
            max_events=100,
        )

        # Filter for specific technology
        relevant_events = [
            e for e in tech_events
            if technology_focus.upper() in " ".join(e.themes).upper()
        ]

        # Build context
        context = await self.context_builder.build_context(
            focus_area=f"Technology Disruption: {technology_focus}",
            query=technology_focus,
            lookback_hours=lookback_hours,
        )

        # Generate synthesis
        synthesis_prompt = f"""Synthesize {technology_focus} disruption signals:

Technology Events ({len(relevant_events)} relevant of {len(tech_events)} total):
{self._summarize_events(relevant_events[:15])}

Sectors Affected: {', '.join(sectors)}

Current Narratives:
- Dominant: {', '.join(context.dominant_narratives[:3]) if context.dominant_narratives else 'None'}
- Emerging: {', '.join(context.emerging_narratives[:3]) if context.emerging_narratives else 'None'}

Analyze:
1. Current adoption momentum and key indicators
2. Sector-specific disruption patterns
3. Labor market implications
4. Investment and market sentiment
5. Timeline and phase of disruption cycle
6. What traditional technology forecasts miss about emergence patterns"""

        synthesis = await self.llm_client.generate(synthesis_prompt)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "technology_focus": technology_focus,
            "lookback_hours": lookback_hours,
            "sectors": sectors,
            "events_analyzed": len(relevant_events),
            "total_tech_events": len(tech_events),
            "synthesis": synthesis,
            "context_id": context.id,
        }

    async def generate_economic_briefing(
        self,
        focus_areas: Optional[list[str]] = None,
        lookback_hours: int = 24,
    ) -> dict[str, Any]:
        """
        Generate a comprehensive economic briefing.

        Synthesizes multiple analyses into a coherent intelligence
        briefing for strategic decision-making.

        Args:
            focus_areas: Specific areas to cover
            lookback_hours: Data lookback period

        Returns:
            Comprehensive economic briefing
        """
        focus_areas = focus_areas or [
            "monetary_policy",
            "labor_market",
            "technology",
            "markets",
        ]

        # Run multiple analyses concurrently
        market_psych = asyncio.create_task(
            self.synthesize_market_psychology(lookback_hours=lookback_hours)
        )
        stress = asyncio.create_task(
            self.track_economic_stress(lookback_hours=lookback_hours)
        )
        tech = asyncio.create_task(
            self.synthesize_technology_disruption(lookback_hours=lookback_hours)
        )

        results = await asyncio.gather(
            market_psych, stress, tech,
            return_exceptions=True
        )

        # Build comprehensive context
        main_context = await self.context_builder.build_context(
            focus_area="Comprehensive Economic Analysis",
            lookback_hours=lookback_hours,
            max_events=150,
        )

        # Run main analysis
        intelligence = await self.analyzer.analyze(
            query="Comprehensive economic situation assessment",
            focus_area="Global Economic Intelligence",
            lookback_hours=lookback_hours,
            analysis_depth="deep",
        )

        # Compile briefing
        briefing_prompt = f"""Generate an executive economic briefing based on:

MAIN INTELLIGENCE:
{intelligence.detailed_analysis[:2000]}

MARKET PSYCHOLOGY:
{results[0].get('synthesis', 'N/A')[:500] if isinstance(results[0], dict) else 'Analysis unavailable'}

ECONOMIC STRESS:
{results[1].get('synthesis', 'N/A')[:500] if isinstance(results[1], dict) else 'Analysis unavailable'}

TECHNOLOGY DISRUPTION:
{results[2].get('synthesis', 'N/A')[:500] if isinstance(results[2], dict) else 'Analysis unavailable'}

EMERGENCE SIGNALS: {len(intelligence.emergence_signals)} detected
PHASE TRANSITION RISK: {intelligence.phase_transition_risk:.2f}

Create a structured executive briefing with:
1. Executive Summary (3-4 sentences)
2. Key Developments (bullet points)
3. Risk Assessment
4. Opportunities Identified
5. Strategic Recommendations
6. Monitoring Priorities"""

        briefing = await self.llm_client.generate(briefing_prompt)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "lookback_hours": lookback_hours,
            "focus_areas": focus_areas,
            "executive_briefing": briefing,
            "market_psychology": results[0] if isinstance(results[0], dict) else None,
            "economic_stress": results[1] if isinstance(results[1], dict) else None,
            "technology_disruption": results[2] if isinstance(results[2], dict) else None,
            "main_intelligence_id": intelligence.id,
            "emergence_signals_count": len(intelligence.emergence_signals),
            "phase_transition_risk": intelligence.phase_transition_risk,
        }

    def _format_sector_sentiments(
        self,
        sectors: list[str],
        sentiments: list[EconomicSentiment],
    ) -> str:
        """Format sector sentiments for prompt."""
        lines = []
        for sector, sentiment in zip(sectors, sentiments):
            if isinstance(sentiment, EconomicSentiment):
                lines.append(
                    f"- {sector}: tone={sentiment.overall_tone:.1f}, "
                    f"anxiety={sentiment.anxiety_score:.2f}, "
                    f"uncertainty={sentiment.uncertainty_score:.2f}"
                )
            else:
                lines.append(f"- {sector}: data unavailable")
        return "\n".join(lines)

    def _format_frameworks(self, frameworks: list[str]) -> str:
        """Format relevant frameworks for prompt."""
        from emergentomics.cag.context_builder import TheoreticalFrameworkLibrary

        lines = []
        for name in frameworks:
            fw = TheoreticalFrameworkLibrary.get_framework(name)
            if fw:
                lines.append(f"- {fw['name']}: {fw['description'][:100]}...")
        return "\n".join(lines) if lines else "No specific frameworks identified"

    def _summarize_events(self, events: list) -> str:
        """Summarize events for prompt."""
        if not events:
            return "No events to summarize"

        lines = []
        for event in events[:10]:
            tone = f"tone={event.tone:.1f}" if event.tone else "tone=N/A"
            lines.append(
                f"- {event.event_description or 'No title'} ({tone})"
            )
        return "\n".join(lines)
