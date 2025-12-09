"""
Economic Opportunity Detector.

Algorithms for detecting economic opportunities before they become
obvious to the broader market, using CAG-enhanced pattern recognition.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional

import structlog

from emergentomics.core.config import get_settings
from emergentomics.core.models import (
    EconomicContext,
    EconomicEvent,
    EconomicSentiment,
    EmergenceSignal,
    EmergenceType,
    MedallionLayer,
    OpportunityAlert,
    OpportunityType,
)
from emergentomics.cag.engine import LLMClient
from emergentomics.cag.context_builder import EconomicContextBuilder
from emergentomics.gdelt.collectors import (
    EconomicEventCollector,
    SentimentCollector,
    ThemeCollector,
)

logger = structlog.get_logger(__name__)


class EmergenceDetector:
    """
    Detects emergence signals in economic data.

    Uses pattern recognition and anomaly detection to identify
    early signals of phase transitions, trend accelerations,
    and systemic shifts.
    """

    def __init__(self):
        self.settings = get_settings()

    def detect_sentiment_divergence(
        self,
        sentiment: EconomicSentiment,
        statistical_indicators: Optional[dict[str, float]] = None,
    ) -> Optional[EmergenceSignal]:
        """
        Detect divergence between sentiment and statistics.

        This is a key emergence signal: when how people feel about
        the economy diverges significantly from what metrics show.
        """
        if not sentiment:
            return None

        # Check for significant anxiety despite neutral/positive indicators
        if sentiment.anxiety_score > 0.6 and sentiment.overall_tone > -2:
            return EmergenceSignal(
                id=f"div_anxiety_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                detected_at=datetime.utcnow(),
                emergence_type=EmergenceType.SENTIMENT_DIVERGENCE,
                description="High anxiety despite neutral headline sentiment",
                hypothesis=(
                    "Underlying stress not captured in aggregate metrics. "
                    "May indicate distributional issues or anticipatory anxiety."
                ),
                signal_strength=min(sentiment.anxiety_score, 0.9),
                confidence=0.7,
                supporting_sentiment=[sentiment.id],
                layer=MedallionLayer.GOLD,
            )

        # Check for optimism-uncertainty divergence
        if sentiment.optimism_score > 0.5 and sentiment.uncertainty_score > 0.6:
            return EmergenceSignal(
                id=f"div_optunc_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                detected_at=datetime.utcnow(),
                emergence_type=EmergenceType.SENTIMENT_DIVERGENCE,
                description="High optimism coexisting with high uncertainty",
                hypothesis=(
                    "Market may be pricing in upside without adequately "
                    "pricing risk. Classic pre-correction pattern."
                ),
                signal_strength=min(sentiment.uncertainty_score, 0.8),
                confidence=0.6,
                supporting_sentiment=[sentiment.id],
                layer=MedallionLayer.GOLD,
            )

        return None

    def detect_narrative_shift(
        self,
        context: EconomicContext,
    ) -> Optional[EmergenceSignal]:
        """
        Detect shifts in dominant economic narratives.

        Narrative shifts often precede actual economic shifts.
        """
        if not context.emerging_narratives:
            return None

        # Check if emerging narratives contradict dominant ones
        if context.narrative_conflicts:
            return EmergenceSignal(
                id=f"narr_shift_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                detected_at=datetime.utcnow(),
                emergence_type=EmergenceType.NARRATIVE_SHIFT,
                description=f"Narrative conflict detected: {context.narrative_conflicts[0]}",
                hypothesis=(
                    "Conflicting narratives suggest unstable consensus. "
                    "Market may be approaching inflection point."
                ),
                signal_strength=0.7,
                confidence=0.6,
                aligned_frameworks=["narrative_economics"],
                layer=MedallionLayer.GOLD,
            )

        # Check for rapid emergence of new narratives
        if len(context.emerging_narratives) >= 3:
            return EmergenceSignal(
                id=f"narr_mult_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                detected_at=datetime.utcnow(),
                emergence_type=EmergenceType.NARRATIVE_SHIFT,
                description=f"Multiple emerging narratives: {', '.join(context.emerging_narratives[:3])}",
                hypothesis=(
                    "Multiple new narratives suggest information environment "
                    "is shifting. Traditional consensus may be outdated."
                ),
                signal_strength=0.6,
                confidence=0.5,
                aligned_frameworks=["narrative_economics", "complexity_economics"],
                layer=MedallionLayer.GOLD,
            )

        return None

    def detect_geographic_clustering(
        self,
        events: list[EconomicEvent],
    ) -> Optional[EmergenceSignal]:
        """
        Detect geographic clustering of economic events.

        Concentrated activity in specific regions may signal
        emerging patterns not visible in aggregate data.
        """
        if not events:
            return None

        # Count events by country
        country_counts: dict[str, int] = {}
        for event in events:
            if event.location and event.location.country_code:
                code = event.location.country_code
                country_counts[code] = country_counts.get(code, 0) + 1

        if not country_counts:
            return None

        # Check for significant concentration
        total = sum(country_counts.values())
        for country, count in country_counts.items():
            concentration = count / total
            if concentration > 0.4 and count >= 5:  # 40%+ concentration
                return EmergenceSignal(
                    id=f"geo_cluster_{country}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                    detected_at=datetime.utcnow(),
                    emergence_type=EmergenceType.GEOGRAPHIC_CLUSTERING,
                    description=f"Event concentration in {country} ({concentration:.0%} of coverage)",
                    hypothesis=(
                        f"Significant activity clustering in {country}. "
                        "May indicate regional catalyst or early emergence pattern."
                    ),
                    signal_strength=min(concentration, 0.9),
                    confidence=min(count / 10, 0.8),
                    geographic_scope=country,
                    layer=MedallionLayer.GOLD,
                )

        return None

    def detect_theme_momentum(
        self,
        events: list[EconomicEvent],
        lookback_events: Optional[list[EconomicEvent]] = None,
    ) -> Optional[EmergenceSignal]:
        """
        Detect acceleration in specific themes.

        Theme momentum can signal emerging trends before they
        appear in traditional indicators.
        """
        if len(events) < 10:
            return None

        # Count current theme frequencies
        theme_counts: dict[str, int] = {}
        for event in events:
            for theme in event.themes:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1

        # Find dominant emerging themes
        sorted_themes = sorted(theme_counts.items(), key=lambda x: -x[1])
        if not sorted_themes:
            return None

        top_theme, top_count = sorted_themes[0]
        concentration = top_count / len(events)

        if concentration > 0.3 and top_count >= 5:
            return EmergenceSignal(
                id=f"theme_mom_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                detected_at=datetime.utcnow(),
                emergence_type=EmergenceType.TREND_ACCELERATION,
                description=f"Theme momentum: {top_theme} ({concentration:.0%} of coverage)",
                hypothesis=(
                    f"Strong momentum in {top_theme} theme. "
                    "May indicate accelerating trend or emerging pattern."
                ),
                signal_strength=min(concentration * 1.5, 0.9),
                confidence=min(top_count / 15, 0.8),
                sector_scope=top_theme,
                layer=MedallionLayer.GOLD,
            )

        return None

    def aggregate_signals(
        self,
        context: EconomicContext,
        events: list[EconomicEvent],
    ) -> list[EmergenceSignal]:
        """
        Run all detection algorithms and aggregate signals.

        Args:
            context: Economic context
            events: Economic events

        Returns:
            List of detected emergence signals
        """
        signals = []

        # Check sentiment divergence
        if context.sentiment_snapshot:
            signal = self.detect_sentiment_divergence(context.sentiment_snapshot)
            if signal:
                signals.append(signal)

        # Check narrative shifts
        signal = self.detect_narrative_shift(context)
        if signal:
            signals.append(signal)

        # Check geographic clustering
        signal = self.detect_geographic_clustering(events)
        if signal:
            signals.append(signal)

        # Check theme momentum
        signal = self.detect_theme_momentum(events)
        if signal:
            signals.append(signal)

        return signals


class EconomicOpportunityDetector:
    """
    Detects economic opportunities before they become widely recognized.

    Uses emergence signals, sentiment analysis, and pattern recognition
    to identify potential opportunities for strategic positioning.

    Example:
        detector = EconomicOpportunityDetector()

        opportunities = await detector.scan_opportunities(
            focus_areas=["technology", "emerging_markets"],
            lookback_hours=48,
        )

        for opp in opportunities:
            print(f"{opp.opportunity_type}: {opp.title}")
            print(f"  Confidence: {opp.confidence}")
            print(f"  Description: {opp.description}")
    """

    def __init__(self):
        self.settings = get_settings()
        self.context_builder = EconomicContextBuilder()
        self.emergence_detector = EmergenceDetector()
        self.event_collector = EconomicEventCollector()
        self.sentiment_collector = SentimentCollector()
        self.llm_client = LLMClient()

    async def scan_opportunities(
        self,
        focus_areas: Optional[list[str]] = None,
        lookback_hours: int = 48,
        min_confidence: float = 0.4,
    ) -> list[OpportunityAlert]:
        """
        Scan for economic opportunities across focus areas.

        Args:
            focus_areas: Areas to scan (uses defaults if None)
            lookback_hours: Data lookback period
            min_confidence: Minimum confidence threshold

        Returns:
            List of detected opportunities
        """
        focus_areas = focus_areas or [
            "monetary_policy",
            "technology",
            "labor_market",
            "trade",
            "markets",
        ]

        all_opportunities: list[OpportunityAlert] = []

        for area in focus_areas:
            try:
                opportunities = await self._scan_area(
                    focus_area=area,
                    lookback_hours=lookback_hours,
                    min_confidence=min_confidence,
                )
                all_opportunities.extend(opportunities)
            except Exception as e:
                logger.warning(f"Failed to scan {area}", error=str(e))

        # Sort by confidence
        all_opportunities.sort(key=lambda x: -x.confidence)

        return all_opportunities

    async def _scan_area(
        self,
        focus_area: str,
        lookback_hours: int,
        min_confidence: float,
    ) -> list[OpportunityAlert]:
        """Scan a specific focus area for opportunities."""
        # Build context
        context = await self.context_builder.build_context(
            focus_area=focus_area,
            lookback_hours=lookback_hours,
        )

        # Detect emergence signals
        signals = self.emergence_detector.aggregate_signals(
            context=context,
            events=context.recent_events,
        )

        # Convert relevant signals to opportunities
        opportunities = []

        for signal in signals:
            opportunity = self._signal_to_opportunity(signal, context)
            if opportunity and opportunity.confidence >= min_confidence:
                opportunities.append(opportunity)

        # Add sentiment-based opportunities
        if context.sentiment_snapshot:
            sent_opps = self._detect_sentiment_opportunities(
                context.sentiment_snapshot,
                focus_area,
            )
            opportunities.extend([o for o in sent_opps if o.confidence >= min_confidence])

        return opportunities

    def _signal_to_opportunity(
        self,
        signal: EmergenceSignal,
        context: EconomicContext,
    ) -> Optional[OpportunityAlert]:
        """Convert an emergence signal to an opportunity if applicable."""
        opportunity_map = {
            EmergenceType.TREND_ACCELERATION: OpportunityType.NARRATIVE_MOMENTUM,
            EmergenceType.NARRATIVE_SHIFT: OpportunityType.NARRATIVE_MOMENTUM,
            EmergenceType.SENTIMENT_DIVERGENCE: OpportunityType.SENTIMENT_REVERSAL,
            EmergenceType.TECHNOLOGY_ADOPTION: OpportunityType.TECHNOLOGY_WAVE,
            EmergenceType.POLICY_SIGNAL: OpportunityType.POLICY_ARBITRAGE,
            EmergenceType.SECTOR_CONTAGION: OpportunityType.SECTOR_ROTATION,
            EmergenceType.GEOGRAPHIC_CLUSTERING: OpportunityType.GEOGRAPHIC_ADVANTAGE,
        }

        if signal.emergence_type not in opportunity_map:
            return None

        opp_type = opportunity_map[signal.emergence_type]

        return OpportunityAlert(
            id=f"opp_{signal.id}",
            detected_at=datetime.utcnow(),
            opportunity_type=opp_type,
            title=f"{opp_type.value.replace('_', ' ').title()}: {signal.description[:50]}",
            description=signal.hypothesis,
            confidence=signal.confidence * signal.signal_strength,
            triggering_signals=[signal.id],
            geographic_focus=[signal.geographic_scope] if signal.geographic_scope else [],
            sector_focus=[signal.sector_scope] if signal.sector_scope else [],
            layer=MedallionLayer.GOLD,
        )

    def _detect_sentiment_opportunities(
        self,
        sentiment: EconomicSentiment,
        focus_area: str,
    ) -> list[OpportunityAlert]:
        """Detect opportunities from extreme sentiment readings."""
        opportunities = []

        # Extreme pessimism = potential contrarian opportunity
        if sentiment.anxiety_score > 0.7 and sentiment.overall_tone < -5:
            opportunities.append(OpportunityAlert(
                id=f"opp_contrarian_{datetime.utcnow().strftime('%H%M%S')}",
                detected_at=datetime.utcnow(),
                opportunity_type=OpportunityType.SENTIMENT_REVERSAL,
                title=f"Potential Contrarian Opportunity in {focus_area}",
                description=(
                    f"Extreme negative sentiment (anxiety={sentiment.anxiety_score:.2f}, "
                    f"tone={sentiment.overall_tone:.1f}) may create oversold conditions."
                ),
                urgency="moderate",
                confidence=min(sentiment.anxiety_score * 0.7, 0.7),
                sector_focus=[focus_area],
                risks_to_consider=[
                    "Sentiment may be justified by fundamentals",
                    "Further deterioration possible before reversal",
                ],
                layer=MedallionLayer.GOLD,
            ))

        # Low uncertainty with high conviction = potential timing
        if sentiment.uncertainty_score < 0.3 and abs(sentiment.overall_tone) > 5:
            opportunities.append(OpportunityAlert(
                id=f"opp_timing_{datetime.utcnow().strftime('%H%M%S')}",
                detected_at=datetime.utcnow(),
                opportunity_type=OpportunityType.MARKET_TIMING,
                title=f"High Conviction Window in {focus_area}",
                description=(
                    f"Low uncertainty ({sentiment.uncertainty_score:.2f}) with strong "
                    f"directional sentiment suggests clear positioning opportunity."
                ),
                urgency="high",
                confidence=0.5,
                time_sensitivity="May close as uncertainty rises",
                sector_focus=[focus_area],
                layer=MedallionLayer.GOLD,
            ))

        return opportunities

    async def generate_opportunity_report(
        self,
        opportunities: list[OpportunityAlert],
        include_analysis: bool = True,
    ) -> str:
        """
        Generate a formatted opportunity report.

        Args:
            opportunities: List of detected opportunities
            include_analysis: Whether to include LLM analysis

        Returns:
            Formatted report string
        """
        if not opportunities:
            return "No significant opportunities detected in the current scan."

        report_lines = [
            "# Economic Opportunity Report",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            f"Opportunities Detected: {len(opportunities)}",
            "",
            "---",
        ]

        for i, opp in enumerate(opportunities[:10], 1):
            report_lines.extend([
                f"\n## {i}. {opp.title}",
                f"**Type:** {opp.opportunity_type.value}",
                f"**Confidence:** {opp.confidence:.0%}",
                f"**Urgency:** {opp.urgency}",
                f"\n{opp.description}",
            ])

            if opp.sector_focus:
                report_lines.append(f"\n**Sectors:** {', '.join(opp.sector_focus)}")
            if opp.geographic_focus:
                report_lines.append(f"**Regions:** {', '.join(opp.geographic_focus)}")
            if opp.risks_to_consider:
                report_lines.append("\n**Risks:**")
                for risk in opp.risks_to_consider:
                    report_lines.append(f"  - {risk}")

        if include_analysis and opportunities:
            # Generate LLM analysis of opportunities
            analysis_prompt = f"""Analyze these detected economic opportunities:

{chr(10).join(f'- {o.title}: {o.description}' for o in opportunities[:5])}

Provide:
1. Overall assessment of opportunity quality
2. Priority ranking recommendation
3. Key risks across opportunities
4. Suggested action timeline
5. Monitoring recommendations"""

            analysis = await self.llm_client.generate(analysis_prompt)
            report_lines.extend([
                "\n---",
                "\n## Analysis",
                analysis,
            ])

        return "\n".join(report_lines)

    async def monitor_opportunity(
        self,
        opportunity_id: str,
        lookback_hours: int = 6,
    ) -> dict[str, Any]:
        """
        Monitor an existing opportunity for status changes.

        Args:
            opportunity_id: ID of opportunity to monitor
            lookback_hours: Recent data to check

        Returns:
            Monitoring update
        """
        # This would integrate with persistence layer
        # For now, return a monitoring template
        return {
            "opportunity_id": opportunity_id,
            "monitored_at": datetime.utcnow().isoformat(),
            "status": "active",
            "message": "Opportunity monitoring requires persistence layer integration",
        }
