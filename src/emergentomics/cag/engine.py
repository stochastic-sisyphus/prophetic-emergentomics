"""
Context Augmented Economic Analyzer - The Core CAG Engine.

This is the heart of Prophetic Emergentomics: an engine that takes
theoretical predictions, real-time events, and sentiment data, then
synthesizes them through LLM to produce economic intelligence that
bridges statistical metrics with lived reality.
"""

import asyncio
from datetime import datetime
from typing import Any, Optional

import structlog

from emergentomics.core.config import get_settings
from emergentomics.core.models import (
    EconomicContext,
    EconomicIntelligence,
    EmergenceSignal,
    EmergenceType,
    MedallionLayer,
    OpportunityAlert,
    OpportunityType,
)
from emergentomics.cag.context_builder import EconomicContextBuilder
from emergentomics.cag.prompts import CAGPromptTemplates

logger = structlog.get_logger(__name__)


class LLMClient:
    """
    Abstract LLM client interface.

    Supports multiple providers (Anthropic, OpenAI, LiteLLM) with
    a consistent interface.
    """

    def __init__(self):
        self.settings = get_settings().llm
        self._client = None

    async def _ensure_client(self):
        """Initialize the appropriate LLM client."""
        if self._client is not None:
            return

        if self.settings.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(
                    api_key=self.settings.anthropic_api_key
                )
            except ImportError:
                logger.warning("anthropic package not installed")
                self._client = None

        elif self.settings.provider == "openai":
            try:
                import openai
                self._client = openai.AsyncOpenAI(
                    api_key=self.settings.openai_api_key
                )
            except ImportError:
                logger.warning("openai package not installed")
                self._client = None

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        await self._ensure_client()

        if self._client is None:
            return self._generate_mock_response(prompt)

        max_tokens = max_tokens or self.settings.max_tokens
        temperature = temperature or self.settings.temperature
        system_prompt = system_prompt or CAGPromptTemplates.SYSTEM_PROMPT

        try:
            if self.settings.provider == "anthropic":
                response = await self._client.messages.create(
                    model=self.settings.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text

            elif self.settings.provider == "openai":
                response = await self._client.chat.completions.create(
                    model=self.settings.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content or ""

        except Exception as e:
            logger.error("LLM generation failed", error=str(e))
            return self._generate_mock_response(prompt)

        return ""

    def _generate_mock_response(self, prompt: str) -> str:
        """Generate a mock response when LLM is unavailable."""
        return f"""[Mock Analysis - LLM Not Configured]

Based on the provided context, here are placeholder insights:

1. **Statistical vs. Lived Reality Gap**: Unable to perform real analysis without LLM.

2. **Emergence Signals**: Context data suggests potential patterns but requires LLM synthesis.

3. **Key Uncertainties**: Analysis limited without LLM integration.

4. **Recommendations**: Configure LLM provider (Anthropic or OpenAI) for full analysis.

To enable full analysis:
- Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable
- Configure LLM settings in your .env file

Context contained {len(prompt)} characters of data for analysis."""


class ContextAugmentedEconomicAnalyzer:
    """
    The main CAG engine for economic analysis.

    Brings together context building, LLM synthesis, and intelligence
    extraction to produce actionable economic insights.

    Example:
        analyzer = ContextAugmentedEconomicAnalyzer()
        intelligence = await analyzer.analyze(
            query="What are the emerging signals in the US labor market?",
            focus_area="US Labor Market",
            lookback_hours=48,
        )
        print(intelligence.executive_summary)
    """

    def __init__(self):
        self.settings = get_settings()
        self.context_builder = EconomicContextBuilder()
        self.llm_client = LLMClient()

    async def analyze(
        self,
        query: str,
        focus_area: Optional[str] = None,
        lookback_hours: int = 24,
        max_events: int = 100,
        countries: Optional[list[str]] = None,
        analysis_depth: str = "moderate",
        include_emergence_detection: bool = True,
        include_opportunity_detection: bool = True,
    ) -> EconomicIntelligence:
        """
        Perform comprehensive CAG analysis.

        Args:
            query: The specific question or analysis request
            focus_area: Economic domain to focus on (uses query if not provided)
            lookback_hours: How far back to collect data
            max_events: Maximum events to analyze
            countries: Optional country filter
            analysis_depth: shallow, moderate, or deep
            include_emergence_detection: Whether to detect emergence signals
            include_opportunity_detection: Whether to detect opportunities

        Returns:
            EconomicIntelligence with synthesized analysis
        """
        focus_area = focus_area or query

        logger.info(
            "Starting CAG analysis",
            query=query,
            focus_area=focus_area,
            lookback_hours=lookback_hours,
        )

        # Build context
        context = await self.context_builder.build_context(
            focus_area=focus_area,
            query=query,
            lookback_hours=lookback_hours,
            max_events=max_events,
            countries=countries,
        )

        # Run main analysis
        analysis_prompt = CAGPromptTemplates.build_analysis_prompt(
            context=context,
            query=query,
            analysis_depth=analysis_depth,
        )

        analysis_response = await self.llm_client.generate(analysis_prompt)

        # Detect emergence signals if requested
        emergence_signals = []
        if include_emergence_detection:
            emergence_signals = await self._detect_emergence(context)

        # Detect opportunities if requested
        opportunities = []
        if include_opportunity_detection:
            opportunities = await self._detect_opportunities(context)

        # Extract key insights from analysis
        key_insights = self._extract_insights(analysis_response)

        # Calculate phase transition risk
        phase_risk = self._calculate_phase_transition_risk(
            context, emergence_signals
        )

        # Create intelligence object
        intelligence_id = f"intel_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        intelligence = EconomicIntelligence(
            id=intelligence_id,
            created_at=datetime.utcnow(),
            query=query,
            analysis_scope=focus_area,
            executive_summary=self._create_executive_summary(analysis_response),
            detailed_analysis=analysis_response,
            key_insights=key_insights,
            blind_spots_identified=self._identify_blind_spots(context),
            narrative_vs_metrics_gap=self._analyze_narrative_metrics_gap(context),
            emergence_signals=emergence_signals,
            phase_transition_risk=phase_risk,
            scenarios=self._generate_scenarios(context, emergence_signals),
            monitoring_triggers=self._generate_monitoring_triggers(
                context, emergence_signals
            ),
            uncertainty_factors=self._identify_uncertainties(context),
            opportunities_detected=[
                {
                    "type": opp.opportunity_type.value,
                    "title": opp.title,
                    "confidence": opp.confidence,
                }
                for opp in opportunities
            ],
            strategic_recommendations=self._generate_recommendations(
                context, emergence_signals, opportunities
            ),
            context_used=context.id,
            model_used=self.settings.llm.model,
            synthesis_confidence=context.context_completeness,
            data_sources=["GDELT"],
            event_count_analyzed=len(context.recent_events),
            sentiment_snapshots_used=1 if context.sentiment_snapshot else 0,
            layer=MedallionLayer.GOLD,
        )

        logger.info(
            "CAG analysis complete",
            intelligence_id=intelligence.id,
            events_analyzed=intelligence.event_count_analyzed,
            emergence_signals=len(emergence_signals),
        )

        return intelligence

    async def _detect_emergence(
        self,
        context: EconomicContext,
    ) -> list[EmergenceSignal]:
        """Detect emergence signals in the context."""
        prompt = CAGPromptTemplates.build_emergence_detection_prompt(context)
        response = await self.llm_client.generate(prompt)

        # Parse emergence signals from response
        # This is a simplified parser - in production, use structured output
        signals = []

        emergence_types = [
            ("phase_transition", EmergenceType.PHASE_TRANSITION),
            ("trend_acceleration", EmergenceType.TREND_ACCELERATION),
            ("narrative_shift", EmergenceType.NARRATIVE_SHIFT),
            ("policy_signal", EmergenceType.POLICY_SIGNAL),
            ("sentiment_divergence", EmergenceType.SENTIMENT_DIVERGENCE),
            ("geographic_clustering", EmergenceType.GEOGRAPHIC_CLUSTERING),
            ("sector_contagion", EmergenceType.SECTOR_CONTAGION),
            ("technology_adoption", EmergenceType.TECHNOLOGY_ADOPTION),
        ]

        response_lower = response.lower()
        for type_name, emergence_type in emergence_types:
            if type_name.replace("_", " ") in response_lower:
                signal = EmergenceSignal(
                    id=f"signal_{type_name}_{datetime.utcnow().strftime('%H%M%S')}",
                    detected_at=datetime.utcnow(),
                    emergence_type=emergence_type,
                    description=f"Potential {type_name.replace('_', ' ')} detected",
                    hypothesis=f"Evidence suggests emerging {type_name.replace('_', ' ')} pattern",
                    signal_strength=0.6,
                    confidence=0.5,
                    supporting_events=[e.id for e in context.recent_events[:5]],
                    aligned_frameworks=context.relevant_frameworks,
                    layer=MedallionLayer.GOLD,
                )
                signals.append(signal)

        return signals

    async def _detect_opportunities(
        self,
        context: EconomicContext,
    ) -> list[OpportunityAlert]:
        """Detect opportunities in the context."""
        prompt = CAGPromptTemplates.build_opportunity_detection_prompt(context)
        response = await self.llm_client.generate(prompt)

        # Parse opportunities from response
        opportunities = []

        opportunity_types = [
            ("market_timing", OpportunityType.MARKET_TIMING),
            ("sector_rotation", OpportunityType.SECTOR_ROTATION),
            ("narrative_momentum", OpportunityType.NARRATIVE_MOMENTUM),
            ("technology_wave", OpportunityType.TECHNOLOGY_WAVE),
            ("sentiment_reversal", OpportunityType.SENTIMENT_REVERSAL),
        ]

        response_lower = response.lower()
        for type_name, opp_type in opportunity_types:
            if type_name.replace("_", " ") in response_lower:
                opportunity = OpportunityAlert(
                    id=f"opp_{type_name}_{datetime.utcnow().strftime('%H%M%S')}",
                    detected_at=datetime.utcnow(),
                    opportunity_type=opp_type,
                    title=f"{type_name.replace('_', ' ').title()} Opportunity",
                    description=f"Potential {type_name.replace('_', ' ')} opportunity identified",
                    confidence=0.5,
                    triggering_signals=[],
                    layer=MedallionLayer.GOLD,
                )
                opportunities.append(opportunity)

        return opportunities

    def _create_executive_summary(self, full_analysis: str) -> str:
        """Extract executive summary from full analysis."""
        # Take first 2-3 sentences as summary
        sentences = full_analysis.split(". ")
        summary_sentences = sentences[:3]
        return ". ".join(summary_sentences) + "." if summary_sentences else full_analysis[:500]

    def _extract_insights(self, analysis: str) -> list[str]:
        """Extract key insights from analysis text."""
        insights = []

        # Look for numbered points or bullet points
        lines = analysis.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith(("1.", "2.", "3.", "4.", "5.", "-", "*", "•")):
                clean_line = line.lstrip("0123456789.-*• ").strip()
                if len(clean_line) > 20:
                    insights.append(clean_line)

        return insights[:10]

    def _identify_blind_spots(self, context: EconomicContext) -> list[str]:
        """Identify potential blind spots in traditional analysis."""
        blind_spots = []

        if context.narrative_conflicts:
            blind_spots.append(
                "Conflicting narratives suggest uncertainty not captured in point forecasts"
            )

        if context.sentiment_snapshot:
            if context.sentiment_snapshot.anxiety_score > 0.6:
                blind_spots.append(
                    "High anxiety levels may not be reflected in headline statistics"
                )
            if context.sentiment_snapshot.uncertainty_score > 0.6:
                blind_spots.append(
                    "Elevated uncertainty suggests wider outcome distribution than models assume"
                )

        if context.emerging_narratives:
            blind_spots.append(
                "Emerging narratives may signal shifts before they appear in data"
            )

        return blind_spots

    def _analyze_narrative_metrics_gap(self, context: EconomicContext) -> Optional[str]:
        """Analyze gap between narratives and metrics."""
        if not context.sentiment_snapshot:
            return None

        sentiment = context.sentiment_snapshot

        if sentiment.anxiety_score > 0.5 and sentiment.optimism_score < 0.3:
            return (
                "Sentiment suggests economic anxiety despite potentially positive headline metrics. "
                "This gap often reflects distributional effects or lagging lived experience."
            )

        if sentiment.uncertainty_score > 0.6:
            return (
                "High uncertainty in coverage suggests forecasting confidence may be overstated. "
                "Markets and models may be underpricing volatility."
            )

        return None

    def _calculate_phase_transition_risk(
        self,
        context: EconomicContext,
        signals: list[EmergenceSignal],
    ) -> float:
        """Calculate risk of imminent phase transition."""
        risk = 0.0

        # Check for phase transition signals
        phase_signals = [
            s for s in signals
            if s.emergence_type == EmergenceType.PHASE_TRANSITION
        ]
        if phase_signals:
            risk += 0.3

        # High uncertainty is a precursor to transitions
        if context.sentiment_snapshot and context.sentiment_snapshot.uncertainty_score > 0.7:
            risk += 0.2

        # Narrative conflicts suggest instability
        if len(context.narrative_conflicts) >= 2:
            risk += 0.1

        # Multiple strong signals suggest systemic stress
        strong_signals = [s for s in signals if s.signal_strength > 0.7]
        if len(strong_signals) >= 3:
            risk += 0.2

        return min(risk, 1.0)

    def _generate_scenarios(
        self,
        context: EconomicContext,
        signals: list[EmergenceSignal],
    ) -> list[dict[str, Any]]:
        """Generate possible future scenarios."""
        scenarios = []

        # Base case scenario
        scenarios.append({
            "name": "Base Case",
            "probability": 0.5,
            "description": "Continuation of current trends with gradual evolution",
            "key_assumptions": ["No major shocks", "Policy continuity"],
        })

        # Check for acceleration signals
        acceleration_signals = [
            s for s in signals
            if s.emergence_type == EmergenceType.TREND_ACCELERATION
        ]
        if acceleration_signals:
            scenarios.append({
                "name": "Acceleration Scenario",
                "probability": 0.25,
                "description": "Current trends accelerate beyond linear projections",
                "key_assumptions": ["Positive feedback loops activate"],
            })

        # Check for disruption signals
        if context.narrative_conflicts or len(signals) >= 3:
            scenarios.append({
                "name": "Disruption Scenario",
                "probability": 0.15,
                "description": "Significant departure from current trajectory",
                "key_assumptions": ["Major narrative shift", "Policy response"],
            })

        return scenarios

    def _generate_monitoring_triggers(
        self,
        context: EconomicContext,
        signals: list[EmergenceSignal],
    ) -> list[str]:
        """Generate triggers to monitor."""
        triggers = []

        if context.emerging_narratives:
            triggers.append(
                f"Monitor momentum of emerging narratives: {', '.join(context.emerging_narratives[:2])}"
            )

        if context.sentiment_snapshot:
            if context.sentiment_snapshot.anxiety_score > 0.5:
                triggers.append("Watch for anxiety score crossing 0.7 threshold")
            if context.sentiment_snapshot.uncertainty_score > 0.5:
                triggers.append("Monitor uncertainty score for further increases")

        for signal in signals[:3]:
            triggers.append(f"Track evolution of {signal.emergence_type.value} signal")

        return triggers

    def _identify_uncertainties(self, context: EconomicContext) -> list[str]:
        """Identify key uncertainties."""
        uncertainties = []

        if context.data_freshness_hours > 12:
            uncertainties.append("Data freshness may not reflect latest developments")

        if context.source_diversity_score < 0.5:
            uncertainties.append("Limited source diversity may bias analysis")

        if context.context_completeness < 0.7:
            uncertainties.append("Incomplete context may miss relevant signals")

        if context.narrative_conflicts:
            uncertainties.append(
                "Conflicting narratives create uncertainty about market direction"
            )

        return uncertainties

    def _generate_recommendations(
        self,
        context: EconomicContext,
        signals: list[EmergenceSignal],
        opportunities: list[OpportunityAlert],
    ) -> list[str]:
        """Generate strategic recommendations."""
        recommendations = []

        # Based on emergence signals
        if any(s.emergence_type == EmergenceType.PHASE_TRANSITION for s in signals):
            recommendations.append(
                "Consider stress-testing strategies against phase transition scenarios"
            )

        if any(s.emergence_type == EmergenceType.NARRATIVE_SHIFT for s in signals):
            recommendations.append(
                "Monitor narrative evolution and consider positioning for narrative momentum"
            )

        # Based on sentiment
        if context.sentiment_snapshot:
            if context.sentiment_snapshot.anxiety_score > 0.6:
                recommendations.append(
                    "High anxiety environment - consider defensive positioning"
                )
            if context.sentiment_snapshot.uncertainty_score > 0.6:
                recommendations.append(
                    "Elevated uncertainty - consider volatility strategies"
                )

        # Based on opportunities
        for opp in opportunities[:2]:
            recommendations.append(
                f"Evaluate {opp.opportunity_type.value} opportunity: {opp.title}"
            )

        return recommendations[:5]

    async def analyze_gap(
        self,
        statistical_indicators: dict[str, float],
        focus_area: str,
        lookback_hours: int = 24,
    ) -> str:
        """
        Analyze the gap between statistical indicators and lived reality.

        This is the core CAG insight: explaining why metrics and
        experience diverge.

        Args:
            statistical_indicators: Dict of indicator name to value
            focus_area: Economic domain
            lookback_hours: Data lookback period

        Returns:
            Analysis of the gap
        """
        context = await self.context_builder.build_context(
            focus_area=focus_area,
            lookback_hours=lookback_hours,
        )

        prompt = CAGPromptTemplates.build_gap_analysis_prompt(
            context=context,
            statistical_indicators=statistical_indicators,
        )

        return await self.llm_client.generate(prompt)
