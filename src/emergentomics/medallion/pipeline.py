"""
Medallion Data Pipeline for Prophetic Emergentomics.

Orchestrates the flow of data through Bronze → Silver → Gold layers,
applying transformations and enrichments at each stage.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional

import structlog

from emergentomics.core.config import get_settings
from emergentomics.core.models import (
    EconomicContext,
    EconomicEvent,
    EconomicIntelligence,
    EconomicSentiment,
    EmergenceSignal,
    MedallionLayer,
    OpportunityAlert,
)
from emergentomics.cag.engine import ContextAugmentedEconomicAnalyzer
from emergentomics.cag.context_builder import EconomicContextBuilder
from emergentomics.gdelt.collectors import EconomicEventCollector, SentimentCollector
from emergentomics.intelligence.detector import EconomicOpportunityDetector, EmergenceDetector
from emergentomics.medallion.storage import BronzeStore, SilverStore, GoldStore

logger = structlog.get_logger(__name__)


class MedallionPipeline:
    """
    Orchestrates data flow through medallion architecture.

    Pipeline stages:
    1. Ingest: Collect raw data from GDELT → Bronze
    2. Enrich: Clean, consolidate, add context → Silver
    3. Synthesize: LLM analysis, pattern detection → Gold

    Example:
        pipeline = MedallionPipeline()

        # Run full pipeline for a focus area
        result = await pipeline.run(
            focus_area="US Labor Market",
            lookback_hours=24,
        )

        print(f"Intelligence ID: {result['gold']['intelligence_id']}")
        print(f"Signals detected: {len(result['gold']['signals'])}")
    """

    def __init__(self):
        self.settings = get_settings()

        # Storage layers
        self.bronze = BronzeStore()
        self.silver = SilverStore()
        self.gold = GoldStore()

        # Processing components
        self.event_collector = EconomicEventCollector()
        self.sentiment_collector = SentimentCollector()
        self.context_builder = EconomicContextBuilder()
        self.analyzer = ContextAugmentedEconomicAnalyzer()
        self.emergence_detector = EmergenceDetector()
        self.opportunity_detector = EconomicOpportunityDetector()

    async def run(
        self,
        focus_area: str,
        query: Optional[str] = None,
        lookback_hours: int = 24,
        countries: Optional[list[str]] = None,
        skip_bronze: bool = False,
        skip_gold: bool = False,
    ) -> dict[str, Any]:
        """
        Run the full medallion pipeline.

        Args:
            focus_area: Economic domain to analyze
            query: Specific search query
            lookback_hours: Data lookback period
            countries: Country filter
            skip_bronze: Skip bronze ingestion (use cached data)
            skip_gold: Skip gold synthesis (faster for data exploration)

        Returns:
            Dict with results from each layer
        """
        logger.info(
            "Starting medallion pipeline",
            focus_area=focus_area,
            lookback_hours=lookback_hours,
        )

        result = {
            "focus_area": focus_area,
            "started_at": datetime.utcnow().isoformat(),
            "bronze": {},
            "silver": {},
            "gold": {},
        }

        # Bronze: Ingest raw data
        if not skip_bronze:
            bronze_result = await self._run_bronze_stage(
                query=query or focus_area,
                lookback_hours=lookback_hours,
                countries=countries,
            )
            result["bronze"] = bronze_result
        else:
            logger.info("Skipping bronze stage")

        # Silver: Build context
        silver_result = await self._run_silver_stage(
            focus_area=focus_area,
            query=query,
            lookback_hours=lookback_hours,
            countries=countries,
        )
        result["silver"] = silver_result

        # Gold: Synthesize intelligence
        if not skip_gold:
            gold_result = await self._run_gold_stage(
                context_id=silver_result.get("context_id"),
                focus_area=focus_area,
                query=query,
                lookback_hours=lookback_hours,
            )
            result["gold"] = gold_result
        else:
            logger.info("Skipping gold stage")

        result["completed_at"] = datetime.utcnow().isoformat()

        return result

    async def _run_bronze_stage(
        self,
        query: str,
        lookback_hours: int,
        countries: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Bronze stage: Ingest raw data.

        Collects events and sentiment from GDELT, stores in Bronze layer.
        """
        logger.info("Running bronze stage")

        # Collect events
        events = await self.event_collector.collect_custom_query(
            query=query,
            lookback_hours=lookback_hours,
            countries=countries,
            max_events=200,
        )

        # Store events
        event_ids = await self.bronze.save_events(events)

        # Collect sentiment
        sentiment = await self.sentiment_collector.collect_global_sentiment(
            lookback_hours=lookback_hours,
            max_events=200,
        )

        # Store sentiment
        sentiment_id = await self.bronze.save_sentiment(sentiment)

        logger.info(
            "Bronze stage complete",
            events_collected=len(events),
            sentiment_id=sentiment_id,
        )

        return {
            "events_collected": len(events),
            "event_ids": event_ids[:10],  # First 10 for reference
            "sentiment_id": sentiment_id,
            "layer": MedallionLayer.BRONZE.value,
        }

    async def _run_silver_stage(
        self,
        focus_area: str,
        query: Optional[str],
        lookback_hours: int,
        countries: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Silver stage: Build enriched context.

        Cleans data, builds context, identifies patterns.
        """
        logger.info("Running silver stage")

        # Build comprehensive context
        context = await self.context_builder.build_context(
            focus_area=focus_area,
            query=query,
            lookback_hours=lookback_hours,
            countries=countries,
        )

        # Store context
        context_id = await self.silver.save_context(context)

        # Detect emergence signals
        signals = self.emergence_detector.aggregate_signals(
            context=context,
            events=context.recent_events,
        )

        logger.info(
            "Silver stage complete",
            context_id=context_id,
            events_in_context=len(context.recent_events),
            signals_detected=len(signals),
        )

        return {
            "context_id": context_id,
            "events_in_context": len(context.recent_events),
            "frameworks_identified": context.relevant_frameworks,
            "dominant_narratives": context.dominant_narratives[:3],
            "emerging_narratives": context.emerging_narratives[:3],
            "signals_detected": len(signals),
            "signal_types": [s.emergence_type.value for s in signals],
            "layer": MedallionLayer.SILVER.value,
        }

    async def _run_gold_stage(
        self,
        context_id: Optional[str],
        focus_area: str,
        query: Optional[str],
        lookback_hours: int,
    ) -> dict[str, Any]:
        """
        Gold stage: Synthesize intelligence.

        LLM analysis, opportunity detection, final intelligence product.
        """
        logger.info("Running gold stage")

        # Run full CAG analysis
        intelligence = await self.analyzer.analyze(
            query=query or f"Comprehensive analysis of {focus_area}",
            focus_area=focus_area,
            lookback_hours=lookback_hours,
            analysis_depth="moderate",
        )

        # Store intelligence
        intel_id = await self.gold.save_intelligence(intelligence)

        # Store emergence signals
        signal_ids = []
        for signal in intelligence.emergence_signals:
            signal_id = await self.gold.save_signal(signal)
            signal_ids.append(signal_id)

        # Detect and store opportunities
        opportunities = await self.opportunity_detector.scan_opportunities(
            focus_areas=[focus_area.lower().replace(" ", "_")],
            lookback_hours=lookback_hours,
        )

        opportunity_ids = []
        for opp in opportunities:
            opp_id = await self.gold.save_opportunity(opp)
            opportunity_ids.append(opp_id)

        logger.info(
            "Gold stage complete",
            intelligence_id=intel_id,
            signals=len(signal_ids),
            opportunities=len(opportunity_ids),
        )

        return {
            "intelligence_id": intel_id,
            "executive_summary": intelligence.executive_summary,
            "key_insights": intelligence.key_insights[:5],
            "signals": signal_ids,
            "phase_transition_risk": intelligence.phase_transition_risk,
            "opportunities": opportunity_ids,
            "opportunities_count": len(opportunities),
            "recommendations": intelligence.strategic_recommendations[:3],
            "layer": MedallionLayer.GOLD.value,
        }

    async def run_continuous(
        self,
        focus_areas: list[str],
        interval_minutes: int = 60,
        max_iterations: Optional[int] = None,
    ):
        """
        Run pipeline continuously at specified intervals.

        Args:
            focus_areas: Areas to monitor
            interval_minutes: Minutes between runs
            max_iterations: Maximum runs (None = infinite)
        """
        iteration = 0

        while max_iterations is None or iteration < max_iterations:
            logger.info(
                "Starting continuous pipeline iteration",
                iteration=iteration,
                focus_areas=focus_areas,
            )

            for area in focus_areas:
                try:
                    await self.run(
                        focus_area=area,
                        lookback_hours=interval_minutes // 60 + 1,
                    )
                except Exception as e:
                    logger.error(
                        "Pipeline iteration failed",
                        focus_area=area,
                        error=str(e),
                    )

            iteration += 1

            if max_iterations is None or iteration < max_iterations:
                logger.info(
                    "Waiting for next iteration",
                    wait_minutes=interval_minutes,
                )
                await asyncio.sleep(interval_minutes * 60)

    async def cleanup(self) -> dict[str, int]:
        """
        Clean up old data based on retention settings.

        Returns:
            Dict with cleanup counts per layer
        """
        settings = self.settings.medallion

        bronze_deleted = await self.bronze.events.cleanup_old(settings.bronze_retention_days)
        silver_deleted = await self.silver.contexts.cleanup_old(settings.silver_retention_days)
        gold_deleted = await self.gold.intelligence.cleanup_old(settings.gold_retention_days)

        logger.info(
            "Cleanup complete",
            bronze_deleted=bronze_deleted,
            silver_deleted=silver_deleted,
            gold_deleted=gold_deleted,
        )

        return {
            "bronze": bronze_deleted,
            "silver": silver_deleted,
            "gold": gold_deleted,
        }

    async def get_recent_intelligence(
        self,
        lookback_hours: int = 24,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get recent intelligence reports."""
        return await self.gold.get_recent_intelligence(
            lookback_hours=lookback_hours,
            limit=limit,
        )

    async def get_active_opportunities(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get active opportunities."""
        return await self.gold.get_active_opportunities(limit=limit)
