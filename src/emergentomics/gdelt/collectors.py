"""
GDELT Data Collectors for continuous economic intelligence gathering.

These collectors provide structured interfaces for gathering specific
types of economic intelligence from GDELT.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional

import structlog

from emergentomics.core.config import get_settings
from emergentomics.core.models import EconomicEvent, EconomicSentiment
from emergentomics.gdelt.client import GDELTClient
from emergentomics.gdelt.parsers import GDELTEventParser, GDELTSentimentParser

logger = structlog.get_logger(__name__)


class EconomicEventCollector:
    """
    Collector for economic events from GDELT.

    Provides domain-specific methods for gathering different types
    of economic events.
    """

    # Pre-defined economic topic queries
    TOPIC_QUERIES = {
        "monetary_policy": [
            "central bank", "interest rate", "federal reserve", "ECB",
            "monetary policy", "inflation target", "rate hike", "rate cut",
        ],
        "fiscal_policy": [
            "government spending", "tax policy", "fiscal stimulus",
            "budget deficit", "national debt", "fiscal policy",
        ],
        "labor_market": [
            "unemployment", "job growth", "labor market", "hiring",
            "layoffs", "workforce", "employment rate", "wage growth",
        ],
        "trade": [
            "trade war", "tariff", "export", "import", "trade deficit",
            "trade agreement", "WTO", "trade policy",
        ],
        "technology": [
            "AI economy", "automation", "tech sector", "digital economy",
            "fintech", "cryptocurrency", "tech layoffs", "tech investment",
        ],
        "markets": [
            "stock market", "bond market", "equity", "bull market",
            "bear market", "market crash", "IPO", "market volatility",
        ],
        "recession_risk": [
            "recession", "economic downturn", "depression", "slowdown",
            "contraction", "negative growth", "hard landing",
        ],
        "growth": [
            "GDP growth", "economic expansion", "boom", "recovery",
            "soft landing", "economic growth", "prosperity",
        ],
    }

    def __init__(self, client: Optional[GDELTClient] = None):
        self.client = client
        self.parser = GDELTEventParser()
        self.settings = get_settings().gdelt

    async def _ensure_client(self) -> GDELTClient:
        """Ensure we have an active client."""
        if self.client is None:
            self.client = GDELTClient()
        return self.client

    async def collect_by_topic(
        self,
        topic: str,
        lookback_hours: int = 24,
        max_events: int = 100,
        countries: Optional[list[str]] = None,
    ) -> list[EconomicEvent]:
        """
        Collect events for a specific economic topic.

        Args:
            topic: One of the predefined topics (monetary_policy, labor_market, etc.)
            lookback_hours: How far back to search
            max_events: Maximum events to return
            countries: Optional country filter

        Returns:
            List of EconomicEvent models
        """
        if topic not in self.TOPIC_QUERIES:
            available = ", ".join(self.TOPIC_QUERIES.keys())
            raise ValueError(f"Unknown topic '{topic}'. Available: {available}")

        query_terms = self.TOPIC_QUERIES[topic]
        query = " OR ".join(f'"{term}"' for term in query_terms)

        start_date = datetime.utcnow() - timedelta(hours=lookback_hours)

        async with GDELTClient() as client:
            articles = await client.search_economic_events(
                query=query,
                countries=countries,
                start_date=start_date,
                max_records=max_events,
            )

        events = self.parser.parse_articles(articles)
        logger.info(
            "Collected economic events",
            topic=topic,
            count=len(events),
            lookback_hours=lookback_hours,
        )

        return events

    async def collect_all_topics(
        self,
        lookback_hours: int = 24,
        max_events_per_topic: int = 50,
        countries: Optional[list[str]] = None,
    ) -> dict[str, list[EconomicEvent]]:
        """
        Collect events across all predefined economic topics.

        Args:
            lookback_hours: How far back to search
            max_events_per_topic: Max events per topic
            countries: Optional country filter

        Returns:
            Dict mapping topic names to event lists
        """
        results: dict[str, list[EconomicEvent]] = {}

        # Run collections concurrently (with rate limiting in client)
        async with GDELTClient() as client:
            self.client = client
            tasks = [
                self.collect_by_topic(topic, lookback_hours, max_events_per_topic, countries)
                for topic in self.TOPIC_QUERIES.keys()
            ]
            topic_results = await asyncio.gather(*tasks, return_exceptions=True)

            for topic, result in zip(self.TOPIC_QUERIES.keys(), topic_results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to collect {topic}", error=str(result))
                    results[topic] = []
                else:
                    results[topic] = result

        self.client = None
        return results

    async def collect_custom_query(
        self,
        query: str,
        lookback_hours: int = 24,
        max_events: int = 100,
        countries: Optional[list[str]] = None,
        themes: Optional[list[str]] = None,
    ) -> list[EconomicEvent]:
        """
        Collect events using a custom search query.

        Args:
            query: Custom GDELT search query
            lookback_hours: How far back to search
            max_events: Maximum events to return
            countries: Optional country filter
            themes: Optional theme filter

        Returns:
            List of EconomicEvent models
        """
        start_date = datetime.utcnow() - timedelta(hours=lookback_hours)

        async with GDELTClient() as client:
            articles = await client.search_economic_events(
                query=query,
                themes=themes,
                countries=countries,
                start_date=start_date,
                max_records=max_events,
            )

        return self.parser.parse_articles(articles)


class SentimentCollector:
    """
    Collector for economic sentiment snapshots.

    Aggregates event data into sentiment indicators.
    """

    def __init__(self):
        self.event_collector = EconomicEventCollector()
        self.sentiment_parser = GDELTSentimentParser()

    async def collect_global_sentiment(
        self,
        lookback_hours: int = 24,
        max_events: int = 200,
    ) -> EconomicSentiment:
        """
        Collect global economic sentiment snapshot.

        Args:
            lookback_hours: How far back to analyze
            max_events: Max events to aggregate

        Returns:
            EconomicSentiment snapshot
        """
        events = await self.event_collector.collect_custom_query(
            query="",  # Empty query = all economic content
            lookback_hours=lookback_hours,
            max_events=max_events,
        )

        return self.sentiment_parser.aggregate_sentiment(
            events=events,
            scope="global",
            aggregation_period=f"{lookback_hours}h",
        )

    async def collect_country_sentiment(
        self,
        country_code: str,
        lookback_hours: int = 24,
        max_events: int = 100,
    ) -> EconomicSentiment:
        """
        Collect sentiment for a specific country.

        Args:
            country_code: ISO country code (e.g., "US", "UK", "DE")
            lookback_hours: How far back to analyze
            max_events: Max events to aggregate

        Returns:
            EconomicSentiment for the country
        """
        events = await self.event_collector.collect_custom_query(
            query="",
            lookback_hours=lookback_hours,
            max_events=max_events,
            countries=[country_code],
        )

        return self.sentiment_parser.aggregate_sentiment(
            events=events,
            scope="country",
            country_code=country_code,
            aggregation_period=f"{lookback_hours}h",
        )

    async def collect_sector_sentiment(
        self,
        sector: str,
        lookback_hours: int = 24,
        max_events: int = 100,
    ) -> EconomicSentiment:
        """
        Collect sentiment for a specific economic sector.

        Args:
            sector: Sector to analyze (finance, technology, labor, etc.)
            lookback_hours: How far back to analyze
            max_events: Max events to aggregate

        Returns:
            EconomicSentiment for the sector
        """
        # Map sectors to relevant topics
        sector_topics = {
            "finance": ["markets", "monetary_policy"],
            "technology": ["technology"],
            "labor": ["labor_market"],
            "trade": ["trade"],
            "fiscal": ["fiscal_policy"],
        }

        topics = sector_topics.get(sector, [])
        all_events: list[EconomicEvent] = []

        for topic in topics:
            try:
                events = await self.event_collector.collect_by_topic(
                    topic=topic,
                    lookback_hours=lookback_hours,
                    max_events=max_events // len(topics) if topics else max_events,
                )
                all_events.extend(events)
            except Exception as e:
                logger.warning(f"Failed to collect {topic} for sector {sector}", error=str(e))

        return self.sentiment_parser.aggregate_sentiment(
            events=all_events,
            scope="sector",
            sector=sector,
            aggregation_period=f"{lookback_hours}h",
        )

    async def collect_multi_country_comparison(
        self,
        country_codes: list[str],
        lookback_hours: int = 24,
    ) -> dict[str, EconomicSentiment]:
        """
        Collect and compare sentiment across multiple countries.

        Args:
            country_codes: List of ISO country codes
            lookback_hours: How far back to analyze

        Returns:
            Dict mapping country codes to sentiment
        """
        tasks = [
            self.collect_country_sentiment(code, lookback_hours)
            for code in country_codes
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            code: result if not isinstance(result, Exception) else None
            for code, result in zip(country_codes, results)
        }


class ThemeCollector:
    """
    Collector for tracking specific economic themes over time.

    Useful for monitoring narratives and tracking theme momentum.
    """

    def __init__(self):
        self.client: Optional[GDELTClient] = None

    async def get_theme_momentum(
        self,
        theme_query: str,
        lookback_days: int = 7,
    ) -> dict[str, Any]:
        """
        Track momentum (volume/attention) for a theme over time.

        Args:
            theme_query: GDELT theme or search query
            lookback_days: Days to analyze

        Returns:
            Momentum data with volume timeline
        """
        async with GDELTClient() as client:
            timeline = await client.get_timeline_volume(
                query=theme_query,
                timespan=f"{lookback_days}d",
            )

        return {
            "theme": theme_query,
            "lookback_days": lookback_days,
            "timeline": timeline.get("timeline", []),
            "collected_at": datetime.utcnow().isoformat(),
        }

    async def get_theme_sentiment_trend(
        self,
        theme_query: str,
        lookback_days: int = 7,
    ) -> dict[str, Any]:
        """
        Track sentiment evolution for a theme over time.

        Args:
            theme_query: GDELT theme or search query
            lookback_days: Days to analyze

        Returns:
            Sentiment trend data
        """
        async with GDELTClient() as client:
            timeline = await client.get_timeline_tone(
                query=theme_query,
                timespan=f"{lookback_days}d",
            )

        return {
            "theme": theme_query,
            "lookback_days": lookback_days,
            "tone_timeline": timeline.get("timeline", []),
            "collected_at": datetime.utcnow().isoformat(),
        }

    async def get_emerging_themes(
        self,
        base_query: str = "",
        lookback_hours: int = 24,
    ) -> dict[str, Any]:
        """
        Identify emerging themes in economic coverage.

        Args:
            base_query: Base query to filter (empty = all economic)
            lookback_hours: How far back to analyze

        Returns:
            Word cloud / theme frequency data
        """
        # Build economic-focused query
        settings = get_settings().gdelt
        if not base_query:
            base_query = " OR ".join(
                f'theme:{t}' for t in settings.economic_themes[:5]
            )

        async with GDELTClient() as client:
            themes = await client.get_word_cloud(
                query=base_query,
                mode="wordcloud",
            )

        return {
            "base_query": base_query,
            "lookback_hours": lookback_hours,
            "themes": themes.get("wordcloud", []),
            "collected_at": datetime.utcnow().isoformat(),
        }
