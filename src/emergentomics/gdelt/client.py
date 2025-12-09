"""
GDELT API Client for real-time economic intelligence.

The GDELT Project monitors print, broadcast, and web news globally,
identifying events, sentiment, and themes in real-time.

API Documentation: https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional
from urllib.parse import urlencode

import httpx
import structlog

from emergentomics.core.config import get_settings, GDELTSettings
from emergentomics.core.models import EconomicEvent, EconomicSentiment, MedallionLayer

logger = structlog.get_logger(__name__)


class GDELTRateLimiter:
    """Simple rate limiter for GDELT API requests."""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time: Optional[float] = None
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait if necessary to respect rate limits."""
        async with self._lock:
            if self.last_request_time is not None:
                elapsed = asyncio.get_event_loop().time() - self.last_request_time
                if elapsed < self.min_interval:
                    await asyncio.sleep(self.min_interval - elapsed)
            self.last_request_time = asyncio.get_event_loop().time()


class GDELTClient:
    """
    Async client for GDELT 2.0 API.

    Provides access to:
    - DOC API: Full-text article search with sentiment/theme analysis
    - GEO API: Geographic event mapping
    - TV API: Television news monitoring (if available)

    Example:
        async with GDELTClient() as client:
            events = await client.search_economic_events(
                query="inflation recession",
                start_date=datetime.now() - timedelta(days=1)
            )
    """

    def __init__(self, settings: Optional[GDELTSettings] = None):
        self.settings = settings or get_settings().gdelt
        self._client: Optional[httpx.AsyncClient] = None
        self._rate_limiter = GDELTRateLimiter(self.settings.requests_per_minute)

    async def __aenter__(self) -> "GDELTClient":
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'async with GDELTClient()' context.")
        return self._client

    async def _request(
        self,
        url: str,
        params: dict[str, Any],
        retries: int = 3,
    ) -> dict[str, Any]:
        """Make a rate-limited request to GDELT API."""
        await self._rate_limiter.acquire()

        for attempt in range(retries):
            try:
                response = await self.client.get(url, params=params)
                response.raise_for_status()

                # GDELT returns JSON for most queries
                return response.json()

            except httpx.HTTPStatusError as e:
                logger.warning(
                    "GDELT API error",
                    status_code=e.response.status_code,
                    attempt=attempt + 1,
                    url=url,
                )
                if attempt < retries - 1:
                    await asyncio.sleep(self.settings.retry_delay_seconds * (2**attempt))
                else:
                    raise

            except httpx.RequestError as e:
                logger.warning(
                    "GDELT request failed",
                    error=str(e),
                    attempt=attempt + 1,
                    url=url,
                )
                if attempt < retries - 1:
                    await asyncio.sleep(self.settings.retry_delay_seconds * (2**attempt))
                else:
                    raise

        return {}

    def _build_economic_query(
        self,
        base_query: str = "",
        themes: Optional[list[str]] = None,
        countries: Optional[list[str]] = None,
    ) -> str:
        """Build a GDELT query string focused on economic content."""
        query_parts = []

        if base_query:
            query_parts.append(base_query)

        # Add economic theme filters
        if themes is None:
            themes = self.settings.economic_themes

        if themes:
            theme_query = " OR ".join(f'theme:{t}' for t in themes[:10])  # Limit themes
            query_parts.append(f"({theme_query})")

        # Add country filters
        if countries:
            country_query = " OR ".join(f'sourcecountry:{c}' for c in countries)
            query_parts.append(f"({country_query})")

        return " ".join(query_parts)

    async def search_documents(
        self,
        query: str,
        mode: str = "artlist",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_records: Optional[int] = None,
        source_lang: Optional[str] = None,
        source_country: Optional[str] = None,
        sort: str = "datedesc",
        format: str = "json",
    ) -> dict[str, Any]:
        """
        Search GDELT DOC 2.0 API for articles.

        Args:
            query: Search query (supports GDELT query syntax)
            mode: artlist, artgallery, timelinevol, etc.
            start_date: Start of date range
            end_date: End of date range
            max_records: Maximum records to return
            source_lang: Filter by source language (e.g., 'english')
            source_country: Filter by source country code
            sort: Sort order (datedesc, dateasc, toneasc, tonedesc)
            format: Output format (json, html)

        Returns:
            API response as dict
        """
        params: dict[str, Any] = {
            "query": query,
            "mode": mode,
            "format": format,
            "sort": sort,
        }

        if start_date:
            params["startdatetime"] = start_date.strftime("%Y%m%d%H%M%S")
        if end_date:
            params["enddatetime"] = end_date.strftime("%Y%m%d%H%M%S")
        if max_records:
            params["maxrecords"] = min(max_records, self.settings.max_records_per_query)
        if source_lang:
            params["sourcelang"] = source_lang
        if source_country:
            params["sourcecountry"] = source_country

        return await self._request(self.settings.doc_api_url, params)

    async def search_economic_events(
        self,
        query: str = "",
        themes: Optional[list[str]] = None,
        countries: Optional[list[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_records: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Search for economic events using GDELT DOC API.

        Automatically applies economic theme filters to focus results.

        Args:
            query: Base search query
            themes: Economic themes to filter (uses defaults if None)
            countries: Country codes to filter
            start_date: Start of date range
            end_date: End of date range
            max_records: Maximum records to return

        Returns:
            List of article/event records
        """
        if start_date is None:
            start_date = datetime.utcnow() - timedelta(hours=self.settings.lookback_hours)

        economic_query = self._build_economic_query(query, themes, countries)

        logger.info(
            "Searching GDELT for economic events",
            query=economic_query,
            start_date=start_date.isoformat(),
        )

        response = await self.search_documents(
            query=economic_query,
            mode="artlist",
            start_date=start_date,
            end_date=end_date,
            max_records=max_records,
        )

        articles = response.get("articles", [])
        logger.info("Retrieved economic events", count=len(articles))

        return articles

    async def get_timeline_volume(
        self,
        query: str,
        timespan: str = "7d",
        resolution: str = "hour",
    ) -> dict[str, Any]:
        """
        Get volume timeline for a query.

        Useful for tracking attention/coverage over time.

        Args:
            query: Search query
            timespan: Time span (e.g., "7d", "30d", "1y")
            resolution: Time resolution (hour, day, week, month)

        Returns:
            Timeline data with volume counts
        """
        params = {
            "query": query,
            "mode": "timelinevol",
            "timezoom": "yes",
            "timespan": timespan,
            "format": "json",
        }

        return await self._request(self.settings.doc_api_url, params)

    async def get_timeline_tone(
        self,
        query: str,
        timespan: str = "7d",
    ) -> dict[str, Any]:
        """
        Get sentiment/tone timeline for a query.

        Tracks how sentiment around a topic evolves over time.

        Args:
            query: Search query
            timespan: Time span

        Returns:
            Timeline data with tone measurements
        """
        params = {
            "query": query,
            "mode": "timelinetone",
            "timespan": timespan,
            "format": "json",
        }

        return await self._request(self.settings.doc_api_url, params)

    async def get_word_cloud(
        self,
        query: str,
        mode: str = "wordcloud",
    ) -> dict[str, Any]:
        """
        Get word/theme cloud for a query.

        Useful for identifying dominant narratives and themes.

        Args:
            query: Search query
            mode: wordcloud or themegraph

        Returns:
            Word/theme frequency data
        """
        params = {
            "query": query,
            "mode": mode,
            "format": "json",
        }

        return await self._request(self.settings.doc_api_url, params)

    async def get_geo_data(
        self,
        query: str,
        mode: str = "pointdata",
    ) -> dict[str, Any]:
        """
        Get geographic distribution of events.

        Maps where events are occurring globally.

        Args:
            query: Search query
            mode: pointdata, heatmap, etc.

        Returns:
            Geographic event data
        """
        params = {
            "query": query,
            "mode": mode,
            "format": "json",
        }

        return await self._request(self.settings.geo_api_url, params)

    async def get_economic_sentiment_snapshot(
        self,
        themes: Optional[list[str]] = None,
        countries: Optional[list[str]] = None,
        lookback_hours: int = 24,
    ) -> dict[str, Any]:
        """
        Get a snapshot of current economic sentiment.

        Combines multiple GDELT queries to build a comprehensive
        picture of economic sentiment.

        Args:
            themes: Economic themes to track
            countries: Countries to focus on
            lookback_hours: How far back to look

        Returns:
            Aggregated sentiment data
        """
        if themes is None:
            themes = self.settings.economic_themes[:5]  # Top 5 themes

        start_date = datetime.utcnow() - timedelta(hours=lookback_hours)

        # Build comprehensive economic query
        economic_query = self._build_economic_query("", themes, countries)

        # Get multiple data points concurrently
        articles_task = self.search_documents(
            query=economic_query,
            mode="artlist",
            start_date=start_date,
            max_records=100,
        )

        tone_task = self.get_timeline_tone(
            query=economic_query,
            timespan=f"{lookback_hours}h" if lookback_hours <= 72 else "7d",
        )

        # themes_task = self.get_word_cloud(
        #     query=economic_query,
        #     mode="wordcloud",
        # )

        articles, tone = await asyncio.gather(
            articles_task,
            tone_task,
            return_exceptions=True,
        )

        # Handle potential errors
        if isinstance(articles, Exception):
            logger.warning("Failed to get articles", error=str(articles))
            articles = {"articles": []}
        if isinstance(tone, Exception):
            logger.warning("Failed to get tone", error=str(tone))
            tone = {}

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "lookback_hours": lookback_hours,
            "articles": articles.get("articles", []) if isinstance(articles, dict) else [],
            "tone_timeline": tone.get("timeline", []) if isinstance(tone, dict) else [],
            "query_used": economic_query,
        }


# Synchronous convenience wrapper
def create_gdelt_client() -> GDELTClient:
    """Create a GDELT client (must be used with async context manager)."""
    return GDELTClient()
