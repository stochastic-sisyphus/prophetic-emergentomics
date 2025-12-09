"""
Prophetic Emergentomics - Quickstart Example

This example demonstrates the core capabilities of the CAG + GDELT
economic intelligence platform.

Requirements:
    pip install -e .
    export ANTHROPIC_API_KEY=your-key  # or OPENAI_API_KEY

Usage:
    python examples/quickstart.py
"""

import asyncio
from datetime import datetime

from emergentomics import (
    ContextAugmentedEconomicAnalyzer,
    GDELTClient,
    EconomicSynthesisEngine,
    EconomicOpportunityDetector,
)
from emergentomics.medallion import MedallionPipeline
from emergentomics.utils import setup_logging


async def example_gdelt_data_collection():
    """Example: Collect economic events from GDELT."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: GDELT Data Collection")
    print("=" * 60)

    async with GDELTClient() as client:
        # Search for economic events
        events = await client.search_economic_events(
            query="inflation interest rates",
            max_records=10,
        )
        print(f"\nFound {len(events)} events about inflation/interest rates")

        # Get economic sentiment snapshot
        snapshot = await client.get_economic_sentiment_snapshot(
            lookback_hours=24,
        )
        print(f"Analyzed {len(snapshot.get('articles', []))} articles for sentiment")


async def example_cag_analysis():
    """Example: Run Context Augmented Generation analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: CAG Economic Analysis")
    print("=" * 60)

    analyzer = ContextAugmentedEconomicAnalyzer()

    intelligence = await analyzer.analyze(
        query="What are the emerging signals in the US technology sector?",
        focus_area="US Technology Sector",
        lookback_hours=24,
        analysis_depth="moderate",
    )

    print(f"\n[Intelligence ID: {intelligence.id}]")
    print(f"\nExecutive Summary:")
    print(f"  {intelligence.executive_summary[:500]}...")

    print(f"\nEmergence Signals Detected: {len(intelligence.emergence_signals)}")
    for signal in intelligence.emergence_signals[:3]:
        print(f"  - {signal.emergence_type.value}: {signal.description}")

    print(f"\nPhase Transition Risk: {intelligence.phase_transition_risk:.0%}")


async def example_opportunity_detection():
    """Example: Scan for economic opportunities."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Opportunity Detection")
    print("=" * 60)

    detector = EconomicOpportunityDetector()

    opportunities = await detector.scan_opportunities(
        focus_areas=["technology", "labor_market"],
        lookback_hours=48,
        min_confidence=0.3,
    )

    print(f"\nFound {len(opportunities)} potential opportunities:")
    for opp in opportunities[:5]:
        print(f"\n  {opp.opportunity_type.value}: {opp.title}")
        print(f"    Confidence: {opp.confidence:.0%}")
        print(f"    Description: {opp.description[:200]}...")


async def example_synthesis_engine():
    """Example: Use the synthesis engine for specialized analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Economic Synthesis")
    print("=" * 60)

    engine = EconomicSynthesisEngine()

    # Market psychology analysis
    psychology = await engine.synthesize_market_psychology(
        sectors=["technology", "finance"],
        lookback_hours=24,
    )

    print(f"\nMarket Psychology Analysis:")
    print(f"  Dominant Mood: {psychology['dominant_mood']}")
    print(f"  Anxiety: {psychology['aggregate_metrics']['anxiety']:.2f}")
    print(f"  Optimism: {psychology['aggregate_metrics']['optimism']:.2f}")
    print(f"  Uncertainty: {psychology['aggregate_metrics']['uncertainty']:.2f}")


async def example_medallion_pipeline():
    """Example: Run the full medallion pipeline."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Medallion Pipeline (Bronze → Silver → Gold)")
    print("=" * 60)

    pipeline = MedallionPipeline()

    result = await pipeline.run(
        focus_area="Labor Market Dynamics",
        lookback_hours=24,
        skip_gold=False,  # Set to True for faster exploration
    )

    print(f"\nPipeline Results:")
    print(f"  Bronze: {result['bronze'].get('events_collected', 'N/A')} events collected")
    print(f"  Silver: {result['silver'].get('events_in_context', 'N/A')} events in context")
    print(f"  Silver: {result['silver'].get('signals_detected', 'N/A')} signals detected")
    print(f"  Gold: Intelligence ID {result['gold'].get('intelligence_id', 'N/A')}")
    print(f"  Gold: {result['gold'].get('opportunities_count', 0)} opportunities detected")


async def example_gap_analysis():
    """Example: Analyze gap between statistics and lived reality."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Statistical-Reality Gap Analysis")
    print("=" * 60)

    analyzer = ContextAugmentedEconomicAnalyzer()

    # Example: GDP shows growth but people feel recession
    analysis = await analyzer.analyze_gap(
        statistical_indicators={
            "GDP_growth": 2.5,
            "unemployment_rate": 3.8,
            "inflation_rate": 3.2,
            "wage_growth": 4.1,
        },
        focus_area="US Economic Conditions",
        lookback_hours=48,
    )

    print(f"\nGap Analysis:")
    print(f"  {analysis[:800]}...")


async def main():
    """Run all examples."""
    setup_logging(level="WARNING")  # Reduce noise

    print("\n" + "#" * 60)
    print("# PROPHETIC EMERGENTOMICS - QUICKSTART EXAMPLES")
    print("# Event-Driven Economic Intelligence Platform")
    print("#" * 60)
    print(f"\nTimestamp: {datetime.utcnow().isoformat()}")

    # Run examples (comment out any you want to skip)
    await example_gdelt_data_collection()
    await example_cag_analysis()
    await example_opportunity_detection()
    await example_synthesis_engine()
    await example_medallion_pipeline()
    await example_gap_analysis()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
