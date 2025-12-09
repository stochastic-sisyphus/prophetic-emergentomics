"""
Command Line Interface for Prophetic Emergentomics.

Provides commands for running analyses, managing pipelines,
and generating economic intelligence.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from emergentomics.core.config import get_settings
from emergentomics.utils.logging import setup_logging


console = Console()


def print_banner():
    """Print the application banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║           PROPHETIC EMERGENTOMICS                             ║
║     Event-Driven Economic Intelligence Platform               ║
║                                                               ║
║   CAG Framework + GDELT Real-Time Intelligence                ║
║   "Traditional econometrics give us the skeleton;             ║
║    LLMs add the nervous system"                               ║
╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="blue"))


async def run_analysis(
    query: str,
    focus_area: Optional[str] = None,
    lookback_hours: int = 24,
    depth: str = "moderate",
    output_format: str = "text",
):
    """Run a CAG analysis."""
    from emergentomics.cag.engine import ContextAugmentedEconomicAnalyzer

    console.print(f"\n[bold blue]Running Analysis[/bold blue]")
    console.print(f"Query: {query}")
    console.print(f"Focus Area: {focus_area or query}")
    console.print(f"Lookback: {lookback_hours}h")
    console.print(f"Depth: {depth}\n")

    analyzer = ContextAugmentedEconomicAnalyzer()

    with console.status("Analyzing..."):
        intelligence = await analyzer.analyze(
            query=query,
            focus_area=focus_area,
            lookback_hours=lookback_hours,
            analysis_depth=depth,
        )

    if output_format == "json":
        print(json.dumps(intelligence.model_dump(mode="json"), indent=2, default=str))
    else:
        # Pretty print results
        console.print(Panel(
            intelligence.executive_summary,
            title="Executive Summary",
            style="green",
        ))

        if intelligence.key_insights:
            console.print("\n[bold]Key Insights:[/bold]")
            for i, insight in enumerate(intelligence.key_insights[:5], 1):
                console.print(f"  {i}. {insight}")

        if intelligence.emergence_signals:
            console.print(f"\n[bold yellow]Emergence Signals: {len(intelligence.emergence_signals)}[/bold yellow]")
            for signal in intelligence.emergence_signals[:3]:
                console.print(f"  • {signal.emergence_type.value}: {signal.description}")

        console.print(f"\n[bold]Phase Transition Risk:[/bold] {intelligence.phase_transition_risk:.0%}")

        if intelligence.strategic_recommendations:
            console.print("\n[bold]Recommendations:[/bold]")
            for rec in intelligence.strategic_recommendations[:3]:
                console.print(f"  → {rec}")


async def run_pipeline(
    focus_area: str,
    lookback_hours: int = 24,
    skip_gold: bool = False,
):
    """Run the medallion pipeline."""
    from emergentomics.medallion.pipeline import MedallionPipeline

    console.print(f"\n[bold blue]Running Medallion Pipeline[/bold blue]")
    console.print(f"Focus Area: {focus_area}")
    console.print(f"Lookback: {lookback_hours}h\n")

    pipeline = MedallionPipeline()

    with console.status("Processing Bronze → Silver → Gold..."):
        result = await pipeline.run(
            focus_area=focus_area,
            lookback_hours=lookback_hours,
            skip_gold=skip_gold,
        )

    # Display results
    table = Table(title="Pipeline Results")
    table.add_column("Layer", style="cyan")
    table.add_column("Metric", style="magenta")
    table.add_column("Value", style="green")

    table.add_row("Bronze", "Events Collected", str(result["bronze"].get("events_collected", "N/A")))
    table.add_row("Silver", "Events in Context", str(result["silver"].get("events_in_context", "N/A")))
    table.add_row("Silver", "Signals Detected", str(result["silver"].get("signals_detected", "N/A")))

    if not skip_gold:
        table.add_row("Gold", "Intelligence ID", result["gold"].get("intelligence_id", "N/A"))
        table.add_row("Gold", "Opportunities", str(result["gold"].get("opportunities_count", 0)))

    console.print(table)

    if not skip_gold and result["gold"].get("executive_summary"):
        console.print(Panel(
            result["gold"]["executive_summary"],
            title="Intelligence Summary",
            style="green",
        ))


async def scan_opportunities(
    lookback_hours: int = 48,
    min_confidence: float = 0.4,
):
    """Scan for economic opportunities."""
    from emergentomics.intelligence.detector import EconomicOpportunityDetector

    console.print(f"\n[bold blue]Scanning for Opportunities[/bold blue]")
    console.print(f"Lookback: {lookback_hours}h")
    console.print(f"Min Confidence: {min_confidence:.0%}\n")

    detector = EconomicOpportunityDetector()

    with console.status("Scanning..."):
        opportunities = await detector.scan_opportunities(
            lookback_hours=lookback_hours,
            min_confidence=min_confidence,
        )

    if not opportunities:
        console.print("[yellow]No significant opportunities detected.[/yellow]")
        return

    console.print(f"[bold green]Found {len(opportunities)} opportunities:[/bold green]\n")

    for i, opp in enumerate(opportunities[:10], 1):
        console.print(f"[bold]{i}. {opp.title}[/bold]")
        console.print(f"   Type: {opp.opportunity_type.value}")
        console.print(f"   Confidence: {opp.confidence:.0%}")
        console.print(f"   Urgency: {opp.urgency}")
        console.print(f"   {opp.description}\n")


async def generate_briefing(
    lookback_hours: int = 24,
):
    """Generate an economic briefing."""
    from emergentomics.synthesis.engine import EconomicSynthesisEngine

    console.print(f"\n[bold blue]Generating Economic Briefing[/bold blue]")
    console.print(f"Lookback: {lookback_hours}h\n")

    engine = EconomicSynthesisEngine()

    with console.status("Generating briefing..."):
        briefing = await engine.generate_economic_briefing(
            lookback_hours=lookback_hours,
        )

    console.print(Panel(
        briefing.get("executive_briefing", "No briefing generated"),
        title=f"Economic Briefing - {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        style="blue",
    ))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prophetic Emergentomics - Event-Driven Economic Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  emergentomics analyze "US labor market trends"
  emergentomics analyze --focus "Technology Sector" "AI impact on employment"
  emergentomics pipeline --focus "Monetary Policy" --lookback 48
  emergentomics opportunities --lookback 72 --min-confidence 0.5
  emergentomics briefing
        """,
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run CAG analysis")
    analyze_parser.add_argument("query", help="Analysis query")
    analyze_parser.add_argument("--focus", "-f", help="Focus area for analysis")
    analyze_parser.add_argument("--lookback", "-l", type=int, default=24, help="Lookback hours")
    analyze_parser.add_argument(
        "--depth", "-d",
        choices=["shallow", "moderate", "deep"],
        default="moderate",
        help="Analysis depth",
    )

    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run medallion pipeline")
    pipeline_parser.add_argument("--focus", "-f", required=True, help="Focus area")
    pipeline_parser.add_argument("--lookback", "-l", type=int, default=24, help="Lookback hours")
    pipeline_parser.add_argument("--skip-gold", action="store_true", help="Skip gold synthesis")

    # Opportunities command
    opp_parser = subparsers.add_parser("opportunities", help="Scan for opportunities")
    opp_parser.add_argument("--lookback", "-l", type=int, default=48, help="Lookback hours")
    opp_parser.add_argument("--min-confidence", "-c", type=float, default=0.4, help="Minimum confidence")

    # Briefing command
    brief_parser = subparsers.add_parser("briefing", help="Generate economic briefing")
    brief_parser.add_argument("--lookback", "-l", type=int, default=24, help="Lookback hours")

    args = parser.parse_args()

    # Setup
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(level=log_level)

    if not args.json:
        print_banner()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Run appropriate command
    try:
        if args.command == "analyze":
            asyncio.run(run_analysis(
                query=args.query,
                focus_area=args.focus,
                lookback_hours=args.lookback,
                depth=args.depth,
                output_format="json" if args.json else "text",
            ))

        elif args.command == "pipeline":
            asyncio.run(run_pipeline(
                focus_area=args.focus,
                lookback_hours=args.lookback,
                skip_gold=args.skip_gold,
            ))

        elif args.command == "opportunities":
            asyncio.run(scan_opportunities(
                lookback_hours=args.lookback,
                min_confidence=args.min_confidence,
            ))

        elif args.command == "briefing":
            asyncio.run(generate_briefing(
                lookback_hours=args.lookback,
            ))

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if args.debug:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
