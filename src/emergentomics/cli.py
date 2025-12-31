"""
Command Line Interface for Prophetic Emergentomics.

Provides commands for running emergence detection on GDELT data
and exporting Observable-compatible visualizations.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from emergentomics.core.config import get_settings
from emergentomics.detection.emergence import EmergenceDetector


def print_banner():
    """Print the application banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           PROPHETIC EMERGENTOMICS                             ║
║     ML/DL-Driven Emergence Detection                          ║
║                                                               ║
║   GDELT Behavioral Data + ML Detection                        ║
║   "Economies behave like ecologies, not machines"             ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def cmd_detect(args):
    """Run emergence detection on sample or provided data."""
    print(f"\n[Emergence Detection]")
    print(f"  Data points: {args.n_points}")
    print(f"  Output: {args.output or 'stdout'}\n")

    # Generate sample data or load from file
    if args.input:
        print(f"Loading data from {args.input}...")
        with open(args.input) as f:
            data_dict = json.load(f)
        data = np.array(data_dict.get("values", []))
        timestamps = None
    else:
        print("Using synthetic data for demonstration...")
        # Generate synthetic data with some anomalies and regime changes
        np.random.seed(42)
        n = args.n_points

        # Base signal with regime change
        t = np.arange(n)
        regime_change_point = n // 2
        data = np.where(
            t < regime_change_point,
            np.sin(t / 10) * 0.5 + np.random.randn(n) * 0.2,
            np.sin(t / 10) * 0.5 + 1.5 + np.random.randn(n) * 0.3,
        )

        # Add some anomalies
        anomaly_idx = np.random.choice(n, size=int(n * 0.1), replace=False)
        data[anomaly_idx] += np.random.randn(len(anomaly_idx)) * 2

        timestamps = [
            datetime.utcnow().replace(hour=i % 24, minute=0, second=0, microsecond=0)
            for i in range(n)
        ]

    # Run detection
    detector = EmergenceDetector(
        anomaly_threshold=args.threshold,
    )

    print("Running emergence detection...")
    report = detector.analyze(data, timestamps=timestamps)

    # Export to Observable format
    observable_data = detector.to_observable(report)

    # Output
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(observable_data, f, indent=2)
        print(f"\nResults written to {output_path}")
    else:
        print("\n" + "=" * 60)
        print("EMERGENCE DETECTION RESULTS")
        print("=" * 60)

        meta = observable_data["metadata"]
        summary = observable_data["summary"]

        print(f"\nReport ID: {meta['report_id']}")
        print(f"Generated: {meta['generated_at']}")
        print(f"Confidence: {meta['overall_confidence']:.2f}")

        print(f"\nSummary:")
        print(f"  Signals Detected: {summary['total_signals']}")
        print(f"  Phase Transitions: {summary['phase_transitions']}")
        print(f"  Anomaly Rate: {summary['anomaly_rate']:.1%}")
        print(f"  Clusters: {summary['n_clusters']}")

        if observable_data["signals"]:
            print(f"\nSignals:")
            for sig in observable_data["signals"]:
                print(f"  - [{sig['type']}] {sig['description'][:60]}...")
                print(f"    Confidence: {sig['confidence']:.2f}, Method: {sig['method']}")

        print("\n" + "=" * 60)


def cmd_export_dashboard(args):
    """Export sample data for the Observable dashboard."""
    print(f"\nExporting dashboard data to {args.output}...")

    # Generate comprehensive sample data
    np.random.seed(42)

    observable_data = {
        "metadata": {
            "report_id": f"demo_{datetime.utcnow().strftime('%Y%m%d')}",
            "generated_at": datetime.utcnow().isoformat(),
            "overall_confidence": 0.73,
            "models_used": ["isolation_forest", "hdbscan", "trend_analysis"],
        },
        "summary": {
            "total_signals": 5,
            "phase_transitions": 1,
            "anomaly_rate": 0.12,
            "n_clusters": 4,
        },
        "anomalies": [
            {
                "timestamp": (datetime.utcnow().replace(day=i+1)).isoformat(),
                "score": float(0.4 + np.sin(i/5) * 0.3 + np.random.rand() * 0.2),
                "is_anomaly": np.random.rand() < 0.12,
            }
            for i in range(30)
        ],
        "clusters": {
            "scatter": [
                {
                    "x": float(np.random.randn() + [[-1, -1], [1, -1], [-1, 1], [1, 1]][i % 4][0]),
                    "y": float(np.random.randn() + [[-1, -1], [1, -1], [-1, 1], [1, 1]][i % 4][1]),
                    "cluster": i % 4,
                }
                for i in range(100)
            ],
            "n_clusters": 4,
            "silhouette": 0.65,
        },
        "signals": [
            {
                "type": "anomaly_cascade",
                "description": "Detected correlated anomalies in labor market indicators",
                "confidence": 0.82,
                "method": "isolation_forest",
            },
            {
                "type": "cluster_formation",
                "description": "New economic regime cluster identified",
                "confidence": 0.71,
                "method": "hdbscan",
            },
            {
                "type": "phase_transition",
                "description": "Significant shift in primary indicators",
                "confidence": 0.68,
                "method": "trend_analysis",
            },
        ],
    }

    with open(args.output, "w") as f:
        json.dump(observable_data, f, indent=2)

    print(f"Dashboard data exported to {args.output}")
    print("Use this with the Observable dashboard at docs/index.html")


def cmd_info(args):
    """Show system information."""
    settings = get_settings()

    print("\nProphetic Emergentomics Configuration")
    print("=" * 40)
    print(f"App Name: {settings.app_name}")
    print(f"Version: {settings.app_version}")
    print(f"Debug: {settings.debug}")
    print(f"Log Level: {settings.log_level}")

    print("\nFeature Flags:")
    print(f"  GDELT Integration: {settings.enable_gdelt_integration}")
    print(f"  Anomaly Detection: {settings.enable_anomaly_detection}")
    print(f"  Clustering: {settings.enable_clustering}")
    print(f"  GNN: {settings.enable_gnn}")

    print("\nML Settings:")
    print(f"  Anomaly Method: {settings.ml.anomaly_method}")
    print(f"  Clustering Method: {settings.ml.clustering_method}")
    print(f"  Emergence Threshold: {settings.ml.emergence_threshold}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prophetic Emergentomics - ML-Driven Emergence Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Suppress banner output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # detect command
    detect_parser = subparsers.add_parser(
        "detect",
        help="Run emergence detection on data",
    )
    detect_parser.add_argument(
        "-n", "--n-points",
        type=int,
        default=100,
        help="Number of data points for synthetic data (default: 100)",
    )
    detect_parser.add_argument(
        "-i", "--input",
        type=str,
        help="Input JSON file with data",
    )
    detect_parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file for Observable JSON (default: stdout)",
    )
    detect_parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.7,
        help="Anomaly threshold (default: 0.7)",
    )
    detect_parser.set_defaults(func=cmd_detect)

    # export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export sample data for Observable dashboard",
    )
    export_parser.add_argument(
        "-o", "--output",
        type=str,
        default="docs/data/sample.json",
        help="Output file path",
    )
    export_parser.set_defaults(func=cmd_export_dashboard)

    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show configuration info",
    )
    info_parser.set_defaults(func=cmd_info)

    args = parser.parse_args()

    if not args.no_banner:
        print_banner()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
