# Copilot Instructions for Prophetic Emergentomics

## Repository Overview

This repository implements **The Prophecy of the Emergent Economy** — an ML/DL-driven emergence detection platform for complex economic systems. GDELT provides behavioral data; ML models detect phase transitions and regime changes.

**Core Philosophy**: Economies behave like ecologies, not machines. We detect emergence, not predict outcomes. Uncertainty is structure, not error.

## Technical Architecture

### Implemented Stack

```
src/emergentomics/
├── core/           # Configuration and data models (Pydantic)
├── gdelt/          # GDELT API integration (async httpx)
├── detection/      # ML-based emergence detection
│   ├── anomaly.py     # Isolation forest, statistical anomaly detection
│   ├── clustering.py  # HDBSCAN, K-means regime identification
│   └── emergence.py   # Combined emergence detection
└── cli.py          # Command-line interface
```

### Key Dependencies
- **ML/DL Core**: `scikit-learn` (anomaly detection, clustering)
- **Async HTTP**: `httpx`, `aiohttp` for GDELT API calls
- **Data Processing**: `pandas`, `polars`, `numpy`
- **Network Analysis**: `networkx` for economic graphs
- **Visualization**: `plotly` for output generation
- **Validation**: `pydantic`, `pydantic-settings`
- **Optional ML**: `torch`, `hdbscan`, `umap-learn`, `torch-geometric`

## Development Guidelines

### Code Style

1. **Type Hints**: Full type annotations with Pydantic models
2. **Async for I/O**: GDELT operations use async/await
3. **Configuration**: Environment-based settings via `pydantic-settings`
4. **Observable Output**: All detection results export to JSON for visualization

### Data Models

Core models in `src/emergentomics/core/models.py`:
- `GDELTEvent` — Event from GDELT behavioral data stream
- `GDELTSentiment` — Aggregated sentiment from news coverage
- `AnomalyScore` — Anomaly detection result
- `ClusteringResult` — Regime clustering output
- `EmergenceSignal` — Detected emergence pattern (8 types)
- `PhaseTransition` — Detected regime change
- `EmergenceReport` — Complete analysis output

### Data Flow

```python
# 1. Collect behavioral data from GDELT
events = await gdelt_client.search_economic_events(query)

# 2. Extract features
tones = [e.tone for e in events]
data = np.array(tones)

# 3. Run ML detection
detector = EmergenceDetector()
report = detector.analyze(data, gdelt_events=events)

# 4. Export for Observable visualization
observable_json = detector.to_observable(report)
```

### Emergence Signal Types

The platform detects 8 emergence signal types:
1. `phase_transition` — Regime change detected
2. `trend_acceleration` — Exponential departure from baseline
3. `cluster_formation` — New grouping in feature space
4. `anomaly_cascade` — Propagating anomalies across network
5. `network_restructuring` — Topology shift in economic graph
6. `sentiment_divergence` — Behavior vs indicators gap
7. `contagion_pattern` — Cross-sector/region spread
8. `adaptation_signal` — System adjusting to new equilibrium

### ML Methods

**Anomaly Detection:**
- Isolation Forest (sklearn)
- Statistical Z-score fallback
- Feature contribution analysis

**Clustering:**
- HDBSCAN for density-based regime detection
- K-means for partitional clustering
- PCA/UMAP for dimensionality reduction

**Network Analysis (planned):**
- Graph Neural Networks for economic network structure
- Centrality metrics for critical node identification

## Conceptual Vocabulary

- **Emergentomics**: Economic analysis through self-organizing, non-linear interactions
- **Complexity Economics**: Systems as evolving networks, not equilibrium machines
- **Narrative Economics**: How collective beliefs drive market behaviors
- **Prophetic Economics**: Structural inference under uncertainty

## GDELT Integration

GDELT serves as the primary behavioral data source — capturing the information ecosystem's traces rather than official statistics.

### Rate Limiting
- Configurable requests per minute
- Automatic backoff on 429 responses
- Request queuing for burst operations

### Economic Theme Queries
```python
themes = [
    "ECON_BANKRUPTCY", "ECON_PRICECHANGE", "ECON_DEBT",
    "TAX_", "TRADE_", "INFLATION", "UNEMPLOYMENT"
]
```

## Observable Dashboard

The repository includes an interactive visualization at `docs/index.html`:
- Observable Plot for charts (anomaly timeline, cluster scatter)
- D3.js for network visualization
- Sentiment distribution histogram
- Emergence signal cards
- Hosted via GitHub Pages

## Security Guidelines

- **No hardcoded credentials**: Use environment variables
- **API keys via `.env`**: Never commit secrets
- **Rate limit awareness**: Respect GDELT API limits
- **Data validation**: All external data passes through Pydantic models

## Testing

```bash
# Run tests
pytest tests/

# Type checking
mypy src/emergentomics/

# Linting
ruff check src/
```

## CLI Usage

```bash
# Install
pip install -e .

# Install with full ML dependencies
pip install -e ".[ml-full]"
```

## Epistemological Constraints

When contributing, remember:
1. **Emergence over equilibrium**: Economic systems evolve, they don't settle
2. **Uncertainty as structure**: Embrace irreducible uncertainty
3. **Behavioral traces over official stats**: GDELT captures what people write, not what institutions report
4. **Detection over prediction**: Find patterns forming, don't forecast outcomes
5. **Observable output**: All results should be visualizable

## Contribution Focus Areas

Priority areas for enhancement:
- Additional GDELT theme parsers
- New anomaly detection algorithms
- GNN implementation for network analysis
- Enhanced clustering methods
- Real-time streaming capabilities
- Additional alternative data source integrations
