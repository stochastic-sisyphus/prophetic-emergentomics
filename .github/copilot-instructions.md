# Copilot Instructions for Prophetic Emergentomics

## Repository Overview

This repository implements **The Prophecy of the Emergent Economy** — an event-driven economic intelligence platform that combines Context Augmented Generation (CAG) with GDELT real-time global event monitoring. The platform bridges the gap between what economic indicators say and what people actually experience.

**Core Philosophy**: Traditional econometrics give us the skeleton; LLMs add the nervous system. GDP says growth while people feel recession — we bridge that gap.

## Technical Architecture

### Implemented Stack

```
src/emergentomics/
├── core/           # Configuration and data models (Pydantic)
├── cag/            # Context Augmented Generation engine
├── gdelt/          # GDELT API integration (async httpx)
├── synthesis/      # LLM-powered economic synthesis
├── intelligence/   # Emergence detection algorithms
├── medallion/      # Bronze → Silver → Gold data pipeline
├── patterns/       # Pattern discovery modules
└── cli.py          # Command-line interface
```

### Key Dependencies
- **Async HTTP**: `httpx`, `aiohttp` for GDELT API calls
- **LLM Integration**: `anthropic`, `openai`, `litellm`
- **Data Processing**: `pandas`, `polars`, `numpy`
- **Validation**: `pydantic`, `pydantic-settings`
- **Logging**: `structlog`, `rich`

## Development Guidelines

### Code Style

1. **Async-First**: All GDELT and LLM operations use async/await
2. **Type Hints**: Full type annotations with Pydantic models
3. **Configuration**: Environment-based settings via `pydantic-settings`
4. **Error Handling**: Structured exceptions with proper logging

### Data Models

Core models in `src/emergentomics/core/models.py`:
- `EconomicEvent` — GDELT event with economic classification
- `EconomicSentiment` — Multi-source sentiment aggregation
- `EconomicContext` — Rich context for CAG synthesis
- `EmergenceSignal` — Detected emergence patterns (8 types)
- `EconomicIntelligence` — Gold-layer synthesized intelligence
- `OpportunityAlert` — Actionable market opportunities

### Medallion Architecture

Follow the Bronze → Silver → Gold pattern:

```python
# Bronze: Raw GDELT data
bronze_events = await gdelt_client.search_economic_events(query)

# Silver: Enriched with sentiment, themes, classification
silver_data = await enrichment_pipeline.process(bronze_events)

# Gold: LLM-synthesized intelligence
gold_intelligence = await cag_engine.synthesize(silver_data)
```

### Emergence Signal Types

The platform detects 8 emergence signal types:
1. `phase_transition` — Systemic state changes
2. `narrative_shift` — Collective belief changes
3. `sentiment_divergence` — Indicator-reality gaps
4. `cascade_initiation` — Chain reaction triggers
5. `complexity_spike` — System complexity increases
6. `adaptation_pattern` — Behavioral adjustments
7. `resilience_test` — System stress responses
8. `emergence_crystallization` — New pattern formation

### CAG Implementation

Context Augmented Generation assembles multi-layer context:
1. **Theoretical Layer**: Complexity economics frameworks
2. **Event Layer**: Real-time GDELT events
3. **Sentiment Layer**: Global sentiment analysis
4. **Narrative Layer**: Dominant economic narratives
5. **Statistical Layer**: Traditional indicators

```python
context = await context_builder.build_comprehensive_context(
    query=query,
    events=events,
    sentiment=sentiment,
    frameworks=["emergence_theory", "complexity_economics"]
)
intelligence = await cag_engine.synthesize(context)
```

## Conceptual Vocabulary

- **Emergentomics**: Economic analysis through self-organizing, non-linear interactions
- **Complexity Economics**: Systems as evolving, interdependent networks (not equilibrium)
- **Narrative Economics**: How collective beliefs drive market behaviors
- **Prophetic Economics**: Foresight embracing uncertainty and emergence
- **CAG (Context Augmented Generation)**: LLM synthesis with structured context injection

## Theoretical Frameworks

Five foundational frameworks guide analysis:

1. **Emergence Theory**: Macro patterns from micro interactions
2. **Complexity Economics**: Adaptive systems, not mechanical equilibria
3. **Narrative Economics**: Stories drive economic behavior
4. **Adaptive Markets**: Evolution meets efficient markets
5. **Prophetic Economics**: Uncertainty as structure, not error

## GDELT Integration

### Rate Limiting
GDELT enforces strict rate limits. The client implements:
- Configurable requests per minute
- Automatic backoff on 429 responses
- Request queuing for burst operations

### Query Patterns
```python
# Economic event queries
queries = [
    "inflation OR deflation",
    "unemployment OR jobs",
    "trade tariff sanctions",
    "monetary policy central bank",
    "recession depression economic crisis"
]

# Theme-based queries
themes = ["ECON_BANKRUPTCY", "ECON_PRICECHANGE", "ECON_DEBT"]
```

## Interactive Dashboard

The repository includes an immersive visualization dashboard at `docs/index.html`:
- Plotly.js interactive charts
- GSAP scroll animations
- Live emergence signal feed
- Geographic economic stress mapping
- Hosted via GitHub Pages

## Security Guidelines

- **No hardcoded credentials**: Use environment variables
- **API keys via `.env`**: Never commit secrets
- **Rate limit awareness**: Respect GDELT and LLM provider limits
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

# Analyze a query
emergentomics analyze "semiconductor supply chain disruption"

# Run full pipeline
emergentomics pipeline --focus "energy markets"

# Detect opportunities
emergentomics opportunities

# Generate briefing
emergentomics briefing
```

## Epistemological Constraints

When contributing, remember:
1. **Emergence over equilibrium**: Economic systems don't settle, they evolve
2. **Uncertainty as structure**: Embrace irreducible uncertainty
3. **Real-time over synthetic**: Prioritize live GDELT data over simulations
4. **Gap analysis**: Always compare indicators to lived experience
5. **Probabilistic thinking**: Confidence intervals, not point predictions

## Contribution Focus Areas

Priority areas for enhancement:
- Additional GDELT theme parsers
- New emergence detection algorithms
- Expanded theoretical framework library
- Enhanced opportunity scoring models
- Real-time streaming capabilities
- Multi-language sentiment analysis
