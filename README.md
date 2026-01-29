# CogFinAgents

# The Cost of Cognition: Deconstructing Noise, Narratives, and Rationality in Financial Agents

This repository implements a configurable multi-agent financial market simulation. Agents observe daily multi-source signals (market, news, social, macro, company, on-chain), optionally retrieve episodic memory, and make trading decisions via an LLM.

> For the public GitHub version, the `data/` folder contains **one month of preprocessed data** so that all experiments can be run without shipping the full proprietary dataset.


## 1) Quick Start

### Requirements

- Python 3.10+ (recommended)
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Run

From the project root:

```bash
python simulation.py
```

By default, the simulation reads:

- `config/simulation.yaml` (experiment switches, assets, dates, output dir, LLM config)
- `config/agents.yaml` (agent profiles + memory parameters)
- `config/data_schema.yaml` (signal schemas + truncation rules)

---

## 2) Data Layout (1-Month Public Subset)

The GitHub repo is configured to use a **1-month subset** at `data_1month/`:

```text
data_1month/
  BTC/
    processed_market.csv
    processed_news.csv
    processed_tweets.csv          # downsampled to 10k rows for this month
    processed_company_data.csv
    processed_company_data_earnings.csv
    processed_on_chain.csv
    processed_trump.csv
  NVDA/
    ...
  GLD/
    ...
  DOGE/
    ...
  KO/
    ...
  global/
    processed_macro.csv
    processed_macro_reports.csv
    processed_trump.csv
```

In `config/simulation.yaml`, the assets root is already set to this subset:

```yaml
assets_root: "./data_1month"
```

If you have the full dataset locally with the same structure under `data/`, you can switch back by changing:

```yaml
assets_root: "./data"
```

---

## 3) Config-Driven Experiments (Three Types)

All experiments are launched via the same command:

```bash
python simulation.py
```

You select **which experiment to run** purely by editing `config/simulation.yaml`. There are three main experiment types:

1. **Full Simulation Baseline**
2. **Trump Social Experiment**
3. **Ablation & Case-Study Data-Drop Experiments**

### 3.1. Experiment A — Full Simulation Baseline

**Goal**: run the full pipeline with all standard data sources and memory enabled.

In `config/simulation.yaml`:

```yaml
trump_social:
  enabled: false

ablation:
  memory_disabled: false
  remove_social_news: false

data_drop_rules:   # keep all rules disabled for baseline
  - name: "case1_drop_macro_D_GLD"
    enabled: false
  - name: "case2_drop_onchain_CD_BTC_DOGE"
    enabled: false
  - name: "case3_drop_company_CD_NVDA_KO"
    enabled: false
```

Then run:

```bash
python simulation.py
```

### 3.2. Experiment B — Trump Social Experiment

**Goal**: replace Reddit-style social data with **Trump Truth Social posts only**, while keeping other signals the same.

In `config/simulation.yaml`:

```yaml
trump_social:
  enabled: true
  path: "./data_1month/global/processed_trump.csv"

ablation:
  memory_disabled: false
  remove_social_news: false

data_drop_rules:
  - name: "case1_drop_macro_D_GLD"
    enabled: false
  - name: "case2_drop_onchain_CD_BTC_DOGE"
    enabled: false
  - name: "case3_drop_company_CD_NVDA_KO"
    enabled: false
```

Internally:

- When `trump_social.enabled: true`:
  - `DataProvider` uses **Trump data only** for `social_data` (no Reddit merge).
  - Social truncation uses `truncation.social_data.max_rows` from `config/data_schema.yaml`.
  - Ranking weight is `favourites_count + reblogs_count + num_comments`.

Run:

```bash
python simulation.py
```

Result filenames get a `_trump` suffix, e.g. `res_gpt-5-mini_2024-11-01_2024-11-30_trump.csv`.

### 3.3. Experiment C — Ablation & Case Study (Data Masking)

**Goal**: remove specific information sources, or mask modules for specific agents/assets/date ranges, to study behavioral differences.

#### 3.3.1. Ablation 1: No-Memory

In `config/simulation.yaml`:

```yaml
ablation:
  memory_disabled: true
  remove_social_news: false
```

Effects:
- Memory retrieval is disabled and a **no-memory prompt** is used in `trader/trading_agent.py`.
- Agents only see current-day signals (no episodic history).

Result filenames get an `_ablation1-memory` suffix.

#### 3.3.2. Ablation 2: No Social + No News

In `config/simulation.yaml`:

```yaml
ablation:
  memory_disabled: false
  remove_social_news: true
```

Effects:
- `DataProvider` clears both `social_data` and `news_data` before constructing the LLM input.
- Memory still exists but is formed from non-social/non-news signals.

Result filenames get an `_ablation2-socialnews` suffix.

#### 3.3.3. Case Study: Data Masking via `data_drop_rules`

**Goal**: for a specific period, selectively drop modules (e.g., macro, on-chain, company data) for specific agents and assets.

In `config/simulation.yaml`:

```yaml
data_drop_rules:
  - name: "case2_drop_onchain_CD_BTC_DOGE"
    enabled: true
    agents: ["C", "D"]
    assets: ["BTC", "DOGE"]
    drop_modules: ["on_chain_data"]
    date_start: "2024-10-01"
    date_end: "2024-12-01"
```

Behavior:
- A rule matches by `(agent_type, asset, date in [date_start, date_end])`.
- For matches, all modules in `drop_modules` are removed from that agent/day/asset input.
- When any case study rule is enabled, the simulation runs over the rule’s asset/date range (it is not constrained by the global range).

Result filenames include the rule name, e.g. `..._case2_drop_onchain_CD_BTC_DOGE.csv`.

---

## 4) Social / News Truncation (Sampling Strategy)

Signal truncation is configured in `config/data_schema.yaml` under `truncation`:

```yaml
truncation:
  social_data:
    max_rows: 10
  news_data:
    max_rows: 10
    sentiment_distribution:
      Positive: 4
      Negative: 4
      Neutral: 2
  macro_data:
    max_text_chars: 3000
```

- **Social data**: keep up to `max_rows` rows per day after ranking.
- **News**: stratified sampling by sentiment with the per-class caps above.
- **Macro**: cap large text fields by character count to avoid LLM overflow.

---

## 5) Agent Memory Hyperparameters

Memory policy is configured per agent type in `config/agents.yaml`:

```yaml
profiles:
  A:
    memory:
      window_days: 10
      sort_key: "recency"
      top_k: 5
```

Common fields:
- `window_days`: trading-day lookback window for candidate episodes.
- `top_k`: max number of episodes to retrieve.
- `sort_key`: ranking criterion (`recency`, `abs_return`, `abs_pnl`, `abs_error`, etc.).

All memory retrievals obey a strict **t-2 constraint** (on day *t*, agents can only see episodes up to *t-2*), implemented in `trader/MemoryPolicy.py`.

---

## 6) Outputs

Outputs are written under `result_dir` from `config/simulation.yaml`, e.g.:

```yaml
result_dir: "./simulation_results"
```

The system writes:
- **Result CSV** (`res_...csv`): one row per episode, with decisions and ex-post enrichment metrics (e.g., pnl, realized_return, prediction_error).
- **Optional LLM I/O traces** (`input_...jsonl` / similar): for reproducibility and auditing when enabled.

Filenames include suffixes for enabled experiment switches (e.g., `_trump`, `_ablation1-memory`, `_ablation2-socialnews`, `_{rule.name}`).

---

