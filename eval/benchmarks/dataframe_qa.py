#!/usr/bin/env python3
"""
DataFrame QA Benchmark — Jupyter-notebook-style data analysis tasks.

Simulates the real-world scenario: a quant loads a large CSV dataset in their
Jupyter notebook (cell 1: pd.read_csv()), and needs to answer analytical
questions about it. The dataset is too large to fit in an LLM's context window.

This is where RLMs shine: the model sees only metadata about the dataset,
writes Python code to chunk/search/aggregate, and uses llm_query() to
interpret sub-sections.

Task types (ordered by difficulty):
- O(1) Lookup:      "What is the price of AAPL on 2024-03-15?"
- O(N) Aggregation: "What is the average return of tech stocks?"
- O(N) Ranking:     "Which stock had the highest volatility?"
- O(N²) Cross:      "Which pair of stocks has the highest correlation?"
- Multi-step:       "Find the best-performing sector, then list its top 3 stocks"

Context format (stored as `context` variable in REPL):
```
QUESTION: <analytical question>

DATASET:
date,ticker,open,high,low,close,volume,sector
2024-01-02,AAPL,185.5,186.7,184.2,186.1,50234000,Technology
2024-01-02,GOOGL,140.1,141.3,139.8,140.9,25678000,Technology
...
```

The model must parse the CSV, compute the answer programmatically.
"""

from __future__ import annotations

import random
import math
from dataclasses import dataclass, field


# Sectors and their stocks
SECTORS = {
    "Technology": ["AAPL", "GOOGL", "MSFT", "NVDA", "META", "AMD", "INTC", "CRM", "ORCL", "ADBE"],
    "Finance": ["JPM", "BAC", "GS", "MS", "WFC", "C", "AXP", "BLK", "SCHW", "USB"],
    "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "LLY", "BMY", "AMGN"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL"],
    "Consumer": ["AMZN", "WMT", "HD", "PG", "KO", "PEP", "COST", "NKE", "MCD", "SBUX"],
}

ALL_TICKERS = [t for tickers in SECTORS.values() for t in tickers]
TICKER_SECTOR = {t: s for s, tickers in SECTORS.items() for t in tickers}


@dataclass
class DataFrameQATask:
    """A single DataFrame QA task."""
    task_id: str
    prompt: str
    question: str
    expected_answer: str
    task_type: str  # lookup, aggregation, ranking, cross, multi_step
    n_rows: int
    n_tickers: int
    difficulty: str  # easy, medium, hard
    # Ground truth data for scoring
    data: list[dict] = field(default_factory=list, repr=False)


def _generate_price_data(
    tickers: list[str],
    n_days: int,
    seed: int,
) -> list[dict]:
    """Generate synthetic stock price data."""
    rng = random.Random(seed)

    # Base prices per ticker
    base_prices = {}
    for t in tickers:
        base_prices[t] = rng.uniform(20, 500)

    # Generate daily data
    rows = []
    start_year = 2024
    current_prices = dict(base_prices)

    for day_idx in range(n_days):
        # Skip weekends (roughly)
        month = (day_idx // 22) + 1
        if month > 12:
            month = ((month - 1) % 12) + 1
            start_year = 2024 + (day_idx // 264)
        day_of_month = (day_idx % 22) + 1
        if day_of_month > 28:
            day_of_month = 28

        date_str = f"{start_year}-{month:02d}-{day_of_month:02d}"

        for ticker in tickers:
            # Random walk with drift
            sector = TICKER_SECTOR.get(ticker, "Technology")
            # Sector-specific trends
            drift = {"Technology": 0.0003, "Finance": 0.0001, "Healthcare": 0.0002,
                     "Energy": -0.0001, "Consumer": 0.0002}.get(sector, 0.0001)

            daily_return = rng.gauss(drift, 0.02)  # ~2% daily vol
            current_prices[ticker] *= (1 + daily_return)

            price = current_prices[ticker]
            high = price * (1 + abs(rng.gauss(0, 0.005)))
            low = price * (1 - abs(rng.gauss(0, 0.005)))
            open_price = price * (1 + rng.gauss(0, 0.003))
            volume = int(rng.gauss(30_000_000, 15_000_000))
            volume = max(1_000_000, volume)

            rows.append({
                "date": date_str,
                "ticker": ticker,
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(price, 2),
                "volume": volume,
                "sector": sector,
            })

    return rows


def _data_to_csv(rows: list[dict]) -> str:
    """Convert rows to CSV string."""
    header = "date,ticker,open,high,low,close,volume,sector"
    lines = [header]
    for r in rows:
        lines.append(
            f"{r['date']},{r['ticker']},{r['open']},{r['high']},{r['low']},"
            f"{r['close']},{r['volume']},{r['sector']}"
        )
    return "\n".join(lines)


def _generate_lookup_question(rows: list[dict], rng: random.Random) -> tuple[str, str]:
    """O(1) lookup: find a specific value."""
    target_row = rng.choice(rows)
    ticker = target_row["ticker"]
    date = target_row["date"]
    field = rng.choice(["close", "volume", "high", "low"])

    question = f"What was the {field} price of {ticker} on {date}?"
    if field == "volume":
        question = f"What was the trading volume of {ticker} on {date}?"
        answer = str(target_row["volume"])
    else:
        answer = str(target_row[field])

    return question, answer


def _generate_aggregation_question(rows: list[dict], rng: random.Random) -> tuple[str, str]:
    """O(N) aggregation: compute statistics over data."""
    question_type = rng.choice(["avg_price", "total_volume", "sector_avg", "max_price"])

    if question_type == "avg_price":
        ticker = rng.choice(list(set(r["ticker"] for r in rows)))
        prices = [r["close"] for r in rows if r["ticker"] == ticker]
        avg = sum(prices) / len(prices)
        question = f"What is the average closing price of {ticker} across all dates? Round to 2 decimal places."
        answer = f"{avg:.2f}"

    elif question_type == "total_volume":
        ticker = rng.choice(list(set(r["ticker"] for r in rows)))
        total = sum(r["volume"] for r in rows if r["ticker"] == ticker)
        question = f"What is the total trading volume of {ticker} across all dates?"
        answer = str(total)

    elif question_type == "sector_avg":
        sector = rng.choice(list(set(r["sector"] for r in rows)))
        prices = [r["close"] for r in rows if r["sector"] == sector]
        avg = sum(prices) / len(prices)
        question = f"What is the average closing price across all {sector} stocks and dates? Round to 2 decimal places."
        answer = f"{avg:.2f}"

    elif question_type == "max_price":
        ticker = rng.choice(list(set(r["ticker"] for r in rows)))
        max_price = max(r["high"] for r in rows if r["ticker"] == ticker)
        question = f"What is the highest price (high) ever reached by {ticker}?"
        answer = f"{max_price}"

    return question, answer


def _generate_ranking_question(rows: list[dict], rng: random.Random) -> tuple[str, str]:
    """O(N) ranking: find top/bottom items."""
    question_type = rng.choice(["highest_avg", "most_volatile", "highest_volume"])

    tickers = list(set(r["ticker"] for r in rows))

    if question_type == "highest_avg":
        avgs = {}
        for t in tickers:
            prices = [r["close"] for r in rows if r["ticker"] == t]
            avgs[t] = sum(prices) / len(prices)
        best = max(avgs, key=avgs.get)
        question = "Which stock has the highest average closing price? Return just the ticker symbol."
        answer = best

    elif question_type == "most_volatile":
        vols = {}
        for t in tickers:
            prices = [r["close"] for r in rows if r["ticker"] == t]
            if len(prices) > 1:
                returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                mean_r = sum(returns) / len(returns)
                var = sum((r - mean_r)**2 for r in returns) / (len(returns) - 1)
                vols[t] = math.sqrt(var)
            else:
                vols[t] = 0
        most_vol = max(vols, key=vols.get)
        question = "Which stock has the highest daily return volatility (standard deviation of daily returns)? Return just the ticker symbol."
        answer = most_vol

    elif question_type == "highest_volume":
        total_vols = {}
        for t in tickers:
            total_vols[t] = sum(r["volume"] for r in rows if r["ticker"] == t)
        highest = max(total_vols, key=total_vols.get)
        question = "Which stock has the highest total trading volume across all dates? Return just the ticker symbol."
        answer = highest

    return question, answer


def _generate_sector_question(rows: list[dict], rng: random.Random) -> tuple[str, str]:
    """O(N) sector analysis: aggregate by sector."""
    sectors = list(set(r["sector"] for r in rows))

    question_type = rng.choice(["best_sector", "sector_count", "sector_volume"])

    if question_type == "best_sector":
        sector_returns = {}
        for s in sectors:
            sector_tickers = list(set(r["ticker"] for r in rows if r["sector"] == s))
            returns = []
            for t in sector_tickers:
                prices = sorted(
                    [(r["date"], r["close"]) for r in rows if r["ticker"] == t],
                    key=lambda x: x[0]
                )
                if len(prices) >= 2:
                    ret = (prices[-1][1] - prices[0][1]) / prices[0][1]
                    returns.append(ret)
            sector_returns[s] = sum(returns) / len(returns) if returns else 0
        best = max(sector_returns, key=sector_returns.get)
        question = "Which sector had the best average return (from first date to last date across its stocks)? Return just the sector name."
        answer = best

    elif question_type == "sector_count":
        sector_counts = {}
        for s in sectors:
            sector_counts[s] = len(set(r["ticker"] for r in rows if r["sector"] == s))
        question = "How many unique stocks are in each sector? Return as 'Sector: N' per line, sorted alphabetically."
        lines = [f"{s}: {sector_counts[s]}" for s in sorted(sectors)]
        answer = "\n".join(lines)

    elif question_type == "sector_volume":
        sector_vols = {}
        for s in sectors:
            sector_vols[s] = sum(r["volume"] for r in rows if r["sector"] == s)
        highest = max(sector_vols, key=sector_vols.get)
        question = "Which sector has the highest total trading volume? Return just the sector name."
        answer = highest

    return question, answer


def _generate_multi_step_question(rows: list[dict], rng: random.Random) -> tuple[str, str]:
    """Multi-step analysis: requires chaining 2+ computations.

    These simulate real Jupyter notebook workflows where you do:
    1. Filter/aggregate to find something
    2. Use that result to ask a follow-up question
    """
    tickers = list(set(r["ticker"] for r in rows))
    sectors = list(set(r["sector"] for r in rows))

    question_type = rng.choice(["best_sector_top_stock", "volatile_stock_max_day", "worst_performer_sector"])

    if question_type == "best_sector_top_stock":
        # Step 1: Find best performing sector
        sector_returns = {}
        for s in sectors:
            sector_tickers = list(set(r["ticker"] for r in rows if r["sector"] == s))
            returns = []
            for t in sector_tickers:
                prices = sorted(
                    [(r["date"], r["close"]) for r in rows if r["ticker"] == t],
                    key=lambda x: x[0]
                )
                if len(prices) >= 2:
                    ret = (prices[-1][1] - prices[0][1]) / prices[0][1]
                    returns.append((t, ret))
            sector_returns[s] = returns
        best_sector = max(sector_returns, key=lambda s: sum(r for _, r in sector_returns[s]) / max(len(sector_returns[s]), 1))

        # Step 2: Find the top stock in that sector
        best_stock = max(sector_returns[best_sector], key=lambda x: x[1])
        question = f"First find the sector with the best average return (first date to last date). Then within that sector, which stock had the highest individual return? Return just the ticker symbol."
        answer = best_stock[0]

    elif question_type == "volatile_stock_max_day":
        # Step 1: Find most volatile stock
        vols = {}
        for t in tickers:
            prices = [r["close"] for r in rows if r["ticker"] == t]
            if len(prices) > 1:
                returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                mean_r = sum(returns) / len(returns)
                var = sum((r - mean_r)**2 for r in returns) / (len(returns) - 1)
                vols[t] = math.sqrt(var)
            else:
                vols[t] = 0
        most_vol = max(vols, key=vols.get)

        # Step 2: Find its highest single-day gain
        ticker_rows = sorted([r for r in rows if r["ticker"] == most_vol], key=lambda r: r["date"])
        max_gain = 0
        max_gain_date = ticker_rows[0]["date"]
        for i in range(1, len(ticker_rows)):
            gain = (ticker_rows[i]["close"] - ticker_rows[i-1]["close"]) / ticker_rows[i-1]["close"]
            if gain > max_gain:
                max_gain = gain
                max_gain_date = ticker_rows[i]["date"]
        question = f"First find the most volatile stock (highest std of daily returns). Then find the date when that stock had its biggest single-day gain. Return just the date (YYYY-MM-DD)."
        answer = max_gain_date

    elif question_type == "worst_performer_sector":
        # Step 1: Find worst performing stock
        stock_returns = {}
        for t in tickers:
            prices = sorted(
                [(r["date"], r["close"]) for r in rows if r["ticker"] == t],
                key=lambda x: x[0]
            )
            if len(prices) >= 2:
                stock_returns[t] = (prices[-1][1] - prices[0][1]) / prices[0][1]
        worst_stock = min(stock_returns, key=stock_returns.get)

        # Step 2: Return its sector
        sector = next(r["sector"] for r in rows if r["ticker"] == worst_stock)
        question = f"Which sector does the worst-performing stock (lowest total return from first to last date) belong to? Return just the sector name."
        answer = sector

    return question, answer


def generate_dataframe_qa_task(
    task_idx: int,
    n_tickers: int = 10,
    n_days: int = 60,
    task_type: str = "mixed",
    seed: int | None = None,
) -> DataFrameQATask:
    """Generate a single DataFrame QA task."""
    seed = seed if seed is not None else task_idx + 90000
    rng = random.Random(seed)

    # Select tickers from multiple sectors
    selected_tickers = []
    sectors_to_use = rng.sample(list(SECTORS.keys()), min(len(SECTORS), max(2, n_tickers // 3)))
    for sector in sectors_to_use:
        n_from_sector = max(1, n_tickers // len(sectors_to_use))
        selected_tickers.extend(rng.sample(SECTORS[sector], min(n_from_sector, len(SECTORS[sector]))))
    selected_tickers = selected_tickers[:n_tickers]

    # Generate data
    data = _generate_price_data(selected_tickers, n_days, seed)
    csv_str = _data_to_csv(data)

    # Generate question
    if task_type == "mixed":
        task_type_actual = rng.choice(["lookup", "aggregation", "ranking", "sector", "multi_step"])
    else:
        task_type_actual = task_type

    if task_type_actual == "lookup":
        question, answer = _generate_lookup_question(data, rng)
    elif task_type_actual == "aggregation":
        question, answer = _generate_aggregation_question(data, rng)
    elif task_type_actual == "ranking":
        question, answer = _generate_ranking_question(data, rng)
    elif task_type_actual == "sector":
        question, answer = _generate_sector_question(data, rng)
    elif task_type_actual == "multi_step":
        question, answer = _generate_multi_step_question(data, rng)
    else:
        question, answer = _generate_aggregation_question(data, rng)

    difficulty = "easy" if n_tickers <= 5 and n_days <= 30 else \
                 "medium" if n_tickers <= 15 and n_days <= 90 else "hard"

    prompt = f"QUESTION: {question}\n\nDATASET:\n{csv_str}"

    task_id = f"dfqa_{task_idx:03d}_{n_tickers}t_{n_days}d_{task_type_actual}"

    return DataFrameQATask(
        task_id=task_id,
        prompt=prompt,
        question=question,
        expected_answer=answer,
        task_type=task_type_actual,
        n_rows=len(data),
        n_tickers=n_tickers,
        difficulty=difficulty,
        data=data,
    )


def score_dataframe_qa(answer: str | None, expected: str, task_type: str) -> dict:
    """Score a DataFrame QA answer.

    Returns dict with 'score' (0-1), 'match_type', and details.
    """
    if answer is None:
        return {"score": 0.0, "match_type": "no_answer"}

    answer = str(answer).strip()
    expected = str(expected).strip()

    # Exact match
    if answer == expected:
        return {"score": 1.0, "match_type": "exact"}

    # Case-insensitive match
    if answer.lower() == expected.lower():
        return {"score": 1.0, "match_type": "case_insensitive"}

    # For numeric answers, try fuzzy matching
    try:
        ans_num = float(answer.replace(",", ""))
        exp_num = float(expected.replace(",", ""))
        if exp_num != 0:
            rel_error = abs(ans_num - exp_num) / abs(exp_num)
            if rel_error < 0.001:  # Within 0.1%
                return {"score": 1.0, "match_type": "numeric_close"}
            elif rel_error < 0.01:  # Within 1%
                return {"score": 0.8, "match_type": "numeric_approx"}
            elif rel_error < 0.05:  # Within 5%
                return {"score": 0.5, "match_type": "numeric_rough"}
    except (ValueError, ZeroDivisionError):
        pass

    # Contains match — expected answer found within the answer text
    if expected.lower() in answer.lower():
        return {"score": 0.8, "match_type": "contains"}

    # For multi-line answers (sector_count), check partial matches
    if "\n" in expected:
        expected_lines = set(l.strip().lower() for l in expected.split("\n") if l.strip())
        answer_lines = set(l.strip().lower() for l in answer.split("\n") if l.strip())
        if expected_lines:
            overlap = len(expected_lines & answer_lines)
            score = overlap / len(expected_lines)
            return {"score": score, "match_type": "partial_lines", "overlap": overlap, "total": len(expected_lines)}

    return {"score": 0.0, "match_type": "no_match"}


def generate_dataframe_qa_suite(
    n_tasks: int = 20,
    seed_offset: int = 90000,
) -> list[DataFrameQATask]:
    """Generate a DataFrame QA benchmark suite.

    Configurations:
    - Small dataset (5 tickers, 30 days, ~150 rows, ~8K chars): easy
    - Medium dataset (15 tickers, 60 days, ~900 rows, ~50K chars): medium
    - Large dataset (30 tickers, 120 days, ~3600 rows, ~200K chars): hard
    - XL dataset (50 tickers, 250 days, ~12500 rows, ~700K chars): very hard
    """
    configs = [
        # (n_tickers, n_days, count)
        (5, 30, 5),      # Easy: ~8K chars
        (15, 60, 5),     # Medium: ~50K chars
        (30, 120, 5),    # Hard: ~200K chars
        (50, 250, 5),    # Very hard: ~700K chars
    ]

    tasks = []
    idx = 0
    for n_tickers, n_days, count in configs:
        for _ in range(count):
            if len(tasks) >= n_tasks:
                return tasks
            tasks.append(generate_dataframe_qa_task(
                task_idx=idx,
                n_tickers=n_tickers,
                n_days=n_days,
                seed=idx + seed_offset,
            ))
            idx += 1

    return tasks


if __name__ == "__main__":
    # Quick test
    tasks = generate_dataframe_qa_suite(n_tasks=4)
    for t in tasks:
        print(f"{t.task_id}: {t.question}")
        print(f"  Expected: {t.expected_answer}")
        print(f"  Rows: {t.n_rows}, Chars: {len(t.prompt)}")
        print()
