import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

class EpisodeEnricher:
    """
    Ex-post enrichment/backtesting module.
    Responsibilities:
    - Join episodes with price feeds and benchmarks.
    - Compute realized_return, pnl, prediction_error (if model-implied return available).
    - Fill ex-post fields: price_open, price_next (default: open/close of same day, or custom policy).
    - Optionally compute position transitions (position_before/position_after) if a position book is provided.
    """

    def __init__(self,
                 price_feed: pd.DataFrame,
                 benchmark_feed: Optional[pd.DataFrame] = None,
                 settle_policy: str = "open-next_open2"):
        """
        price_feed must include columns: ['date', 'asset', 'open', 'close'] at minimum.
        benchmark_feed (optional) same structure for excess return.
        settle_policy:
          - "open-close": realized_return = close/open - 1 within same date
          - "close-close": price_open = previous day's close, price_next = current day's close, realized_return = (current day close / previous day close) - 1
          - "open-next_open2": price_open = next trading day (open_{t+1}), price_next = day after next (open_{t+2}), realized_return = (open_{t+2}/open_{t+1}) - 1
          - "close-next_close": requires next day close; not implemented in this minimal version
        """
        self.price_feed = price_feed.copy() if price_feed is not None else None
        self.benchmark_feed = benchmark_feed.copy() if benchmark_feed is not None else None
        self.settle_policy = settle_policy

        if self.price_feed is not None:
            self.price_feed['date'] = pd.to_datetime(self.price_feed['date']).dt.strftime("%Y-%m-%d")
        if self.benchmark_feed is not None:
            self.benchmark_feed['date'] = pd.to_datetime(self.benchmark_feed['date']).dt.strftime("%Y-%m-%d")

    def enrich(self, episodes: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Returns a DataFrame with ex-post fields populated:
        - price_open, price_next, realized_return, pnl, prediction_error (if available)
        - price_trend (sign of realized_return)
        """
        df = pd.DataFrame(episodes).copy()
        if df.empty:
            return df

        df['date'] = pd.to_datetime(df['date']).dt.strftime("%Y-%m-%d")

        if self.price_feed is None or self.price_feed.empty:
            # No price enrichment possible; return as-is
            df['realized_return'] = np.nan
            df['pnl'] = np.nan
            df['price_open'] = np.nan
            df['price_next'] = np.nan
            df['price_trend'] = np.nan
            return df

        # Prepare price table with forward opens (t+1, t+2) for open-next_open2 policy
        price_tbl = self.price_feed[['date', 'asset', 'open', 'close']].drop_duplicates(subset=['asset', 'date']).copy()
        price_tbl = price_tbl.sort_values(['asset', 'date'])
        price_tbl['next_open'] = price_tbl.groupby('asset')['open'].shift(-1)
        price_tbl['next2_open'] = price_tbl.groupby('asset')['open'].shift(-2)

        # Merge prices
        merged = pd.merge(
            df,
            price_tbl,
            on=['date', 'asset'],
            how='left',
            suffixes=('', '_mkt')
        )

        # Debug: check merge results
        if merged['close'].isna().any():
            missing_count = merged['close'].isna().sum()
            total_count = len(merged)
            import warnings
            warnings.warn(f"[EpisodeEnricher] Warning: {missing_count}/{total_count} episodes have missing price data after merge. "
                         f"Check if price_feed contains data for all (date, asset) combinations in episodes.")

        # Settle policy: calculate price and return according to strategy
        if self.settle_policy == "open-close":
            # Use open-close: realized_return = close/open - 1
            merged['price_open'] = merged['open']
            merged['price_next'] = merged['close']
            # Handle division by zero and NaN: if price_open is 0 or NaN, set realized_return to NaN
            merged['realized_return'] = np.where(
                (merged['price_open'].notna()) & (merged['price_open'] != 0) & (merged['price_next'].notna()),
                merged['price_next'] / merged['price_open'] - 1,
                np.nan
            )
        elif self.settle_policy == "close-close":
            # Use close-close strategy: price_open = previous day's close, price_next = current day's close
            # realized_return = (current day close / previous day close) - 1
            # Problem: same asset on same day may have multiple episodes, need to deduplicate by date first, then shift
            # Solution: first build unique (asset, date) -> close mapping, then shift date, then merge back
            merged = merged.sort_values(['asset', 'date'])
            
            # Build unique (asset, date) -> close mapping (take first close value for each (asset, date) combination)
            unique_dates = merged[['asset', 'date', 'close']].drop_duplicates(subset=['asset', 'date'], keep='first')
            unique_dates = unique_dates.sort_values(['asset', 'date'])
            # Group by asset, shift date (get previous day's close)
            unique_dates['prev_close'] = unique_dates.groupby('asset')['close'].shift(1)
            
            # Merge prev_close back to original merged DataFrame
            merged = merged.merge(
                unique_dates[['asset', 'date', 'prev_close']],
                on=['asset', 'date'],
                how='left',
                suffixes=('', '_prev')
            )
            
            # price_open = previous day's close, price_next = current day's close
            merged['price_open'] = merged['prev_close']
            merged['price_next'] = merged['close']
            
            # Calculate realized_return: if price_open and price_next both have values and price_open is not 0, calculate return
            # Debug: check if merge succeeded and if price data is correct
            if merged['close'].isna().all():
                import warnings
                warnings.warn(f"[EpisodeEnricher] Warning: All 'close' prices are NaN after merge. Check if price_feed matches episodes by date and asset.")
            
            merged['realized_return'] = np.where(
                (merged['price_open'].notna()) & (merged['price_open'] != 0) & (merged['price_next'].notna()),
                merged['price_next'] / merged['price_open'] - 1,
                np.nan
            )
            
            # Clean temporary columns
            merged = merged.drop(columns=['prev_close'], errors='ignore')
        elif self.settle_policy == "open-next_open2":
            # Use open-open forward-looking two days: price_open = t+1 open, price_next = t+2 open
            # realized_return = (open_{t+2} / open_{t+1}) - 1
            merged = merged.sort_values(['asset', 'date'])

            # next_open and next2_open are already pre-calculated in price_tbl
            if merged[['next_open', 'next2_open']].isna().any().any():
                import warnings
                missing = merged[['next_open', 'next2_open']].isna().sum().sum()
                warnings.warn(f"[EpisodeEnricher] Warning: {missing} rows missing next_open/next2_open; returns set to NaN where unavailable.")

            merged['price_open'] = merged['next_open']
            merged['price_next'] = merged['next2_open']
            merged['realized_return'] = np.where(
                (merged['price_open'].notna()) & (merged['price_open'] != 0) & (merged['price_next'].notna()),
                merged['price_next'] / merged['price_open'] - 1,
                np.nan
            )

            # Clean temporary columns
            merged = merged.drop(columns=['next_open', 'next2_open'], errors='ignore')
        else:
            # Default use open-close
            merged['price_open'] = merged['open']
            merged['price_next'] = merged['close']
            merged['realized_return'] = np.where(
                (merged['price_open'].notna()) & (merged['price_open'] != 0) & (merged['price_next'].notna()),
                merged['price_next'] / merged['price_open'] - 1,
                np.nan
            )

        # PnL: target_position × realized_return (unit notional)
        # target_position is stored as string in Episode (from LLM output), explicitly convert to float here
        # Fix: merged is DataFrame, should use merged['target_position'] instead of merged.get()
        tp = pd.to_numeric(merged['target_position'], errors='coerce')
        # Handle NaN: if tp or realized_return is NaN, set pnl to NaN
        merged['pnl'] = np.where(
            (tp.notna()) & (merged['realized_return'].notna()),
            tp * merged['realized_return'],
            np.nan
        )

        # Trend sign: handle NaN values
        merged['price_trend'] = np.where(
            merged['realized_return'].notna(),
            np.sign(merged['realized_return']),
            np.nan
        )

        # Prediction error: if an implied return proxy exists (optional)
        if 'implied_return' in merged.columns:
            # Handle NaN: if realized_return or implied_return is NaN, set prediction_error to NaN
            merged['prediction_error'] = np.where(
                (merged['realized_return'].notna()) & (merged['implied_return'].notna()),
                merged['realized_return'] - merged['implied_return'],
                np.nan
            )
        else:
            # fallback: difference between realized_return and mapped belief_score (optional)
            if 'belief_score' in merged.columns:
                # belief_score ∈ [-1,1] maps to expected return
                # Use more reasonable mapping: belief_score * 0.05 represents 5% maximum expected return (more consistent with actual volatility range)
                # Or can dynamically adjust based on historical volatility, here use fixed ratio
                bs = pd.to_numeric(merged['belief_score'], errors='coerce')
                # Handle NaN: if realized_return or bs is NaN, set prediction_error to NaN
                # prediction_error = realized_return - expected_return
                # expected_return = belief_score * max_expected_return (use 5% as maximum expected return)
                max_expected_return = 0.05  # 5% maximum expected daily return
                expected_return = bs * max_expected_return
                merged['prediction_error'] = np.where(
                    (merged['realized_return'].notna()) & (bs.notna()),
                    merged['realized_return'] - expected_return,
                    np.nan
                )
            else:
                merged['prediction_error'] = np.nan

        return merged

    def compute_excess(self, enriched_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds excess_return vs. benchmark (annualized mean difference), simplified.
        """
        if self.benchmark_feed is None or self.benchmark_feed.empty:
            enriched_df['excess_return'] = np.nan
            return enriched_df

        bench = self.benchmark_feed[['date', 'asset', 'open', 'close']].copy()
        bench['date'] = pd.to_datetime(bench['date']).dt.strftime("%Y-%m-%d")
        # Handle division by zero and NaN: if open is 0 or NaN, set bench_ret to NaN
        bench['bench_ret'] = np.where(
            (bench['open'].notna()) & (bench['open'] != 0) & (bench['close'].notna()),
            bench['close'] / bench['open'] - 1,
            np.nan
        )

        out = pd.merge(
            enriched_df,
            bench[['date', 'asset', 'bench_ret']],
            on=['date', 'asset'],
            how='left'
        )

        # Annualized excess: ann_return - ann_bench
        # This requires grouping over periods; here we keep a per-row bench_ret for downstream aggregation.
        # Handle NaN: if realized_return or bench_ret is NaN, set excess_return to NaN
        out['excess_return'] = np.where(
            (out['realized_return'].notna()) & (out['bench_ret'].notna()),
            out['realized_return'] - out['bench_ret'],
            np.nan
        )
        return out