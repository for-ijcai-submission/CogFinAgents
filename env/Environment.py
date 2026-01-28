import pandas as pd
from typing import Dict, List, Any, Callable, Optional
from data.DataProvider import DataProvider
from config.ConfigLoader import SimulationConfig, DataSchemas

class Environment:
    """
    Orchestrates date progression, assets, and provides standardized merged signal views per date.
    """

    def __init__(self,
                 sim_cfg: SimulationConfig,
                 schema_cfg: DataSchemas,
                 hooks: Optional[Dict[str, List[Callable]]] = None):
        self.sim_cfg = sim_cfg
        self.assets = sim_cfg.assets
        # Pass Trump social data configuration to DataProvider
        use_trump = sim_cfg.trump_social.enabled if sim_cfg.trump_social else False
        trump_path = sim_cfg.trump_social.path if sim_cfg.trump_social else None
        self.provider = DataProvider(sim_cfg.assets_root, schema_cfg, 
                                     use_trump_social=use_trump, 
                                     trump_social_path=trump_path)
        self.date_start = pd.to_datetime(sim_cfg.date_start).date()
        self.date_end = pd.to_datetime(sim_cfg.date_end).date()
        self.hooks = hooks or {
            'before_day': [],
            'after_day': [],
            'before_decision': [],
            'after_decision': [],
        }
        all_dates = set()
        for a in self.assets:
            all_dates.update(self.provider.list_available_dates(a))
        # Convert Timestamp to date for comparison
        self.sim_dates = [d for d in sorted(all_dates) if self.date_start <= d.date() <= self.date_end]

    def iter_dates(self) -> List[pd.Timestamp]:
        return self.sim_dates

    def signals_for(self, asset: str, dstr: str, access_level: str, market_window_days: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Return merged signals list per category for specified access_level ('basic' or 'advanced').
        Each category returns a list of records (all rows for that date), even if only one row exists.
        market_window_days: If specified, market_data returns historical window_days days of data
        """
        remove_social_news = self.sim_cfg.ablation.remove_social_news if self.sim_cfg.ablation else False
        return self.provider.signals_for(asset, dstr, access_level=access_level, market_window_days=market_window_days, remove_social_news=remove_social_news)
    
    def is_trading_day(self, asset: str, date: Any) -> bool:
        """
        Check if an asset is a trading day on a given date.
        Only based on dates with data in the asset's processed_market.csv.
        
        Args:
            asset: Asset code
            date: Date (can be date object, string, or Timestamp)
        
        Returns:
            True if the asset is a trading day on this date, False otherwise
        """
        return self.provider.is_trading_day(asset, date)

    def run_hook(self, hook_name: str, *args, **kwargs):
        for fn in self.hooks.get(hook_name, []):
            try:
                fn(*args, **kwargs)
            except Exception:
                pass