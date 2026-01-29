import os
import glob
import pandas as pd
from typing import Dict, List, Any, Optional
import re
from config.ConfigLoader import DataSchemas

class DataProvider:
    """
    Load standardized, preprocessed data per asset and provide structured signal views.

    Field selection (basic/advanced) is driven by DataSchemas config.
    This version centralizes basic/advanced merging: callers specify access level
    ('basic' or 'advanced'), and DataProvider returns an already-merged list of dicts
    per signal category (no nested {'basic','advanced'}).

    Example output for signals_for(..., access_level='advanced'):
    {
        'market_data': [{'open':..., 'close':..., 'vix':..., 'fng':...}],
        'macro_data': [{...}, {...}],  # May have multiple rows
        'news_data': [{...}, {...}],   # May have multiple rows
        ...
    }
    """

    def __init__(self, assets_root: str, schema_cfg: DataSchemas, use_trump_social: bool = False, trump_social_path: Optional[str] = None):
        self.assets_root = assets_root
        self.schema_cfg = schema_cfg
        self._cache: Dict[str, Dict[str, Any]] = {}
        # Project root directory: parent directory of assets_root (text_path is relative to project root)
        self.root_dir = os.path.dirname(os.path.abspath(assets_root))
        # Trump social data configuration
        self.use_trump_social = use_trump_social
        self.trump_social_path = trump_social_path or os.path.join(self.root_dir, 'data', 'global', 'processed_trump.csv')
        self._trump_social_df: Optional[pd.DataFrame] = None
        if self.use_trump_social:
            self._load_trump_social()

    def load_asset(self, asset: str) -> Dict[str, Any]:
        """
        Load preprocessed CSVs for an asset plus global macro CSVs.
        Only CSV (and text via text_path references inside company data) are considered.
        """
        if asset in self._cache:
            return self._cache[asset]

        folder = os.path.join(self.assets_root, asset)
        data: Dict[str, Any] = {}

        # Per-asset CSVs
        kinds = [
            'processed_market',
            'processed_news',
            'processed_tweets',
            'processed_company_data',
            'processed_company_data_earnings',
            'processed_on_chain',
        ]
        for kind in kinds:
            csv_path = os.path.join(folder, f'{kind}.csv')
            if os.path.exists(csv_path):
                data[kind] = pd.read_csv(csv_path)
            else:
                data[kind] = None

        # Global macro CSVs
        global_folder = os.path.join(self.assets_root, 'global')
        for macro_kind in ['processed_macro', 'processed_macro_reports']:
            csv_path = os.path.join(global_folder, f'{macro_kind}.csv')
            data[macro_kind] = pd.read_csv(csv_path) if os.path.exists(csv_path) else None

        self._cache[asset] = data
        return data

    @staticmethod
    def _slice_date(df: Optional[pd.DataFrame], dstr: str) -> List[Dict[str, Any]]:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return []
        date_col = next((c for c in df.columns if c.lower() == 'date'), None)
        if not date_col:
            return []
        # To accommodate various date formats (e.g. "YYYY-MM-DD" and "YYYY-MM-DD 00:00:00"), uniformly use pandas parsing before comparison
        try:
            target_date = pd.to_datetime(dstr, errors='coerce')
        except Exception:
            return []
        if pd.isna(target_date):
            return []

        # Also uniformly parse dates in data as Timestamp, then match by exact time point
        parsed_dates = pd.to_datetime(df[date_col], errors='coerce')
        mask = parsed_dates == target_date
        return df[mask].to_dict(orient='records')
    
    def _slice_date_range(self, df: Optional[pd.DataFrame], dstr: str, window_days: int) -> List[Dict[str, Any]]:
        """
        Get data from dstr going back window_days trading days, sorted by date from old to new
        Based on actual trading days count in data source, not calendar days
        """
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return []
        date_col = next((c for c in df.columns if c.lower() == 'date'), None)
        if not date_col:
            return []
        
        # Convert to date type for comparison and sorting
        df = df.copy()
        df['_date_parsed'] = pd.to_datetime(df[date_col], errors='coerce')
        target_date = pd.to_datetime(dstr)
        
        # Filter invalid dates, only keep data <= target_date
        df_valid = df[
            (df['_date_parsed'].notna()) & 
            (df['_date_parsed'] <= target_date)
        ].copy()
        
        if df_valid.empty:
            return []
        
        # Sort by date from old to new
        df_valid = df_valid.sort_values('_date_parsed', ascending=True)
        
        # Deduplicate: if multiple records for same date, keep last one (usually latest data)
        df_valid = df_valid.drop_duplicates(subset=['_date_parsed'], keep='last')
        
        # Based on trading days count: take last window_days trading days
        # If data is insufficient for window_days, return all available data
        num_records = len(df_valid)
        if num_records <= window_days:
            filtered_df = df_valid.copy()
        else:
            # Take last window_days records
            filtered_df = df_valid.iloc[-window_days:].copy()
        
        # Delete temporary columns, but keep original date column
        filtered_df = filtered_df.drop(columns=['_date_parsed'])
        
        return filtered_df.to_dict(orient='records')

    def _applicable(self, asset: str, key: str) -> bool:
        # Company data only for specific assets; on_chain only for crypto
        if key == 'company_data':
            return asset in self.schema_cfg.applicability.get('company_assets', [])
        if key == 'on_chain_data':
            return asset in self.schema_cfg.applicability.get('onchain_assets', [])
        return True

    def _load_text_from_path(self, text_path: Optional[str]) -> Optional[str]:
        """Read txt file content pointed to by text_path"""
        if not text_path or not isinstance(text_path, str):
            return None
        
        # Handle path: may be relative path (relative to project root) or absolute path
        if os.path.isabs(text_path):
            file_path = text_path
        else:
            # Relative path: based on project root
            file_path = os.path.join(self.root_dir, text_path)
        
        # Normalize path separators (Windows/Unix compatible)
        file_path = os.path.normpath(file_path)
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return content
        except Exception:
            return None

    def _merge_fields(self, record: Dict[str, Any], schema_key: str, access_level: str) -> Dict[str, Any]:
        """
        Merge fields from schema according to access level.
        - 'basic': include only schema.basic
        - 'advanced': include schema.basic + schema.advanced
        - If contains text_path, read file content as text_content
        - For market_data, always keep date field (key identifier for time series)
        """
        merged: Dict[str, Any] = {}
        schema = self.schema_cfg.schemas.get(schema_key)
        if not schema:
            return merged
        
        # For market_data, always keep date field (even if not in schema)
        if schema_key == 'market_data' and 'date' in record:
            merged['date'] = record.get('date')
        
        # basic first
        for k in schema.basic:
            if k in record and record.get(k) is not None:
                merged[k] = record.get(k)
        # advanced if applicable
        if access_level == 'advanced':
            for k in schema.advanced:
                if k in record and record.get(k) is not None:
                    merged[k] = record.get(k)
            # If contains text_path, read file content
            if 'text_path' in merged and merged['text_path']:
                text_content = self._load_text_from_path(merged['text_path'])
                if text_content:
                    # If text is too long, apply truncation (use macro truncation logic)
                    trunc_cfg = self.schema_cfg.truncation
                    if trunc_cfg and trunc_cfg.macro_data:
                        max_chars = trunc_cfg.macro_data.max_text_chars
                        text_content = self._truncate_macro_text(text_content, max_chars)
                    merged['text_content'] = text_content
        return merged

    def _truncate_social_data(self, recs: List[Dict[str, Any]], schema_key: str) -> List[Dict[str, Any]]:
        """
        Intelligent sampling:
        - post: sort by score * num_comments weighted
        - comment: sort by score
        Then merge and take top N
        """
        trunc_cfg = self.schema_cfg.truncation
        if not trunc_cfg or not trunc_cfg.social_data:
            return recs
        
        max_rows = trunc_cfg.social_data.max_rows
        if len(recs) <= max_rows:
            return recs
        
        df = pd.DataFrame(recs)
        
        # Separate posts and comments
        if 'kind' in df.columns:
            posts = df[df['kind'] == 'post'].copy()
            comments = df[df['kind'] == 'comment'].copy()
            others = df[~df['kind'].isin(['post', 'comment'])].copy()
        else:
            # If no kind field, treat all as others
            posts = pd.DataFrame()
            comments = pd.DataFrame()
            others = df.copy()
        
        # For posts: sort by score * num_comments (original strategy)
        if not posts.empty:
            score = posts['score'].fillna(0) if 'score' in posts.columns else pd.Series([0] * len(posts))
            num_comments = posts['num_comments'].fillna(0) if 'num_comments' in posts.columns else pd.Series([0] * len(posts))
            posts['weight'] = score * num_comments
            posts = posts.sort_values('weight', ascending=False)
            posts = posts.drop(columns=['weight'])
        
        # For comments: sort by score
        if not comments.empty:
            score = comments['score'].fillna(0) if 'score' in comments.columns else pd.Series([0] * len(comments))
            comments = comments.sort_values('score', ascending=False)
        
        # Merge: first posts, then comments, finally others
        combined = pd.concat([posts, comments, others], ignore_index=True)
        
        # Take top N
        result = combined.head(max_rows)
        return result.to_dict(orient='records')

    def _truncate_trump_social_data(self, recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Trump experiment's social_data truncation logic:
        - Reuse social_data truncation config (max_rows)
        - Scoring weight uses favourites_count + reblogs_count + num_comments
        """
        trunc_cfg = self.schema_cfg.truncation
        if not trunc_cfg or not trunc_cfg.social_data:
            return recs

        max_rows = trunc_cfg.social_data.max_rows
        if len(recs) <= max_rows:
            return recs

        df = pd.DataFrame(recs)
        # When three fields are missing, treat as 0
        fav = df['favourites_count'].fillna(0) if 'favourites_count' in df.columns else pd.Series([0] * len(df))
        reblogs = df['reblogs_count'].fillna(0) if 'reblogs_count' in df.columns else pd.Series([0] * len(df))
        num_comments = df['num_comments'].fillna(0) if 'num_comments' in df.columns else pd.Series([0] * len(df))
        df['weight'] = fav + reblogs + num_comments
        df = df.sort_values('weight', ascending=False)
        df = df.drop(columns=['weight'])
        return df.head(max_rows).to_dict(orient='records')

    def _truncate_news_data(self, recs: List[Dict[str, Any]], schema_key: str) -> List[Dict[str, Any]]:
        """Intelligent sampling: stratified sampling by sentiment (take some from Positive/Negative/Neutral each)"""
        trunc_cfg = self.schema_cfg.truncation
        if not trunc_cfg or not trunc_cfg.news_data:
            return recs
        
        max_rows = trunc_cfg.news_data.max_rows
        sentiment_dist = trunc_cfg.news_data.sentiment_distribution
        
        if len(recs) <= max_rows:
            return recs
        
        df = pd.DataFrame(recs)
        if 'sentiment' not in df.columns:
            return recs[:max_rows]
        
        sampled = []
        for sentiment, limit in sentiment_dist.items():
            # Handle NaN values, only match non-empty and matching sentiment
            mask = df['sentiment'].notna() & (df['sentiment'].astype(str).str.upper() == sentiment.upper())
            subset = df[mask].head(limit)
            if not subset.empty:
                sampled.append(subset)
        
        result_df = pd.concat(sampled, ignore_index=True) if sampled else df.head(max_rows)
        # If still exceeds max_rows after sampling, truncate again by sentiment importance
        if len(result_df) > max_rows:
            result_df = result_df.head(max_rows)
        
        return result_df.to_dict(orient='records')

    def _truncate_macro_text(self, text: Optional[str], max_chars: int) -> Optional[str]:
        """Truncate macro text: keep beginning and end"""
        # Handle NaN values (pandas converts empty values to float NaN when reading CSV)
        if text is None:
            return None
        # Check if NaN (float type NaN)
        try:
            if isinstance(text, float) and pd.isna(text):
                return None
        except (TypeError, AttributeError):
            pass
        # Ensure string type
        if not isinstance(text, str):
            text = str(text) if text else None
            if not text:
                return None
        if len(text) <= max_chars:
            return text
        
        # Keep first 60% and last 40%
        head_chars = int(max_chars * 0.6)
        tail_chars = max_chars - head_chars
        return text[:head_chars] + "\n[... truncated ...]\n" + text[-tail_chars:]

    def _truncate_macro_data(self, recs: List[Dict[str, Any]], schema_key: str) -> List[Dict[str, Any]]:
        """Truncate text field of macro data"""
        trunc_cfg = self.schema_cfg.truncation
        if not trunc_cfg or not trunc_cfg.macro_data:
            return recs
        
        max_chars = trunc_cfg.macro_data.max_text_chars
        result = []
        for rec in recs:
            new_rec = rec.copy()
            # Check if text field exists and is not empty (including handling NaN values)
            if 'text' in new_rec:
                text_value = new_rec['text']
                # Handle NaN values (pandas converts empty values to float NaN when reading CSV)
                is_nan = False
                try:
                    if isinstance(text_value, float) and pd.isna(text_value):
                        is_nan = True
                except (TypeError, AttributeError):
                    pass
                
                if text_value is not None and not is_nan:
                    new_rec['text'] = self._truncate_macro_text(text_value, max_chars)
                else:
                    # If NaN or None, set to None
                    new_rec['text'] = None
            result.append(new_rec)
        return result

    def signals_for(self, asset: str, dstr: str, access_level: str = 'basic', market_window_days: Optional[int] = None, remove_social_news: bool = False) -> Dict[str, Any]:
        """
        Return merged signals per category for the given asset/date and access level.

        access_level: 'basic' or 'advanced'
        market_window_days: If specified, market_data returns historical window_days days of data (going back from dstr)
        Output (each category is a list of records, even if single row):
        {
          'market_data': [{date: '2025-04-01', open: ..., close: ...}, {date: '2025-04-02', ...}, ...],  # Historical data list
          'macro_data': [{...}, {...}], 'news_data': [{...}, {...}],
          'social_data': [{...}, {...}], 'company_data': [{...}], 'on_chain_data': [{...}]
        }
        """
        if access_level not in ('basic', 'advanced'):
            raise ValueError(f"access_level must be 'basic' or 'advanced', got {access_level}")

        data = self.load_asset(asset)

        def pack_single(kind_file_key: str, schema_key: str) -> List[Dict[str, Any]]:
            # Skip category if not applicable for asset
            if not self._applicable(asset, schema_key):
                return []
            recs = self._slice_date(data.get(kind_file_key), dstr)
            if not recs:
                return []
            # Field merging
            merged_recs = [self._merge_fields(rec, schema_key, access_level) for rec in recs]
            # Intelligent truncation
            if schema_key == 'social_data':
                merged_recs = self._truncate_social_data(merged_recs, schema_key)
            elif schema_key == 'news_data':
                merged_recs = self._truncate_news_data(merged_recs, schema_key)
            return merged_recs

        def pack_multi(kind_file_keys: List[str], schema_key: str) -> List[Dict[str, Any]]:
            # Merge multiple data sources, return list of all rows
            all_recs = []
            for k in kind_file_keys:
                recs = self._slice_date(data.get(k), dstr)
                if recs:
                    # Merge fields for each row and add to result list
                    all_recs.extend([self._merge_fields(rec, schema_key, access_level) for rec in recs])
            # Intelligent truncation
            if schema_key == 'macro_data':
                all_recs = self._truncate_macro_data(all_recs, schema_key)
            return all_recs

        def pack_market_history(kind_file_key: str, schema_key: str, window_days: int) -> List[Dict[str, Any]]:
            """Pack historical data for market_data"""
            if not self._applicable(asset, schema_key):
                return []
            # Get historical window_days days of data
            recs = self._slice_date_range(data.get(kind_file_key), dstr, window_days)
            if not recs:
                return []
            # Field merging
            merged_recs = [self._merge_fields(rec, schema_key, access_level) for rec in recs]
            return merged_recs

        # Special handling for market_data: if market_window_days is specified, return historical data
        if market_window_days is not None and market_window_days > 0:
            market_data = pack_market_history('processed_market', 'market_data', market_window_days)
        else:
            market_data = pack_single('processed_market', 'market_data')

        # Ablation 2: If remove_social_news is True, remove social and news data
        if remove_social_news:
            all_social = []
            all_news = []
        else:
            # Special handling for social_data:
            # - If use_trump_social = true: only use Trump data (apply same truncation config as regular social_data, but sorting weight based on favourites_count/reblogs_count/num_comments)
            # - If use_trump_social = false: only use Reddit data (truncated)
            if self.use_trump_social:
                # Trump experiment mode: only use Trump data, apply specialized truncation logic
                trump_social_raw = self._get_trump_social_for_date(dstr)
                # Filter useless fields (permalink, post_id, comment_id, etc.), but keep other original fields
                all_social = []
                if trump_social_raw:
                    # List of fields to filter
                    fields_to_remove = ['permalink', 'post_id', 'comment_id', 'parent_id', 'body', 'url', 'asset']
                    for rec in trump_social_raw:
                        filtered_rec = {k: v for k, v in rec.items() if k not in fields_to_remove}
                        all_social.append(filtered_rec)
                # Apply Trump truncation logic (share max_rows config with social_data)
                if all_social:
                    all_social = self._truncate_trump_social_data(all_social)
            else:
                # Baseline mode: only use Reddit data (already truncated)
                all_social = pack_single('processed_tweets', 'social_data')
            
            all_news = pack_single('processed_news', 'news_data')

        return {
            'market_data': market_data,
            'macro_data': pack_multi(['processed_macro', 'processed_macro_reports'], 'macro_data'),
            'news_data': all_news,
            'social_data': all_social,
            'company_data': pack_multi(['processed_company_data', 'processed_company_data_earnings'], 'company_data'),
            'on_chain_data': pack_single('processed_on_chain', 'on_chain_data'),
        }

    def list_available_dates(self, asset: str) -> List[pd.Timestamp]:
        """
        Return list of available trading days for asset.
        Only extract dates from market_data, because only market_data guarantees only trading days (no weekend trading).
        Other data sources (news, social, etc.) may contain weekend data and should not be used to determine simulation dates.
        """
        dates = set()
        data = self.load_asset(asset)
        # Prefer market_data dates (only contains trading days)
        market_df = data.get('processed_market')
        if market_df is not None and isinstance(market_df, pd.DataFrame) and not market_df.empty:
            date_col = next((c for c in market_df.columns if c.lower() == 'date'), None)
            if date_col:
                for d in pd.to_datetime(market_df[date_col], errors='coerce').dt.date.dropna():
                    dates.add(d)
        else:
            # If no market_data, fallback to other data sources (but this should not happen)
        for df in data.values():
            if isinstance(df, pd.DataFrame) and not df.empty:
                date_col = next((c for c in df.columns if c.lower() == 'date'), None)
                if date_col:
                    for d in pd.to_datetime(df[date_col], errors='coerce').dt.date.dropna():
                        dates.add(d)
        return sorted([pd.Timestamp(d) for d in dates])
    
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
        data = self.load_asset(asset)
        market_df = data.get('processed_market')
        if market_df is None or not isinstance(market_df, pd.DataFrame) or market_df.empty:
            return False
        
        date_col = next((c for c in market_df.columns if c.lower() == 'date'), None)
        if not date_col:
            return False
        
        # Convert input date to date object
        try:
            if isinstance(date, str):
                target_date = pd.to_datetime(date).date()
            elif isinstance(date, pd.Timestamp):
                target_date = date.date()
            else:
                target_date = date
        except Exception:
            return False
        
        # Check if this date exists in market_data
        market_dates = pd.to_datetime(market_df[date_col], errors='coerce').dt.date
        return target_date in market_dates.values
    
    def _load_trump_social(self) -> None:
        """Load Trump social data CSV file"""
        if not os.path.exists(self.trump_social_path):
            print(f"[WARN] Trump social data file does not exist: {self.trump_social_path}")
            self._trump_social_df = pd.DataFrame()
            return
        
        try:
            self._trump_social_df = pd.read_csv(self.trump_social_path, encoding='utf-8-sig')
            print(f"[TRUMP] Loaded Trump social data: {len(self._trump_social_df)} records")
        except Exception as e:
            print(f"[ERROR] Failed to load Trump social data: {e}")
            self._trump_social_df = pd.DataFrame()
    
    def _get_trump_social_for_date(self, dstr: str) -> List[Dict[str, Any]]:
        """
        Get Trump social data for specified date
        
        Args:
            dstr: Date string (YYYY-MM-DD)
        
        Returns:
            List of Trump social data for this date (dictionary format)
        """
        if self._trump_social_df is None or self._trump_social_df.empty:
            return []
        
        # Filter by date
        date_col = next((c for c in self._trump_social_df.columns if c.lower() == 'date'), None)
        if not date_col:
            return []
        
        # Ensure date format consistency: convert date column to string and strip spaces, also normalize input date
        target_date = str(dstr).strip()
        # Convert date column to string and strip spaces for matching
        date_series = self._trump_social_df[date_col].astype(str).str.strip()
        
        # Try multiple matching methods
        filtered_df = self._trump_social_df[date_series == target_date].copy()
        
        # If direct match fails, try converting to date object then compare
        if filtered_df.empty:
            try:
                target_date_obj = pd.to_datetime(target_date).date()
                date_series_parsed = pd.to_datetime(self._trump_social_df[date_col], errors='coerce').dt.date
                filtered_df = self._trump_social_df[date_series_parsed == target_date_obj].copy()
            except Exception:
                pass
        
        # Debug info: for 2024-11-01, if data not found, print detailed info
        if filtered_df.empty and ('2024-11-01' in str(dstr) or str(dstr).strip() == '2024-11-01'):
            print(f"[DEBUG] Trump data not found: date={repr(dstr)}, target_date={repr(target_date)}")
            print(f"[DEBUG] DataFrame size: {len(self._trump_social_df)}, date_col={date_col}")
            if len(self._trump_social_df) > 0:
                sample_dates = self._trump_social_df[date_col].head(10).tolist()
                print(f"[DEBUG] Sample dates in DataFrame: {sample_dates}")
                # Check if 2024-11-01 data exists (using different matching methods)
                test_match1 = len(self._trump_social_df[date_series == '2024-11-01'])
                test_match2 = len(self._trump_social_df[date_series.str.contains('2024-11-01')])
                print(f"[DEBUG] Direct match count: {test_match1}, Contains match count: {test_match2}")
        
        if filtered_df.empty:
            return []
        
        # Convert to list of dictionaries
        return filtered_df.to_dict(orient='records')