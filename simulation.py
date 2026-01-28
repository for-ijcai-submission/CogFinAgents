import os
import pandas as pd
from typing import Dict, List, Any, Optional
import dspy
import traceback
import json

from env.Environment import Environment
from trader.trading_agent import TradingAgent, normalize_decision_field
from config.ConfigLoader import load_simulation_config, load_agents_config, load_data_schema
from evaluation.EpisodeEnricher import EpisodeEnricher

def normalize_field(value: Any, default: Any = None) -> str:
    """Normalize field formatting: remove quotes, convert to plain string"""
    if value is None:
        return str(default) if default is not None else ''
    # If string, remove leading/trailing quotes if present
    if isinstance(value, str):
        value = value.strip()
        # Remove leading/trailing single or double quotes
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        return value
    # If numeric, convert directly to string (without quotes)
    return str(value)

def simulation(sim_cfg_path: str, agents_cfg_path: str, schema_cfg_path: str, save_result: bool = True):
    # Load configs
    sim_cfg = load_simulation_config(sim_cfg_path)
    agents_cfg = load_agents_config(agents_cfg_path)
    schema_cfg = load_data_schema(schema_cfg_path)

    # ---------------- Case study configuration priority rules ----------------
    # When enabled data_drop_rules (Case study) exist:
    # - Only run assets and time ranges specified by these rules
    # - Main experiment's assets / date_start / date_end automatically disabled
    # - Case study parameters can be independent of main experiment config, no subset constraint required
    active_case_rules = [
        rule for rule in (getattr(sim_cfg, "data_drop_rules", None) or [])
        if getattr(rule, "enabled", False)
    ]
    if active_case_rules:
        if len(active_case_rules) > 1:
            # To avoid ambiguity, currently only one case study rule can be enabled per run
            names = [r.name for r in active_case_rules]
            raise ValueError(
                f"Multiple case study rules enabled ({names}). "
                f"Current design supports only one enabled data_drop_rule per run. Please keep only one enabled: true."
            )
        rule = active_case_rules[0]

        # Override simulation scope: only run assets and date range from case study
        # Case study configuration is completely independent, not constrained by main experiment config
        sim_cfg.assets = list(rule.assets)
        sim_cfg.date_start = rule.date_start
        sim_cfg.date_end = rule.date_end
        
        # Override agent_type_count: only create agent types specified in case study rule
        # Keep the count from main experiment config, but filter to only rule.agents
        case_agent_types = set(getattr(rule, "agents", []))
        if case_agent_types:
            original_agent_type_count = sim_cfg.agent_type_count
            filtered_agent_type_count = {
                atype: count 
                for atype, count in original_agent_type_count.items()
                if atype in case_agent_types
            }
            # Validate that all rule.agents exist in original config
            missing_types = case_agent_types - set(original_agent_type_count.keys())
            if missing_types:
                raise ValueError(
                    f"Case study rule specifies agent types {sorted(missing_types)} "
                    f"that are not defined in main experiment agent_type_count. "
                    f"Available types: {sorted(original_agent_type_count.keys())}"
                )
            sim_cfg.agent_type_count = filtered_agent_type_count
            print(f"[CASE STUDY] Agent types filtered to: {sorted(filtered_agent_type_count.keys())} "
                  f"(from rule: {sorted(case_agent_types)})")

    # Configure LLM
    llm = dspy.LM(
        api_base=sim_cfg.llm.api_base,
        api_key=sim_cfg.llm.api_key,
        model=sim_cfg.llm.model,
        temperature=sim_cfg.llm.temperature,
        cache=sim_cfg.llm.cache
    )
    dspy.configure(lm=llm)

    # Environment
    env = Environment(sim_cfg=sim_cfg, schema_cfg=schema_cfg, hooks={
        'before_day': [],
        'after_day': [],
        'before_decision': [],
        'after_decision': [],
    })

    # Build agents
    agents: List[TradingAgent] = []
    debug_llm_io = bool(getattr(sim_cfg.execution, "debug_llm_io", False))
    for atype, n in sim_cfg.agent_type_count.items():
        for i in range(n):
            agents.append(
                TradingAgent(
                    agent_id=f'{atype}-{i}',
                    agent_type=atype,
                    agents_cfg=agents_cfg,
                    debug_llm_io=debug_llm_io,
                )
            )

    history_rows: List[Dict] = []
    # Isolate memory by (agent_id, asset) to avoid cross-contamination between different assets
    episodes_store: Dict[str, Dict[str, List[Dict]]] = {
        a.agent_id: {asset: [] for asset in sim_cfg.assets} 
        for a in agents
    }

    num_threads = sim_cfg.execution.max_workers

    # Incremental save: append save after processing each day
    result_dir = sim_cfg.result_dir if save_result else None
    json_path = None
    csv_path = None
    input_csv_path = None  # CSV file path for dynamic input
    price_feed_path = None  # CSV file path for price data
    existing_data: Dict[tuple, Dict] = {}  # key: (agent_id, date, asset), value: row dict
    existing_input_data: Dict[tuple, Dict] = {}  # key: (agent_id, date, asset), value: input row dict
    model_slug = None
    base_name = None
    
    # Preload price data: check if price_feed.csv exists
    price_feed: Optional[pd.DataFrame] = None
    if result_dir:
        price_feed_path = os.path.join(result_dir, 'price_feed.csv')
        if os.path.exists(price_feed_path) and os.path.getsize(price_feed_path) > 0:
            try:
                print(f"[LOAD] Loading price feed from {price_feed_path}...")
                price_feed = pd.read_csv(price_feed_path, encoding='utf-8-sig')
                # Ensure date format is unified as string
                price_feed['date'] = pd.to_datetime(price_feed['date'], errors='coerce').dt.strftime("%Y-%m-%d")
                # Validate required columns
                required_cols = ['date', 'asset', 'open', 'close']
                if not all(col in price_feed.columns for col in required_cols):
                    print(f"[WARN] Price feed missing required columns. Expected: {required_cols}, got: {list(price_feed.columns)}")
                    price_feed = None
                else:
                    print(f"[LOAD] Price feed loaded: {len(price_feed)} records, assets: {sorted(price_feed['asset'].unique().tolist())}, "
                          f"date range: {price_feed['date'].min()} to {price_feed['date'].max()}")
            except Exception as e:
                print(f"[WARN] Failed to load price feed from {price_feed_path}: {e}")
                traceback.print_exc()
                price_feed = None
        else:
            print(f"[INFO] Price feed not found at {price_feed_path}, will load from DataProvider during enrichment")
    
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
        # Generate concise filename: include model name and start/end dates
        model_slug = str(sim_cfg.llm.model).replace('/', '-').replace(':', '-').replace(' ', '_')
        base_name = f"res_{model_slug}_{sim_cfg.date_start}_{sim_cfg.date_end}"
        
        # Add suffix based on experiment type
        experiment_suffixes = []
        # Trump experiment
        if sim_cfg.trump_social and sim_cfg.trump_social.enabled:
            experiment_suffixes.append("trump")
        # Ablation 1: Disable memory
        if sim_cfg.ablation and sim_cfg.ablation.memory_disabled:
            experiment_suffixes.append("ablation1-memory")
        # Ablation 2: Remove social and news
        if sim_cfg.ablation and sim_cfg.ablation.remove_social_news:
            experiment_suffixes.append("ablation2-socialnews")
        # Case study / general data drop experiments: enabled data_drop_rules names as suffix
        if getattr(sim_cfg, "data_drop_rules", None):
            active_rule_names = [rule.name for rule in sim_cfg.data_drop_rules if getattr(rule, "enabled", False)]
            experiment_suffixes.extend(active_rule_names)
        
        # If experiment tags exist, add to filename
        if experiment_suffixes:
            experiment_tag = "_".join(experiment_suffixes)
            base_name = f"{base_name}_{experiment_tag}"
        
        json_path = os.path.join(result_dir, f'{base_name}.json')
        csv_path = os.path.join(result_dir, f'{base_name}.csv')
        input_csv_path = os.path.join(result_dir, f'input_{base_name}.csv')  # CSV file for dynamic input
        
        # If CSV file exists, read existing data and load into episodes_store
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            try:
                print(f"[LOAD] Loading existing data from {csv_path}...")
                existing_df = pd.read_csv(csv_path, encoding='utf-8-sig')
                if not existing_df.empty:
                    # Convert existing data to list of dictionaries
                    # Ensure date format is unified as string (YYYY-MM-DD)
                    if 'date' in existing_df.columns:
                        existing_df['date'] = pd.to_datetime(existing_df['date'], errors='coerce').dt.strftime("%Y-%m-%d")
                    existing_rows = existing_df.to_dict('records')
                    # Build deduplication dictionary: key = (agent_id, date, asset)
                    # Also filter out non-trading day data
                    for row in existing_rows:
                        # Ensure date is string format
                        date_val = row.get('date', '')
                        date_str = str(date_val) if date_val is not None else ''
                        # Handle possible NaN or invalid dates
                        if not date_str or date_str == 'NaT' or date_str == 'nan' or date_str == 'None':
                            continue
                        
                        asset = str(row.get('asset', ''))
                        # Verify if the asset is a trading day on this date
                        if asset and not env.is_trading_day(asset, date_str):
                            print(f"[LOAD] Filtered out non-trading day record: {date_str} {asset}")
                            continue
                        
                        key = (str(row.get('agent_id', '')), date_str, asset)
                        existing_data[key] = row
                        
                        # Load into episodes_store (for memory)
                        agent_id = str(row.get('agent_id', ''))
                        if agent_id in episodes_store and asset in episodes_store[agent_id]:
                            # Convert to Episode format dictionary
                            episode_dict = {
                                'date': row.get('date'),
                                'asset': asset,
                                'agent_id': agent_id,
                                'belief_prior': normalize_field(row.get('belief_prior'), ''),
                                'belief_score': normalize_field(row.get('belief_score'), ''),
                                'action': normalize_field(row.get('action'), ''),
                                'target_position': normalize_field(row.get('target_position'), ''),
                                'selected_signals': row.get('selected_signals'),
                                'selected_memories': row.get('selected_memories'),
                                'reason': row.get('reason'),
                                'memory_update_note': row.get('memory_update_note'),
                                'price_open': row.get('price_open'),
                                'price_next': row.get('price_next'),
                                'realized_return': row.get('realized_return'),
                                'pnl': row.get('pnl'),
                                'prediction_error': row.get('prediction_error'),
                                'position_before': row.get('position_before'),
                                'position_after': row.get('position_after'),
                            }
                            episodes_store[agent_id][asset].append(episode_dict)
                    
                    print(f"[LOAD] Loaded {len(existing_data)} existing records, restored memory for {len([a for a in episodes_store.values() for episodes in a.values() if episodes])} episodes")
            except Exception as e:
                print(f"[WARN] Failed to load existing data from {csv_path}: {e}")
                traceback.print_exc()
                existing_data = {}
        
        # Load existing dynamic input data (if exists)
        if input_csv_path and os.path.exists(input_csv_path) and os.path.getsize(input_csv_path) > 0:
            try:
                print(f"[LOAD] Loading existing input data from {input_csv_path}...")
                existing_input_df = pd.read_csv(input_csv_path, encoding='utf-8-sig')
                if not existing_input_df.empty:
                    if 'date' in existing_input_df.columns:
                        existing_input_df['date'] = pd.to_datetime(existing_input_df['date'], errors='coerce').dt.strftime("%Y-%m-%d")
                    existing_input_rows = existing_input_df.to_dict('records')
                    for row in existing_input_rows:
                        date_val = row.get('date', '')
                        date_str = str(date_val) if date_val is not None else ''
                        if not date_str or date_str == 'NaT' or date_str == 'nan' or date_str == 'None':
                            continue
                        key = (str(row.get('agent_id', '')), date_str, str(row.get('asset', '')))
                        existing_input_data[key] = row
                    print(f"[LOAD] Loaded {len(existing_input_data)} existing input records")
            except Exception as e:
                print(f"[WARN] Failed to load existing input data from {input_csv_path}: {e}")
                traceback.print_exc()
                existing_input_data = {}

    # Track whether input has been printed for each agent_type (print only once per agent_type)
    printed_agent_types: set = set()

    def apply_data_drop_rules_for_signals(
        signals: Dict[str, Any],
        agent_type: str,
        asset: str,
        date_str: str,
    ) -> Dict[str, Any]:
        """
        Apply module-level masking to signals for specified agent_type × asset × date based on sim_cfg.data_drop_rules:
        Set matching rule modules to empty list.
        """
        rules = getattr(sim_cfg, "data_drop_rules", None)
        if not rules:
            return signals

        try:
            dt = pd.to_datetime(date_str, errors="coerce")
        except Exception:
            return signals
        if pd.isna(dt):
            return signals

        out = dict(signals) if isinstance(signals, dict) else signals
        for rule in rules:
            if not getattr(rule, "enabled", False):
                continue
            if agent_type not in getattr(rule, "agents", []):
                continue
            if asset not in getattr(rule, "assets", []):
                continue
            try:
                start_dt = pd.to_datetime(rule.date_start)
                end_dt = pd.to_datetime(rule.date_end)
            except Exception:
                continue
            if not (start_dt <= dt <= end_dt):
                continue
            # Rule matched: clear corresponding module
            for module in getattr(rule, "drop_modules", []):
                if isinstance(out, dict) and module in out:
                    out[module] = []  # All data modules in signals are lists
        return out
    
    try:
        for d in env.iter_dates():
            # Unify date string format to YYYY-MM-DD, avoid "YYYY-MM-DD 00:00:00"
            dstr = pd.to_datetime(d, errors='coerce').strftime("%Y-%m-%d")
            env.run_hook('before_day', date=dstr)
            day_rows: List[Dict] = []
            day_input_rows: List[Dict] = []  # Dynamic input records for the day

            for agent in agents:
                try:
                    access_level = agent.profile.info_access
                    merged_signals_by_asset: Dict[str, Dict[str, Any]] = {}
                    memory_by_asset: Dict[str, List[Dict[str, Any]]] = {}
                    # Only process assets that are trading days on this date
                    trading_assets = [asset for asset in sim_cfg.assets if env.is_trading_day(asset, d)]
                    if not trading_assets:
                        print(f"[INFO] No assets trading on {dstr}, skipping agent {agent.agent_id}")
                        continue

                    for asset in trading_assets:
                        try:
                            env.run_hook('before_decision', date=dstr, asset=asset, agent_id=agent.agent_id)  # type: ignore
                            # Use agent's window_days config to limit market_data history days
                            market_window_days = agent.profile.memory.window_days
                            raw_signals = env.signals_for(
                                asset,
                                dstr,
                                access_level=access_level,
                                market_window_days=market_window_days,
                            )
                            # Apply general data drop rules (case study / ablation)
                            merged_signals_by_asset[asset] = apply_data_drop_rules_for_signals(
                                raw_signals,
                                agent_type=agent.agent_type,
                                asset=asset,
                                date_str=dstr,
                            )
                            # Isolate memory by asset: each asset only sees its own historical memory
                            # Ablation 1: If memory_disabled is True, force memory to be empty
                            if sim_cfg.ablation and sim_cfg.ablation.memory_disabled:
                                memory_by_asset[asset] = []  # Force empty
                            else:
                                memory_by_asset[asset] = agent.select_memory(
                                    episodes_store[agent.agent_id][asset],
                                    window_date=dstr,
                                )
                        except Exception as e:
                            print(f"[ERROR] Failed to get signals for {agent.agent_id}/{asset} on {dstr}: {e}")
                            merged_signals_by_asset[asset] = {}
                            memory_by_asset[asset] = []

            # Batch decide via unified TradingAgent interface
                    # Print input only once per agent_type
                    should_print_input = debug_llm_io and (agent.agent_type not in printed_agent_types)
                    if should_print_input:
                        printed_agent_types.add(agent.agent_type)

                    try:
                        # Ablation 1: Pass memory_disabled flag to batch_decide
                        memory_disabled = sim_cfg.ablation.memory_disabled if sim_cfg.ablation else False
                        results = agent.batch_decide(
                            date=dstr,
                            assets=trading_assets,  # Only pass trading day assets
                            merged_signals_by_asset=merged_signals_by_asset,
                            memory_by_asset=memory_by_asset,
                            num_threads=num_threads,
                            should_print_input=should_print_input,
                            memory_disabled=memory_disabled,
                        )
                    except Exception as e:
                        print(f"[ERROR] Agent {agent.agent_id} batch_decide failed on {dstr}: {e}")
                        traceback.print_exc()
                        # Create default decisions only for trading day assets (use normalize_decision_field to ensure type consistency)
                        # Return format must match batch_decide normal return format: {'pred': ..., 'episode': ...}
                        results = {}
                        for asset in trading_assets:
                            default_decision = {
                                'selected_signals': normalize_decision_field([], 'selected_signals'),
                                'selected_memories': normalize_decision_field([], 'selected_memories'),
                                'belief_prior': normalize_decision_field(0.0, 'belief_prior'),
                                'belief_score': normalize_decision_field(0.0, 'belief_score'),
                                'action': normalize_decision_field('HOLD', 'action'),
                                'target_position': normalize_decision_field(0.0, 'target_position'),
                                'reason': normalize_decision_field(f'Agent decision failed: {str(e)[:100]}', 'reason'),
                                'memory_update_note': normalize_decision_field('', 'memory_update_note'),
                            }
                            from model.Episode import Episode

                            episode = Episode.from_decision(default_decision, date=dstr, asset=asset, agent_id=agent.agent_id)
                            # Create default pred object, matching batch_decide return format
                            default_pred = type('DefaultPred', (), {
                                'selected_signals': default_decision['selected_signals'],
                                'selected_memories': default_decision['selected_memories'],
                                'belief_prior': default_decision['belief_prior'],
                                'belief_score': default_decision['belief_score'],
                                'action': default_decision['action'],
                                'target_position': default_decision['target_position'],
                                'reason': default_decision['reason'],
                                'memory_update_note': default_decision['memory_update_note'],
                            })()
                            results[asset] = {'pred': default_pred, 'episode': episode}

                    # Collect results - reference dspy_sample.py style, flattened output
                    # Only process results for trading day assets
                    for asset in trading_assets:
                        try:
                            # Verify again that the asset is a trading day on this date (double check)
                            if not env.is_trading_day(asset, d):
                                print(f"[WARN] Asset {asset} is not a trading day on {dstr}, skipping result collection")
                                continue

                            if asset not in results:
                                print(f"[WARN] Missing result for {agent.agent_id}/{asset} on {dstr}, skipping")
                                continue

                            pred = results[asset]['pred']
                            episode = results[asset]['episode']
                            episode_dict = episode.model_dump()

                            # Store memory isolated by asset
                            episodes_store[agent.agent_id][asset].append(episode_dict)

                            # Reference dspy_sample.py: directly access dspy output fields, flatten to CSV columns
                            # Convert list fields to comma-separated strings for CSV readability
                            selected_signals = getattr(pred, 'selected_signals', [])
                            selected_memories = getattr(pred, 'selected_memories', [])

                            row = {
                                # Basic identification fields
                                'date': dstr,
                                'asset': asset,
                                'agent_id': agent.agent_id,
                                'agent_type': agent.agent_type,
                                # InvestmentDecisionPrompt output fields (reference prompts.py)
                                'selected_signals': ', '.join(str(s) for s in selected_signals) if selected_signals else '',
                                'selected_memories': ', '.join(str(m) for m in selected_memories) if selected_memories else '',
                                'belief_prior': normalize_field(getattr(pred, 'belief_prior', 0.0), 0.0),
                                'belief_score': normalize_field(getattr(pred, 'belief_score', 0.0), 0.0),
                                'action': normalize_field(getattr(pred, 'action', 'HOLD'), 'HOLD'),
                                'target_position': normalize_field(getattr(pred, 'target_position', 0.0), 0.0),
                                'reason': normalize_field(getattr(pred, 'reason', ''), ''),
                                'memory_update_note': normalize_field(getattr(pred, 'memory_update_note', ''), ''),
                                # Other Episode fields (ex-post and state fields)
                                'price_open': episode_dict.get('price_open'),
                                'price_next': episode_dict.get('price_next'),
                                'realized_return': episode_dict.get('realized_return'),
                                'pnl': episode_dict.get('pnl'),
                                'prediction_error': episode_dict.get('prediction_error'),
                                'position_before': episode_dict.get('position_before'),
                                'position_after': episode_dict.get('position_after'),
                            }
                            day_rows.append(row)
                            history_rows.append(row)

                            # Collect dynamic input (only save successful decisions, not default decisions)
                            if 'dynamic_input' in results[asset]:
                                dynamic_input = results[asset]['dynamic_input']
                                input_row = {
                    'date': dstr,
                    'asset': asset,
                    'agent_id': agent.agent_id,
                    'agent_type': agent.agent_type,
                                    'memory_notes': json.dumps(dynamic_input.get('memory_notes', ''), ensure_ascii=False),
                                    'data_inputs': json.dumps(dynamic_input.get('data_inputs', {}), ensure_ascii=False),
                                }
                                day_input_rows.append(input_row)

                            env.run_hook('after_decision', date=dstr, asset=asset, agent_id=agent.agent_id)  # type: ignore

                        except Exception as e:
                            print(f"[ERROR] Failed to collect result for {agent.agent_id}/{asset} on {dstr}: {e}")
                            traceback.print_exc()
                            continue

                except Exception as e:
                    print(f"[ERROR] Agent {agent.agent_id} failed on {dstr}: {e}")
                    traceback.print_exc()
                    continue

            # Real-time calculation of ex-post metrics: enrich episodes t-2 and earlier (if price_feed available)
            # Note: For "open-next_open2" strategy, need t+1 and t+2 prices, so can only calculate episodes t-2 and earlier
            if price_feed is None or price_feed.empty:
                # Try to load price_feed from DataProvider (if not yet loaded)
                try:
                    print(f"[ENRICH] Loading price feed from DataProvider for real-time enrichment...")
                    price_feed_list = []
                    for asset in sim_cfg.assets:
                        asset_data = env.provider.load_asset(asset)
                        market_df = asset_data.get('processed_market')
                        if market_df is not None and not market_df.empty:
                            if 'date' in market_df.columns and 'open' in market_df.columns and 'close' in market_df.columns:
                                market_df = market_df[['date', 'open', 'close']].copy()
                                market_df['asset'] = asset
                                price_feed_list.append(market_df)
                    if price_feed_list:
                        price_feed = pd.concat(price_feed_list, ignore_index=True)
                        price_feed['date'] = pd.to_datetime(price_feed['date']).dt.strftime("%Y-%m-%d")
                        print(f"[ENRICH] Price feed loaded: {len(price_feed)} records")
                except Exception as e:
                    print(f"[WARN] Failed to load price feed for real-time enrichment: {e}")

            if price_feed is not None and not price_feed.empty and existing_data:
                try:
                    # Find episodes that can calculate ex-post metrics (t-2 and earlier)
                    # Current date is d, need to calculate episodes d-2 and earlier
                    from datetime import datetime, timedelta
                    current_date = pd.to_datetime(dstr).date()
                    # Get all trading dates, find date corresponding to d-2
                    all_trading_dates = sorted([pd.to_datetime(d).date() for d in env.sim_dates])
                    cutoff_idx = None
                    for i, td in enumerate(all_trading_dates):
                        if td >= current_date:
                            cutoff_idx = i - 2  # Position of d-2
                            break

                    if cutoff_idx is not None and cutoff_idx >= 0:
                        cutoff_date = all_trading_dates[cutoff_idx]
                        cutoff_str = cutoff_date.strftime("%Y-%m-%d")

                        # Filter episodes that need enrichment (t-2 and earlier, and don't have ex-post metrics yet)
                        episodes_to_enrich = []
                        for key, row in existing_data.items():
                            row_date_str = row.get('date', '')
                            if row_date_str:
                                try:
                                    row_date = pd.to_datetime(row_date_str).date()
                                    if row_date <= cutoff_date:
                                        # Check if ex-post metrics already exist
                                        if row.get('realized_return') is None or pd.isna(row.get('realized_return')):
                                            episodes_to_enrich.append(row)
                                except Exception:
                                    continue

                        if episodes_to_enrich:
                            print(f"[ENRICH] Real-time enriching {len(episodes_to_enrich)} episodes (date <= {cutoff_str})...")
                            enricher = EpisodeEnricher(price_feed=price_feed, settle_policy="open-next_open2")
                            enriched_df = enricher.enrich(episodes_to_enrich)

                            # Update ex-post fields in existing_data
                            enriched_dicts = enriched_df.to_dict('records')
                            enriched_by_key = {}
                            for enriched_row in enriched_dicts:
                                key = (
                                    str(enriched_row.get('agent_id', '')),
                                    str(enriched_row.get('date', '')),
                                    str(enriched_row.get('asset', ''))
                                )
                                enriched_by_key[key] = enriched_row

                            for key, row in existing_data.items():
                                if key in enriched_by_key:
                                    enriched_row = enriched_by_key[key]
                                    # Update ex-post fields
                                    existing_data[key]['price_open'] = enriched_row.get('price_open')
                                    existing_data[key]['price_next'] = enriched_row.get('price_next')
                                    existing_data[key]['realized_return'] = enriched_row.get('realized_return')
                                    existing_data[key]['pnl'] = enriched_row.get('pnl')
                                    existing_data[key]['prediction_error'] = enriched_row.get('prediction_error')

                            # Update corresponding episode_dict in episodes_store
                            for enriched_row in enriched_dicts:
                                agent_id = str(enriched_row.get('agent_id', ''))
                                asset = str(enriched_row.get('asset', ''))
                                date_str = str(enriched_row.get('date', ''))

                                if agent_id in episodes_store and asset in episodes_store[agent_id]:
                                    # Find corresponding episode_dict and update
                                    for episode_dict in episodes_store[agent_id][asset]:
                                        if episode_dict.get('date') == date_str:
                                            episode_dict['price_open'] = enriched_row.get('price_open')
                                            episode_dict['price_next'] = enriched_row.get('price_next')
                                            episode_dict['realized_return'] = enriched_row.get('realized_return')
                                            episode_dict['pnl'] = enriched_row.get('pnl')
                                            episode_dict['prediction_error'] = enriched_row.get('prediction_error')
                                            break

                            print(f"[ENRICH] Real-time enrichment completed: {len(episodes_to_enrich)} episodes updated")
                except Exception as e:
                    print(f"[WARN] Failed to perform real-time enrichment: {e}")
                    traceback.print_exc()

            # Incremental save: merge and save after processing each day (deduplicate: overwrite same agent_id+date+asset)
            # Filter out non-trading day data (double check)
            valid_day_rows = []
            for row in day_rows:
                asset = row.get('asset', '')
                date_str = row.get('date', '')
                if asset and date_str and env.is_trading_day(asset, date_str):
                    valid_day_rows.append(row)
                else:
                    print(f"[WARN] Filtered out non-trading day record: {date_str} {asset}")

            if save_result and valid_day_rows:
                try:
                    # Update existing_data: new data overwrites old data (same key)
                    # Only save trading day data
                    for row in valid_day_rows:
                        key = (str(row.get('agent_id', '')), str(row.get('date', '')), str(row.get('asset', '')))
                        existing_data[key] = row

                    # Merge and save all data (including new data and retained old data)
                    all_rows = list(existing_data.values())
                    if all_rows:
                        all_df = pd.DataFrame(all_rows)
                        # Sort by date and agent_id to ensure consistent output order
                        if 'date' in all_df.columns and 'agent_id' in all_df.columns:
                            all_df = all_df.sort_values(['date', 'agent_id', 'asset'], ascending=[True, True, True])

                        # Save complete data (overwrite mode)
                        all_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                        # JSON: overwrite mode (one JSON object per line)
                        with open(json_path, 'w', encoding='utf-8') as f:
                            for _, row in all_df.iterrows():
                                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + '\n')

                        print(f"[SAVED] Day {dstr}: {len(valid_day_rows)} new records, total {len(all_rows)} records saved")
                except Exception as e:
                    print(f"[ERROR] Failed to save results for {dstr}: {e}")
                    traceback.print_exc()

            # Save dynamic input (only save successful decisions, not default decisions)
            if save_result and day_input_rows and input_csv_path:
                try:
                    # Update existing_input_data: new data overwrites old data (same key)
                    for row in day_input_rows:
                        key = (str(row.get('agent_id', '')), str(row.get('date', '')), str(row.get('asset', '')))
                        existing_input_data[key] = row

                    # Merge and save all input data (including new data and retained old data)
                    all_input_rows = list(existing_input_data.values())
                    if all_input_rows:
                        all_input_df = pd.DataFrame(all_input_rows)
                        # Sort by date and agent_id
                        if 'date' in all_input_df.columns and 'agent_id' in all_input_df.columns:
                            all_input_df = all_input_df.sort_values(['date', 'agent_id', 'asset'], ascending=[True, True, True])

                        # Save dynamic input data
                        all_input_df.to_csv(input_csv_path, index=False, encoding='utf-8-sig')
                        print(f"[SAVED] Day {dstr}: {len(day_input_rows)} new input records, total {len(all_input_rows)} input records saved")
                except Exception as e:
                    print(f"[ERROR] Failed to save input data for {dstr}: {e}")
                    traceback.print_exc()

        env.run_hook('after_day', date=dstr)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Simulation interrupted by user")
    except Exception as e:
        print(f"\n[FATAL] Simulation failed: {e}")
        traceback.print_exc()
    finally:
        # Final save: ensure all data is saved (merge new and old data, deduplicate)
        if save_result:
            try:
                # Merge all data: existing_data already contains all data (new data already updated)
                all_rows = list(existing_data.values())

                if all_rows:
                    # Use EpisodeEnricher to fill ex-post fields (only for new data or data that needs updating)
                    # For efficiency, only enrich new data in history_rows
                    if history_rows:
                        print(f"[ENRICH] Enriching {len(history_rows)} new episodes with ex-post fields...")

                        # Use preloaded price_feed, if not exists then load from DataProvider
                        if price_feed is None or price_feed.empty:
                            print(f"[ENRICH] Price feed not preloaded, loading from DataProvider...")
                            # Build price_feed: get price data for all assets from DataProvider
                            price_feed_list = []
                            for asset in sim_cfg.assets:
                                asset_data = env.provider.load_asset(asset)
                                market_df = asset_data.get('processed_market')
                                if market_df is not None and not market_df.empty:
                                    # Ensure date, open, close columns exist
                                    if 'date' in market_df.columns and 'open' in market_df.columns and 'close' in market_df.columns:
                                        market_df = market_df[['date', 'open', 'close']].copy()
                                        market_df['asset'] = asset
                                        price_feed_list.append(market_df)

                            if price_feed_list:
                                price_feed = pd.concat(price_feed_list, ignore_index=True)
                                price_feed['date'] = pd.to_datetime(price_feed['date']).dt.strftime("%Y-%m-%d")
                                print(
                                    f"[ENRICH] Price feed loaded from DataProvider: {len(price_feed)} records, "
                                    f"assets: {sorted(price_feed['asset'].unique().tolist())}, "
                                    f"date range: {price_feed['date'].min()} to {price_feed['date'].max()}"
                                )
                            else:
                                print(f"[WARN] No price data available from DataProvider")

                        if price_feed is not None and not price_feed.empty:
                            print(f"[ENRICH] Episodes to enrich: {len(history_rows)} records")
                            # Use open-next_open2 strategy for final enrichment
                            enricher = EpisodeEnricher(price_feed=price_feed, settle_policy="open-next_open2")
                            enriched_df = enricher.enrich(history_rows)
                            # Debug: check enrichment results
                            non_zero_returns = enriched_df['realized_return'].notna().sum()
                            zero_or_nan_returns = (enriched_df['realized_return'].isna() | (enriched_df['realized_return'] == 0)).sum()
                            print(f"[ENRICH] Enrichment result: {non_zero_returns} non-zero returns, {zero_or_nan_returns} zero/NaN returns")
                            if zero_or_nan_returns > 0:
                                # Show some examples to help diagnose issues
                                sample_issues = enriched_df[
                                    enriched_df['realized_return'].isna() | (enriched_df['realized_return'] == 0)
                                ].head(3)
                                if not sample_issues.empty:
                                    print(f"[ENRICH] Sample issues (first 3):")
                                    for idx, row in sample_issues.iterrows():
                                        print(
                                            f"  - {row.get('date', 'N/A')} {row.get('asset', 'N/A')}: "
                                            f"price_open={row.get('price_open', 'N/A')}, "
                                            f"price_next={row.get('price_next', 'N/A')}, "
                                            f"close={row.get('close', 'N/A')}"
                                        )
                            # Convert DataFrame back to list of dictionaries, update ex-post fields
                            enriched_dicts = enriched_df.to_dict('records')
                            # Update ex-post fields for corresponding records in existing_data
                            for enriched_row in enriched_dicts:
                                key = (
                                    str(enriched_row.get('agent_id', '')),
                                    str(enriched_row.get('date', '')),
                                    str(enriched_row.get('asset', ''))
                                )
                                if key in existing_data:
                                    existing_data[key]['price_open'] = enriched_row.get('price_open')
                                    existing_data[key]['price_next'] = enriched_row.get('price_next')
                                    existing_data[key]['realized_return'] = enriched_row.get('realized_return')
                                    existing_data[key]['pnl'] = enriched_row.get('pnl')
                                    existing_data[key]['prediction_error'] = enriched_row.get('prediction_error')
                            print(f"[ENRICH] Ex-post fields enriched successfully")
                        else:
                            print(f"[WARN] No price data available for enrichment")

                    # Save final results (all merged data)
                    # Filter out non-trading day data again (ensure final saved data are all trading day data)
                    final_rows = []
                    for row in existing_data.values():
                        asset = row.get('asset', '')
                        date_str = row.get('date', '')
                        if asset and date_str and env.is_trading_day(asset, date_str):
                            final_rows.append(row)
                    df = pd.DataFrame(final_rows)
                    # Sort by date and agent_id to ensure consistent output order
                    if 'date' in df.columns and 'agent_id' in df.columns:
                        df = df.sort_values(['date', 'agent_id', 'asset'], ascending=[True, True, True])

                    df.to_json(json_path, orient='records', lines=True, force_ascii=False)
                    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                    print(f"[FINAL] All results saved: {len(final_rows)} records (including {len(history_rows)} new) to {result_dir}")
                else:
                    print(f"[FINAL] No data to save")
            except Exception as e:
                print(f"[ERROR] Failed to save final results: {e}")
                traceback.print_exc()

        print(f"Simulation finished. Total records: {len(history_rows)}. Output at {result_dir}.")

if __name__ == '__main__':
    simulation(
        sim_cfg_path="config/simulation.yaml",
        agents_cfg_path="config/agents.yaml",
        schema_cfg_path="config/data_schema.yaml",
        save_result=True
    )