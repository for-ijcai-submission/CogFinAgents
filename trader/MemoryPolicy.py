from typing import List, Dict
import pandas as pd
from config.ConfigLoader import AgentProfile

class MemoryPolicy:
    """
    Select memory episodes according to agent profile config (window_days, sort_key, top_k).
    Format memory notes with agent-type-specific detail levels.
    """

    @staticmethod
    def select(agent_id: str, agent_profile: AgentProfile, episodes: List[Dict], window_date: str) -> List[Dict]:
        """
        Select memory episodes according to agent configuration.
        
        Logic:
        1. Filter all episodes within window_days date range (counted by trading days, not calendar days)
        2. Sort by sort_key
        3. Take top_k episodes
        
        Args:
            agent_id: Agent ID
            agent_profile: Agent configuration (contains memory.window_days, memory.sort_key, memory.top_k)
            episodes: List of all available episodes
            window_date: Current date (string format, e.g. "2025-04-10")
            
        Returns:
            Filtered episodes list (at most top_k)
        """
        if not episodes:
            return []
        
        try:
            df = pd.DataFrame(episodes)
            if df.empty:
                return []
            
            # Parse dates, filter invalid dates
            df['dt'] = pd.to_datetime(df['date'], errors='coerce')
            cutoff = pd.to_datetime(window_date, errors='coerce')
            
            # Check if date parsing succeeded
            if pd.isna(cutoff):
                return []
            
            # Filter: only keep records with valid dates and <= cutoff
            df_valid = df[df['dt'].notna() & (df['dt'] <= cutoff)].copy()
            if df_valid.empty:
                return []
            
            # Sort by date from newest to oldest
            df_valid = df_valid.sort_values('dt', ascending=False)
            
            # Strict memory retrieval limit: current decision day t can only see content from t-2 and earlier
            # Implementation: first discard "the most recent trading day" (usually t-1) from available trading days list,
            # then take the most recent window_days from remaining dates as memory window.
            unique_dates = df_valid['dt'].drop_duplicates()
            if len(unique_dates) <= 1:
                # When there is only one historical trading day or less, do not provide memory to avoid seeing t-1
                return []
            # Discard the most recent trading day (t-1), only keep t-2 and earlier trading days
            unique_dates_excl_latest = unique_dates[1:]
            # From remaining trading days, take the most recent window_days
            window_dates = unique_dates_excl_latest.head(agent_profile.memory.window_days)
            window = df_valid[df_valid['dt'].isin(window_dates)].copy()
            
            if window.empty:
                return []
            
            # Sort by sort_key and take top_k
            sort_key = agent_profile.memory.sort_key
            top_k = agent_profile.memory.top_k
            
            if sort_key == 'abs_error':
                # Type D: sort by absolute prediction error
                if 'prediction_error' in window.columns:
                    window = window.assign(
                        abs_err=pd.to_numeric(window['prediction_error'], errors='coerce').fillna(0).abs()
                    ).nlargest(top_k, 'abs_err')
                else:
                    # If column doesn't exist, sort by date and take most recent
                    window = window.head(top_k)
                    
            elif sort_key == 'abs_return':
                # Type B: sort by absolute daily return
                if 'realized_return' in window.columns:
                    window = window.assign(
                        abs_ret=pd.to_numeric(window['realized_return'], errors='coerce').fillna(0).abs()
                    ).nlargest(top_k, 'abs_ret')
                else:
                    window = window.head(top_k)
                    
            elif sort_key == 'abs_pnl':
                # Type C: sort by absolute PnL
                if 'pnl' in window.columns:
                    window = window.assign(
                        abs_pnl=pd.to_numeric(window['pnl'], errors='coerce').fillna(0).abs()
                    ).nlargest(top_k, 'abs_pnl')
                else:
                    window = window.head(top_k)
                    
            elif sort_key == 'recency':
                # Type A: sort by date from newest to oldest, then by abs_return as secondary sort
                if 'realized_return' in window.columns:
                    window = window.assign(
                        abs_ret=pd.to_numeric(window['realized_return'], errors='coerce').fillna(0).abs()
                    ).sort_values(['dt', 'abs_ret'], ascending=[False, False]).head(top_k)
                else:
                    # If no realized_return, only sort by date
                    window = window.head(top_k)
            else:
                # Default: sort by date and take most recent top_k
                window = window.head(top_k)
            
            # Delete temporary columns
            if 'abs_err' in window.columns:
                window = window.drop(columns=['abs_err'])
            if 'abs_ret' in window.columns:
                window = window.drop(columns=['abs_ret'])
            if 'abs_pnl' in window.columns:
                window = window.drop(columns=['abs_pnl'])
            if 'dt' in window.columns:
                window = window.drop(columns=['dt'])
            
            return window.to_dict(orient='records')
            
        except Exception as e:
            # Exception handling: return empty list to avoid affecting main flow
            print(f"[WARN] MemoryPolicy.select failed for agent {agent_id} on {window_date}: {e}")
            return []

    @staticmethod
    def format_memory_notes(agent_type: str, memory_episodes: List[Dict], window_days: int, top_k: int) -> str:
        """
        Format memory_notes, providing different amounts and quality of information based on agent type:
        - A: Minimal information, only focus on recent large moves and emotional triggers (only show large moves, ignore moderate days)
        - B: Medium information, focus on technical patterns and price paths (top_k episodes, technical statistics)
        - C: More information, focus on emotional stories and PnL (top_k episodes, detailed emotional descriptions)
        - D: Maximum information, focus on prediction errors and learning evidence (top_k episodes, complete statistics)
        
        Args:
            agent_type: Agent type ('A', 'B', 'C', 'D')
            memory_episodes: Filtered memory episodes (provided by MemoryPolicy.select)
            window_days: Memory window days
            top_k: Maximum number of episodes to display (consistent with MemoryPolicy.select's top_k)
        """
        # Memory Construction and Use instructions
        memory_instructions = {
            'A': """Memory Construction and Use (M_t):
- Your memory M_t contains only the most recent few days (or the largest moves within a very short window).
- You almost never recall older, moderate days; you live "in the moment".
- Recent large gains can trigger overconfidence and aggressive risk-taking.
- Recent large losses can trigger sudden risk aversion or overreaction, even if fundamentals have not changed.""",
            'B': """Memory Construction and Use (M_t):
- Your memory M_t contains a small number of past days with the LARGEST absolute daily returns (|realized_return|) over a recent window.
- Treat these episodes as key technical events (major breakouts, crashes, spikes).
- Use them to recognise recurring technical patterns (e.g. how similar large moves behaved afterwards).
- You focus on price paths and regimes; you do not care about the fundamental stories behind them.""",
            'C': """Memory Construction and Use (M_t):
- Your memory M_t contains a small number of past days with the LARGEST absolute PnL (|pnl|) over a recent window.
- These are your most emotionally powerful experiences: very large gains or very large losses.
- You treat them as vivid stories: "the time I made a fortune on AI news", "the crash after ignoring a warning".
- When today's situation reminds you of such a story, you tend to over-weight that memory in your belief and decision.""",
            'D': """Memory Construction and Use (M_t):
- Your memory M_t contains a small number of past days with the LARGEST absolute forecast errors (|prediction_error|) over a recent window.
- Use these episodes as statistical evidence about when your previous views were wrong.
- When similar conditions appear again, you should recall these errors, adjust your confidence, and, if appropriate, be more cautious.
- You do NOT overreact to individual stories; you interpret memories as samples for learning about model performance."""
        }
        
        def safe_float(value, default=0.0):
            """Safely convert to float"""
            if value is None:
                return default
            try:
                if isinstance(value, str):
                    value = value.strip()
                    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                return float(value) if value else default
            except (ValueError, TypeError):
                return default
        
        parts = []
        if agent_type in memory_instructions:
            parts.append(memory_instructions[agent_type])
        
        if not memory_episodes:
            parts.append(f"\n[Past {window_days} Trading Days Summary]")
            parts.append("No past trading episodes available.")
            return "\n".join(parts)
        
        # Determine information detail level based on agent type
        if agent_type == 'A':
            # Type A: only focus on large moves, ignore moderate days (consistent with "almost never recall older, moderate days")
            # Only show large move episodes (|return|>5%)
            large_move_episodes = [
                ep for ep in memory_episodes 
                if abs(safe_float(ep.get('realized_return'))) > 0.05
            ]
            # If large move episodes are fewer than top_k, supplement with recent episodes (but prioritize large moves)
            if len(large_move_episodes) < top_k:
                # Supplement with recent episodes, but only take first top_k
                episodes_to_show = large_move_episodes + [
                    ep for ep in memory_episodes 
                    if ep not in large_move_episodes
                ][:top_k - len(large_move_episodes)]
            else:
                episodes_to_show = large_move_episodes[:top_k]
            
            parts.append(f"\n[Past {window_days} Trading Days Summary]")
            returns = [safe_float(ep.get('realized_return')) for ep in episodes_to_show]
            large_moves = sum(1 for r in returns if abs(r) > 0.05)
            parts.append(f"Recent large moves: {large_moves} (ignoring moderate days)")
            if episodes_to_show:
                parts.append("\n[Key Episodes - Large Moves Only]")
                for i, ep in enumerate(episodes_to_show, 1):
                    date = ep.get('date', 'N/A')
                    action = ep.get('action', 'N/A')
                    ret = safe_float(ep.get('realized_return'))
                    memory_note = ep.get('memory_update_note', '').strip()
                    if memory_note:
                        parts.append(f"{i}. {date}: {action}, Return: {ret:.4f} [LARGE MOVE] | Learned: {memory_note}")
                    else:
                        parts.append(f"{i}. {date}: {action}, Return: {ret:.4f} [LARGE MOVE]")
            else:
                parts.append("\n[Key Episodes]")
                parts.append("No large moves in recent memory.")
        
        elif agent_type == 'B':
            # Type B: focus on technical patterns and price paths (top_k episodes, technical statistics)
            episodes_to_show = memory_episodes[:top_k]
            parts.append(f"\n[Past {window_days} Trading Days Summary]")
            returns = [safe_float(ep.get('realized_return')) for ep in episodes_to_show]
            avg_abs_return = sum(abs(r) for r in returns) / len(returns) if returns else 0
            parts.append(f"Key technical events: {len(episodes_to_show)}, Avg |return|: {avg_abs_return:.4f}")
            parts.append("\n[Key Episodes - Technical Patterns]")
            for i, ep in enumerate(episodes_to_show, 1):
                date = ep.get('date', 'N/A')
                action = ep.get('action', 'N/A')
                ret = safe_float(ep.get('realized_return'))
                memory_note = ep.get('memory_update_note', '').strip()
                base_line = f"{i}. {date}: {action}, Return: {ret:.4f} (abs: {abs(ret):.4f})" + (" [TECHNICAL EVENT]" if abs(ret) > 0.05 else "")
                if memory_note:
                    parts.append(f"{base_line} | Pattern learned: {memory_note}")
                else:
                    parts.append(base_line)
        
        elif agent_type == 'C':
            # Type C: focus on emotional stories and PnL (top_k episodes, detailed emotional descriptions)
            episodes_to_show = memory_episodes[:top_k]
            parts.append(f"\n[Past {window_days} Trading Days Summary]")
            pnls = [safe_float(ep.get('pnl')) for ep in episodes_to_show]
            avg_pnl = sum(pnls) / len(pnls) if pnls else 0
            big_wins = sum(1 for p in pnls if p > 0.1)
            big_losses = sum(1 for p in pnls if p < -0.1)
            parts.append(f"Emotional episodes: {len(episodes_to_show)}, Avg PnL: {avg_pnl:.4f}, Big wins: {big_wins}, Big losses: {big_losses}")
            parts.append("\n[Key Episodes - Vivid Stories]")
            for i, ep in enumerate(episodes_to_show, 1):
                date = ep.get('date', 'N/A')
                action = ep.get('action', 'N/A')
                pnl = safe_float(ep.get('pnl'))
                ret = safe_float(ep.get('realized_return'))
                emotion = " [BIG WIN]" if pnl > 0.1 else (" [BIG LOSS]" if pnl < -0.1 else "")
                memory_note = ep.get('memory_update_note', '').strip()
                base_line = f"{i}. {date}: {action}, PnL: {pnl:.4f}, Return: {ret:.4f}{emotion}"
                if memory_note:
                    parts.append(f"{base_line} | Story: {memory_note}")
                else:
                    parts.append(base_line)
        
        elif agent_type == 'D':
            # Type D: focus on prediction errors and learning evidence (top_k episodes, complete statistics)
            # memory_episodes have already been filtered to top_k by MemoryPolicy.select, use directly here
            episodes_to_show = memory_episodes[:top_k]
            parts.append(f"\n[Past {window_days} Trading Days Summary]")
            returns = [safe_float(ep.get('realized_return')) for ep in episodes_to_show]
            pnls = [safe_float(ep.get('pnl')) for ep in episodes_to_show]
            errors = [safe_float(ep.get('prediction_error')) for ep in episodes_to_show if ep.get('prediction_error') is not None]
            avg_return = sum(returns) / len(returns) if returns else 0
            avg_pnl = sum(pnls) / len(pnls) if pnls else 0
            avg_error = sum(abs(e) for e in errors) / len(errors) if errors else 0
            max_error = max((abs(e) for e in errors), default=0) if errors else 0
            parts.append(f"Total episodes analyzed: {len(episodes_to_show)}")
            parts.append(f"- Avg return: {avg_return:.4f}, Avg PnL: {avg_pnl:.4f}")
            parts.append(f"- Avg |prediction_error|: {avg_error:.4f} (max: {max_error:.4f})")
            parts.append("\n[Key Episodes - Learning Evidence]")
            for i, ep in enumerate(episodes_to_show, 1):
                date = ep.get('date', 'N/A')
                action = ep.get('action', 'N/A')
                ret = safe_float(ep.get('realized_return'))
                pnl = safe_float(ep.get('pnl'))
                err = safe_float(ep.get('prediction_error'))
                memory_note = ep.get('memory_update_note', '').strip()
                base_line = f"{i}. {date}: {action}, Return: {ret:.4f}, PnL: {pnl:.4f}, Error: {err:.4f}"
                if memory_note:
                    parts.append(f"{base_line} | Lesson: {memory_note}")
                else:
                    parts.append(base_line)
        
        else:
            # Default: medium detail level
            episodes_to_show = memory_episodes[:top_k]
            parts.append(f"\n[Past {window_days} Trading Days Summary]")
            parts.append(f"Episodes: {len(episodes_to_show)}")
            parts.append("\n[Key Episodes]")
            for i, ep in enumerate(episodes_to_show, 1):
                date = ep.get('date', 'N/A')
                action = ep.get('action', 'N/A')
                ret = safe_float(ep.get('realized_return'))
                parts.append(f"{i}. {date}: {action}, Return: {ret:.4f}")
        
        return "\n".join(parts)