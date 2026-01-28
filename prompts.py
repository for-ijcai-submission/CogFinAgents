import dspy
from typing import Dict, List

def profile_to_text(profile: Dict, memory_disabled: bool = False) -> str:
    """
    Serialize a structured AgentProfile (dict-like) into text for LLM.
    Includes core profile_text (behavioral rules) plus structured config.
    Expected keys: name, info_access, memory:{window_days, sort_key, top_k}, profile_text.
    
    Args:
        profile: Agent profile dictionary
        memory_disabled: If True, exclude memory configuration from the profile text
    """
    name = profile.get('name', '')
    info = profile.get('info_access', '')
    mem = profile.get('memory', {})
    profile_text = profile.get('profile_text', '').strip()
    
    # Core behavioral description (profile_text) is primary
    parts = []
    if profile_text:
        parts.append(profile_text)
    
    # Append structured config as metadata
    if memory_disabled:
        # Ablation 1: Exclude memory configuration
        config_summary = f"\n[Config] Access={info}"
    else:
        config_summary = (
            f"\n[Config] Access={info}; "
            f"Memory(window_days={mem.get('window_days')}, sort_key={mem.get('sort_key')}, top_k={mem.get('top_k')})"
        )
    parts.append(config_summary)
    
    return "\n".join(parts) if parts else f"Agent={name}"


class InvestmentDecisionPrompt(dspy.Signature):
    """
    You are an investment agent in a trading market.

    INPUTS:
    - agent_profile: Agent-specific cognitive rules (info access, attention, belief).
    - memory_notes: Your past trading performance summary over the recent window, including Memory Construction and Use rules, performance statistics, and key episodes. Review this to understand your recent patterns and what you should focus on.
    - data_inputs: Today's visible signals for your info access.

    TASK:
    1) Review memory_notes to understand your recent trading performance patterns and what matters to you (as defined by your Memory Construction and Use rules).
    2) Select attention (A_t) from I_t (respect capacity/rules). Each selected signal must be prefixed with its data type (e.g., 'macro_data.vix', 'market_data.close', 'news_data.sentiment', 'social_data.content', 'company_data.fiscalDateEnding').
    3) Update belief: start at prior, compute belief_score in [-1,1] using only selected_signals (+ explicit memories if used).
    4) Decide action LONG/SHORT/FLAT/HOLD and target_position in [-1,1] (sign=direction, magnitude=conviction).
    5) Provide brief reason and memory_update_note: a summary and reflection on your past trading performance (as shown in memory_notes), following your Memory Construction and Use rules. This note will be included when this episode is recalled in future memory_notes.

    """
    agent_profile: str = dspy.InputField(desc="")
    memory_notes: str = dspy.InputField(desc="Your past trading performance summary over the recent window, including Memory Construction and Use rules, performance statistics, and key episodes.")
    data_inputs: dict = dspy.InputField(desc="Today's merged signals (I_t), pre-filtered for access level.")

    selected_signals: list = dspy.OutputField(desc="Identifiers of signals from I_t that were attended to. Each signal must be prefixed with its data type  (e.g., 'macro_data.vix', 'market_data.close', 'news_data.sentiment', 'social_data.content', 'company_data.fiscalDateEnding'). Format: 'data_type.signal_name' or 'data_type.field_name'.")
    selected_memories: list = dspy.OutputField(desc="Short descriptions of referenced memory.")
    belief_prior: str = dspy.OutputField(desc="Belief before today's signals, in [-1,1].")
    belief_score: str = dspy.OutputField(desc="Updated belief after processing, in [-1,1].")
    action: str = dspy.OutputField(
        desc="""Trading action, must match target_position:
        - LONG: target_position > 0 (go long or increase long)
        - SHORT: target_position < 0 (go short or increase short)  
        - FLAT: target_position = 0 (close all positions, return to zero)
        - HOLD: target_position = current_position (maintain last position, no change)
        """
    )

    target_position: str = dspy.OutputField(
        desc="""Desired position in [-1, 1], where:
        Sign = direction (+ long, - short)
        Magnitude = conviction (0.0-0.3 low, 0.3-0.7 medium, 0.7-1.0 high)
        Examples: -1.0 (max short), -0.5 (moderate short), 0.0 (flat), +0.5 (moderate long), +1.0 (max long)
        """
    )
    reason: str = dspy.OutputField(desc="Concise reasoning (3–6 sentences) grounded in signals/memories.")
    memory_update_note: str = dspy.OutputField(desc="1–3 sentences: summary and reflection on your past trading performance (shown in memory_notes), following your Memory Construction and Use rules. This note will be included when this episode is recalled in future memory_notes.")


class NoMemoryInvestmentDecisionPrompt(dspy.Signature):
    """
    You are an investment agent in a trading market. You have NO memory of past trading decisions.

    INPUTS:
    - agent_profile: Agent-specific cognitive rules (info access, attention, belief).
    - data_inputs: Today's visible signals for your info access.

    TASK:
    1) Select attention (A_t) from I_t (respect capacity/rules). Each selected signal must be prefixed with its data type (e.g., 'macro_data.vix', 'market_data.close', 'news_data.sentiment', 'social_data.content', 'company_data.fiscalDateEnding').
    2) Update belief: start at prior, compute belief_score in [-1,1] using only selected_signals.
    3) Decide action LONG/SHORT/FLAT/HOLD and target_position in [-1,1] (sign=direction, magnitude=conviction).
    4) Provide brief reason for your decision.

    """
    agent_profile: str = dspy.InputField(desc="Concise agent profile text.")
    data_inputs: dict = dspy.InputField(desc="Today's merged signals (I_t), pre-filtered for access level.")

    selected_signals: list = dspy.OutputField(desc="Identifiers of signals from I_t that were attended to. Each signal must be prefixed with its data type (e.g., 'macro_data.vix', 'market_data.close', 'news_data.sentiment', 'social_data.content', 'company_data.fiscalDateEnding'). Format: 'data_type.signal_name' or 'data_type.field_name'.")
    belief_prior: str = dspy.OutputField(desc="Belief before today's signals, in [-1,1].")
    belief_score: str = dspy.OutputField(desc="Updated belief after processing, in [-1,1].")
    action: str = dspy.OutputField(
        desc="""Trading action, must match target_position:
        - LONG: target_position > 0 (go long or increase long)
        - SHORT: target_position < 0 (go short or increase short)  
        - FLAT: target_position = 0 (close all positions, return to zero)
        - HOLD: target_position = current_position (maintain last position, no change)
        """
    )

    target_position: str = dspy.OutputField(
        desc="""Desired position in [-1, 1], where:
        Sign = direction (+ long, - short)
        Magnitude = conviction (0.0-0.3 low, 0.3-0.7 medium, 0.7-1.0 high)
        Examples: -1.0 (max short), -0.5 (moderate short), 0.0 (flat), +0.5 (moderate long), +1.0 (max long)
        """
    )
    reason: str = dspy.OutputField(desc="Concise reasoning (3–6 sentences) grounded in signals.")