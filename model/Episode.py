from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Episode(BaseModel):
    # [Identification]
    date: str                                  # "YYYY-MM-DD", decision date t
    asset: str                                 # asset code, e.g. "NVDA", "BTC-USD"
    agent_id: str                              # agent unique ID ("A"|"B"|"C"|"D" or extended)

    # [Ex-ante belief and decision  (from LLM output + code-checked)]
    # Here we allow runtime storage as string format to be compatible with LLM direct output like "0.55"
    belief_prior: Optional[str] = Field(default=None)              # before today's signals, as string in [-1,1]
    belief_score: Optional[str] = Field(default=None)              # after today's signals, as string in [-1,1]
    action: Optional[str] = Field(default=None)                    # "BUY"|"SELL"|"HOLD"
    target_position: Optional[str] = Field(default=None)           # in [-1,1], sign: long/short, magnitude=conviction, stored as string
    selected_signals: Optional[str] = Field(default=None)      # IDs/names of I_t signals used (comma-separated string)
    selected_memories: Optional[str] = Field(default=None)     # IDs/descriptions of M_t memory used (comma-separated string)
    reason: Optional[str] = Field(default=None)                    # concise reasoning
    memory_update_note: Optional[str] = Field(default=None)        # 1–3 sentences, new learning

    # [Ex-post prices and performance  (computed by code, not seen by the agent when deciding)]
    price_open: Optional[float] = Field(default=None)              # market price at decision/reference (e.g. open/close)
    price_next: Optional[float] = Field(default=None)              # settlement price (e.g. close of t+1)
    realized_return: Optional[float] = Field(default=None)         # realized return (price_next / price_open - 1)
    pnl: Optional[float] = Field(default=None)                     # profit and loss for this agent-asset
    prediction_error: Optional[float] = Field(default=None)        # realized_return minus belief-implied

    # [Positions and state  (computed by code)]
    position_before: Optional[float] = Field(default=None)         # position held just before today
    position_after: Optional[float] = Field(default=None)          # position after executing decision

    class Config:
        extra = "allow"

    @classmethod
    def from_decision(cls, decision: Dict[str, Any], *, date: str, asset: str, agent_id: str) -> "Episode":
        """
        Construct an Episode from a decision dict (ex-ante fields) + context,
        with pydantic runtime validation.
        Converts types: lists -> comma-separated strings, numbers -> strings.
        Handles all possible LLM output types robustly.
        """
        def to_str(value: Any) -> Optional[str]:
            """
            Convert value to string, handling all possible types.
            Robust conversion for LLM outputs that may vary in format.
            """
            if value is None:
                return None
            
            # Handle boolean values
            if isinstance(value, bool):
                return str(value).lower()
            
            # Handle list/tuple types
            if isinstance(value, (list, tuple)):
                if not value:
                    return None
                # Handle nested lists/tuples
                flattened = []
                for item in value:
                    if isinstance(item, (list, tuple)):
                        flattened.extend(str(v) for v in item if v is not None)
                    else:
                        if item is not None:
                            flattened.append(str(item))
                return ', '.join(flattened) if flattened else None
            
            # Handle numeric types
            if isinstance(value, (int, float)):
                return str(value)
            
            # Handle string types
            if isinstance(value, str):
                stripped = value.strip()
                return stripped if stripped else None
            
            # Handle other types (dict, set, etc.) - convert to string representation
            try:
                str_value = str(value)
                return str_value if str_value else None
            except Exception:
                return None
        
        payload = dict(
            date=date,
            asset=asset,
            agent_id=agent_id,
            belief_prior=to_str(decision.get("belief_prior")),
            belief_score=to_str(decision.get("belief_score")),
            action=decision.get("action"),
            target_position=to_str(decision.get("target_position")),
            selected_signals=to_str(decision.get("selected_signals")),
            selected_memories=to_str(decision.get("selected_memories")),
            reason=decision.get("reason"),
            memory_update_note=decision.get("memory_update_note"),
        )
        return cls(**payload)