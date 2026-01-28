import dspy
import json
from typing import List, Dict, Any, Optional
from trader.MemoryPolicy import MemoryPolicy
from config.ConfigLoader import AgentProfile, AgentsConfig
from prompts import profile_to_text, InvestmentDecisionPrompt, NoMemoryInvestmentDecisionPrompt
from model.Episode import Episode


def normalize_decision_field(value: Any, field_name: str) -> Any:
    """
    Normalize decision field to expected type for Episode model.
    This function ensures consistency before passing data to Episode.from_decision.
    Uses the same logic as Episode.from_decision.to_str for consistency.
    """
    if value is None:
        return None
    
    # Handle boolean values
    if isinstance(value, bool):
        return str(value).lower()
    
    # Handle list/tuple types (for selected_signals, selected_memories)
    if isinstance(value, (list, tuple)):
        if not value:
            return []  # Keep as empty list, will be converted to None by Episode.from_decision
        # Return as-is for now, Episode.from_decision will convert to comma-separated string
        return value
    
    # Handle numeric types (for belief_prior, belief_score, target_position)
    if isinstance(value, (int, float)):
        # Keep as numeric for now, Episode.from_decision will convert to string
        return value
    
    # Handle string types
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    
    # Handle other types - convert to string representation
    try:
        str_value = str(value)
        return str_value if str_value else None
    except Exception:
        return None


class TradingAgent:
    """
    Holds agent configuration and orchestrates memory selection + decision via policies.
    Unified decision interface uses batch_decide; dict→Episode construction centralized here.
    """

    def __init__(self, agent_id: str, agent_type: str, agents_cfg: AgentsConfig, debug_llm_io: bool = False):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.profile: AgentProfile = agents_cfg.profiles[agent_type]
        self.debug_llm_io = debug_llm_io

    def select_memory(self, episodes_store: List[Dict[str, Any]], window_date: str) -> List[Dict[str, Any]]:
        return MemoryPolicy.select(self.agent_id, self.profile, episodes_store, window_date)

    def batch_decide(self,
                     date: str,
                     assets: List[str],
                     merged_signals_by_asset: Dict[str, Dict[str, Any]],
                     memory_by_asset: Dict[str, List[Dict[str, Any]]],
                     num_threads: int,
                     should_print_input: bool = False,
                     memory_disabled: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Unified batch decision interface:
        - Builds dspy examples per asset (prompts.profile_to_text).
        - Executes batch prediction with concurrency.
        - Returns mapping asset -> {'decision': dict, 'episode': Episode}.
        - memory_by_asset: memory isolated by asset to avoid cross-contamination
        - should_print_input: whether to print LLM input (controlled by simulation.py, print only once per agent_type)
        """
        # Ablation 1: If memory is disabled, use no-memory prompt and exclude memory from profile
        agent_profile_text = profile_to_text(self.profile.model_dump(), memory_disabled=memory_disabled)

        examples: List[dspy.Example] = []
        asset_order: List[str] = []
        for asset in assets:
            if memory_disabled:
                # Ablation 1: No memory - use simplified prompt without memory fields
                ex = dspy.Example(
                    agent_profile=agent_profile_text,
                    data_inputs=merged_signals_by_asset[asset]
                ).with_inputs("agent_profile", "data_inputs")
            else:
                # Normal mode: include memory
                asset_memory = memory_by_asset.get(asset, [])
                # Format memory_notes: includes Memory Construction and Use instructions and trading performance summary
                formatted_memory = MemoryPolicy.format_memory_notes(
                    agent_type=self.agent_type,
                    memory_episodes=asset_memory,
                    window_days=self.profile.memory.window_days,
                    top_k=self.profile.memory.top_k
                )
            ex = dspy.Example(
                agent_profile=agent_profile_text,
                    memory_notes=formatted_memory,
                data_inputs=merged_signals_by_asset[asset]
            ).with_inputs("agent_profile", "memory_notes", "data_inputs")
            examples.append(ex)
            asset_order.append(asset)

        if self.debug_llm_io and should_print_input:
            print(f"\n==== LLM INPUT | agent_type={self.agent_type} | date={date} | memory_disabled={memory_disabled} ====")
            print(f"[agent_profile]")
            print(agent_profile_text)
            for asset, ex in zip(asset_order, examples):
                print(f"\n-- asset: {asset} --")
                print("[data_inputs]")
                try:
                    print(json.dumps(ex.data_inputs, ensure_ascii=False, indent=2))
                except Exception:
                    print("[data_inputs] (non-serializable)")
                    print(str(ex.data_inputs)[:2000])
                if not memory_disabled:
                    print("[memory_notes]")
                    # memory_notes is now string format, print directly
                    print(ex.memory_notes)

        # Use appropriate prompt based on memory_disabled flag
        if memory_disabled:
            predict = dspy.Predict(NoMemoryInvestmentDecisionPrompt)
        else:
            predict = dspy.Predict(InvestmentDecisionPrompt)
        try:
            predictions = predict.batch(examples, num_threads=num_threads)
        except Exception as e:
            # LLM call failed: return default decisions for all assets
            print(f"[ERROR] Agent {self.agent_id} batch_decide failed on {date}: {e}")
            predictions = []
            for _ in asset_order:
                # Create default prediction object (use normalize_decision_field to ensure type consistency)
                default_pred_dict = {
                    'selected_signals': normalize_decision_field([], 'selected_signals'),
                    'belief_prior': normalize_decision_field(0.0, 'belief_prior'),
                    'belief_score': normalize_decision_field(0.0, 'belief_score'),
                    'action': normalize_decision_field('HOLD', 'action'),
                    'target_position': normalize_decision_field(0.0, 'target_position'),
                    'reason': normalize_decision_field(f'LLM call failed, using default decision: {str(e)[:100]}', 'reason'),
                }
                if not memory_disabled:
                    default_pred_dict['selected_memories'] = normalize_decision_field([], 'selected_memories')
                    default_pred_dict['memory_update_note'] = normalize_decision_field('', 'memory_update_note')
                default_pred = type('DefaultPred', (), default_pred_dict)()
                predictions.append(default_pred)

        if self.debug_llm_io:
            print(f"\n==== LLM OUTPUT | agent={self.agent_id} | date={date} ====")
            for asset, pred in zip(asset_order, predictions):
                print(f"\n-- asset: {asset} --")
                try:
                    output_dict = {
                        "selected_signals": getattr(pred, "selected_signals", []),
                        "belief_prior": getattr(pred, "belief_prior", 0.0),
                        "belief_score": getattr(pred, "belief_score", 0.0),
                        "action": getattr(pred, "action", "HOLD"),
                        "target_position": getattr(pred, "target_position", 0.0),
                        "reason": getattr(pred, "reason", ""),
                    }
                    if not memory_disabled:
                        output_dict["selected_memories"] = getattr(pred, "selected_memories", [])
                        output_dict["memory_update_note"] = getattr(pred, "memory_update_note", "")
                    print(json.dumps(output_dict, ensure_ascii=False, indent=2))
                except Exception:
                    print(str(pred))

        # Reference dspy_sample.py: directly return dspy prediction object, flattened output
        results_by_asset: Dict[str, Any] = {}
        for asset, pred in zip(asset_order, predictions):
            try:
                # Episode.from_decision needs decision dict, but return pred object directly
                # Preprocess all fields to ensure type consistency (first layer of multi-layer protection)
                decision = {
                    'selected_signals': normalize_decision_field(
                        getattr(pred, 'selected_signals', []), 'selected_signals'
                    ),
                    'selected_memories': normalize_decision_field(
                        [] if memory_disabled else getattr(pred, 'selected_memories', []), 'selected_memories'
                    ),
                    'belief_prior': normalize_decision_field(
                        getattr(pred, 'belief_prior', 0.0), 'belief_prior'
                    ),
                    'belief_score': normalize_decision_field(
                        getattr(pred, 'belief_score', 0.0), 'belief_score'
                    ),
                    'action': normalize_decision_field(
                        getattr(pred, 'action', 'HOLD'), 'action'
                    ),
                    'target_position': normalize_decision_field(
                        getattr(pred, 'target_position', 0.0), 'target_position'
                    ),
                    'reason': normalize_decision_field(
                        getattr(pred, 'reason', ''), 'reason'
                    ),
                    'memory_update_note': normalize_decision_field(
                        '' if memory_disabled else getattr(pred, 'memory_update_note', ''), 'memory_update_note'
                    ),
            }
                # Episode.from_decision will perform second layer type conversion (second layer of multi-layer protection)
                episode = Episode.from_decision(decision, date=date, asset=asset, agent_id=self.agent_id)
                # Collect dynamic input (excluding static agent_profile, etc.)
                ex = examples[asset_order.index(asset)]
                dynamic_input = {
                    'data_inputs': ex.data_inputs      # Today's data input (signals)
                }
                if not memory_disabled:
                    dynamic_input['memory_notes'] = ex.memory_notes  # Formatted memory notes string
                # Directly return pred object and dynamic input, reference dspy_sample.py style
                results_by_asset[asset] = {'pred': pred, 'episode': episode, 'dynamic_input': dynamic_input}
            except Exception as e:
                # Single asset processing failed: return default decision
                print(f"[ERROR] Agent {self.agent_id} failed to process asset {asset} on {date}: {e}")
                # Use normalize_decision_field to ensure default decision type consistency
                default_decision = {
                    'selected_signals': normalize_decision_field([], 'selected_signals'),
                    'selected_memories': normalize_decision_field([], 'selected_memories'),
                    'belief_prior': normalize_decision_field(0.0, 'belief_prior'),
                    'belief_score': normalize_decision_field(0.0, 'belief_score'),
                    'action': normalize_decision_field('HOLD', 'action'),
                    'target_position': normalize_decision_field(0.0, 'target_position'),
                    'reason': normalize_decision_field(f'Processing failed, using default decision: {str(e)[:100]}', 'reason'),
                    'memory_update_note': normalize_decision_field('', 'memory_update_note'),
                }
                episode = Episode.from_decision(default_decision, date=date, asset=asset, agent_id=self.agent_id)
                # Create default pred object, matching batch_decide return format
                default_pred = type('DefaultPred', (), {
                    'selected_signals': [],
                    'selected_memories': [],
                    'belief_prior': 0.0,
                    'belief_score': 0.0,
                    'action': 'HOLD',
                    'target_position': 0.0,
                    'reason': default_decision['reason'],
                    'memory_update_note': '',
                })()
                results_by_asset[asset] = {'pred': default_pred, 'episode': episode}
        return results_by_asset