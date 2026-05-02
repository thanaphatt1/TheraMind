import os
import sys
import json
from datetime import datetime

# Import constants and utilities from thesis-framework
# We assume theramind and thesis-framework are siblings on the Desktop
THESIS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "thesis-framework"))
sys.path.insert(0, THESIS_DIR)

try:
    from eval.metrics import MetricsHandler
    from utils.io_utils import save_cost_report
except ImportError:
    # Fallback if the path is different
    MetricsHandler = None
    save_cost_report = None

class StandaloneCostTracker:
    """
    A lightweight cost tracker for scripts outside the thesis-framework.
    Logs usage into the central thesis-framework/total_costs.csv.
    """
    def __init__(self, model_condition="theramind_standalone"):
        self.model_condition = model_condition
        self.metrics = MetricsHandler() if MetricsHandler else None
        
    def record_call(self, agent_name, prompt, response_text, usage_metadata, model_name):
        if self.metrics:
            self.metrics.manual_record(
                agent_name=agent_name,
                prompt=prompt,
                response=response_text,
                usage_metadata=usage_metadata,
                model_name=model_name
            )
            
    def save_report(self, patient_id, session_num, total_turns):
        if not self.metrics or not save_cost_report:
            return
            
        trace_data = self.metrics.get_trace()
        total_cost = trace_data.get("total_cost", 0)
        total_input = sum(u.get("prompt_tokens", u.get("input_tokens", 0)) for u in trace_data.get("token_usage", []))
        total_output = sum(u.get("completion_tokens", u.get("output_tokens", 0)) for u in trace_data.get("token_usage", []))
        total_thoughts = sum(u.get("thoughts_token_count", u.get("thought_tokens", 0)) for u in trace_data.get("token_usage", []))
        
        report = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "model_condition": self.model_condition,
            "profile_id": patient_id,
            "session_num": session_num,
            "total_turns": total_turns,
            "total_cost": total_cost,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "thought_tokens": total_thoughts,
            "module_breakdown": self.metrics.get_module_costs()
        }
        
        # Save to central project root in thesis-framework
        try:
            save_cost_report(report, THESIS_DIR, filename="total_costs.csv")
            print(f"[CostTracker] Saved unified report to {THESIS_DIR}/total_costs.csv")
        except Exception as e:
            print(f"[CostTracker] Warning: Failed to save central report: {e}")
            
    def reset(self):
        if self.metrics:
            self.metrics.reset()
