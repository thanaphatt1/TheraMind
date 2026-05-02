import sys
import argparse
from pathlib import Path
import os
from openai import OpenAI
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# Add current directory to sys.path for relative imports
sys.path.append(str(Path(__file__).parent))

# Subdirectory prefix for all four output directories.
# Set via --run-id before any StrictMemoryManager is instantiated for patient data.
_RUN_ID = ""

from memory import StrictMemoryManager
from initialization import TherapistInitializer
from evaluation import TherapistEvaluator
import time
from datetime import datetime
from collections import Counter
import random
import re
from metrics_utils import StandaloneCostTracker

config = StrictMemoryManager().get_config()
client = OpenAI(
    base_url=config["api_config"]["openai"]["base_url"],
    api_key=config["api_config"]["openai"]["api_key"],
)

# Therapist-specific client — defaults to the same DeepSeek backend.
# Overridden at startup when --therapist-model specifies a different LLM.
therapist_client = client
therapist_model = config["api_config"]["openai"]["model"]


class PatientAgent:
    def __init__(self, medical_record: Dict, patient_id: str, cost_tracker: StandaloneCostTracker = None, guidance_file: str = None, replay_attitudes: bool = False):
        self.medical_record = medical_record
        self.patient_id = patient_id
        self.cost_tracker = cost_tracker
        self.guidance_file = guidance_file  # None → default new_data_translate.json
        self.replay_attitudes = replay_attitudes
        self._label_data = self._load_label_data() if replay_attitudes else {}
        self.session_guides = self._load_session_guides()
        self.current_session_num = 1
        self.dialogue_count = 0
        self.memory_manager = StrictMemoryManager()
        self.last_attitude = None

    def _get_all_historical_dialogs(self) -> List[str]:
        all_dialogs = []
        memory_manager = StrictMemoryManager()
        full_record = memory_manager.get_full_record(self.patient_id)
        
        sessions_data = {}
        for session_key, session_data in full_record.get("sessions", {}).items():
            sessions_data[session_key] = {
                "dialogs": session_data.get("dialogs", [])
            }
        
        return sessions_data if sessions_data else {"session_1": {"dialogs": []}}
    
    def _load_label_data(self) -> Dict:
        label_dir = os.path.join("label_data", _RUN_ID) if _RUN_ID else "label_data"
        filepath = os.path.join(label_dir, f"label_{self.patient_id}.json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        print(f"[WARN] No label data found for {self.patient_id} — falling back to random sampling.")
        return {}

    def _load_session_guides(self) -> Dict[int, str]:
        if self.guidance_file:
            guidance_path = self.guidance_file
        else:
            guidance_path = Path(__file__).parent.parent / "new_data_translate.json"
        with open(guidance_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
            patient_data = all_data.get(self.patient_id, {})
            if not patient_data:
                return {}
                
            session_guides = patient_data.get("Conversation guidance", {})
            processed_guides = {}
                
            for k, v in session_guides.items():
                if k.startswith("session_"):
                    session_num = int(k.split("_")[1])
                    guide_content = v.split(":", 1)[1].strip() if ":" in v else v.strip()
                    processed_guides[session_num] = guide_content                
            return processed_guides
                
    
    def update_session(self, session_num: int):
        self.current_session_num = session_num
        self.dialogue_count = 0 

    def generate_response(self, therapist_message):
        self.dialogue_count += 1
        session_guide = self.session_guides.get(self.current_session_num, "")
        historical_dialogs = self._get_all_historical_dialogs()
        client_information = self.medical_record
        session_number = self.current_session_num
        if self.replay_attitudes:
            session_key = f"session_{self.current_session_num}"
            speak_key = f"speak_{self.dialogue_count}"
            session_data = self._label_data.get(session_key, {})
            stored_attitude = session_data.get(speak_key, {}).get("attitude")
            if stored_attitude:
                attitude = stored_attitude
            elif session_data:
                max_speak = max(int(k.split("_")[1]) for k in session_data.keys())
                attitude = session_data[f"speak_{max_speak}"]["attitude"]
                print(f"[WARN] {session_key}/{speak_key} beyond baseline ({max_speak} turns) — extending last attitude '{attitude}'.")
            else:
                print(f"[WARN] No label data for {session_key} at all — sampling randomly.")
                attitude = random.choices(["positive", "negative"], weights=[70, 30])[0]
        else:
            attitude = random.choices(["positive", "negative"], weights=[70, 30])[0]
        self.last_attitude = attitude
        dialogue_count = self.dialogue_count

        prompt = f"""
You are a patient participating in psychological counseling. Please give a response.
  - This is your personal information: {client_information}
  - This is round_{dialogue_count} in your session_{session_number} with the counselor.
  - The conselor just said: {therapist_message}
  - Previous consultation conversation history record: {historical_dialogs}
##Requirements:
1.Your speech should revolve around the guidance for this session: {session_guide}.
2.You must finish this session within 4 to 5 rounds strictly. Now is round_{dialogue_count} in this session. 
  - Please pay attention to the round number and reasonably adjust dialogue rhythm and content each time. 
  - In your last round of speaking, you must proactively, naturally, and very clearly convey the meaning of ending the conversation.
3.Your response should clearly reflect your attitude: {attitude}. You can refer to the following ways to explicitly convey your current attitude.
  - Positive reactions: such as expressing emotions, introspection, exploration, insight, self acceptance, positive action, etc
  - Negative reactions: such as denial, external blame, withdrawal, confused expression, resistance etc
4.On the basis of responding to doctors, you should also pay attention to your own consultation needs. Try to make a balance between them.
5.Your expression should be in line with the patient's speaking style, as colloquial and natural as possible. Limit your speech to 1 to 2 sentences each time, and be faithful to your personal information when speaking.
##Constraints:
Generate your response in English. Your response should be no more than 60 words. 
Do not include any additional text, explanations, word count or formatting outside the JSON object. 
Strictly return a JSON object, like this:
{{
    "patient_response": "your response here"
}}
"""
        import time
        
        gen_config = config.get("api_config", {}).get("profiles", {}).get("generation", {"temperature": 0.7, "top_p": 1.0})
        
        completion = client.chat.completions.create(
            model=config["api_config"]["openai"]["model"],
            messages=[
                {"role": "system", "content": "You are a patient receiving psychological counseling."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=gen_config.get("temperature", 0.7),
            top_p=gen_config.get("top_p", 1.0),
            max_tokens=300
        )

        raw_response = completion.choices[0].message.content
        cleaned_response = re.sub(r'^\s*```(json)?|```\s*$', '', raw_response, flags=re.IGNORECASE).strip()
        response = json.loads(cleaned_response)
        response["attitude"] = attitude  

        if self.cost_tracker and completion.usage:
            self.cost_tracker.record_call(
                agent_name="TheraMindPatient",
                prompt=prompt,
                response_text=response["patient_response"],
                usage_metadata=completion.usage.dict(),
                model_name=config["api_config"]["openai"]["model"]
            )

        self._save_label_data(
            session_num=self.current_session_num,
            speak_num=self.dialogue_count,
            attitude=attitude
        )
            
        return {
            "text": response["patient_response"],  
            "attitude": attitude  
        }  
            
    def _save_label_data(self, session_num: int, speak_num: int, attitude: str):
        label_dir = os.path.join("label_data", _RUN_ID) if _RUN_ID else "label_data"
        os.makedirs(label_dir, exist_ok=True)
        filename = f"label_{self.patient_id}.json"
        filepath = os.path.join(label_dir, filename)
        

        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}
              
        session_key = f"session_{session_num}"
        speak_key = f"speak_{speak_num}"
            
        if session_key not in data:
            data[session_key] = {}
        data[session_key][speak_key] = {"attitude": attitude}
            
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
                


class TherapistAgent:
    def __init__(self, cost_tracker: StandaloneCostTracker = None, records_file: str = None):
        self.memory_manager = StrictMemoryManager()
        self.cost_tracker = cost_tracker
        self.initializer = TherapistInitializer(self.memory_manager, records_file=records_file)
        self.evaluator = TherapistEvaluator(self.memory_manager, cost_tracker=cost_tracker)
        self.current_session_id = None
        self.current_patient_id = None
        self.dialog_count = 0  
        self.session_counter = 0
        self.current_therapy = "unspecified therapy"  

    def auto_conversation(self, patient_id: str, medical_record: Dict, patient_agent: PatientAgent, max_rounds: int = 20):
        session = self.start_new_session(patient_id)
        print(f"DOCTOR: {session['therapist_response']}")
        
        while True:  
            patient_response = patient_agent.generate_response(session['therapist_response'])
            print(f"PATIENT: {patient_response['text']}")
        
            session = self.process_patient_input(patient_response)
            print(f"DOCTOR: {session['therapist_response']}")
        
            if session['session_ended']:
                print("[SYSTEM] This conversation ended normally.")
                break
            
            if self.dialog_count >= max_rounds:  
                print("[SYSTEM] The maximum number of dialogue rounds is reached, the conversation will be forcibly ended.")
                self._end_current_session()
                break
            
            time.sleep(1)
        

    def _end_current_session(self):
        if not hasattr(self, 'current_session_id') or not self.current_session_id:
            return
        self.last_session_turns = self.dialog_count  # save before reset (Bug 2 fix)
        self.current_session_id = None
        self.dialog_count = 0

    def start_new_session(self, patient_id: str) -> Dict[str, Any]:
        self.current_patient_id = patient_id
        self.dialog_count = 0
        
        full_record = self.memory_manager.get_full_record(patient_id)
        if not full_record or not full_record.get("patient_record"):
            medical_info = self.initializer._get_initial_record(patient_id)

            self.memory_manager.create_patient_record(patient_id, medical_info)
            full_record = self.memory_manager.get_full_record(patient_id)  
        
        evaluation_result = self.evaluator.cross_session_evaluate(patient_id)
        self.current_therapy = evaluation_result.get("new_therapy", "") or self._determine_therapy_for_new_session(patient_id)
        
        session_num = self.memory_manager.add_session(patient_id, self.current_therapy)
        self.current_session_id = f"{patient_id}_{session_num}"

        if session_num == 1:
            greeting = "Hello, I'm your psychological counselor. Nice to meet you. Today we can talk about your recent situation."
        else:
            last_session = self.memory_manager.get_session(patient_id, session_num-1)
            last_therapy = "unspecified therapy"
            if last_session is not None:
                last_therapy = last_session.get("therapy", "unspecified therapy")
            greeting = f"Hello! Nice to see you again. How are you feeling today?"
        
        self.evaluator._save_therapy_reason(
            patient_id=patient_id,
            session_num=session_num,
            reason=evaluation_result.get("reason", "")
        )

        return {
            'patient_id': patient_id,
            'session_id': self.current_session_id,
            'therapist_response': greeting,
            'current_therapy': self.current_therapy
        }
    
    def _determine_therapy_for_new_session(self, patient_id: str) -> str:
        full_record = self.memory_manager.get_full_record(patient_id)
        if not full_record.get("sessions"):
            return self.evaluator.select_initial_therapy(full_record["patient_record"])
        evaluation = self.evaluator.cross_session_evaluate(patient_id)
    
        return evaluation.get("new_therapy", "")
    
    def process_patient_input(self, patient_response: Dict) -> Dict[str, Any]:
        patient_text = patient_response["text"]
        patient_attitude = patient_response["attitude"]

        if not hasattr(self, 'current_session_id') or not self.current_session_id:
            return {
                "therapist_response": "There is no active session.",
                "session_ended": False,
                "current_therapy": "unspecified therapy",
                "strategy": {"strategy": "", "strategy_text": ""}
            }
            
        session_num = int(self.current_session_id.split("_")[-1])
        self.evaluator._current_session_num = session_num 
        self.dialog_count += 1

        self.memory_manager.add_dialog(
            self.current_patient_id,
            session_num,
            "patient",
            patient_text
        )

        should_end = self.evaluator.should_end_session(patient_text, self.dialog_count)
        is_rejecting = self.evaluator.evaluate_client_reaction(patient_text)
        emotion_result = self.evaluator.assess_emotion(patient_text)
        if not isinstance(emotion_result, dict):
            emotion_result = {}
        emotion_data = {
            "primary_emotion": emotion_result.get("primary_emotion", ""),
            "emotional_intensity": float(emotion_result.get("emotional_intensity", 0.0))
        }       
        
        full_record = self.memory_manager.get_full_record(self.current_patient_id)
        memory_result = self.evaluator.should_use_memory(
            all_sessions_memory=full_record.get("sessions", {}),
            patient_input=patient_text
        )
        
        current_stage = self.memory_manager.get_current_stage(self.current_patient_id)
    
        strategy_result = self.evaluator.update_response_strategy(
            emotion_data = emotion_data, 
            is_rejecting = is_rejecting,
            patient_input=patient_text,
            patient_id=self.current_patient_id
        )
        strategy = {
            "strategy": strategy_result.get("strategy", ""),
            "strategy_text": strategy_result.get("strategy_text", "")
        }

        # --- Module Output Tracking (Console & File) ---
        log_dir = os.path.join("module_logs", _RUN_ID) if _RUN_ID else "module_logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"trace_{self.current_patient_id}.txt")
        
        log_entry = (
            f"\n[MODULE TRACKING] Session {session_num} | Round {self.dialog_count}\n"
            f"  > [EMOTION]: {emotion_data['primary_emotion']} (Intensity: {emotion_data['emotional_intensity']})\n"
            f"  > [REJECTION]: {'YES' if is_rejecting else 'NO'}\n"
            f"  > [MEMORY]: {memory_result[:150]}...\n" 
            f"  > [STAGE]: {current_stage}\n"
            f"  > [STRATEGY]: {strategy['strategy']}\n"
            f"  > [STRATEGY DETAIL]: {strategy['strategy_text']}\n"
            f"  > [THERAPY]: {self.current_therapy}\n"
            f"  > [END SESSION?]: {'YES' if should_end else 'NO'}\n"
            f"{'-' * 40}\n"
        )
        
        # Print to console
        print(log_entry)
        
        # Save to file
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)

        response = self._generate_response(
            patient_input=patient_text,
            emotion_data=emotion_data,
            current_therapy=self.current_therapy
        )
        
        self.memory_manager.add_dialog(
            self.current_patient_id,
            session_num,
            "doctor",
            response
        )
        
        decision_data = {
            "Memory Invoke": memory_result,
            "Whether to Reject or Deviate": is_rejecting,
            "Current Therapy": self.current_therapy,
            "Current Stage": current_stage,
            "Primary Emotion": emotion_result.get("primary_emotion", ""),
            "Emotional Intensity": emotion_result.get("emotional_intensity", 0.0),
            "Response Strategy": strategy_result.get("strategy", ""),
            "Strategy Description": strategy_result.get("strategy_text", ""),
            "Attitude": patient_attitude
        }  
        self.memory_manager.save_decision_data(
            patient_id=self.current_patient_id,
            session_num=session_num,
            response_num=self.dialog_count,
            decision_data=decision_data
        )

        session_ended = should_end 
        if session_ended:
            self._end_current_session()
        
        return {
            "therapist_response": response,
            "session_ended": session_ended,
            "current_therapy": self.current_therapy,
            "strategy": strategy
        }
    
    def _get_current_therapy(self) -> str:
        if not hasattr(self, 'current_session_id') or not self.current_session_id:
            return "unspecified therapy"
        
        session_num = int(self.current_session_id.split("_")[-1])
        session = self.memory_manager.get_session(self.current_patient_id, session_num)
        return session.get("therapy", "unspecified therapy")
    

    def _generate_response(self, 
                     patient_input: str,   
                     emotion_data: Dict[str, Any],
                     current_therapy: str) -> str:
        config = self.memory_manager.get_config()
                
        session_memory = {}
        if self.current_session_id:
            try:
                session_num = int(self.current_session_id.split("_")[-1])
                full_record = self.memory_manager.get_full_record(self.current_patient_id)
                session_key = f"session_{session_num}"
                if session_key in full_record.get("sessions", {}):
                    session_memory = {
                        "dialogs": full_record["sessions"][session_key].get("dialogs", [])
                    }
            except Exception as e:
                session_memory = {"dialogs": []}

        current_stage = self.memory_manager.get_current_stage(self.current_patient_id)
        full_record = self.memory_manager.get_full_record(self.current_patient_id)
        memory_result = self.evaluator.should_use_memory(
            all_sessions_memory=full_record.get("sessions", {}),
            patient_input=patient_input
        )
        
        primary_emotion = emotion_data.get("primary_emotion", "unknown")
        emotional_intensity = emotion_data.get("emotional_intensity", 0.0)
        
        is_rejecting = self.evaluator.evaluate_client_reaction(patient_input)
        strategy_result = self.evaluator.update_response_strategy(
            emotion_data = emotion_data, 
            is_rejecting = is_rejecting,
            patient_input=patient_input,
            patient_id=self.current_patient_id
        )
        
        current_strategy = strategy_result.get('strategy', '')
        current_strategy_text = strategy_result.get('strategy_text', '')

        prompt = f"""
##Role:
You are a professional and empathetic psychological counselor. 
Your job is to respond to the patient compassionately and offer support in psychological counseling based on the following information and requirements.
##Requirements:
  1.Patient current words: {patient_input}.
  2.Historical memories you may need to be referred to: {memory_result}.
  3.The patient current primary emotion is {primary_emotion}, with an intensity of {emotional_intensity}.
  4.The therapy you should adopt is {current_therapy}. And please refer to analysis of the current treatment stage {current_stage}. 
  5.The response strategy you should adopt is {current_strategy} and the detailed guidance is {current_strategy_text}.
  6.Your expression should be in line with the psychological counselor's speaking style, as colloquial and natural as possible.
  7.Don't always directly repeat or quote what the patient has said. Just empathize the patient with as little words as possible. Ensure the smooth of the conversation.
  8.You must use diverse and different sentence patterns to reply each time to avoid a single reply mode. To avoid using the same sentence pattern, please refer to your previous replies from the conversation records for this session:{session_memory}.
  9.When the patient expresses a clear desire to end this conversation, please also provide a response to end the conversation in a declarative tone.
##Constraints:
Directly generate your response in English. Your response should be no more than 60 words. Do not provide any word count, analysis or explanation. 
"""

        gen_config = config.get("api_config", {}).get("profiles", {}).get("generation", {"temperature": 0.7, "top_p": 1.0})
        
        try:
            completion = therapist_client.chat.completions.create(
                model=therapist_model,
                messages=[
                    {"role": "system", "content": "You are a experienced and empathetic psychological counselor. "},
                    {"role": "user", "content": prompt}
                ],
                temperature=gen_config.get("temperature", 0.7),
                top_p=gen_config.get("top_p", 1.0),
                max_tokens=4096
            )
            raw_response = completion.choices[0].message.content
            clean_response = raw_response.replace('\\"', '"')
            clean_response = clean_response.strip('"')

            if self.cost_tracker and completion.usage:
                self.cost_tracker.record_call(
                    agent_name="TheraMindTherapist",
                    prompt=prompt,
                    response_text=clean_response,
                    usage_metadata=completion.usage.dict(),
                    model_name=therapist_model
                )
        
            return clean_response
        except Exception as e:
            return "Sorry, I'm temporarily unable to process your request, please try again."
        

class AutoDialogueRunner:
    def __init__(self, guidance_file: str = None, records_file: str = None, fix_seed: bool = False, replay_attitudes: bool = False):
        self.guidance_file = guidance_file
        self.records_file = records_file
        self.fix_seed = fix_seed
        self.replay_attitudes = replay_attitudes
        self.tracker = StandaloneCostTracker(model_condition="theramind_standalone")
        self.agent = TherapistAgent(cost_tracker=self.tracker, records_file=records_file)
        self.patient_records = self._load_all_patient_records()

    def _load_all_patient_records(self) -> Dict[str, Dict]:
        initializer = TherapistInitializer(StrictMemoryManager(), records_file=self.records_file)
        all_records = initializer._load_client_record()
        return {
            pid: initializer._get_initial_record(pid)
            for pid in all_records.keys()
        }
    
    def run(self, num_sessions=6, max_rounds_per_session=8, patient_ids: List[str] = None):
        print("\n" + "="*60)
        print("Psychological counseling conversation simulation system".center(40))
        print("="*60 + "\n")
        
        target_patients = patient_ids if patient_ids else self.patient_records.keys()
        
        for patient_id in target_patients:
            if patient_id not in self.patient_records:
                print(f"[WARNING] patient_id {patient_id} not found in records.")
                continue

            if self.fix_seed:
                random.seed(patient_id)

            medical_record = self.patient_records[patient_id]
            print(f"\n[SYSTEM] Start the consultation with {patient_id}.")

            patient_agent = PatientAgent(
                medical_record=medical_record,
                patient_id=patient_id,
                cost_tracker=self.tracker,
                guidance_file=self.guidance_file,
                replay_attitudes=self.replay_attitudes
            )
            
            for session_num in range(1, num_sessions+1):
                # Resume protection: skip sessions that already have dialogs (Bug 5 fix)
                full_record = self.agent.memory_manager.get_full_record(patient_id)
                existing_dialogs = (full_record.get("sessions", {})
                                    .get(f"session_{session_num}", {})
                                    .get("dialogs", []))
                if existing_dialogs:
                    print(f"[SKIP] {patient_id} session {session_num} already completed ({len(existing_dialogs)} turns). Skipping.")
                    patient_agent.update_session(session_num)
                    continue

                print(f"\n[SYSTEM] Session {session_num} begins.")
                patient_agent.update_session(session_num)

                self.agent.auto_conversation(
                    patient_id=patient_id,
                    medical_record=medical_record,
                    patient_agent=patient_agent,
                    max_rounds=max_rounds_per_session
                )
                print(f"[SYSTEM] Session {session_num} ends.")

                # Save cost report using last_session_turns (Bug 2 fix)
                final_turns = getattr(self.agent, 'last_session_turns', 0)
                self.tracker.save_report(patient_id, session_num, final_turns)
                self.tracker.reset()
                
                time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TheraMind dialogue simulation.")
    parser.add_argument("--sessions",         type=int,   default=6,    help="Number of sessions per patient (default: 6)")
    parser.add_argument("--rounds",           type=int,   default=8,    help="Max rounds per session (default: 8)")
    parser.add_argument("--patients",         nargs="+",  default=None, help="Patient IDs to run (default: all)")
    parser.add_argument("--guidance",         default=None,             help="Path to guidance JSON (default: new_data_translate.json)")
    parser.add_argument("--records",          default=None,             help="Path to client records JSON (default: client_records_translate.json)")
    parser.add_argument("--fix-seed",         action="store_true",      help="Seed RNG with patient_id before each patient for reproducible attitude sampling")
    parser.add_argument("--replay-attitudes", action="store_true",      help="Replay attitudes from label_data/ instead of sampling (use after baseline run)")
    parser.add_argument("--workers",          type=int,   default=1,    help="Number of parallel patient workers (default: 1 = sequential)")
    parser.add_argument("--therapist-model",  default=None,             help="Override therapist LLM (e.g. gemini-2.0-flash). Patient stays on DeepSeek.")
    parser.add_argument("--run-id",           default="",               help="Subdirectory prefix for all output dirs (save_data, eval_data, label_data, module_logs). Prevents overwriting existing runs.")
    args = parser.parse_args()

    if args.run_id:
        import memory as _memory_mod
        _RUN_ID = args.run_id
        _memory_mod._RUN_ID = args.run_id

    if args.therapist_model:
        import os as _os
        _model = args.therapist_model
        if "gemini" in _model:
            _api_key = _os.environ.get("GEMINI_API_KEY") or _os.environ.get("GOOGLE_API_KEY", "")
            _base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        else:
            _api_key = config["api_config"]["openai"]["api_key"]
            _base_url = config["api_config"]["openai"]["base_url"]
        therapist_client = OpenAI(base_url=_base_url, api_key=_api_key)
        therapist_model = _model
        print(f"[Therapist] Backend overridden -> model={therapist_model}, base_url={_base_url}")

    def _make_runner():
        return AutoDialogueRunner(
            guidance_file=args.guidance,
            records_file=args.records,
            fix_seed=args.fix_seed,
            replay_attitudes=args.replay_attitudes,
        )

    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        # Resolve full patient list once before spawning threads
        bootstrap = _make_runner()
        all_patient_ids = args.patients if args.patients else list(bootstrap.patient_records.keys())
        print(f"[SYSTEM] Running {len(all_patient_ids)} patients with {args.workers} parallel workers.")

        def _run_one(pid):
            runner = _make_runner()
            runner.run(
                num_sessions=args.sessions,
                max_rounds_per_session=args.rounds,
                patient_ids=[pid],
            )

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_run_one, pid): pid for pid in all_patient_ids}
            for future in as_completed(futures):
                pid = futures[future]
                exc = future.exception()
                if exc:
                    print(f"[ERROR] {pid} raised an exception: {exc}")
                else:
                    print(f"[DONE] {pid} completed.")
    else:
        simulator = _make_runner()
        simulator.run(num_sessions=args.sessions, max_rounds_per_session=args.rounds, patient_ids=args.patients)