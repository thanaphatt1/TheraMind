import sys
import os
from pathlib import Path
from multiprocessing import Process
from typing import List

# Add agent directory to sys.path to enable imports from agent folder
current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, 'agent'))

# Ensure we can import from agent.main
try:
    from main import AutoDialogueRunner
except ImportError:
    # Alternative import path
    sys.path.append(current_dir)
    from agent.main import AutoDialogueRunner

def run_simulation_subset(process_id: int, patient_ids: List[str], num_sessions: int, max_rounds: int):
    """Function to be run in a separate process."""
    print(f"\n[PROCESS {process_id}] Starting simulation for: {patient_ids}")
    runner = AutoDialogueRunner()
    runner.run(num_sessions=num_sessions, max_rounds_per_session=max_rounds, patient_ids=patient_ids)
    print(f"\n[PROCESS {process_id}] Completed tasks.")

def main():
    # 1. Load all patients to determine the workload
    print("[SYSTEM] Initializing Parallel Runner...")
    temp_runner = AutoDialogueRunner()
    all_patients = sorted(list(temp_runner.patient_records.keys()))
    
    if not all_patients:
        print("[ERROR] No patients found in client_records_translate.json.")
        return

    print(f"[SYSTEM] Found {len(all_patients)} patients in total.")
    
    # 2. Split patients into 2 concurrent chunks
    num_processes = 2
    chunk_size = (len(all_patients) + num_processes - 1) // num_processes
    chunks = [all_patients[i:i + chunk_size] for i in range(0, len(all_patients), chunk_size)]
    
    # 3. Simulation Parameters
    num_sessions = 6
    max_rounds = 8
    
    print(f"[SYSTEM] Launching {len(chunks)} concurrent processes...")
    processes = []
    for i, chunk in enumerate(chunks):
        # We pass process_id (i+1) for clearer logging
        p = Process(target=run_simulation_subset, args=(i+1, chunk, num_sessions, max_rounds))
        p.start()
        processes.append(p)
        print(f"[SYSTEM] Process {i+1} assigned patients: {', '.join(chunk)}")

    # 4. Wait for all processes to finish
    print("[SYSTEM] Parallel execution in progress. Please wait for completion.")
    for p in processes:
        p.join()
    
    print("\n" + "="*60)
    print("[SYSTEM] SUCCESS: All parallel simulation processes have completed.")
    print("="*60 + "\n")

if __name__ == "__main__":
    # On Windows, multiprocessing requires the if __name__ == "__main__": guard
    main()
