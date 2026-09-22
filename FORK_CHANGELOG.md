# Fork Changelog & Upstream Differences

This repository is a maintained fork of [Emo-gml/TheraMind](https://github.com/Emo-gml/TheraMind) (accompanying the WWW 2026 paper *"TheraMind: A Strategic and Adaptive Agent for Longitudinal Psychological Counseling"*).

This document outlines the bug fixes, architectural enhancements, and compatibility adjustments implemented in this fork.

---

## 1. Strategy Memory Retrieval & Diversity Prompt Bug Fix

### Location
`agent/tm_evaluation.py` (`TherapistEvaluator._get_session_strategy_memory`)

### Upstream Issue
1. **Hardcoded Relative Path**: Upstream hardcoded `filepath = f"eval_data/decision basis_{patient_id}.json"`. When running simulations from any directory other than the project root or when using custom evaluation output directories, the file was not found, causing strategy memory to silently fail.
2. **Off-by-One Session Indexing**: Upstream computed the current session index as:
   ```python
   current_session_num = len(full_record.get("sessions", {}))
   ```
   Because session records are stored with keys formatted as `"session_1"`, `"session_2"`, etc., during Session 1 the dictionary is empty before session completion (`len == 0`), causing the code to search for `"session_0"`. 
   Consequently, **100% of strategy selection queries across all turns returned `"No strategy memory available"`**, preventing the model's strategy diversity prompt from ever observing or avoiding repetitive therapeutic techniques within a session.

### Fix Applied
- File paths are now resolved dynamically via `self.memory_manager.eval_dir / f"decision basis_{patient_id}.json"`.
- The session lookup correctly indexes the active session (`len(sessions) + 1` or dynamic match against the current session).
- Turn 1 correctly returns `"No strategy memory available"`, while Turns 2+ retrieve all previously deployed strategies in the active session (`"Strategies used in this session: ..."`), restoring the intended dynamic strategy adaptation.

---

## 2. Namespace Collision Prevention

### Location
- Renamed: `agent/evaluation.py` $\rightarrow$ `agent/tm_evaluation.py`
- Updated imports in: `agent/main.py`, `agent/memory.py`

### Upstream Issue
When integrating or benchmarking TheraMind inside broader multi-agent research frameworks (e.g., benchmark test runners with their own `evaluation/` packages), Python's module resolution would collide with `theramind/agent/evaluation.py`.

### Fix Applied
Renamed the module to `tm_evaluation.py` and updated all internal references (`from tm_evaluation import TherapistEvaluator`), ensuring clean and isolated imports across all evaluation runners.

---

## 3. Robust LLM Response & JSON Parsing

### Location
`agent/main.py` (`TherapistAgent.process_patient_input`, `_safe_float`, and extraction utilities)

### Upstream Issue
Upstream relied strictly on standard `json.loads` within Markdown code blocks. When evaluated against diverse LLM backends (e.g., Google Gemini, Anthropic Claude, or local quantized models via Ollama/vLLM), models frequently emitted single-quoted dictionaries or malformed floats for intensity ratings, resulting in fatal runtime exceptions.

### Fix Applied
- **`_safe_float` Parsing**: Implemented robust numeric extraction for patient emotion and resistance intensities, safely handling edge cases like string prefixes or ranges.
- **AST Literal Fallback**: Added fallback parsing via Python's `ast.literal_eval` for valid dictionary outputs formatted with single quotes.
- **Natural Termination Handling**: Added handling for session termination tokens (such as `[/END]`) to allow clean multi-session transitions without infinite dialogue loops.

---

## 4. Elimination of Redundant Recomputations

### Location
`agent/main.py` (`TherapistAgent.process_patient_input`, `TherapistAgent._generate_response`)

### Upstream Issue
`_generate_response()` previously re-evaluated perception and stage components even after `process_patient_input()` had already computed them for the current turn, doubling prompt tokens and occasionally introducing state drift between perception and response generation.

### Fix Applied
Passed precomputed state (`memory_result`, `is_rejecting`, `strategy_result`, `current_stage`) directly from `process_patient_input` into `_generate_response`, halving redundant token costs and ensuring strict perceptual alignment.

---

## Summary of Modified Files

| File | Status | Description |
| :--- | :--- | :--- |
| `agent/tm_evaluation.py` | Renamed / Modified | Renamed from `evaluation.py`; fixed `_get_session_strategy_memory` path and session indexing. |
| `agent/main.py` | Modified | Updated imports to `tm_evaluation`; added `_safe_float`, AST fallback, token handling, and reused precomputed turn state. |
| `agent/memory.py` | Modified | Updated imports from `evaluation` to `tm_evaluation`. |
| `FORK_CHANGELOG.md` | Added | Detailed log of differences and bug fixes against upstream. |
| `README.md` | Modified | Added fork notice and link to `FORK_CHANGELOG.md`. |
