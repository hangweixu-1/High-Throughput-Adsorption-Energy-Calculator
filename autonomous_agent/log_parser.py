import re

_PATTERNS = [
  ("oom", re.compile(r"(CUDA out of memory|out of memory|OOM|std::bad_alloc)", re.I)),
  ("timeout", re.compile(r"(TIMEOUT|timed out|Time limit)", re.I)),
  ("scf", re.compile(r"(SCF.*not.*conver|did not converge)", re.I)),
  ("nan", re.compile(r"(nan|NaN|floating point exception)", re.I)),
  ("missing_module", re.compile(r"(ModuleNotFoundError: No module named)", re.I)),
  ("gpaw_setup", re.compile(r"(GPAW.*setups|PAW setups|install-data)", re.I)),
]

def classify(text):
    t=text or ""
    for name,rx in _PATTERNS:
        if rx.search(t):
            return name
    return None

# safe, whitelisted overrides (we ONLY adjust CLI flags)
HEAL_ACTIONS = {
  "oom": [
    {"max_candidates": 8},
    {"max_candidates": 6, "mace_dtype": "float32"},
    {"max_candidates": 4, "mace_dtype": "float32", "mace_device": "cpu"},
  ],
  "timeout": [
    {"max_candidates": 6},
    {"max_candidates": 4},
  ],
  "scf": [
    {"max_candidates": 8},
  ],
  "nan": [
    {"max_candidates": 6, "mace_dtype": "float32"},
  ],
}

def suggest_overrides(class_name):
    return HEAL_ACTIONS.get(class_name, [])
