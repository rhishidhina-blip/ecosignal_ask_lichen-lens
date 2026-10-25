# Validation of completed multimodal continuation

{
  "context_version": "observation-context-v1",
  "site_baseline_version": 27,
  "site_tests": 149,
  "python_distinct_checks": 39,
  "python_top_level_tests": 15,
  "nested_semantic_checks": 25,
  "selection_combinations_per_engine": 127,
  "direct_path": "injected provider and actual JPEG decode/encode verified; live not run: Site key absent",
  "render_path": "new code injected SDK request + API verified; old deployed text HTTP200 with4citations in6.8s; new backend not yet deployed",
  "scientific_validation": "not_run",
  "browser": "cloud preview desktop empty collection, question display and safe retry checked; exact mobile and real saved attachment interaction not run",
  "secrets_or_observed_data_bundled": false
}

Synthetic fixtures verify software only. See tests/test_multimodal.py and semantic_checks.py. SDK constructor test isolates external proxy environment; no network is used by that test.
