"""Frozen snapshot of the notebook's ARCHITECTURES dict (cell index 3).

Committed as Python source so test failures surface as readable diffs
(THE SPEC §13.6 fixture table — notebook_reference.py row).
"""

NOTEBOOK_ARCHITECTURES = {
    "shallow":            {"name": "shallow",            "depth": 2, "nodes": 8,  "attention": False, "descriptors": []},
    "shallow_attn":       {"name": "shallow_attn",       "depth": 2, "nodes": 8,  "attention": True,  "descriptors": []},
    "medium":             {"name": "medium",             "depth": 3, "nodes": 16, "attention": False, "descriptors": []},
    "medium_attn":        {"name": "medium_attn",        "depth": 3, "nodes": 16, "attention": True,  "descriptors": []},
    "deep":               {"name": "deep",               "depth": 4, "nodes": 32, "attention": False, "descriptors": []},
    "deep_attn":          {"name": "deep_attn",          "depth": 4, "nodes": 32, "attention": True,  "descriptors": []},
    "deep_cusp":          {"name": "deep_cusp",          "depth": 4, "nodes": 32, "attention": False, "descriptors": ["cusp"]},
    "deep_cusp_attn":     {"name": "deep_cusp_attn",     "depth": 4, "nodes": 32, "attention": True,  "descriptors": ["cusp"]},
    "deep_dm":            {"name": "deep_dm",            "depth": 4, "nodes": 32, "attention": False, "descriptors": ["dm_statistics"]},
    "deep_dm_attn":       {"name": "deep_dm_attn",       "depth": 4, "nodes": 32, "attention": True,  "descriptors": ["dm_statistics"]},
    "deep_combined":      {"name": "deep_combined",      "depth": 4, "nodes": 32, "attention": False, "descriptors": ["dm_statistics", "cusp"]},
    "deep_combined_attn": {"name": "deep_combined_attn", "depth": 4, "nodes": 32, "attention": True,  "descriptors": ["dm_statistics", "cusp"]},
    # 2026-05-29: notransform variants — bare deep arch (no DM/Cusp extras)
    # with the Dick XCDiff input log-transform explicitly disabled. Used as
    # the no-log control in the descriptor ablation sweep.
    "deep_notransform":      {"name": "deep_notransform",      "depth": 4, "nodes": 32, "attention": False, "descriptors": []},
    "deep_notransform_attn": {"name": "deep_notransform_attn", "depth": 4, "nodes": 32, "attention": True,  "descriptors": []},
}
