# Revealed preference: inferring cost ratios from operating points

Extension of the ROC/cost idea in the parent folder, applied to AI refusal behaviour.

Key result: at the cost-minimising threshold the ROC slope equals the cost ratio,
dTPR/dFPR = c_FP / c_FN. If you observe where a lab places its operating point
(refusal rate vs. harmful-content catch rate) across several model generations,
and you know or can bound each generation's ROC curve, the common tangent slope
reveals the lab's implied ratio between the cost of over-refusing and the cost
of letting harmful content through.

## Scripts

- `revealed_preference.py` - core idea: operating points across model generations plus their ROC curves share one tangent slope, which is the revealed cost ratio. Output: `revealed_preference.png`.
- `revealed_preference_graphical.py` - graphical solution showing how each constraint slices the feasible region for `c_FP`, giving a final interval. Also reports AUC per generation. Output: `revealed_preference_graphical.png`.
- `revealed_preference_simulation.py` - how well can `k = c_FP / c_FN` be inferred from operating points alone, using trivial-classifier bounds and pairwise dominance, swept over true `k`, minimum AUC and number of generations. Output: `revealed_preference_simulation.png`.
- `ai_utility_reveal.py` - production 2D and 3D ROC cost-surface plots. Outputs: `roc_utility_2d.png`, `roc_utility_3d.png`.
- `interactive_3d.py` - interactive explorer with sliders for the cost-surface parameters; used to tune the settings in `ai_utility_reveal.py`.

All scripts write their figures next to themselves, so run them from anywhere.
