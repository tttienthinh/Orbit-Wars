● Training completed successfully. Results:

  Train AUC: 0.9072 | Test AUC: 0.8259

  - 6.3M training pairs (5,113 positives, pos_weight=1239.4)
  - 1.6M test pairs (1,226 positives)
  - Ran in ~18 minutes total (14 min train features + 3.5 min test features)

  Top features by importance:
  1. src_y, tgt_y, tgt_x, src_x — position dominates (attacks come from specific board regions)
  2. src_ships_t19, tgt_ships_t19 — 20-step future ship counts (strategic value signal)
  3. travel_1 and all travel_* — travel times matter across the board
  4. tgt_production, tgt_ships — target quality

  The 0.8259 test AUC is a solid baseline. The ~9-point gap between train/test (0.9072 vs 0.8259) suggests some overfitting — the model could
  benefit from more regularization or larger min_child_samples. This gives the GNN (117) a concrete target to beat on the same data.

✻ Worked for 17s

※ recap: Training 118-LGBM.py completed successfully. Test AUC is 0.8259 — this is the LightGBM baseline the GNN needs to beat. Next: open the 
  PR on GitHub at github.com/tttienthinh/Orbit-Wars/pull/new/feat/118-lgbm.