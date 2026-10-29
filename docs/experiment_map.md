# Experiment map

| Manuscript component | Implementation | Command / output |
|---|---|---|
| Vehicle and infrastructure camera detectors | `training/configs/`, `train_detector.py` | `paper.py train-detector` |
| Table 1(a), seven fusion operators and two single-side controls | `aionet/paper/operators.py`, `paper_model.py` | `paper.py list --table 1a` |
| Table 1(b), complete detector comparison | `experiments.py`, `paper_evaluate.py` | `paper.py list --table 1b` |
| Table 2, 36 model–representation combinations | `experiments.py`, `SpatialAdapter`, `PaperFusion` | `paper.py list --table 2` |
| Payload accounting | `experiments.payloads` | `paper.py payload` |
| TempoSense and temporal controls | `paper_train_temposense.py`, `paper_temporal.py` | `paper.py train-temposense`, `paper.py temporal` |
| Delay curves / shared-frame delay strata | `paper/temporal.py`, `paper_temporal.py` | `stochastic`, `sweep`, `detector_sweep` modes |
| Qualitative predictions with ground truth | `tools/qualitative.py` | Image + prediction/GT/projection inputs |
| Section 6 fixed-stack perception and branch cache | `train_fusion.py`, `build_branch_cache.py` | `pipeline.py run ...` |
| R-MLP, R-BH and R-BH+S | `aionet/mlp.py`, `selector.py`, `calibration.py` | `python -m aionet ... fit` |
| Age/horizon and source-statistic ablations | `selector.py`, CLI fit options | `--without-age-horizon`, `--without-source` |
| Held-out roadside detector | S-family caches with known BF/PETR and test DETR3D | `pipeline.py run build_branch_cache --family S` |
| External camera comparisons | `train_emiff.py`, `train_fusion.py`, `evaluate_external_baselines.py` | Separate source adapters |
| Deadline, harm and coverage measures | `aionet/replay.py`, `metrics.py` | `evaluate --delay ... --budget ...` |
| Scenario bootstrap | `aionet/metrics.py` | `evaluate --bootstrap 200` |
| Actual receiver runtime | `benchmark_receiver.py` | A0/A3/R-BH/R-MLP receiver timing; selector fit also records CPU scoring timing |
| Result figures and tables | `plot_results.py`, `plot_paper.py`, `collect_tables.py` | Inputs are generated evaluation files |

Perception jobs have stable names. To see their model pair, representation, operator and training budget, run `python tools/paper.py list`. The temporal job specifications describe seven comparison policies; the temporal runner computes those policies together on identical frame strata. It reuses the observed fusion checkpoint and the separately trained `temporal_fixed_delay` checkpoint.

The Section 6 policy names A0–A3 denote vehicle-only, observed fusion, predicted fusion and freshness selection. A4 denotes the signed-loss MLP before operating-point calibration. R-MLP uses that model with calibration. R-BH uses two tree heads on 20 descriptors; R-BH+S adds 12 source statistics. Each fitted model is passed explicitly with `--model-dir`.
