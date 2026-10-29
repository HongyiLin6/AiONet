# Evaluation contract

## Perception protocol

The perception workflow retains **Car** and evaluates the forward region `x ∈ [0,51.2] m`, `y ∈ [-51.2,51.2] m` in vehicle LiDAR coordinates. Boxes use geometric centers, dimensions `(length,width,height)` and a yaw angle in radians. Ground-truth and prediction boxes must share this reference frame.

- AP25/AP50 use all-point interpolated precision–recall curves with 3D IoU matching at 0.25/0.50.
- Center-distance mAP uses the official nuScenes devkit AP interpolation at 0.5, 1, 2 and 4 m, with its 0.1 minimum recall/precision convention. It is distinct from IoU AP.
- NDS combines center-distance mAP with translation, scale, orientation, velocity and attribute errors measured at the 2 m match threshold: `(5*mAP + Σ max(0,1-error))/10`.
- NDS70/NDS80 use IoU matching at 0.70/0.80 with the same five error terms and weights. This is an explicitly defined IoU-conditioned study metric, rather than an official nuScenes leaderboard score.

The nuScenes devkit is the metric implementation dependency. Car is mapped to its lowercase `car` category. This dataset adaptation does not introduce other nuScenes class filters.

## Velocity and attribute inputs

The SPD converter estimates velocity from adjacent observations of the same annotated track. Its `velocity_valid` mask identifies objects supported by an adequate temporal interval. The legacy training format uses zero-filled velocity fields when that estimate is unavailable. Composite evaluation instead rejects those invalid targets unless the caller provides an explicit valid evaluation velocity; it does not count the padding as stationary ground truth.

Five-term NDS additionally requires per-object attribute labels for both predictions and ground truth. A JSON sidecar is indexed by cooperative frame ID. Each entry contains `gt_attributes` and `pred_attributes`, in the exact order of evaluated boxes. Optional `gt_velocity` and `pred_velocity` arrays override velocity fields for evaluation when supported by the caller's annotation protocol. Velocities are in m/s, shape N×2. Attributes use valid nuScenes names such as `vehicle.moving`, `vehicle.stopped` or `vehicle.parked`.

The sidecar's order must match ROI-filtered ground truth and the prediction NPZ produced by the chosen experiment. Run the ordinary evaluator first to inspect prediction order. Record the attribute annotation protocol with the run. The evaluator checks field presence, array lengths and finite velocities.

## Selection protocol

The selection workflow uses seven classes and five scenario-disjoint partitions: perception training, selector fitting, development, calibration and test. Per-class AP is equally averaged over classes with ground-truth support in the evaluated sample, as implemented in `aionet/metrics.py`; unsupported classes are excluded from that sample's mean. Preserve this convention across compared policies. The fixed test scenario IDs are identical to those used by the perception workflow, while the training partitions and checkpoints differ.

Harm compares the committed output's loss against the same frame's vehicle-only loss. `H_all` divides harmful committed frames by all test frames; `H_acc` divides by cooperative commits. `C_coop` is the number of cooperative commits divided by all test frames. For positive cooperation coverage, `H_all = C_coop * H_acc`. An output deadline miss has its own metric and detection outcome.

Calibration chooses an operating threshold using calibration scenarios after development choices. The AP floor, minimum coverage and harm target are empirical operating constraints. Threshold selection can fall back to the gain/coverage feasible set; its mode is recorded with the chosen threshold. These constraints are not a statistical guarantee on future deployments.

## Timing and source identity

Cache replay uses the measured stage costs associated with that run. `benchmark_receiver.py` measures actual batch-one wall time, including image preparation and serialization, under a frozen causal snapshot. Those are different scopes and should be reported separately. The per-frame receiver benchmark does not establish sustained queue throughput.

Retain the source, configuration and checkpoint hashes with each table, together with the split and run identifier.
