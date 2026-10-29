# Source origin and reconstruction choices

Version 0.2.1 is an independent reconstruction of the manuscript's experimental workflows, rather than the original experiment snapshot. The source revisions and choices below define this implementation. Software tests verify interfaces and calculations; reproducing numerical tables requires a complete run with the corresponding data and evaluation annotations.

## Source components

The recovered MetroVision/AiONet repository is pinned at `ad3ee116741163fa1591de3fe746da90c4fe1433`. Its detector families, object decoder and MMDetection3D integration are retained with original licenses. Data preparation, registered BEV fusion, temporal prediction, branch caching and selection derive from the separately recorded SPD implementation. Portable launchers resolve all experiment paths from the chosen working directory.

Newly reconstructed components are in `aionet/paper/`, the `paper_*.py` training scripts and `tools/paper.py`. They supply the Car-only protocol, seven fusion operators, 36 model–representation configurations, temporal comparisons and composite metric interfaces.

## Explicit implementation choices

1. **PETR/DETR3D BEV adapters.** Final decoder query features are projected at their predicted 3D positions by bilinear splatting. A learned convolutional adapter refines the grid. Native BEVFormer maps use a learned channel projection. Adapters are trained separately for each registered pair.
2. **Representation grid.** A BEV transmission row uses vehicle queries to attend to roadside BEV features. Query transmission concatenates projected vehicle and roadside queries. Box transmission embeds geometric detections. BEV+query transmission uses adaptive BEV fusion and query attention.
3. **Conventional fusion reference.** The Table 1(b) reference is concatenation of aligned BEV fields followed by a learned projection. This operator is a declared choice of this reconstruction.
4. **Training selection.** Perception detectors and fusion modules use the fixed final epoch of a 24-epoch schedule. The selection workflow has separate development and calibration scenarios. The test split is excluded from parameter fitting and threshold selection.
5. **Temporal controls.** The observed control uses the last received packet. The queue control uses the freshest observation among arrived packets. Their causal arrival ordering and common delay strata are defined in `paper_temporal.py`.
6. **Metric inputs.** Five-term NDS uses validated velocities and per-object attribute labels. The IoU-conditioned variant retains the five-error weights and replaces the matching criterion. `docs/evaluation.md` specifies the input schema.
7. **External comparisons.** EMIFF uses an SPD camera adapter. V2X-ViT and Select2Col use registered camera BEV interfaces. Configuration and source revisions identify these adaptations.

Report this software version and the generated run fingerprints when comparing experimental results. English comments and documentation replace localized duplicates; original license notices are retained. The MMDetection3D dependency metadata uses Numba 0.56.4 to match the pinned NumPy environment.
