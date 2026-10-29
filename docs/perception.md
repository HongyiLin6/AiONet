# Perception, compatibility and temporal experiments

Use the full Linux/CUDA environment from the README. Obtain and extract the official SPD metadata and both camera image archives into one dataset root:

```text
/datasets/V2X-Seq-SPD/
    cooperative/data_info.json
    infrastructure-side/data_info.json
    infrastructure-side/image/...
    vehicle-side/data_info.json
    vehicle-side/image/...
```

The metadata ZIP retains the official `V2X-Seq-SPD/` prefix. Images are read from the extracted directory; annotations and calibration are read from the ZIP. The scenario IDs are in `configs/perception_split.json`. Car annotations are retained for this workflow.

## Prepare and train the camera models

```bash
python tools/paper.py --workdir workdir_perception init --data-root /datasets/V2X-Seq-SPD
python tools/paper.py --workdir workdir_perception prepare --metadata-zip /datasets/V2X-Seq-SPD.zip
python tools/paper.py --workdir workdir_perception run download_backbones

for model in bevformer petr detr3d; do
  for side in vehicle infrastructure; do
    python tools/paper.py --workdir workdir_perception --gpu 4 train-detector --name "${model}_${side}"
  done
done
```

Each command exposes one GPU. The example executes jobs sequentially; independent commands can be distributed over GPUs 4–7. ResNet initialization comes from the official OpenMMLab ImageNet files with hash checks. The generated six detector configurations use one Car class and 24 epochs. The final epoch is selected without using test labels. Resume an interrupted detector explicitly with `--resume /absolute/path/to/latest.pth`.

## Extract predictions and native representations

```bash
for model in bevformer petr detr3d; do
  for side in vehicle infrastructure; do
    for part in perception_train test; do
      python tools/paper.py --workdir workdir_perception --gpu 4 cache \
        --name "${model}_${side}" --partition "$part"
    done
  done
done
```

Caches contain final decoder query features and predicted positions, geometric boxes, confidence, velocity, timestamps and hashes. BEVFormer additionally exports its native BEV maps. The PETR/DETR3D spatial adapters learn BEV fields from their query features during registered fusion training. Reserve storage for FP16 maps of 150×150×256 values per frame. Completed cache partitions are checked and reused.

## Fusion comparisons and the 36-configuration grid

Inspect each job before running it:

```bash
python tools/paper.py list --table 1a
python tools/paper.py list --table 1b
python tools/paper.py list --table 2
```

A job trains only the adapter, fusion module and shared object decoder. Detector features remain fixed. Example:

```bash
python tools/paper.py --workdir workdir_perception --gpu 4 train --experiment fusion_adaptive
python tools/paper.py --workdir workdir_perception --gpu 4 evaluate --experiment fusion_adaptive
python tools/paper.py --workdir workdir_perception --gpu 4 evaluate --experiment fusion_vehicle_only
```

Run the entire representation grid with:

```bash
for job in $(python tools/paper.py list --table 2 | python -c 'import json,sys; print(" ".join(x["name"] for x in json.load(sys.stdin)))'); do
  python tools/paper.py --workdir workdir_perception --gpu 4 train --experiment "$job"
  python tools/paper.py --workdir workdir_perception --gpu 4 evaluate --experiment "$job"
done
```

Use the same pattern for Tables 1(a) and 1(b), omitting `train` for jobs whose operator is `vehicle_only` or `infrastructure_only`. A fusion job can resume with `train --experiment NAME --resume`. The Table 1(b) reference operator is explicitly `bev_concat`; its source interpretation is recorded in `SOURCE_ORIGIN.md`.

The default evaluation writes AP25/AP50 and center-distance mAP. To include five-error NDS and IoU-conditioned NDS70/NDS80, supply the velocity/attribute evidence described in [evaluation.md](evaluation.md):

```bash
python tools/paper.py --workdir workdir_perception --gpu 4 evaluate \
  --experiment fusion_adaptive --nds --attributes /datasets/evaluation_attributes.json
python tools/collect_tables.py --results workdir_perception/results/paper --output outputs/perception.csv
```

## Temporal prediction and causal replay

Train TempoSense on real roadside BEV histories and fit the fixed-delay fusion control. The declared temporal comparison uses the BEVFormer pair and the adaptive operator.

```bash
python tools/paper.py --workdir workdir_perception --gpu 5 train-temposense
python tools/paper.py --workdir workdir_perception --gpu 6 train --experiment temporal_fixed_delay
python tools/paper.py --workdir workdir_perception --gpu 4 train --experiment fusion_adaptive

for mode in stochastic sweep detector_sweep; do
  python tools/paper.py --workdir workdir_perception --gpu 4 temporal --mode "$mode" \
    --temposense workdir_perception/runs/paper_temposense/bevformer_infrastructure/final.pt
done
```

Skip the last training command if `fusion_adaptive` has already completed. Stochastic replay evaluates vehicle-only, infrastructure-only, observed fusion, fixed-delay training, the freshness queue, constant-velocity box propagation and TempoSense on common frame strata. The fixed-delay training value is the sum of the three declared segment means. Controlled sweeps use 0–1000 ms. Forecasts use three real observations, a maximum 150 ms history gap, a 2 s horizon and at most 20 rollout steps. Only already-arrived packets enter inference.

Observed fusion consumes the last arrival; queue and TempoSense consume the freshest observation among arrivals. An unavailable infrastructure-only input yields an empty detection set; cooperative policies retain the vehicle result. These causal conventions are explicit reconstruction choices.

```bash
python tools/plot_paper.py --kind temporal --input workdir_perception/results/paper_temporal/sweep.json --output outputs/delay_curve.png
python tools/plot_paper.py --kind delay --output outputs/delay_distribution.png
python tools/paper.py payload
```

An interrupted TempoSense fit resumes with `train-temposense --resume`. Epoch checkpoints are written atomically, and epoch seeds are restored consistently for resumed fusion and temporal training.

The delay-distribution figure illustrates the specified synthetic network model. Measured detection plots read the actual evaluation JSON.

## Qualitative overlays

`tools/qualitative.py` reads a camera image, predicted and ground-truth boxes, and the matching LiDAR-to-image projection. NPZ files use `boxes` in geometric-center `[x,y,z,length,width,height,yaw]` order; predictions also contain `scores`. The projection NPY is 3×4 or 4×4 and must correspond to the displayed image resolution.

```bash
python tools/qualitative.py --image frame.jpg --predictions predictions.npz \
  --ground-truth ground_truth.npz --projection lidar_to_image.npy --output outputs/qualitative.png
```
