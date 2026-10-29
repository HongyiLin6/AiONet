# Selection, ablations and held-out integration

This workflow uses a separate seven-class perception stack and `configs/split.json`. All paths below assume execution from the repository root. Use the full environment for data preparation, detectors, branch construction and MLP fitting. After branch export, tree fitting and cached evaluation also run in the smaller CPU environment.

## Prepare the five-part protocol

```bash
python tools/pipeline.py --workdir workdir_selection init --data-root /datasets/V2X-Seq-SPD
python tools/pipeline.py --workdir workdir_selection prepare --metadata-zip /datasets/V2X-Seq-SPD.zip
python tools/pipeline.py --workdir workdir_selection run download_backbones
```

The partitions are perception training (32 scenarios), selector fitting (8), development (8), calibration (9) and test (10). Test labels do not enter fitting, hyperparameter selection or calibration.

## Detectors and native feature caches

```bash
python tools/pipeline.py --workdir workdir_selection --gpu 4 train --name bevformer_vehicle
python tools/pipeline.py --workdir workdir_selection --gpu 5 train --name bevformer_infrastructure
python tools/pipeline.py --workdir workdir_selection --gpu 6 train --name petr_infrastructure
python tools/pipeline.py --workdir workdir_selection --gpu 7 train --name detr3d_infrastructure
```

These are independent jobs and can be scheduled concurrently. Each has a 12-epoch budget. BEVFormer/PETR use the first half of a 24-epoch cosine schedule; DETR3D uses a 12-epoch schedule. Development AP25 selects checkpoints every two epochs. Resume explicitly with `--resume /absolute/path/to/latest.pth`.

```bash
for part in perception_train selector_fit development calibration; do
  python tools/pipeline.py --workdir workdir_selection --gpu 4 run extract_detector_cache --name bevformer_vehicle --partition "$part"
  python tools/pipeline.py --workdir workdir_selection --gpu 5 run extract_detector_cache --name bevformer_infrastructure --partition "$part"
done
for part in selector_fit development calibration; do
  python tools/pipeline.py --workdir workdir_selection --gpu 6 run extract_detector_cache --name petr_infrastructure --partition "$part"
done
```

The F path uses the BEVFormer vehicle/infrastructure pair. The S receiver is fitted and calibrated with known BEVFormer/PETR infrastructure outputs. The DETR3D detector is trained on the perception-training scenes but held out from the receiving system's fitting, development and calibration.

## Fusion, prediction and branches

```bash
python tools/pipeline.py --workdir workdir_selection --gpu 4 run train_temposense
python tools/pipeline.py --workdir workdir_selection --gpu 5 run train_fusion --method aionet
python tools/pipeline.py --workdir workdir_selection --gpu 6 run choose_structured_parameters

for part in selector_fit development calibration; do
  python tools/pipeline.py --workdir workdir_selection --gpu 4 run build_branch_cache --family F --source bevformer_infrastructure --partition "$part"
  python tools/pipeline.py --workdir workdir_selection --gpu 5 run build_branch_cache --family S --source bevformer_infrastructure --partition "$part"
  python tools/pipeline.py --workdir workdir_selection --gpu 6 run build_branch_cache --family S --source petr_infrastructure --partition "$part"
done
python tools/pipeline.py --workdir workdir_selection --gpu 4 run choose_execution_parameters --family F
python tools/pipeline.py --workdir workdir_selection --gpu 5 run choose_execution_parameters --family S
python tools/pipeline.py --workdir workdir_selection --gpu 4 run export_portable_cache --family F --output workdir_selection/portable
python tools/pipeline.py --workdir workdir_selection --gpu 5 run export_portable_cache --family S --output workdir_selection/portable
```

TempoSense and registered fusion use at most six epochs and development patience two. Branch generation preserves real observation history and packet arrival times. It calculates fitting labels only after branch inference. The portable export contains detection matching events and descriptors computed from that run.

## Fit and calibrate controls and selectors

```bash
for family in F S; do
  python -m aionet --artifacts workdir_selection/portable fit --family "$family" \
    --regressor mlp --output "workdir_selection/runs/selectors/${family}_MLP"
  python -m aionet --artifacts workdir_selection/portable fit --family "$family" \
    --output "workdir_selection/runs/selectors/${family}_BH"
  python -m aionet --artifacts workdir_selection/portable fit --family "$family" \
    --source-evidence --leaves 7 --penalty 1 --output "workdir_selection/runs/selectors/${family}_BHS"
  python -m aionet --artifacts workdir_selection/portable fit --family "$family" \
    --source-evidence --without-source --leaves 7 --penalty 1 \
    --output "workdir_selection/runs/selectors/${family}_BHS_no_source"
done
python -m aionet --artifacts workdir_selection/portable fit --family F \
  --without-age-horizon --leaves 15 --penalty 1 --output workdir_selection/runs/selectors/F_BH_no_time
```

The default tree fit chooses 7/15 leaves and a harm penalty of 1/2/4 on development scenarios. Explicit `--leaves` and `--penalty` define a fixed configuration. Keep ablation capacity matched to its intended comparator. The documented BHS source-statistic ablation retains the 32-field interface and packet timing while zeroing the last 12 features and refitting both heads. Each fit recalibrates on calibration scenarios and records its choices and actual CPU scoring time.

## Optional external comparisons

Prepare these before freezing if they are part of the planned comparison:

```bash
python tools/fetch_baselines.py --workdir workdir_selection --names EMIFF v2x-vit
python tools/pipeline.py --workdir workdir_selection --gpu 4 run prepare_emiff
python tools/pipeline.py --workdir workdir_selection --gpu 4 run train_emiff
python tools/pipeline.py --workdir workdir_selection --gpu 5 run train_fusion --method v2xvit
```

EMIFF retains its camera architecture with SPD loading. V2X-ViT uses registered camera BEV maps. Select2Col uses a camera BEV adapter and can be acquired separately after reviewing its upstream terms:

```bash
python tools/fetch_baselines.py --workdir workdir_selection --names Select2Col --accept-select2col-terms
python tools/pipeline.py --workdir workdir_selection --gpu 7 run train_fusion --method select2col
```

Keep these adaptations identified in result descriptions and retain upstream licenses.

## Freeze and evaluate

Freeze every selector that will be evaluated:

```bash
python tools/pipeline.py --workdir workdir_selection freeze --selector-dir workdir_selection/runs/selectors/*

for name in bevformer_vehicle bevformer_infrastructure detr3d_infrastructure; do
  python tools/pipeline.py --workdir workdir_selection --gpu 4 run extract_detector_cache --name "$name" --partition test
done
python tools/pipeline.py --workdir workdir_selection --gpu 4 run build_branch_cache --family F --source bevformer_infrastructure --partition test
python tools/pipeline.py --workdir workdir_selection --gpu 5 run build_branch_cache --family S --source detr3d_infrastructure --partition test
python tools/pipeline.py --workdir workdir_selection --gpu 4 run export_portable_cache --family F --partitions test --output workdir_selection/portable
python tools/pipeline.py --workdir workdir_selection --gpu 5 run export_portable_cache --family S --partitions test --output workdir_selection/portable

for family in F S; do
  python -m aionet --artifacts workdir_selection/portable evaluate --family "$family" \
    --output "workdir_selection/results/${family}_controls"
  for model in MLP BH BHS BHS_no_source; do
    python -m aionet --artifacts workdir_selection/portable evaluate --family "$family" \
      --model-dir "workdir_selection/runs/selectors/${family}_${model}" \
      --output "workdir_selection/results/${family}_${model}"
  done
done
python -m aionet --artifacts workdir_selection/portable evaluate --family F \
  --model-dir workdir_selection/runs/selectors/F_BH_no_time --output workdir_selection/results/F_BH_no_time
```

`custom.json` identifies the supplied model's metrics, and `custom.decisions.json` records per-frame choices. The run fingerprint and calibration files identify the model. For the uncalibrated A4 control, evaluate the same MLP model with `--threshold 0`. Other prespecified conditions are selected with `--delay 187.3`, `--delay 258.0`, `--budget 50` or `--budget 200`. The default condition is 234.9 ms median delay and a 100 ms deadline. Use `--bootstrap 200` for scenario-level intervals from the same predictions.

External comparisons use identical cached delayed frame pairs:

```bash
for method in emiff v2xvit aionet_component; do
  python tools/pipeline.py --workdir workdir_selection --gpu 4 run evaluate_external_baselines --method "$method"
done
python tools/pipeline.py --workdir workdir_selection --gpu 4 run evaluate_external_baselines \
  --method aionet_selected --model-dir workdir_selection/runs/selectors/F_BH
```

Add `select2col` if its model was fitted. This external comparison is uncensored detection quality; deadline-aware committed outputs are measured by the portable evaluator above.

## Runtime and figures

```bash
python tools/pipeline.py --workdir workdir_selection --gpu 4 run benchmark_receiver --policy A0
python tools/pipeline.py --workdir workdir_selection --gpu 4 run benchmark_receiver --policy A3
python tools/pipeline.py --workdir workdir_selection --gpu 4 run benchmark_receiver \
  --policy R-BH --model-dir workdir_selection/runs/selectors/F_BH
```

The receiver benchmark uses 10 warm-up frames and 128 timed frames from the fixed first test sequence. It includes image preparation, GPU inference, selection, chosen branch computation and output serialization. Run timing jobs separately from training and record the shared-resource conditions. This benchmark measures individual frames, not continuous network throughput.

`tools/plot_results.py` accepts a JSON mapping policy labels to scalar values from generated `summary.json` files. Combine the A1 entry from the controls summary with each model's `custom` entry, using labels `R-MLP`, `R-BH` and `R-BH+S`.
