# AiONet research code

AiONet combines registered representation interfaces, a Feature Bank, TempoSense prediction and receiver-side selection for camera-based vehicle–infrastructure perception.

Version **0.2.1** is an independent reconstruction of the manuscript's perception, compatibility and selection workflows. It combines recovered detector components with experiment drivers and receiver selection. [SOURCE_ORIGIN.md](SOURCE_ORIGIN.md) records the implementation choices and source revisions.

## What is included

- BEVFormer, PETR and DETR3D camera configurations for both vehicle and infrastructure roles.
- A Car-only 57/10 scenario protocol, seven registered fusion operators and a 36-configuration model–representation grid.
- Feature Bank and TempoSense modules, causal delay replay, fixed-delay and constant-velocity controls.
- Car AP25/AP50, center-distance mAP, five-error NDS and IoU-conditioned NDS70/NDS80 evaluators.
- A separate seven-class selection workflow with R-MLP, R-BH, source evidence, calibration, deadline handling and held-out roadside-model testing.
- Camera comparison adapters, runtime tools, real-output plotting, ground-truth overlays and metric aggregation.

The source distribution contains code, configurations, scenario IDs, documentation and licenses. Each experiment writes its checkpoints, cached predictions and measured metrics to a separate working directory.

## Install

The full detector workflow uses Linux, Python 3.8, a CUDA 11.x toolkit and an NVIDIA GPU. Run the installer in a dedicated environment:

```bash
python3.8 -m venv .venv-full
source .venv-full/bin/activate
bash tools/install_full.sh
python -m pip install -e . --no-deps
python -m pytest -q
```

For CPU selection and protocol utilities, Python 3.8–3.11 is supported:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python tools/paper.py list --table 2
python -m pytest -q
```

CPU-only tests skip modules whose optional Torch/MMDetection3D/nuScenes dependencies are absent. MLP fitting requires PyTorch; tree fitting and replay use NumPy/scikit-learn.

## Run the paper workflows

Use separate working directories for the two scenario splits. Commands run from the repository root.

| Workflow | Instructions | Main entry point |
|---|---|---|
| Perception, compatibility and temporal experiments | [Perception workflow](docs/perception.md) | `tools/paper.py` |
| Receiver selection, ablations and held-out integration | [Selection workflow](docs/selection.md) | `tools/pipeline.py`, `python -m aionet` |
| Map paper experiments to implementation | [Experiment map](docs/experiment_map.md) | `aionet/paper/experiments.py` |
| Metrics and input requirements | [Evaluation contract](docs/evaluation.md) | `aionet/paper/metrics.py` |

`--gpu` accepts one physical GPU index or UUID per command. Independent experiments can run on separate GPUs. Adjust the example device indices to your machine. Input and configuration hashes identify generated artifacts and prevent incompatible checkpoint reuse.

## Layout

```text
aionet/             Representation interfaces, temporal modules, selection and metrics
aionet/paper/       Perception experiment definitions and reconstructed fusion operators
training/configs/   Detector configuration templates
training/scripts/   Data conversion, training, inference and causal cache construction
training/vendor/    Recovered detector components and their licenses
configs/            Two scenario splits and source revision metadata
tools/              Launchers, installers, plotting and table collection
tests/              Synthetic geometry, metric, timing and model-interface checks
docs/               Reproduction commands and implementation choices
```

## Data and licenses

Obtain DAIR-V2X-Seq-SPD metadata and camera images from the [official V2X-Seq repository](https://github.com/AIR-THU/DAIR-V2X-Seq) under its terms. The metadata converter preserves calibration, track identity and observation times. The evaluation contract describes the velocity and attribute evidence needed for composite NDS scores.

Original integration code uses MIT. Vendored components retain their own licenses, listed in [THIRD_PARTY.md](THIRD_PARTY.md). Select2Col source is acquired directly from its upstream repository under its NoDistrib terms. `CITATION.cff` identifies this software version; `SOURCE_SHA256.json` identifies its distributed files.
