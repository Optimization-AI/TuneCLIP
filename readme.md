<div align="center">

# TuneCLIP

### Tune-Up Open-Weight CLIP: Optimization Framework for Self-Supervised Fine-tuning of CLIP

**NeurIPS 2026 · Spotlight**

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2601.09859-B31B1B?style=flat-square)](https://arxiv.org/abs/2601.09859)
[![Conference](https://img.shields.io/badge/NeurIPS_2026-Spotlight-6842C2?style=flat-square)](#)
[![Model](https://img.shields.io/badge/Backbone-Open--Weight_CLIP-2563EB?style=flat-square)](#method)

**Recover optimizer statistics. Preserve pretrained knowledge. Fine-tune with HGCL.**

[Overview](#overview) · [Method](#method) · [Installation](#installation) · [Training](#training) · [Evaluation](#datacomp-evaluation) · [Citation](#citation)

</div>

---

## Overview

**TuneCLIP** is a two-stage optimization framework for **self-supervised fine-tuning (SSFT)** of open-weight CLIP models. It first recovers optimizer statistics at the pretrained solution through **Optimizer Statistics Recovery (OSR)**, then fine-tunes the model using **Hinged Global Contrastive Learning (HGCL)**.

Standard self-supervised fine-tuning can degrade a pretrained CLIP model immediately. TuneCLIP addresses the optimizer mismatch and reduces the influence of false negatives, enabling improvements from the first fine-tuning epoch.

<table>
  <tr>
    <th align="center" width="50%">The TuneCLIP Framework</th>
    <th align="center" width="50%">Fine-tuning Without the Initial Drop</th>
  </tr>
  <tr>
    <td align="center" valign="middle">
      <img src="Imgs/TuneCLIP_starter.png" alt="Overview of the two-stage TuneCLIP framework" width="100%">
    </td>
    <td align="center" valign="middle">
      <img src="Imgs/drop-openai-clip.png" alt="Zero-shot classification performance before and during fine-tuning" width="100%">
    </td>
  </tr>
  <tr>
    <td align="center">
      Recover optimizer statistics, then fine-tune with HGCL.
    </td>
    <td align="center">
      Improve pretrained CLIP from the first fine-tuning epoch.
    </td>
  </tr>
</table>

## Method

| Stage | Component | Purpose | Model weights |
| :--- | :--- | :--- | :--- |
| **1** | Optimizer Statistics Recovery (**OSR**) | Recover optimizer and contrastive statistics at the pretrained solution | Fixed |
| **2** | Hinged Global Contrastive Learning (**HGCL**) | Fine-tune while limiting unnecessary repulsion of negatives | Updated |

### Stage 1 · Optimizer Statistics Recovery

OSR addresses cold-start bias while keeping the pretrained model parameters fixed at `ω₀`.

1. Run the SogCLR/FastCLIP update computation over the self-supervised dataset.
2. Recover Adam's first and second moments, `m` and `v`, together with the per-sample statistics `uᵢ,ₓ` and `uᵢ,𝓏`.
3. Save the recovered state for subsequent fine-tuning.

Parameter updates are disabled with `--lr 0.0`. The recovered statistics are saved in the Stage 1 checkpoint and restored in Stage 2.

**In the paper, five OSR epochs are sufficient for stable subsequent fine-tuning.**

### Stage 2 · Hinged Global Contrastive Learning

Web-scale image–text datasets contain **false negatives**: an image and a non-matching caption may still be semantically related. Continuing to push these pairs apart can distort useful pretrained representations.

HGCL uses a **squared hinge surrogate** for the pairwise penalty, applied symmetrically to image- and text-anchored pairs. A negative receives no gradient from this penalty once its similarity is sufficiently below the positive similarity.

The margin controls the trade-off between preserving semantically related pairs and separating true negatives.

**Supplied recipe:** resume the OSR checkpoint and enable HGCL with `--hinge_sogclr`, `--margin 0.1`, and `--margin_smoothing 2`.

## Installation

Run the following commands from the repository root.

**Training dependencies**

```bash
pip install -r requirements-training.txt
```

**Evaluation dependencies**

Install these in the same environment or a separate evaluation environment:

```bash
pip install -r requirements-eval.txt
```

## Training

> [!IMPORTANT]
> Before submitting either job, update the project directory, dataset paths, checkpoint paths, environment name, and SLURM directives in the supplied scripts.

### 1. Recover optimizer statistics

The OSR script starts from an **OpenAI ViT-B/16** checkpoint and recovers optimizer statistics with fixed model weights.

```bash
sbatch Scripts_TuneCLIP/run_osr.sh
```

<details>
<summary><strong>View the OSR training command</strong></summary>

```bash
srun python -u src/training/main.py \
  --save-frequency 1 \
  --train-data "/path/to/data.tar" \
  --pretrained openai \
  --train-num-samples 12025856 --data_size 19280000 \
  --warmup 500 --batch-size 256 --epochs 5 --workers 6 \
  --model ViT-B-16 --name TuneCLIP_OSR_openai_vitb16 \
  --seed 2025 --wd 0.02 --local-loss \
  --fastclip --multiply_tau --temperature_scheme global_constant \
  --lr 0.0 --lr_tau 7.8125e-5 --lr_tau_scheduler step_thresh --rho 11.0 \
  --gamma 0.9 --gamma_schedule cosine --gamma_decay_epochs 4
```

</details>

### 2. Fine-tune with HGCL

Set `--resume` to the Stage 1 checkpoint. This restores the model weights and the optimizer/statistical state recovered by OSR.

```bash
sbatch Scripts_TuneCLIP/run_hgcl.sh
```

<details>
<summary><strong>View the HGCL fine-tuning command</strong></summary>

```bash
srun python -u src/training/main.py \
  --save-frequency 1 --hinge_sogclr \
  --margin 0.1 --margin_schedule constant --margin_smoothing 2 \
  --train-data "/path/to/data.tar" \
  --resume "/path/to/checkpoints/epoch_5.pt" \
  --train-num-samples 12025856 --data_size 19280000 \
  --warmup 500 --batch-size 256 --epochs 10 --workers 6 \
  --model ViT-B-16 --name TuneCLIP_final_vitb16_openai \
  --seed 2025 --wd 0.02 --local-loss \
  --fastclip --multiply_tau --temperature_scheme global_constant \
  --lr 1e-5 --lr_tau 7.8125e-5 --lr_tau_scheduler step_thresh --rho 11.0 \
  --gamma 0.9 --gamma_schedule cosine --gamma_decay_epochs 4
```

</details>

## DataComp Evaluation

Edit `evaluation/slurm_evaluate.sh` to specify the project, output directory, dataset, model, and checkpoint paths.

```bash
cd evaluation
sbatch slurm_evaluate.sh
```

<details>
<summary><strong>View the evaluation command</strong></summary>

The model architecture must match the checkpoint being evaluated. For the ViT-B/16 recipe above:

```bash
python evaluate.py \
  --train_output_dir "/path/to/your/project/train_output_dir" \
  --data_dir "/path/to/your/project/datasets/datacomp" \
  --model ViT-B-16 \
  --task datacomp \
  --use_model "ViT-B-16 /path/to/checkpoints/sample.pt"
```

</details>

Results are saved to the selected output directory as `<task>.jsonl`.

## Repository Structure

```text
.
├── Imgs/                         # Framework and performance figures
├── Scripts_TuneCLIP/
│   ├── run_osr.sh                # Stage 1: recover optimizer statistics
│   └── run_hgcl.sh               # Stage 2: fine-tune with HGCL
├── evaluation/
│   └── slurm_evaluate.sh         # DataComp evaluation
├── src/
│   └── training/                 # Training implementation
├── requirements-training.txt
└── requirements-eval.txt
```

## Citation

If you use TuneCLIP in your research, please cite:

```bibtex
@article{mehta2026breaking,
  title={Breaking the Limits of Open-Weight CLIP: An Optimization Framework for Self-supervised Fine-tuning of CLIP},
  author={Mehta, Anant and Wei, Xiyuan and Chen, Xingyu and Yang, Tianbao},
  journal={arXiv preprint arXiv:2601.09859},
  year={2026}
}
```
