# SRDiff

Official research code for **SRDiff: A Cross-Modal Diffusion Model for Satellite-to-Radar Translation in Precipitation Nowcasting**.

This release contains only the main full-SEVIR experiment:

- `cmca`: SRDiff with the Cross-Modal Conditional Adapter;
- `base`: the retained satellite-adapter variant.

Ablations, cross-region experiments, unrelated baselines, and internal cluster launchers are intentionally excluded.

## 1. Install

The reference environment uses Python 3.10, PyTorch 2.3.1, and CUDA 12.1.

```bash
git clone https://github.com/42xingxing/SRDiff.git
cd SRDiff

conda create -n srdiff python=3.10 -y
conda activate srdiff

conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
python -m pip install -r requirements.txt
```

## 2. Prepare data and VAE

Download SEVIR from the [Registry of Open Data on AWS](https://registry.opendata.aws/sevir/) and keep the four modalities used by SRDiff: `ir069`, `ir107`, `lght`, and `vil`.

```text
work_dirs/data/sevir/
├── CATALOG.csv
└── data/
    ├── ir069/
    ├── ir107/
    ├── lght/
    └── vil/
```

Point the loader to that directory:

```bash
export SEVIR_DATA_DIR="$PWD/work_dirs/data/sevir"
```

Fresh training also requires a compatible frame-wise VAE checkpoint:

```text
work_dirs/checkpoints/vae/vae_framewise_lr.pth
```

Change `model.vae.from_pretrained` in [`configs/srdiff/common.yaml`](configs/srdiff/common.yaml) if it is stored elsewhere. Data, VAE weights, and trained SRDiff weights are not bundled.

Only load checkpoints from a trusted source.

## 3. Train

```bash
# CMCA variant
bash scripts/train.sh cmca "0,1,2,3" 4

# Base variant
bash scripts/train.sh base "0,1,2,3" 4
```

Resume a checkpoint produced with this cleaned code structure:

```bash
bash scripts/train.sh cmca "0,1,2,3" 4 --resume work_dirs/output/srdiff_cmca_sevir/checkpoint/last.ckpt
```

Outputs are written to `work_dirs/output/<experiment-name>/`.

## 4. Evaluate

```bash
bash scripts/evaluate.sh cmca "0,1,2,3" 4 --ckpt work_dirs/checkpoints/srdiff-cmca.ckpt
```

Optional arguments:

```text
--ensemble N
--inference-steps N
--pools 1,2,4
```

Reports are saved under:

```text
work_dirs/output/<experiment-name>/evaluation/<checkpoint-name>/
```

The evaluator reports CSI, POD, success ratio, bias, HSS, SSIM, and CRPS. FVD remains `NaN` because this release does not include a provenance-documented detector and complete FVD protocol.

## Repository layout

```text
configs/srdiff/       # shared config and base/cmca selectors
datasets/sevir/       # SEVIR loading, collation, and metrics
scripts/train.sh      # supported training launcher
scripts/evaluate.sh   # supported evaluation launcher
srdiff/cli.py         # Python entry point
src/                  # model, engine, module, and RF scheduler
```

Original SRDiff contributions are released under the [MIT License](LICENSE).
Modified third-party components remain under their upstream licenses, including
the non-commercial DiT license; see [Third-Party Notices](THIRD_PARTY_NOTICES.md).
