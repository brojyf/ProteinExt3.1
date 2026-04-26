# Commands
Check the [training/README.md](training/README.md) and [data/README.md](data/README.md) for more detailed directory information.
1. [Make CV](#make-cv)
2. [Embedding](#embedding)
3. [Training](#training)
4. [Late Fusion](#late-fusion)
5. [Inference](#inference)
6. [Comparison](#comparison)
7. [Exploration](#exploration)


## Make CV
```bash
python training/data/make_cv.py
```

Optional:
- `--fasta`: input FASTA path; default `training/data/raw/training.fasta`
- `--labels`: input labels TSV path; default `training/data/raw/training.tsv`
- `--obo`: GO OBO path; default `data/go-basic.obo`
- `--out`: CV output directory; default `training/data/cv`
- `--folds`: number of folds; default `5`
- `--seed`: random seed; default `42`
- `--cd-hit-bin`: CD-HIT executable; default `cd-hit`
- `--cd-hit-identity`: CD-HIT identity threshold; default `0.5`
- `--cd-hit-word-size`: CD-HIT word size; default `2`
- `--mem`: CD-HIT memory limit MB; default `16000`
- `--threads`: CD-HIT thread count; default `8`
- `--overwrite`: overwrite existing output; default disabled

## Embedding
```bash
python training/data/embedding.py --plm esm2 --layers 20
```

Required:
- `--plm`: `esm2` or `prott5`
- `--layers`: one or more hidden layer ids, space separated; must be ascending because inference stops at the last layer

Optional:
- `--fasta`: input FASTA path; default `training/data/raw/training.fasta`
- `--out-dir`: embedding output directory; default `training/data/embedding`
- `--device`: `auto`, `cuda`, `mps`, or `cpu`; default `auto`
- `--max-length`: max sequence token length; default `1024`
- `--batch-size`: embedding batch size; default `8`
- `--pooling`: `max`, `mean`, or `both`; default `both`
- `--shard-size`: proteins per shard; default `4096`

## Training
```bash
python training/train.py --method esm2-33 --aspect P
```

Required for one neural CV training run:
- `--method`: `esm2-<layer>` or `prott5`; default training config: `esm2-33`
- `--aspect`: `P`, `F`, or `C`; default training config: `P`

Optional:
- `--fold`: one or more fold ids; default `0 1 2 3 4`
- `--epochs`: training epochs; default `20`
- `--batch-size`: batch size; default `16`
- `--device`: `auto`, `cuda`, `mps`, or `cpu`; default `auto`
- `--threads`: BLAST thread count; default `8`
- `--pooling`: `both`, `mean`, or `max`; default `both`
- `--model-dir`: checkpoint output directory; default `models_raw`
- `--oof-dir`: OOF output directory; default `training/oof`
- `--no-crafted`: disable handcrafted protein features; default crafted features enabled
- `--lr-scheduler`: `cosine` or `plateau`; default `cosine`

Exceptions:
- BLAST does not accept `--aspect`; one run generates `P`, `F`, and `C` OOF files 
- `--final`: train MLP methods on `training/data/raw/training.fasta` and `training/data/raw/training.tsv`; does not require `--fold` or `--oof-dir`, and does not save OOF


## Late Fusion
```bash
python training/late_fusion.py
```

Optional:
- `--aspect`: one or more GO aspects; choices `P`, `F`, `C`; default `P F C`
- `--fold`: one or more fold ids; default `0 1 2 3 4`
- `--oof-dir`: OOF prediction directory; default `training/oof`
- `--output`: output fusion weights CSV path; default `models_raw/latefusion_new.csv`
- `--obo`: GO OBO path; default `data/go-basic.obo`
- `--step`: simplex grid search step size (must divide 1.0); default `0.1`
- `--device`: `auto`, `cuda`, `mps`, or `cpu`; default `auto`
- `--jobs`: parallel aspect jobs; `0` uses one job per requested aspect; default `0`

## Inference
```bash
python predict.py
```

Optional:
- `--in`: input FASTA path; default `data/test.fasta`
- `--out`: output TSV path; default `predictions/ProteinExt3.1.tsv`
- `--method`: `fusion`, `esm2-<layer>`, `prott5`, or `blast`; default `fusion`
- `--aspect`: `P`, `F`, `C`, or `PFC`; default `PFC`
- `--batch-size`: prediction batch size; default `2`
- `--cpu`: CPU thread count; default `8`
- `--model-dir`: model directory; default `models`
- `--weights`: fusion weights path; default `<model-dir>/fusion_weights.csv`
- `--obo`: GO OBO path; default `data/go-basic.obo`
- `--device`: `auto`, `cuda`, `mps`, or `cpu`; default `auto`
- `--propagate`: enable GO score propagation; default disabled
- `--no-threshold`: disable fusion threshold filtering; default threshold enabled

## Comparison
Run following commands 


## Exploration
