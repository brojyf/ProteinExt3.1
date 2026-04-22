# Commands

## Training

```bash
# build cross validation files from raw data
python training/data/make_cv.py --fasta training/data/raw/train.fasta --labels training/data/raw/labels.tsv --overwrite

# optional: precompute embeddings; training/prediction also computes missing embeddings on demand
python training/data/embedding.py --plm esm2 --batch-size 2
python training/data/embedding.py --plm t5 --batch-size 1

# neural chains: train each chain independently per aspect
python training/train.py --method esm2_last --aspect P --fold 0 1 2 3 4 --epochs 20 --batch-size 16 --device auto
python training/train.py --method esm2_l20 --aspect P --fold 0 1 2 3 4 --epochs 20 --batch-size 16 --device auto
python training/train.py --method prott5 --aspect P --fold 0 1 2 3 4 --epochs 20 --batch-size 16 --device auto

# BLAST branch
python training/train.py --method blast --aspect P --fold 0 1 2 3 4

# two-stage OOF late fusion
python training/late_fusion.py --aspect P F C --methods esm2_last esm2_l20 prott5 blast --output models/fusion_weights.csv
```


## Inference
``` bash
# Default values:
## in: data/test.fasta
## out: predictions/predictions.tsv
## batch-size: 2
## method: fusion  (fusion/esm2_last/esm2_l20/prott5/blast)
## obo: data/go-basic.obo
## cpu: 8
## weights: models/fusion_weights.csv
python predict.py
```
