# Commands

## Training

```bash
# embedding
python training/data/embedding.py --plm esm2 --batch-size 2
python training/data/embedding.py --plm t5 --batch-size 1

# training submodels
python training/train.py --method cnn/esm2/t5 --aspect P/F/C

# late fusion weight
python training/late_fusion.py
```


## Inference
``` bash
# Default values:
## in: data/test.fasta
## out: predictions/predictions.tsv
## batch-size: 2
## method: fusion  (esm2/cnn/blast/t5)
## obo: data/gobsic.obo
python predict.py 
```