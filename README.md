# Commands

## Training

```bash
# embedding
python training/data/embedding.py --plm esm2 --batch-size 2
python training/data/embedding.py --plm t5 --batch-size 1

# training submodels
python training/train.py --method cnn/esm2/t5 --aspect P/F/C --fold 0 1 2 3 4 --epochs 20 --batch-size 16 --device cpu/mps/cuda

# late fusion weight
python training/late_fusion.py --aspect P/F/C --methods esm2/cnn/blast/t5 --cores 8 --output path/to/.csv
```


## Inference
``` bash
# Default values:
## in: data/test.fasta
## out: predictions/predictions.tsv
## batch-size: 2
## method: fusion  (esm2/cnn/blast/t5)
## obo: data/gobsic.obo
## cpu: 8
## weights: model/fusion_weights.csv
python predict.py
```