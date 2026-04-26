# Training Directory

```text
training/
├── *.py
├── data/
│   ├── *.py
│   ├── go-basic.obo
│   ├── ic.pkl
│   ├── raw/
│   │   ├── training.fasta
│   │   └── training.tsv
│   ├── propagated/
│   │   ├── training.fasta
│   │   └── training.tsv
│   ├── cv/
│   │   └── fold_<0-4>/
│   │       ├── train.fasta
│   │       ├── train_labels.tsv
│   │       ├── val.fasta
│   │       └── val_labels.tsv
│   ├── embedding/
│   │   └── <plm>/<pooling>/<layer>/
│   │       ├── index.json
│   │       └── shard_<id>.pt
│   ├── label_space/
│   │   └── <aspect>_min<min_count>.npy
│   └── protein_features/
│       └── protein_features.pt
└── oof/
    └── <method>/
        └── <method>_<aspect>_fold_<id>.npz
```
