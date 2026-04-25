# Training Directory

```text
training/
├── README.md
├── *.py
├── data/
│   ├── *.py
│   ├── go-basic.obo
│   ├── raw/
│   │   ├── training.fasta
│   │   └── training.tsv
│   ├── cv/
│   │   └── fold_0-4/
│   │       ├── train.fasta
│   │       ├── train_labels.tsv
│   │       ├── val.fasta
│   │       └── val_labels.tsv
│   ├── embedding/
│   │   └── <plm>/<pooling>/<layer>/
│   │       ├── index.json
│   │       ├── shard_00000.pt
│   │       └── shard_00001.pt
│   ├── label_space/
│   │   └── <aspect>_min<min_count>.npy
│   └── protein_features/
│       └── protein_features.pt
└── oof/
```
