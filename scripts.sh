#!/bin/bash
set -e


echo "Running BLAST OOF with 12 threads..."

python3 training/train.py --method blast --aspect P --blast-threads 10
python3 training/train.py --method blast --aspect F --blast-threads 10
python3 training/train.py --method blast --aspect C --blast-threads 10

echo "Running late fusion..."
python3 training/late_fusion_new.py --aspect P F C --step 0.05 --jobs 3 --output models/late_fusion_new.csv

echo "Done."
