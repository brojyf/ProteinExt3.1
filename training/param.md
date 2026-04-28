# ESM2 Hyperparameter Tuning Plan

## Baseline (Default Config, 5-fold OOF Mean)

All runs used `COMMON_TRAINING_CONFIG` defaults:
`epochs=20, lr=3e-4, weight_decay=2e-4, hidden_dim=2048, bottleneck=1024, dropout=0.3, batch_size=16, lr_scheduler=cosine, min_lr=5e-5, early_stop_patience=6, min_count=20`

| chain | method  | pooling | crafted | Fmax   | Smin    | AUPR   |
|-------|---------|---------|---------|--------|---------|--------|
| P     | esm2-28 | mean    | no      | 0.4428 | 90.7323 | 0.2822 |
| F     | esm2-28 | mean    | no      | 0.6920 | 15.3709 | 0.5083 |
| C     | esm2-33 | mean    | yes     | 0.6903 | 17.6211 | 0.5671 |

## Architecture Recap

```
ChainMLPClassifier
  FeatureEncoder (crafted only): LayerNorm → 63→128 GELU Dropout → 128→256
  MLPHead:
    LayerNorm → input_proj(input_dim → hidden_dim)
    residual block: LayerNorm → GELU → Dropout → Linear(hidden_dim → hidden_dim)
    output_head:  LayerNorm → GELU → Dropout → Linear(hidden_dim → bottleneck)
                → LayerNorm → GELU → Dropout → Linear(bottleneck → num_classes)
```

Input dim per config:
- P/F (esm2-28, mean, no-crafted): 1×1280 = 1280
- C   (esm2-33, mean, +crafted):   1×1280 + 256 = 1536

## Tuning Strategy

Greedy sequential search (Mac, no CUDA, fold_0 quick probe → 5-fold confirm):

1. **Round 1** — lr + weight_decay + dropout + min_count (biggest impact)
2. **Round 2** — hidden_dim + bottleneck (capacity)
3. **Round 3** — epochs fine-tune
4. **Round 4** — 5-fold confirm best config

## Search Space

| param        | candidates                            |
|--------------|---------------------------------------|
| lr           | 1e-4, 2e-4, 3e-4, 5e-4               |
| weight_decay | 1e-4, 2e-4, 3e-4, 5e-4               |
| hidden_dim   | 1024, 1536, 2048                      |
| bottleneck   | 512, 768, 1024                        |
| dropout      | 0.2, 0.25, 0.3, 0.35, 0.4            |
| epochs       | 16, 18, 20, 22, 24, 26, 28, 30       |
| min_count    | P: 20/30/40, F: 10/15/20, C: 5/10/15 |

---

## Round 1 — V1 Config

### V1 Config

| param            | P (esm2-28)     | F (esm2-28)     | C (esm2-33)     |
|------------------|-----------------|-----------------|-----------------|
| lr               | **2e-4** (↓)    | 3e-4            | 3e-4            |
| weight_decay     | **3e-4** (↑)    | 2e-4            | 2e-4            |
| dropout          | **0.35** (↑)    | 0.3             | **0.25** (↓)    |
| hidden_dim       | 2048            | 2048            | **1536** (↓)    |
| bottleneck       | 1024            | 1024            | **768** (↓)     |
| epochs           | **24** (↑)      | **22** (↑)      | 18              |
| min_count        | **30** (↑)      | **15** (↓)      | **10** (↓)      |

### V1 fold_0 Results

| aspect | V1 Fmax | V1 Smin | V1 AUPR | baseline Fmax | Δ Fmax  | best_epoch | early_stop |
|--------|---------|---------|---------|---------------|---------|------------|------------|
| P      | 0.3561  | 95.82   | 0.2135  | 0.4428        | **-0.087** | 9/24    | epoch 15   |
| F      | 0.6230  | 18.10   | 0.4094  | 0.6920        | **-0.069** | 22/22   | no         |
| C      | 0.6920  | 17.31   | 0.5707  | 0.6903        | **+0.002** | 11/18   | epoch 17   |

### V1 Analysis

**P: significantly degraded (−0.087)**
- lr=2e-4 + dropout=0.35 + weight_decay=3e-4 三重正则化叠加导致严重欠拟合
- loss 从未低于 0.0093，fmax 在 0.33–0.36 之间震荡无法突破
- best_epoch=9 且 early stop at 15，说明模型学习能力被过度抑制
- min_count=30（vs baseline 20）减少了标签数，本应更容易，但 lr 太低导致反效果

**F: significantly degraded (−0.069)**
- 虽然 lr/dropout/network 几乎未变，但 min_count=15（vs baseline 20）引入了更多稀有标签
- 更大的标签空间直接导致任务变难，Fmax 显著下降
- 模型在 22 epoch 后仍在缓慢上升（epoch 20: 0.6228 → epoch 22: 0.6230），可能需要更多 epoch
- 但即使收敛完全，也不太可能追平 baseline，min_count 是主要原因

**C: on par (+0.002) ✓**
- 缩小 network + 降低 dropout + crafted features 效果良好
- 尽管 min_count=10 引入更多标签，Fmax 仍持平
- early_stop at 17 (best=11)，收敛健康

---

## Round 2 — V2 Config (Next Step)

### 核心修正

1. **P**: 回退正则化强度，只做小幅调整
   - lr 回到 3e-4（与 baseline 相同）
   - dropout 回到 0.3（与 baseline 相同）
   - weight_decay 回到 2e-4（与 baseline 相同）
   - min_count 保持 30（比 baseline 的 20 稍严格，减少标签空间有助于 P）
   - epochs 增加到 24（给更多学习时间）
   - 本轮 V2-P 实质上 = baseline + min_count=30 + epochs=24

2. **F**: 回退 min_count，增加 epochs
   - min_count 回到 20（与 baseline 相同，确保标签空间一致）
   - epochs 增加到 24（V1 观察到模型在后期仍在上升）
   - 其余保持 baseline 默认

3. **C**: 保持 V1（已证明有效）

### V2 Config

| param            | P (esm2-28)     | F (esm2-28)     | C (esm2-33)     |
|------------------|-----------------|-----------------|-----------------|
| lr               | 3e-4            | 3e-4            | 3e-4            |
| weight_decay     | 2e-4            | 2e-4            | 2e-4            |
| dropout          | 0.3             | 0.3             | 0.25            |
| hidden_dim       | 2048            | 2048            | 1536            |
| bottleneck       | 1024            | 1024            | 768             |
| epochs           | **24** (↑)      | **24** (↑)      | 18              |
| min_count        | **30** (↑)      | 20              | 10              |

---

## Log

| round | aspect | change                        | Fmax   | Δ       | note                              |
|-------|--------|-------------------------------|--------|---------|-----------------------------------|
| 0     | P      | baseline (default, 5f mean)   | 0.4428 | —       | min_count=20                      |
| 0     | F      | baseline (default, 5f mean)   | 0.6920 | —       | min_count=20                      |
| 0     | C      | baseline (default, 5f mean)   | 0.6903 | —       | min_count=20                      |
| 1     | P      | V1 (fold_0 only)              | 0.3561 | -0.087  | lr↓ dropout↑ wd↑ → underfitting   |
| 1     | F      | V1 (fold_0 only)              | 0.6230 | -0.069  | min_count=15 → harder label space |
| 1     | C      | V1 (fold_0 only)              | 0.6920 | +0.002  | ✓ smaller net + crafted works     |
| 2     | P      | V2 (fold_0)                   |        |         |                                   |
| 2     | F      | V2 (fold_0)                   |        |         |                                   |
| 2     | C      | V2 = V1 (skip)                | 0.6920 | +0.002  | keep V1                           |

---

## Raw Logs

<details>
<summary>V1 fold_0 — C (esm2-33, mean, +crafted)</summary>

```
python training/train.py --aspect C --method esm2-33 --pooling mean --fold 0 --model-dir tmp/

fold=fold_0 epoch=1  lr=3.00e-04 loss=0.0133 fmax=0.6642
fold=fold_0 epoch=2  lr=2.98e-04 loss=0.0096 fmax=0.6738
fold=fold_0 epoch=3  lr=2.92e-04 loss=0.0091 fmax=0.6806
fold=fold_0 epoch=4  lr=2.83e-04 loss=0.0087 fmax=0.6828
fold=fold_0 epoch=5  lr=2.71e-04 loss=0.0084 fmax=0.6809
fold=fold_0 epoch=6  lr=2.55e-04 loss=0.0081 fmax=0.6856
fold=fold_0 epoch=7  lr=2.37e-04 loss=0.0079 fmax=0.6893
fold=fold_0 epoch=8  lr=2.18e-04 loss=0.0077 fmax=0.6885
fold=fold_0 epoch=9  lr=1.97e-04 loss=0.0075 fmax=0.6900
fold=fold_0 epoch=10 lr=1.75e-04 loss=0.0073 fmax=0.6910
fold=fold_0 epoch=11 lr=1.53e-04 loss=0.0071 fmax=0.6920 ← best
fold=fold_0 epoch=12 lr=1.32e-04 loss=0.0069 fmax=0.6886
fold=fold_0 epoch=13 lr=1.13e-04 loss=0.0067 fmax=0.6879
...
fold=fold_0 early_stop epoch=17 best_epoch=11 best_fmax=0.6920
→ fmax=0.6920  aupr=0.5707  smin=17.31
```

</details>

<details>
<summary>V1 fold_0 — P (esm2-28, mean, no-crafted)</summary>

```
python training/train.py --aspect P --method esm2-28 --pooling mean --no-crafted --fold 0 --model-dir tmp/

fold=fold_0 epoch=1  lr=2.00e-04 loss=0.0132 fmax=0.3343
fold=fold_0 epoch=2  lr=1.99e-04 loss=0.0097 fmax=0.3296
fold=fold_0 epoch=3  lr=1.97e-04 loss=0.0096 fmax=0.3473
fold=fold_0 epoch=4  lr=1.94e-04 loss=0.0095 fmax=0.3515
fold=fold_0 epoch=5  lr=1.90e-04 loss=0.0095 fmax=0.3470
fold=fold_0 epoch=6  lr=1.85e-04 loss=0.0095 fmax=0.3538
fold=fold_0 epoch=7  lr=1.78e-04 loss=0.0095 fmax=0.3553
fold=fold_0 epoch=8  lr=1.71e-04 loss=0.0094 fmax=0.3363
fold=fold_0 epoch=9  lr=1.63e-04 loss=0.0094 fmax=0.3561 ← best
fold=fold_0 epoch=10 lr=1.54e-04 loss=0.0094 fmax=0.3454
...
fold=fold_0 early_stop epoch=15 best_epoch=9 best_fmax=0.3561
→ fmax=0.3561  aupr=0.2135  smin=95.82
```

</details>

<details>
<summary>V1 fold_0 — F (esm2-28, mean, no-crafted)</summary>

```
python training/train.py --aspect F --method esm2-28 --pooling mean --no-crafted --fold 0 --model-dir tmp/

fold=fold_0 epoch=1  lr=3.00e-04 loss=0.0072 fmax=0.5712
fold=fold_0 epoch=2  lr=2.99e-04 loss=0.0047 fmax=0.5676
fold=fold_0 epoch=3  lr=2.95e-04 loss=0.0046 fmax=0.5857
fold=fold_0 epoch=4  lr=2.89e-04 loss=0.0046 fmax=0.5874
fold=fold_0 epoch=5  lr=2.80e-04 loss=0.0045 fmax=0.5859
...
fold=fold_0 epoch=13 lr=1.57e-04 loss=0.0044 fmax=0.6041
...
fold=fold_0 epoch=17 lr=9.31e-05 loss=0.0041 fmax=0.6180
...
fold=fold_0 epoch=22 lr=5.13e-05 loss=0.0040 fmax=0.6230 ← best (no early stop)
→ fmax=0.6230  aupr=0.4094  smin=18.10
```

</details>

<details>V1 fold_0 — C (esm2-33, mean, +crafted)

## C
python training/train.py --aspect C --method esm2-33 --pooling mean --fold 0 --model-dir tmp/

fold=fold_0 lr_scheduler=cosine
fold=fold_0 epoch=1 lr=3.00e-04 loss=0.0133 fmax=0.6505 fmax_threshold=0.25                                                      
fold=fold_0 epoch=2 lr=2.98e-04 loss=0.0096 fmax=0.6719 fmax_threshold=0.27                                                      
fold=fold_0 epoch=3 lr=2.92e-04 loss=0.0090 fmax=0.6801 fmax_threshold=0.24                                                      
fold=fold_0 epoch=4 lr=2.83e-04 loss=0.0087 fmax=0.6846 fmax_threshold=0.29                                                      
fold=fold_0 epoch=5 lr=2.71e-04 loss=0.0084 fmax=0.6851 fmax_threshold=0.30                                                      
fold=fold_0 epoch=6 lr=2.55e-04 loss=0.0081 fmax=0.6882 fmax_threshold=0.27                                                      
fold=fold_0 epoch=7 lr=2.37e-04 loss=0.0079 fmax=0.6875 fmax_threshold=0.28                                                      
fold=fold_0 epoch=8 lr=2.18e-04 loss=0.0077 fmax=0.6893 fmax_threshold=0.29                                                      
fold=fold_0 epoch=9 lr=1.97e-04 loss=0.0075 fmax=0.6905 fmax_threshold=0.28                                                      
fold=fold_0 epoch=10 lr=1.75e-04 loss=0.0073 fmax=0.6865 fmax_threshold=0.24                                                     
fold=fold_0 epoch=11 lr=1.53e-04 loss=0.0071 fmax=0.6925 fmax_threshold=0.28                                                     
fold=fold_0 epoch=12 lr=1.32e-04 loss=0.0069 fmax=0.6876 fmax_threshold=0.25                                                     
fold=fold_0 epoch=13 lr=1.13e-04 loss=0.0067 fmax=0.6862 fmax_threshold=0.26                                                     
fold=fold_0 epoch=14 lr=9.47e-05 loss=0.0066 fmax=0.6885 fmax_threshold=0.25                                                     
fold=fold_0 epoch=15 lr=7.92e-05 loss=0.0065 fmax=0.6887 fmax_threshold=0.26                                                     
fold=fold_0 epoch=16 lr=6.67e-05 loss=0.0063 fmax=0.6880 fmax_threshold=0.25                                                     
fold=fold_0 epoch=17 lr=5.75e-05 loss=0.0062 fmax=0.6898 fmax_threshold=0.25                                                     
fold=fold_0 early_stop epoch=17 best_epoch=11 best_fmax=0.6925
{                                                                                                                                
  "avg_aupr": 0.5716683866574473,
  "avg_fmax": 0.6924629516200838,
  "avg_smin": 17.31546245027336
}

## F

## P