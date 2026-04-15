# Hyperparameters

训练配置已经从 CLI 挪到 [`training/hparams.py`](./hparams.py)。

当前允许用 CLI 临时覆盖：

- `--method`
- `--aspect`
- `--batch-size`
- `--epochs`
- `--fold`

重点字段：

- `TRAINING_RUNS`
  - 控制要跑哪些 `method/aspect`
  - 可用 `enabled: false` 暂时禁用某个 run
- `METHOD_HPARAMS["esm2"]` / `METHOD_HPARAMS["t5"]`
  - `optimizer.attention_lr`
  - `optimizer.classifier_lr`
  - `dropout`
  - `hidden_dim`
  - `pooling`
  - `go_term_loss_weight`
- `METHOD_HPARAMS["cnn"]`
  - `optimizer.lr`
  - `window_size`
  - `dropout`
  - `hidden_dim`
  - `cnn_hidden_dim`
  - `go_term_loss_weight`
- `COMMON_TRAINING_CONFIG`
  - `epochs`
  - `batch_size`
  - `fold`
  - `scheduler.warmup_ratio`
  - `scheduler.min_lr_ratio`
  - `weight_decay`
  - `go_term_loss_weight`

当前默认值已经按训练目标 `BCE + go_term_soft_f1_loss` 调过一版：

- `go_term_loss_weight` 从激进的全局大权重，改成更保守的公共默认值，并允许各方法单独覆盖
- `esm2` / `t5`
  - 下调 `attention_lr` 和 `classifier_lr`
  - 收紧 `hidden_dim`
  - 保留 attention pooling，但减少过快收敛导致的振荡
- `cnn`
  - 下调主学习率
  - 收紧 `cnn_hidden_dim`
  - 把 `window_size` 调成更平滑的奇数窗口
