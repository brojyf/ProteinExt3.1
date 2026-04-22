# TODO
所有需要有embedding的地方：
  如果没有embedding，就先保存embedding再计算。
  如果有embedding，就直接计算。

训练的时候，查找device: cuda -> mps -> cpu

## Design
1. 方法总览
本方法是一个：
多链路 × 多-aspect PFC + OOF late fusion 框架
核心思想
我们不做一个大模型，而是：
6 个 Neural 模型（3链路 × 3 aspect）
3 个 BLAST 模型（3 aspect）
OOF late fusion（每个 aspect 独立）
2. 模型结构（最核心）
2.1 链路定义（3条）
Chain 1
ESM2 last layer
Chain 2
ESM2 layer 20
Chain 3
ProtT5 last layer
2.2 每条链路 × 每个 aspect 都是一个独立模型
也就是说：
Chain	BP	MF	CC
ESM2 last	✔	✔	✔
ESM2 l20	✔	✔	✔
ProtT5	✔	✔	✔
👉 共 9 个 neural 模型
2.3 BLAST 分支
BLAST 也是：
Aspect	模型
BP	✔
MF	✔
CC	✔
👉 共 3 个 BLAST 模型
3. 输入特征设计
3.1 PLM embedding
ESM2
layer 20
last layer
ProtT5
last layer
3.2 Pooling（固定）
全部统一：
MeanPooling
MaxPooling
3.3 Protein-level feature（63维）
组成：
1️⃣ AAC（20）
2️⃣ length transforms（3）
3️⃣ charge stats（14）
4️⃣ hydrophobicity stats（14）
5️⃣ residue group fractions（12）
总维度 = 63
3.4 Feature encoder（每链路独立）
63
→ LayerNorm
→ Linear(63 → 128)
→ GELU
→ Dropout
→ Linear(128 → 256)
输出：
feature_repr: (B, 256)
4. Neural PFC 模型结构
4.1 每个模型输入
[PLM mean, PLM max, feature_repr]
4.2 维度
ESM2
2 × 1280 + 256 = 2816
ProtT5
2 × 1024 + 256 = 2304
4.3 分类头（统一）
使用你的 MLPHead：
LayerNorm
→ Linear
→ Residual block:
    LayerNorm
    GELU
    Dropout
    Linear
→ Output:
    LayerNorm
    GELU
    Dropout
    Linear → bottleneck
    LayerNorm
    GELU
    Dropout
    Linear → num_classes
4.4 输出
logits → sigmoid → p_chain_aspect
5. 标签空间（关键）
5.1 GO propagation
所有标签向上扩展
5.2 低频过滤
对每个 aspect：
count >= 20
5.3 标签空间固定
全局统一 label space
所有 fold 使用一致维度
6. 训练策略
6.1 Cross Validation
5-fold CV
6.2 每个模型独立训练
例如：
模型：ESM2-last × BP
训练：
input: esm2_last + features
output: BP labels
loss: BCEWithLogitsLoss
总共训练
3 chain × 3 aspect × 5 fold = 45 次训练
7. BLAST 分支
每个 fold × 每个 aspect
构建 DB
train_fold only
推理
对 val_fold：
top-k hits
bitscore 加权
输出
p_blast_aspect
8. OOF 构建（核心）
每个 aspect 独立
对于 BP：
收集：
oof_esm2_last_BP
oof_esm2_l20_BP
oof_prott5_BP
oof_blast_BP
oof_labels_BP
MF / CC 同理
9. Late Fusion（分两层）
🔵 第一层：Neural 内部融合
对每个 aspect
例如 BP：
p 
neural
​	
 =a 
1
​	
 p 
1
​	
 +a 
2
​	
 p 
2
​	
 +a 
3
​	
 p 
3
​	
 
其中：
p1 = ESM2 last
p2 = ESM2 l20
p3 = ProtT5
OOF 搜索
约束：
a 
1
​	
 +a 
2
​	
 +a 
3
​	
 =1
搜索：
maximize micro Fmax
🟡 第二层：Neural + BLAST 融合
p 
final
​	
 =βp 
neural
​	
 +(1−β)p 
blast
​	
 
OOF 搜索
maximize micro Fmax
10. 评估指标
主指标（敲定）
micro Fmax
辅助
macro Fmax
AUPR
AUC
Smin
用途
用途	指标
loss	BCEWithLogitsLoss
early stopping	micro Fmax
fusion	micro Fmax
final report	micro Fmax
11. 训练流程（完整）
Step 1：标签处理
propagation
min_count filtering
固定 label space
Step 2：Embedding cache（全局）
缓存：
esm2 last
esm2 l20
prott5 last
protein features
规则：
有就 load
没有才算
Step 3：训练 9 个模型（5-fold）
每个 fold：
train → train_fold
val → val_fold
保存 OOF predictions
Step 4：BLAST OOF
每个 fold：
train_fold 建 DB
val_fold 查询
Step 5：构建 OOF 数据
对每个 aspect：
oof_chain1
oof_chain2
oof_chain3
oof_blast
oof_labels
Step 6：融合学习
1️⃣ neural fusion
学：
a1, a2, a3
2️⃣ final fusion
学：
beta
Step 7：最终 CV 结果
使用：
OOF fused predictions
计算指标
12. 推理流程
输入
sequence
Neural
3个模型分别预测
BLAST
查询训练数据库
融合
p_neural = a1*p1 + a2*p2 + a3*p3
p_final = beta*p_neural + (1-beta)*p_blast
13. 关键设计优势
✔ 完全解耦
每个 chain 独立
✔ aspect-aware
BP / MF / CC 单独建模
✔ feature 有效利用
63维统计特征增强表达
✔ OOF 无泄漏融合
保证融合可靠
✔ BLAST + 深度模型互补
提升泛化能力
14. 消融实验
Chain
ESM2 last
ESM2 l20
ProtT5
Feature
无 feature
63维 feature
Fusion
单链路
三链路融合
BLAST
Label filtering
min_count = 10 / 20 / 30