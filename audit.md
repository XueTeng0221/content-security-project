0. 结论一句话

  当前代码不是一个 Deepfake 检测器，而是一个 MELD 7 类情感分类器。 项目命名（AVDeepfakeDetector）、文档（PR
  OGRESS.md）、二分类指标（AUC/AP/EER）与实际训练目标（num_classes=7 + CE on MELD emotion
  标签）严重错位。在补 Deepfake 数据与监督信号前，目前所有「ablation / metrics」都没有 deepfake 语义。

  ---
  1. 方法定位差距（与 SOTA 的对比与新意）

  1.1 与三篇核心论文的差距

  维度: 监督信号
  本项目: ❌ 无 real/fake 标签，只用 MELD 情感标签
  AVFF (CVPR'24): 两阶段：自监督 + CE 二分类
  SpeechForensics (NeurIPS'24): 自监督 + 二分类
  DeepShield (ICCV'25): local + global 伪造分支
  ────────────────────────────────────────
  维度: 视觉骨干
  本项目: ImageNet-pretrained ViT-B/16（冻结）
  AVFF (CVPR'24): VideoMAE ViT-B（视频 MAE 预训练）
  SpeechForensics (NeurIPS'24): AV-HuBERT 视觉支路
  DeepShield (ICCV'25): ensemble + self-blended
  ────────────────────────────────────────
  维度: 音频骨干
  本项目: 4 层 Transformer + Conv1d，从头训练
  AVFF (CVPR'24): unimodal transformer，pretrained
  SpeechForensics (NeurIPS'24): AV-HuBERT 音频支路
  DeepShield (ICCV'25): —
  ────────────────────────────────────────
  维度: 跨模态机制
  本项目: 单向 cross-attn（V←A），2 层
  AVFF (CVPR'24): A↔V 双向 contrastive (InfoNCE) + 重建 + adversarial + A2V/V2A transformer
  SpeechForensics (NeurIPS'24): 联合 AV-HuBERT 表征对齐
  DeepShield (ICCV'25): local/global dual-path
  ────────────────────────────────────────
  维度: 训练数据
  本项目: MELD 情感对话视频
  AVFF (CVPR'24): FakeAVCeleb 真实视频做 stage1
  SpeechForensics (NeurIPS'24): FF++（4 种伪造）
  DeepShield (ICCV'25): FF++ 跨操作
  ────────────────────────────────────────
  维度: 报告指标
  本项目: acc / macro-F1（情感）
  AVFF (CVPR'24): AP 99.4–99.8 / AUC 95–99（FakeAVCeleb），FF++→DFDC AUC 95.5
  SpeechForensics (NeurIPS'24): 跨域 AUC ≥95
  DeepShield (ICCV'25): SOTA on 跨操作

  关键认知：
  - AVFF 的核心是 A↔V 双向对比 + 跨模态重建；本项目的 cross-attention
  仅是单向特征融合，没有任何跨模态对齐损失。
  - SpeechForensics 的本质信号是 语音内容/音素—唇形一致性（基于
  AV-HuBERT），不是「情感一致性」。所以本项目和它在信号层面是正交的，不是重复 →
  这其实是个机会，但前提是要明确这一点。
  - DeepShield/GenD/FCG/LAA-Net/NPR 全部用 CLIP ViT + 轻量适配 或 self-blended images 作为 2025-2026
  主流；ImageNet-ViT 冻结已经是 2-3 年前路线。

  1.2 「情感一致性」并非新意 — 必须正面对照 Mittal 2020

  CLAUDE.md 完全没列入 Mittal et al., "Emotions Don't Lie: An Audio-Visual Deepfake Detection Method using
  Affective Cues" (ACM MM 2020, arXiv:2003.06711 (https://arxiv.org/abs/2003.06711))。这正是「情感不一致 →
  检测 deepfake」的开山之作；后续 ACE-Net (2025)、PST 2025
  工作也都在这一脉。任何以「情感一致性」立项的工作如果不引用并量化超越 Mittal 2020，会被一审拒。

  可立得住的差异化方向（按可行性排序）：
  1. 现代骨干替换：以 AV-HuBERT / EmoFAN / Wav2Vec2-Emotion 替换原方法的手工情感特征 → 即「Emotions Don't
  Lie 的 2026 重写」。直接、好讲。
  2. 时序情感轨迹一致性：不只比较 utterance-level 情感分布，而是在帧级建模情感轨迹的 cross-modal Dynamic
  Time Warping 距离（PST 2025 类似思路但更细）。
  3. 与音素/唇形通道的双信号融合：emotion-consistency（本项目）+ phoneme-viseme（SpeechForensics）→
  对抗相互补足，能在不同伪造类型上互补。
  4. 情感一致性 → 跨数据集泛化协议（GenD 风格）：以 FF++ 训，FakeAVCeleb / DFDC / DF-TIMIT
  测，强调情感信号的 domain-invariance。

  ---
  2. 数据集与评估方案

  2.1 致命缺口

  - MELD 不是 Deepfake 数据集（confirmed: ACL 2019, Friends 对话情感）。它没有任何伪造样本，因此
  metrics.py:18 的 evaluate_metrics（AUC/AP/EER）和 train.py:97-99 的 ROC/PR 绘图分支在 num_classes=7
  配置下永远进不去——这些代码当前是死代码。
  - outputs/ 下若已存的 confusion.png 等都是 7 类情感混淆矩阵，与 PROGRESS.md 的「Deepfake 检测」叙事不符。

  2.2 必须补的数据集（按优先级）

  数据集: FakeAVCeleb
  状态: ❌ 未下载
  用途: 主战场（音视频均伪造）
  备注: NeurIPS 2021 D&B；500 real + 19,500 fake；Wav2Lip + Faceswap；与 AVFF 同一基准
  ────────────────────────────────────────
  数据集: FaceForensics++
  状态: ❌ 未下载
  用途: 跨操作泛化
  备注: 1k real + 4k fake；无音频伪造 → 用作纯视觉支路 sanity check
  ────────────────────────────────────────
  数据集: DFDC
  状态: ❌ 未下载
  用途: 跨数据集泛化
  备注: 12 万段；视觉伪造
  ────────────────────────────────────────
  数据集: DF-TIMIT
  状态: 可选
  用途: 经典对照
  备注: 视觉伪造，规模小

  PROGRESS.md 第 80 行已经写「补充 Deepfake 数据集」是 TODO——但当前全部训练流程都依赖
  MELD，没有任何代码骨架可以无缝接 FakeAVCeleb。需要：
  - 新增 src/data/fakeavceleb.py，CSV/list 形式提供 (video_path, label∈{0,1})。
  - train.yaml 增加 dataset: fakeavceleb 字段，num_classes: 2。
  - 共用 _sample_frames + extract_mel，但要支持 25fps 同步采样以保证 A-V
  时间对齐（情感一致性必须在同一时间窗对比，否则比较的不是 alignment 是 distribution）。

  2.3 评估协议建议（对齐 SOTA 通用做法）

  - In-dataset：FakeAVCeleb 给 AUC + AP（report 双指标，因为类别极度不均衡）。
  - Cross-manipulation（FF++ 内）：训 Deepfakes，测 Face2Face / FaceSwap / NeuralTextures。
  - Cross-dataset：FakeAVCeleb → DFDC、FF++ → Celeb-DF（GenD 报告 AUC 96.4 / 86.4 可作为对标 baseline）。
  - EER：当前 compute_eer 用 argmin|fpr-fnr| 是离散近似，FPR/TPR 不连续时会偏；建议改成线性插值（参考
  sklearn 之外的 BOSARIS 风格实现）。

  ---
  3. 模型架构与实现合理性

  3.1 架构层面（按严重程度）

  P0 — 训练目标错位
  - configs/train.yaml:9 num_classes: 7 配 MELD CE，但模型类名 AVDeepfakeDetector、metrics 默认
  class_names=("Real","Fake")、PROGRESS.md 全篇按二分类叙事。两条路线必须二选一：要么把项目重命名为「AV
  情感一致性预训练 → 迁移到 deepfake 二分类」（两阶段），要么直接训二分类，丢掉情感监督。

  P1 — 缺一致性损失
  - 既然命题是「情感一致性」，模型必须显式建模一致性度量。当前只是 cross-attn 后 mean pool 进 MLP，等于把 A
   和 V 当一段共享特征，没有任何「比较」操作。建议加入：
    - 双塔 + InfoNCE：L_align = -log exp(sim(v,a)/τ) / Σ exp(sim(v,a')/τ)（同 AVFF 双向）。
    - 或情感分布 KL：L_consist = KL(p_emo_visual || p_emo_audio)，真实样本 KL 应小、伪造样本 KL 应大。

  P1 — 单向 cross-attn
  - cross_attention.py:35 仅 visual Q / audio KV。AVFF 双向（A2V +
  V2A）才能捕获两个方向的对齐失真。建议改成对称设计或加并行的 audio→visual 分支。

  P2 — 骨干选择落后
  - ImageNet ViT-B/16 冻结：face/lip 域差距大，情感分类相关的人脸特征会被「类别中心」噪声化。
    - 视觉建议：open_clip 的 ViT-B/16 + LayerNorm tuning（参考 GenD），或 EmoFAN / FaceXFormer
  这类人脸专用骨干。
    - 音频建议：用 HuggingFace facebook/wav2vec2-base 或 facebook/hubert-base-ls960 替换从头训练的 4 层
  Transformer。MELD 训练集只有几千条，从头训 4 层 Transformer 严重欠拟合。

  P2 — ViT 部分未冻结的层不一致
  - visual_encoder.py:18-20 只冻 patch_embed 和 encoder 的参数，但 class_token、pos_embed、ln 都默认
  requires_grad=True，这通常不是预期。建议：for p in vit.parameters(): p.requires_grad_(False) 后再单独打开
   proj。

  3.2 训练流程（可复现性 / 工程）

  问题: 全局未设随机种子
  位置: 整个 repo
  严重度: P0
  修复: train.py 头部加 torch.manual_seed(cfg.seed); np.random.seed(...);
    torch.use_deterministic_algorithms(True)
  ────────────────────────────────────────
  问题: ffmpeg 子进程 + 临时文件按样本调用
  位置: meld.py:75-87
  严重度: P1
  修复: I/O 是瓶颈；要么预解码一次写 wav 缓存，要么用 torchaudio.io 直读 MP4 音轨
  ────────────────────────────────────────
  问题: 每 epoch 全量 ckpt
  位置: train.py:75
  严重度: P1
  修复: 改 best-on-val 一份 + last 一份；20 epoch × ~500 MB 会爆盘
  ────────────────────────────────────────
  问题: 无 AMP / grad clipping
  位置: train.py
  严重度: P2
  修复: torch.cuda.amp.autocast + GradScaler；ViT-B 训 batch=8 在单卡上是冒险
  ────────────────────────────────────────
  问题: run_all.sh 空文件、run_all.bat 注释掉 tests
  位置: 根目录
  严重度: P2
  修复: 至少跑 pytest tests/ -q 再训
  ────────────────────────────────────────
  问题: ablation 4 个 variant 共用单训练循环但无相同种子、共用 dataloader 顺序但 shuffle=True 导致每
  variant
    看到不同顺序
  位置: ablation.py:37
  严重度: P2
  修复: 每 variant 前重置 seed
  ────────────────────────────────────────
  问题: compute_eer 离散近似
  位置: metrics.py:11-15
  严重度: P2
  修复: 用插值法或 pyeer
  ────────────────────────────────────────
  问题: config.py:11 用 type(v) 推断类型，v=None fallback 到 str 但实际 _DEFAULTS 中的 lr=1e-4 类型是 float

    — OK；但 freeze_visual: true 经过 yaml 是 bool，CLI 转 type=bool 永远是 True（任何非空字符串 bool 都是
    True）
  位置: config.py:11
  严重度: P2
  修复: bool 字段单独处理 --no-freeze-visual flag

  3.3 测试覆盖

  - tests/test_models.py 仅 shape 检查，未验证：(a) gradient 能反传到 trainable 参数，(b) frozen 参数确实
  requires_grad=False，(c) 双 batch / 单 batch 一致性。建议加
  test_frozen_params_have_no_grad、test_gradient_flow_to_classifier_and_proj。

  ---
  4. 优先修复清单（按价值/成本排序）

  1. P0 — 把项目定位讲清楚：在 PROGRESS.md / CLAUDE.md 里明确两阶段叙事：「Stage 1：MELD 上预训练 AV
  情感一致性表征 / Stage 2：FakeAVCeleb 上 fine-tune 二分类」。否则后续所有改动都名不正言不顺。
  2. P0 — 接入 FakeAVCeleb：写 src/data/fakeavceleb.py，跑通一次二分类训练，输出真正有意义的 AUC/AP。
  3. P1 — 加情感一致性损失：双塔 InfoNCE 或 KL；这是项目「新意」的物理载体。
  4. P1 — 全局 seed + best ckpt + AMP。
  5. P1 — 引用并对比 Mittal 2020 "Emotions Don't Lie"：在报告/CLAUDE.md 里补上，否则差异化讲不通。
  6. P2 — 骨干升级：先把音频换成 Wav2Vec2 frozen，对小数据增益最大。
  7. P2 — 跨数据集协议：FakeAVCeleb→DFDC，对齐 GenD/AVFF 的 cross-dataset 评估栏。

  ---
  5. 高信号文献对照表（建议加入 CLAUDE.md）

  缺失论文: Mittal et al., "Emotions Don't Lie", ACM MM 2020
  重要性: ★★★ 必加
  用途: 情感一致性 deepfake 检测开山，必须对照
  ────────────────────────────────────────
  缺失论文: Hosler et al., "Do Deepfakes Feel Emotions?", CVPRW 2021
  重要性: ★★
  用途: 情感方向延续
  ────────────────────────────────────────
  缺失论文: AV-HuBERT (Shi et al., ICLR 2022)
  重要性: ★★
  用途: 现代 AV 自监督骨干，SpeechForensics 的基础
  ────────────────────────────────────────
  缺失论文: RealForensics (Haliassos et al., CVPR 2022)
  重要性: ★★
  用途: SpeechForensics 直接对照基线

  ---
  过程说明

  - 4 个并行 haiku 子代理共耗 ~12 分钟、~230k
  tokens；本会话上下文里只保留了它们返回的精炼摘要，原始网页未进入主上下文。
  - 未触碰 CLAUDE.md 文本方向（方向一）的 30 篇文献，按你的「同方向 ~10 篇」选择只覆盖视觉/音频 deepfake +
  AV 融合脉络。
  - 没有做的事：跑 pytest、跑 train.py、下载任何模型权重 /
  数据集——审计纯静态。需要我接着做这些任一项请明示。

● 审计完成。最关键的发现：当前 repo 与「Deepfake 检测」名义不符——训练目标是 MELD 7
  类情感分类，整套二分类指标（AUC/AP/EER）是死代码，且未引用情感一致性 deepfake 检测的开山论文 Mittal
  2020。下一步建议先按 §4 的 P0/P1 顺序处理。