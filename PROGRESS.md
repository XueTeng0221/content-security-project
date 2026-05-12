# Project Progress

## 项目定位（两阶段）

**Stage 1 — MELD 情感一致性预训练**
在 MELD 情感对话数据集上，以 7 类情感分类为代理任务，训练 AV 跨模态对齐表征（视觉 Q / 音频 KV cross-attention + InfoNCE 对齐损失）。目标是让模型学会"真实视频中音视频情感应一致"的先验。

**Stage 2 — FakeAVCeleb / FF++ 二分类 fine-tune**
将 Stage 1 的骨干迁移到 Deepfake 二分类（real=0 / fake=1）。主战场：FakeAVCeleb（音视频均伪造）；辅助：FF++（纯视觉伪造，sanity check）。报告指标：AUC、AP、EER。

**核心假设（对照 Mittal et al., "Emotions Don't Lie", ACM MM 2020）**
伪造视频中音视频情感不一致（KL 散度大 / InfoNCE 对比分数低）→ 可作为检测信号。本项目以现代骨干（ViT-B/16 + Transformer AudioEncoder）重写该思路，并加入双向 cross-attention 与显式对齐损失。

---

## 已完成

- [x] 项目骨架：VisualEncoder (ViT-B/16 冻结) + AudioEncoder (4-layer Transformer) + CrossModalAttention
- [x] MELD 数据加载（Stage 1 代理任务）
- [x] Mel 频谱提取（torchaudio）
- [x] 基础训练循环 + seaborn 可视化（loss / confusion / ROC / PR）
- [x] 消融实验框架（ablation.py）
- [x] 代码审计（audit.md）：发现训练目标错位、缺一致性损失、缺 Deepfake 数据

## 进行中

- [ ] **P0** 接入 FakeAVCeleb（src/data/fakeavceleb.py）→ 跑通二分类
- [ ] **P0** 接入 FF++（src/data/ff_plus_plus.py）→ 跨操作泛化
- [ ] **P1** 加 InfoNCE 对齐损失到 detector.py
- [ ] **P1** 全局 seed + best-ckpt 保存 + AMP

## TODO

- [ ] 骨干升级：音频换 Wav2Vec2 frozen（P2）
- [ ] 跨数据集评估协议：FakeAVCeleb → DFDC（P2）
- [ ] EER 改线性插值（P2）
- [ ] 引用并量化超越 Mittal 2020（报告）
