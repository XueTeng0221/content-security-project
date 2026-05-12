# 项目进度文档

**项目名称**：音视频情感一致性驱动的 Deepfake 检测  
**最近更新**：2026-05-12  
**小组**：4 人小组

---

## 一、研究方向

参考方向二（视觉 Deepfake 伪造检测）与方向三（音频内容安全），聚焦于：

> **音视频情感一致性**作为 Deepfake 检测的判别信号——真实视频中人脸表情与语音情感高度对齐，而合成/换脸视频往往存在跨模态不一致。

核心参考论文：AVFF（CVPR 2024）、SpeechForensics（NeurIPS 2024）、DeepShield（ICCV 2025）。

---

## 二、数据集

| 数据集 | 用途 | 状态 |
|--------|------|------|
| MELD（Multimodal EmotionLines Dataset） | 情感一致性预训练 / 评估 | 已下载（dev/test CSV + 视频片段） |

- `data/MELD/dev_sent_emo.csv`、`data/MELD/test_sent_emo.csv` 已就位  
- 训练集视频片段路径：`data/MELD/train/train_splits/`

---

## 三、已完成工作

### 3.1 项目结构

```
src/
  data/
    meld.py          # MELDDataset：视频帧采样 + ffmpeg 音频提取 + Mel 特征
    transforms.py    # video_transform / extract_mel
  models/
    visual_encoder.py   # ViT-B/16 冻结骨干 + 线性投影 → (B,T,D)
    audio_encoder.py    # Conv1d + Transformer + AdaptivePool → (B,T_a,D)
    cross_attention.py  # 跨模态注意力（视觉 Q，音频 K/V），2 层
    detector.py         # AVDeepfakeDetector：融合 → 分类头
    ablation_models.py  # VisualOnly / AudioOnly / ConcatFusion 消融变体
  utils/
    metrics.py       # AUC/AP/EER（二分类）、Accuracy/Macro-F1（多分类）
                     # seaborn 可视化：ROC、PR、混淆矩阵、Loss 曲线
    config.py        # YAML + 命令行参数合并
  train.py           # 主训练循环（AdamW + CosineAnnealingLR）
  ablation.py        # 4 变体消融对比 + barplot 可视化
configs/
  train.yaml         # 默认超参（epochs=20, batch=8, lr=1e-4, embed_dim=256）
  ablation.yaml      # 消融专用超参（epochs=10）
run_all.sh / run_all.bat  # 一键运行训练 + 消融
```

### 3.2 模型架构

```
视频帧 (B,T,3,224,224)
    └─ ViT-B/16 (冻结) + Linear → (B,T,256)
                                        ↘
                               CrossModalAttention (2层)
                                        ↗           → mean pool → MLP → logits
音频 Mel (B,1,80,T_mel)
    └─ Conv1d + Transformer + Pool → (B,16,256)
```

### 3.3 评估指标与可视化

- **二分类**：AUC-ROC、AP、EER；输出 `roc.png`、`pr.png`、`confusion.png`
- **多分类（7 情感）**：Accuracy、Macro-F1；输出 `confusion.png`
- **训练过程**：`loss.png`（train/val loss 曲线）
- **消融**：`ablation.png`（barplot）、`ablation.csv`

---

## 四、待完成工作

- [ ] 补充 Deepfake 数据集（FaceForensics++、FakeAVCeleb）用于二分类检测评估
- [ ] 运行完整训练并记录实验结果
- [ ] 撰写项目报告（待模板下发）

---

## 五、运行方式

```bash
# 训练
python -m src.train

# 消融实验
python -m src.ablation

# 一键运行（Linux/Mac）
bash run_all.sh
```
