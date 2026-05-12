# 2026 信息内容安全 Project 参考文献清单（统一格式版）

最近更新时间：2026.5.11

## CLAUDE 的任务

你的任务是以【4 人小组】为基础：

- 调研最新相关文献（可参考以下资源获取渠道/参考论文清单）并阅读相关内容
- 面向本次信息内容安全 Project，针对上述文献工作在某些方面（如架构/算法）的局限性，以 Pytorch 为准：
  - 基于以上方案，确定可能用到的公开数据集/预训练模型（如果需要自主额外制作数据集，欢迎告诉我）
  - 确定项目结构，并分阶段开始实现
  - 在实现过程中，对于必需的评估指标，请做出 seaborn 可视化
- 在项目过程中，请在项目文档中记录你的工作进度，并定期提交项目报告（等报告的模板下发时，我会告诉你）。

## 一、高质量会议期刊

1. 信息安全四大顶会：IEEE S&P、ACM CCS、USENIX Security、NDSS。

2. 人工智能 / 信息检索：NeurIPS、IJCAI、AAAI、ICLR、ICML、SIGIR。

3. 计算机视觉：CVPR、ECCV、ICCV、TPAMI（期刊）。

4. 语音与音频：ICASSP、Interspeech。

5. 自然语言处理：ACL、EMNLP、COLING。

6. 中文期刊：《计算机学报》《计算机研究与发展》《信息安全学报》《中国图形图像学报》《网络与信息安全学报》。

## 二、相关资源获取

### 1. 学术搜索网站

• Google Scholar：https://scholar.google.com/

• DBLP：https://dblp.org

• Semantic Scholar：https://www.semanticscholar.org/

• Microsoft Academic：https://academic.microsoft.com/

• AMiner：https://www.aminer.org/

### 2. 文献下载平台

• Sci-Hub：https://sci-hub.tw/

• Sci-Hub 镜像：https://tool.yovisun.com/scihub/

• Sci-Hub 镜像：http://www.sci-hub.ac.cn/

### 3. 代码获取方式

• 作者个人主页。

• Papers With Code：https://paperswithcode.com/

• 代码托管平台：GitHub（https://github.com/）、SourceForge（https://sourceforge.net/）、码云（https://gitee.com/）。

• 直接向作者发邮件索要。

## 三、参考论文清单

说明

1. 以下论文仅供参考，选题不限于这些工作。
2. 建议优先选择近三年（2024-2026）的高质量论文，以便了解前沿问题、最新技术和优秀论文写法。
3. 2026 年论文可能暂未公开代码，可根据其参考文献继续追踪相关论文和代码。
4. 本版统一为同一条目格式：题名 来源/年份 方向/备注 代码。其中”代码：未明确公开”表示原清单中未核验到公开官方代码；“代码：无”表示原清单明确标注无代码。

# 方向一：文本内容安全

## 1.1 虚假信息检测

T1-01. Bad Actor, Good Advisor: Exploring the Role of Large Language Models in Fake News Detection
来源/年份：AAAI 2024。
论文链接：[AAAI 页面](https://ojs.aaai.org/index.php/AAAI/article/view/30214)；[arXiv](https://arxiv.org/abs/2309.12247)。
方向/备注：将 LLM 作为 advisor 辅助 fake news detection。
代码链接：[ARG](https://github.com/ictmcg/arg)。

T1-02. On Fake News Detection with LLM Enhanced Semantics Mining
来源/年份：EMNLP 2024。
论文链接：[ACL Anthology](https://aclanthology.org/2024.emnlp-main.31/)。
方向/备注：研究 LLM-enhanced semantics 用于 fake news detection。
代码链接：[LESS4FD](https://github.com/LESS4FD/LESS4FD)。

T1-03. Unified Evidence Enhancement Inference Framework for Fake News Detection
来源/年份：IJCAI 2024。
论文链接：[IJCAI 页面](https://www.ijcai.org/proceedings/2024/0723)；[PDF](https://www.ijcai.org/proceedings/2024/0723.pdf)。
方向/备注：evidence-aware / inference 框架。
代码链接：未明确公开。

T1-04. Natural Language-centered Inference Network for Multi-modal Fake News Detection
来源/年份：IJCAI 2024。
论文链接：[IJCAI 页面](https://www.ijcai.org/proceedings/2024/281)。
方向/备注：多模态 fake news detection。
代码链接：[NLIN](https://github.com/juices6/NLIN)。

T1-05. Generate First, Then Sample: Enhancing Fake News Detection with LLM-Augmented Reinforced Sampling
来源/年份：ACL 2025。
论文链接：[ACL Anthology](https://aclanthology.org/2025.acl-long.1182/)。
方向/备注：先用 LLM 生成，再通过强化采样增强 detector。
代码链接：作者主页提供 code 入口（[作者主页](https://gymbeijing.github.io/)）。

T1-06. DRES: Fake News Detection by Dynamic Representation and Ensemble Selection
来源/年份：EMNLP 2025。
论文链接：[ACL Anthology](https://aclanthology.org/2025.emnlp-main.1013/)；[arXiv](https://arxiv.org/abs/2509.16893)。
方向/备注：动态表示选择 + 动态集成。
代码链接：[FakeNewsDetection_DRES](https://github.com/FFarhangian/FakeNewsDetection_DRES)。

T1-07. Cross-Domain Fake News Detection based on Dual-Granularity Adversarial Training
来源/年份：COLING 2025。
论文链接：[ACL Anthology](https://aclanthology.org/2025.coling-main.631/)；[PDF](https://aclanthology.org/2025.coling-main.631.pdf)。
方向/备注：跨域 fake news detection。
代码链接：未明确公开。

T1-08. Robust Misinformation Detection by Visiting Potential Commonsense Conflict
来源/年份：IJCAI 2025。
论文链接：[IJCAI 页面](https://www.ijcai.org/proceedings/2025/863)；[arXiv](https://arxiv.org/abs/2504.21604)。
方向/备注：用 commonsense conflict 强化 misinformation detection。
代码链接：[MD-PCC](https://github.com/wangbing1416/MD-PCC)。

T1-09. Reasoning About the Unsaid: Misinformation Detection with Omission-Aware Graph Inference
来源/年份：AAAI 2026。
论文链接：[AAAI 页面](https://ojs.aaai.org/index.php/AAAI/article/view/37097)；[arXiv](https://arxiv.org/abs/2512.01728)。
方向/备注：关注“省略 / 遗漏型”误导信息。
代码链接：[OmiGraph](https://github.com/ICTMCG/OmiGraph)。

## 1.2 文本生成检测

T1-10. MAGE: Machine-generated Text Detection in the Wild
来源/年份：ACL 2024。
论文链接：[ACL Anthology](https://aclanthology.org/2024.acl-long.3/)；[arXiv](https://arxiv.org/abs/2305.13242)。
方向/备注：代表性的 machine-generated text detection 基准 / 测试床论文。
代码链接：[MAGE](https://github.com/yafuly/MAGE)。

T1-11. M4GT-Bench: Evaluation Benchmark for Black-Box Machine-Generated Text Detection
来源/年份：ACL 2024。
论文链接：[ACL Anthology](https://aclanthology.org/2024.acl-long.218/)；[arXiv](https://arxiv.org/abs/2402.11175)。
方向/备注：多语言、多领域、多生成器 benchmark。
代码链接：[M4GT-Bench](https://github.com/mbzuai-nlp/M4GT-Bench)。

T1-12. BiScope: AI-generated Text Detection by Checking Memorization of Preceding Tokens
来源/年份：NeurIPS 2024。
论文链接：[NeurIPS 页面](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bc808cf2d2444b0abcceca366b771389-Abstract-Conference.html)。
方向/备注：双向利用 token 预测 / 记忆信号做 AI-generated text detection。
代码链接：[BiScope](https://github.com/MarkGHX/BiScope)。

T1-13. DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios
来源/年份：NeurIPS 2024 Datasets & Benchmarks。
论文链接：[NeurIPS 页面](https://papers.nips.cc/paper/2024/hash/b61bdf7e9f64c04ec75a26e781e2ad51-Abstract-Datasets_and_Benchmarks_Track.html)。
方向/备注：面向真实场景的 LLM-generated text detection benchmark。
代码链接：[DetectRL](https://github.com/junchaoIU/DetectRL)。

T1-14. DPIC: Decoupling Prompt and Intrinsic Characteristics for LLM Generated Text Detection
来源/年份：NeurIPS 2024。
论文链接：[NeurIPS 页面](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1d35af80e775e342f4cd3792e4405837-Abstract-Conference.html)；[OpenReview](https://openreview.net/forum?id=BZh05P2EoN)；[arXiv](https://arxiv.org/abs/2305.12519)。
方向/备注：强调将 prompt 因素和模型内在生成特征解耦。
代码链接：未明确公开。

T1-15. Who Wrote This? The Key to Zero-Shot LLM-Generated Text Detection Is GECScore
来源/年份：COLING 2025。
论文链接：[ACL Anthology](https://aclanthology.org/2025.coling-main.684/)；[arXiv](https://arxiv.org/abs/2405.04286)。
方向/备注：零样本 LLM-generated text detection 代表作之一。
代码链接：[GECScore](https://github.com/junchaoIU/GECScore)。

T1-16. MoSEs: Uncertainty-Aware AI-Generated Text Detection via Mixture of Stylistics Experts with Conditional Thresholds
来源/年份：EMNLP 2025。
论文链接：[ACL Anthology](https://aclanthology.org/2025.emnlp-main.294/)；[arXiv](https://arxiv.org/abs/2509.02499)。
方向/备注：风格专家混合 + 不确定性估计。
代码链接：[MoSEs](https://github.com/creator-xi/MoSEs)。

T1-17. Machine-generated text detection prevents language model collapse
来源/年份：EMNLP 2025。
论文链接：[ACL Anthology](https://aclanthology.org/2025.emnlp-main.1506/)；[arXiv](https://arxiv.org/html/2502.15654v1)。
方向/备注：将文本生成检测与 model collapse 问题联系起来。
代码链接：[model_collapse](https://github.com/GeorgeDrayson/model_collapse)。

T1-18. SpecDetect: Simple, Fast, and Training-Free Detection of LLM-Generated Text via Spectral Analysis
来源/年份：AAAI 2026。
论文链接：[AAAI 页面](https://ojs.aaai.org/index.php/AAAI/article/view/40510)；[arXiv](https://arxiv.org/abs/2508.11343)。
方向/备注：训练自由、频谱分析路线。
代码链接：[SpecDetect](https://github.com/luohaitong/SpecDetect)。

T1-19. MGT-Prism: Enhancing Domain Generalization for Machine-Generated Text Detection via Spectral Alignment
来源/年份：AAAI 2026。
论文链接：[AAAI 页面](https://ojs.aaai.org/index.php/AAAI/article/view/40485)；[arXiv](https://arxiv.org/abs/2508.13768)。
方向/备注：强调 domain generalization。
代码链接：未明确公开。

## 1.3 文本对抗样本生成及检测

T1-20. COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability
来源/年份：ICML 2024。
论文链接：[arXiv](https://arxiv.org/abs/2402.08679)。
方向/备注：代表性的 jailbreak 生成方法，强调 stealthiness 与 controllability。
代码链接：[COLD-Attack](https://github.com/Yu-Fangxu/COLD-Attack)。

T1-21. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models
来源/年份：NeurIPS 2024 Datasets & Benchmarks。
论文链接：[arXiv](https://arxiv.org/abs/2404.01318)；[项目主页](https://jailbreakbench.github.io/)。
方向/备注：兼顾攻击与防御评测的 jailbreak benchmark。
代码链接：[jailbreakbench](https://github.com/JailbreakBench/jailbreakbench)。

T1-22. Mission Impossible: A Statistical Perspective on Jailbreaking LLMs
来源/年份：NeurIPS 2024。
论文链接：[NeurIPS 页面](https://proceedings.neurips.cc/paper_files/paper/2024/hash/439bf902de1807088d8b731ca20b0777-Abstract-Conference.html)；[arXiv](https://arxiv.org/abs/2408.01420)。
方向/备注：从理论 / 统计视角分析 jailbreak。
代码链接：未明确公开。

T1-23. Fight Back Against Jailbreaking via Prompt Adversarial Tuning
来源/年份：NeurIPS 2024。
论文链接：[NeurIPS 页面](https://proceedings.neurips.cc/paper_files/paper/2024/hash/759ca99a82e2a9137c6bef4811c8d378-Abstract-Conference.html)。
方向/备注：代表性的 prompt adversarial tuning 防御方法。
代码链接：[PAT](https://github.com/PKU-ML/PAT)。

T1-24. AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases
来源/年份：NeurIPS 2024。
论文链接：[NeurIPS 页面](https://proceedings.neurips.cc/paper_files/paper/2024/hash/eb113910e9c3f6242541c1652e30dfd6-Abstract-Conference.html)；[arXiv](https://arxiv.org/abs/2407.12784)；[项目主页](https://billchan226.github.io/AgentPoison)。
方向/备注：将对抗从单轮 prompt 扩展到 agent memory / KB poisoning。
代码链接：[AgentPoison](https://github.com/AI-secure/AgentPoison)。

T1-25. Can Indirect Prompt Injection Attacks Be Detected and Removed?
来源/年份：ACL 2025。
论文链接：[ACL Anthology PDF](https://aclanthology.org/2025.acl-long.890.pdf)；[arXiv](https://arxiv.org/abs/2502.16580)。
方向/备注：贴合“检测 + 清除”间接 prompt injection 的方向。
代码链接：[indirect-pia-detection](https://github.com/LukeChen-go/indirect-pia-detection)。

T1-26. Defense Against Prompt Injection Attack by Leveraging Attack Techniques
来源/年份：ACL 2025。
论文链接：[ACL Anthology](https://aclanthology.org/2025.acl-long.897/)；[arXiv PDF](https://arxiv.org/pdf/2411.00459v1)。
方向/备注：把 attack 技术反过来用于 defense。
代码链接：[pia-defense-by-attack](https://github.com/LukeChen-go/pia-defense-by-attack)。

T1-27. Weak-to-Strong Jailbreaking on Large Language Models
来源/年份：ICML 2025。
论文链接：[OpenReview](https://openreview.net/forum?id=7DXaCYUvDN)；[arXiv](https://arxiv.org/abs/2401.17256)。
方向/备注：有影响力的 weak-to-strong jailbreak 方法。
代码链接：[weak-to-strong](https://github.com/XuandongZhao/weak-to-strong)。

T1-28. TopicAttack: An Indirect Prompt Injection Attack via Topic Transition
来源/年份：EMNLP 2025。
论文链接：[ACL Anthology](https://aclanthology.org/2025.emnlp-main.372/)；[arXiv](https://arxiv.org/abs/2507.13686)。
方向/备注：基于 topic transition 的间接 prompt injection 攻击。
代码链接：[topicattack](https://github.com/LukeChen-go/topicattack)。

T1-29. Attacking Misinformation Detection Using Adversarial Examples Generated by Language Models
来源/年份：EMNLP 2025。
论文链接：[ACL Anthology](https://aclanthology.org/2025.emnlp-main.1405/)；[arXiv](https://arxiv.org/abs/2410.20940)。
方向/备注：直接针对 misinformation detector 的对抗攻击，方法名 TREPAT。
代码链接：原清单标注“有官方代码 trepat”，但本轮未检索到可核验的官方仓库链接。

# 方向二：视频/图像内容安全

## 2.1 视觉 Deepfake（合成）技术

V2-01. Baliah S, et al. VFace: A Training-Free Approach for Diffusion-Based Video Face Swapping
来源/年份：WACV 2026。
方向/备注：training-free 视频 face swapping。
代码：https://github.com/Sanoojan/VFace

V2-02. Zhang X, et al. X-NeMo: Expressive Neural Motion Reenactment via Disentangled Latent Attention
来源/年份：ICLR 2025。
方向/备注：表情/动作驱动的人脸重演。
代码：https://github.com/bytedance/x-nemo-inference

V2-03. Ki J, et al. FLOAT: Generative Motion Latent Flow Matching for Audio-driven Talking Portrait
来源/年份：ICCV 2025。
方向/备注：audio-driven talking portrait，flow matching 路线。
代码：https://github.com/deepbrainai-research/float

V2-04. Li Y, et al. InsTaG: Learning Personalized 3D Talking Head from Few-Second Video
来源/年份：CVPR 2025。
方向/备注：few-shot personalized 3D talking head。
代码：https://github.com/Fictionarry/InsTaG

V2-05. Bigata D, et al. KeyFace: Expressive Audio-Driven Facial Animation for Long Sequences via KeyFrame Interpolation
来源/年份：CVPR 2025。
方向/备注：长序列表情驱动 talking face。
代码：https://github.com/antonibigata/keyface_cvpr

V2-06. Luo Y, et al. CanonSwap: High-Fidelity and Consistent Video Face Swapping via Canonical Space Modulation
来源/年份：ICCV 2025。
方向/备注：高保真且时序一致的视频 face swapping。
代码：https://github.com/Pixel-Talk/CanonSwap

V2-07. Baliah S, et al. Realistic and Efficient Face Swapping: A Unified Approach with Diffusion Models
来源/年份：WACV 2025。
方向/备注：基于 diffusion 的高效 face swapping。
代码：https://github.com/sanoojan/reface

V2-08. Luo/相关作者团队. VividFace: A Diffusion-Based Hybrid Framework for High-Fidelity Video Face Swapping
来源/年份：NeurIPS 2025。
方向/备注：高保真视频换脸。
代码：https://github.com/deepcs233/VividFace

V2-09. Peng Z, Hu W, Shi Y, et al. SyncTalk: The Devil is in the Synchronization for Talking Head Synthesis
来源/年份：CVPR 2024: 666-676。
方向/备注：talking head 合成。
代码：https://github.com/ZiqiaoPeng/SyncTalk

V2-10. Mitra A, Mohanty S P, Kougianos E. The World of Generative AI: Deepfakes and Large Language Models
来源/年份：arXiv preprint arXiv:2402.04373, 2024。
方向/备注：综述。
代码：无。

V2-11. Ye R, Li Z, Huang C, et al. Real3D-Portrait: One-shot Realistic 3D Talking Portrait Synthesis
来源/年份：ICLR 2024。
方向/备注：one-shot 3D talking portrait 合成。
代码：https://github.com/yerfor/Real3DPortrait

V2-12. Xu S, Li M, et al. MimicTalk: Mimicking a Personalized and Expressive 3D Talking Face in Minutes
来源/年份：NeurIPS 2024。
方向/备注：个性化 3D talking face 快速建模。
代码：https://github.com/yerfor/MimicTalk/

## 2.2 视觉 Deepfake 伪造检测

V2-13. Deepfake Detection that Generalizes Across Benchmarks
来源/年份：WACV 2026。
方向/备注：跨 benchmark 泛化的 deepfake 检测。
代码：https://github.com/yermandy/GenD

V2-14. Li M, Ahmadiadli Y, Zhang X P. A Survey on Speech Deepfake Detection
来源/年份：ACM Computing Surveys, 2025。
方向/备注：综述。
代码：无。

V2-15. DeepFake-Adapter: Dual-Level Adapter for DeepFake Detection
来源/年份：International Journal of Computer Vision (IJCV), 2025。
方向/备注：adapter-based deepfake detection。
代码：https://github.com/rshaojimmy/DeepFake-Adapter

V2-16. Towards More General Video-based Deepfake Detection through Facial Component Guided Adaptation for Foundation Model
来源/年份：CVPR 2025。
方向/备注：面向 foundation model 的更通用视频 deepfake 检测。
代码：https://github.com/aiiu-lab/DFD-FCG

V2-17. DeepShield: Fortifying Deepfake Video Detection with Local and Global Forgery Analysis
来源/年份：ICCV 2025。
方向/备注：结合局部与全局伪造分析的视频 deepfake 检测。
代码：https://github.com/lijichang/DeepShield

V2-18. Gambín Á F, Yazidi A, Vasilakos A, et al. Deepfakes: Current and Future Trends
来源/年份：Artificial Intelligence Review, 2024, 57(3): 64。
方向/备注：综述。
代码：无。

V2-19. Oorloff T, Koppisetti S, Bonettini N, et al. AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection
来源/年份：CVPR 2024: 27102-27112。
方向/备注：音视频特征融合的视频伪造检测。
代码：https://github.com/JoeLeelyf/OpenAVFF

V2-20. Ba Z, Liu Q, Liu Z, et al. Exposing the Deception: Uncovering More Forgery Clues for Deepfake Detection
来源/年份：AAAI 2024, 38(2): 719-728。
方向/备注：通过更多伪造线索进行 deepfake 检测。
代码：https://github.com/QingyuLiu/Exposing-the-Deception

V2-21. Peng C, Miao Z, Liu D, et al. Where Deepfakes Gaze at? Spatial-Temporal Gaze Inconsistency Analysis for Video Face Forgery Detection
来源/年份：IEEE Transactions on Information Forensics and Security (TIFS), 2024。
方向/备注：基于 gaze inconsistency 的视频伪造检测。
代码：https://github.com/ziminMIAO/DFGaze

V2-22. Zhang/相关作者团队. LAA-Net: Localized Artifact Attention Network for Quality-Agnostic and Generalizable Deepfake Detection
来源/年份：CVPR 2024。
方向/备注：localized artifact attention，泛化 deepfake 检测。
代码：https://github.com/10Ring/LAA-Net

V2-23. Tan/相关作者团队. Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection
来源/年份：CVPR 2024。
方向/备注：利用生成网络上采样痕迹进行泛化检测。
代码：https://github.com/chuangchuangtan/NPR-DeepfakeDetection

V2-24. SpeechForensics: Audio-Visual Speech Representation Learning for Face Forgery Detection
来源/年份：NeurIPS 2024。
方向/备注：音视频 speech representation learning 用于 face forgery detection。
代码：https://github.com/Eleven4AI/SpeechForensics

## 2.3 视觉对抗样本及检测

V2-25. Cheng K, Zhang J, Li H, Zhong Z, Zhang M, Qin Z. PPOM-Attack: A Substitute Model-free Perturbation Prediction and Optimization Method for Black-box Adversarial Attack against Face Recognition
来源/年份：IEEE Transactions on Information Forensics and Security, 2026。
方向/备注：无需替代模型的黑盒人脸识别对抗攻击。
代码：无。

V2-26. Peng F, Liu Y, Zhou G, Long M. AdvDiffusion: Adversarial Patches Generation for Face Recognition with High Transferability in Physical Domain
来源/年份：IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026。
方向/备注：面向物理场景的人脸识别高迁移性对抗补丁生成。
代码：无。

V2-27. Ma H, Jiang X, Xu K, Sun T. PSO-based Black-box Adversarial Patch Attack against Face Recognition
来源/年份：IEEE Transactions on Circuits and Systems for Video Technology, 2026。
方向/备注：基于粒子群优化的黑盒人脸识别对抗补丁攻击。
代码：无。

V2-28. Yang X, Xu L, Pang T, Dong Y, Wang Y, Su H, Zhu J. Face3DAdv: Exploiting Robust Adversarial 3D Patches on Physical Face Recognition
来源/年份：International Journal of Computer Vision, 2025, 133(1): 353-371。
方向/备注：面向物理人脸识别的鲁棒 3D 对抗补丁攻击。
代码：无。

V2-29. Liu Y, Wei H, Jia C, Xiao R, Ruan W, Wei X, Zhou J T, Wang Z. ProjAttacker: A Configurable Physical Adversarial Attack for Face Recognition via Projector
来源/年份：CVPR 2025: 21248-21257。
方向/备注：基于投影仪的可配置物理人脸识别对抗攻击。
代码：无。

V2-30. Zhou F, Yin B, Ling H, Zhou Q, Wang W. Improving the Transferability of Adversarial Attacks on Face Recognition with Diverse Parameters Augmentation
来源/年份：CVPR 2025。
方向/备注：通过多样参数增强提升人脸识别对抗攻击迁移性。
代码：https://github.com/fengfanzhou/DPA

V2-31. Ren M, Wang Y, Zhu Y, Huang Y, Sun Z, Li Q, Tan T. Artificial Immune System of Secure Face Recognition Against Adversarial Attacks
来源/年份：International Journal of Computer Vision, 2024, 132(12): 5718-5740。
方向/备注：面向开放集人脸识别的对抗防御方法。
代码：https://github.com/RenMin1991/SIDE

V2-32. Liu D, Wang X, Peng C, Wang N, Hu R, Gao X. Adv-Diffusion: Imperceptible Adversarial Face Identity Attack via Latent Diffusion Model
来源/年份：AAAI 2024, 38(4): 3585-3593。
方向/备注：基于潜空间扩散模型的不可感知人脸身份对抗攻击。
代码：https://github.com/kopperx/Adv-Diffusion

V2-33. Zhang Q, Guo Q, Gao R, Juefei-Xu F, Yu H, Feng W. Adversarial Relighting Against Face Recognition
来源/年份：IEEE Transactions on Information Forensics and Security, 2024, 19: 9145-9157。
方向/备注：通过对抗性光照变化攻击人脸识别。
代码：无。

V2-34. Hu C, Li Y, Feng Z, Wu X. Toward Transferable Attack via Adversarial Diffusion in Face Recognition
来源/年份：IEEE Transactions on Information Forensics and Security, 2024。
方向/备注：基于扩散模型的人脸识别可迁移对抗攻击。
代码：无。

V2-35. Wu Z, Cheng Y, Zhang S, Ji X, Xu W. UniID: Spoofing Face Authentication System by Universal Identity
来源/年份：NDSS 2024。
方向/备注：通过通用身份实现的人脸认证欺骗攻击。
代码：无。

V2-36. Li M, Wang J, Zhang H, Zhou Z, Hu S, Pei X. Transferable Adversarial Facial Images for Privacy Protection
来源/年份：ACM Multimedia 2024: 10649-10658。
方向/备注：用于隐私保护的可迁移人脸对抗样本生成。
代码：无。

# 方向三：音频内容安全

## 3.1 语音合成 / 语音转换

A3-01. Qi T, Wang S, Lu C, et al. PromptEVC: Controllable Emotional Voice Conversion with Natural Language Prompts
来源/年份：Interspeech 2025: 4588-4592。
方向/备注：语音转换。
代码：未标注。

A3-02. Li F, Wang J, Niu Y, et al. StarVC: A Unified Auto-Regressive Framework for Joint Text and Speech Generation in Voice Conversion
来源/年份：Interspeech 2025: 4593-4597。
方向/备注：语音转换。
代码：未标注。

A3-03. Kaneko T, Kameoka H, Tanaka K, et al. FasterVoiceGrad: Faster One-step Diffusion-Based Voice Conversion with Adversarial Diffusion Conversion Distillation
来源/年份：Interspeech 2025: 4598-4602。
方向/备注：语音转换。
代码：未标注。

A3-04. Lobashev A, Yermekova A, Larchenko M. Training-Free Voice Conversion with Factorized Optimal Transport
来源/年份：Interspeech 2025: 1373-1377。
方向/备注：语音转换。
代码：未标注。

A3-05. Wang K, Guan W, Jiang Z, et al. Discl-VC: Disentangled Discrete Tokens and In-Context Learning for Controllable Zero-Shot Voice Conversion
来源/年份：Interspeech 2025: 1383-1387。
方向/备注：语音转换。
代码：未标注。

A3-06. Ren P, Guan W, Wang K, et al. ReFlow-VC: Zero-shot Voice Conversion Based on Rectified Flow and Speaker Feature Optimization
来源/年份：Interspeech 2025: 1388-1392。
方向/备注：语音转换。
代码：未标注。

A3-07. Jin J, Yang Z, Zhou Y, et al. In This Environment, As That Speaker: A Text-Driven Framework for Multi-Attribute Speech Conversion
来源/年份：Interspeech 2025: 1393-1397。
方向/备注：语音转换。
代码：未标注。

A3-08. Xing J, Li Z, Chen S, et al. EATS-Speech: Emotion-Adaptive Transformation and Priority Synthesis for Zero-Shot Text-to-Speech
来源/年份：Interspeech 2025: 4358-4362。
方向/备注：语音合成。
代码：未标注。

A3-09. Cho D H, Oh H S, Kim S B, et al. DiEmo-TTS: Disentangled Emotion Representations via Self-Supervised Distillation for Cross-Speaker Emotion Transfer in Text-to-Speech
来源/年份：Interspeech 2025: 4373-4377。
方向/备注：语音合成。
代码：未标注。

A3-10. Li X, Xing J, Xing X, et al. SA-RAS: Speaker-Aware Style Retrieval Augmented Generation for Expressive Zero-Shot Text-to-Speech Synthesis
来源/年份：Interspeech 2025: 4388-4392。
方向/备注：语音合成。
代码：未标注。

## 3.2 语音伪造检测

A3-11. Yan Z, Zhao Y, Wang H. VoiceWukong: Benchmarking Deepfake Voice Detection
来源/年份：USENIX Security Symposium 2025。
方向/备注：语音伪造检测。
代码：未标注。

A3-12. Liu Z, Chen A, Zhang Y, et al. Rethinking Fake Speech Detection: A Generalized Framework Leveraging Spectrogram Magnitude
来源/年份：NDSS Symposium 2026。
方向/备注：语音伪造检测。
代码：未标注。

A3-13. Kumari K, Abbasihafshejani M, Pegoraro A, et al. VoiceRadar: Voice Deepfake Detection using Micro-Frequency and Compositional Analysis
来源/年份：NDSS Symposium 2026。
方向/备注：语音伪造检测。
代码：未标注。

A3-14. Kwok C Y, Yip J Q, Qiu Z, et al. Bona fide Cross Testing Reveals Weak Spot in Audio Deepfake Detection Systems
来源/年份：Interspeech 2025: 2230-2234。
方向/备注：语音伪造检测。
代码：未标注。

A3-15. El Kheir Y, Polzehl T, Möller S. BiCrossMamba-ST: Speech Deepfake Detection with Bidirectional Mamba Spectro-Temporal Cross-Attention
来源/年份：Interspeech 2025: 2235-2239。
方向/备注：语音伪造检测。
代码：未标注。

A3-16. Glazer N, Chernin D, Achituve I, et al. Few-Shot Speech Deepfake Detection Adaptation with Gaussian Processes
来源/年份：Interspeech 2025: 2240-2244。
方向/备注：语音伪造检测。
代码：未标注。

A3-17. Müller N, Kawa P, Choong W H, et al. Replay Attacks Against Audio Deepfake Detection
来源/年份：Interspeech 2025: 2245-2249。
方向/备注：语音伪造检测。
代码：未标注。

A3-18. Kim S B, Shin H S, Heo J, et al. Enhancing Audio Deepfake Detection by Improving Representation Similarity of Bonafide Speech
来源/年份：Interspeech 2025: 2250-2254。
方向/备注：语音伪造检测。
代码：未标注。

A3-19. Yang M, Gu Y, He Q, et al. Generalizable Audio Deepfake Detection via Hierarchical Structure Learning and Feature Whitening in Poincaré sphere
来源/年份：Interspeech 2025: 2255-2259。
方向/备注：语音伪造检测。
代码：未标注。

A3-20. Das A, El Kheir Y, Franzreb C, et al. Generalizable Audio Spoofing Detection using Non-Semantic Representations
来源/年份：Interspeech 2025: 4553-4557。
方向/备注：语音伪造检测。
代码：未标注。

## 3.3 语音对抗样本及检测 / 防御

A3-21. Chen M, Wang K, Lu L, et al. Hijacking Large Audio-Language Models via Context-Agnostic and Imperceptible Auditory Prompt Injection
来源/年份：IEEE Symposium on Security and Privacy 2026。
方向/备注：语音对抗样本 / 音频提示注入。
代码：未标注。

A3-22. Zhang Z, Wang D, Yang Q, et al. SafeSpeech: Robust and Universal Voice Protection Against Malicious Speech Synthesis
来源/年份：USENIX Security Symposium 2025。
方向/备注：语音对抗防御 / 语音保护。
代码：未标注。

A3-23. Jin W, Cao Y, Su J, et al. Whispering Under the Eaves: Protecting User Privacy Against Commercial and LLM-powered Automatic Speech Recognition Systems
来源/年份：USENIX Security Symposium 2025。
方向/备注：语音对抗样本 / ASR 隐私保护。
代码：未标注。

A3-24. Ling Z, Hu P, Gao X, et al. Sirens’ Whisper: Inaudible Near-Ultrasonic Jailbreaks of Speech-Driven LLMs
来源/年份：USENIX Security Symposium 2026。
方向/备注：语音对抗攻击 / 音频 jailbreak。
代码：未标注。

A3-25. Zhang Z, Wang D, Mi Y, et al. E2E-VGuard: Adversarial Prevention for Production LLM-based End-To-End Speech Synthesis
来源/年份：NeurIPS 2025。
方向/备注：语音对抗防御 / 语音合成保护。
代码：未标注。

A3-26. Cheng H, Xiao E, Shao J, et al. Jailbreak-AudioBench: In-Depth Evaluation and Analysis of Jailbreak Threats for Large Audio Language Models
来源/年份：NeurIPS 2025 Datasets and Benchmarks Track。
方向/备注：语音对抗检测 / 音频 jailbreak 评测。
代码：未标注。

A3-27. Sankala S, Parvathala V, Gundluru R, et al. Adversarial Attacks on Text-dependent Speaker Verification System
来源/年份：Interspeech 2025: 4558-4562。
方向/备注：语音对抗攻击 / 说话人验证。
代码：未标注。

A3-28. Yu G, Lee J, Kim S, et al. Mimic Blocker: Self-Supervised Adversarial Training for Voice Conversion Defense with Pretrained Feature Extractors
来源/年份：Interspeech 2025: 1663-1667。
方向/备注：语音对抗防御 / 语音转换防护。
代码：未标注。

A3-29. Alexos A, Peri R, Jayanthi S M, et al. Defending Speech-enabled LLMs Against Adversarial Jailbreak Threats
来源/年份：Interspeech 2025: 2048-2052。
方向/备注：语音对抗检测 / 音频 jailbreak 防御。
代码：未标注。

A3-30. Karimov E, Varlamov A, Ivanov D, et al. Novel Loss-Enhanced Universal Adversarial Patches for Sustainable Speaker Privacy
来源/年份：Interspeech 2025: 1513-1517。
方向/备注：语音对抗样本 / 说话人隐私保护。
代码：未标注。