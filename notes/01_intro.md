# 1. 知识蒸馏简介

## 1.1 背景：模型压缩的挑战

随着深度学习模型越来越庞大（如 GPT、ResNet152），直接部署到资源受限的设备（手机、嵌入式）变得困难。**模型压缩** 应运而生，常见方法有：
- 剪枝（Pruning）
- 量化（Quantization）
- 知识蒸馏（Knowledge Distillation）

本节聚焦于知识蒸馏，一种通过“师生学习”实现模型轻量化的技术。

## 1.2 核心思想：从“教师”到“学生”

知识蒸馏由 Hinton 等人在 2015 年提出，核心想法是：
> 让一个小模型（学生）去模仿一个大模型（教师）的输出行为，从而将教师的知识迁移到学生身上。

**形象比喻**：
- 教师模型：经验丰富的专家，能给出细腻的判断（不仅知道“这是猫”，还知道“这有点像狗”）。
- 学生模型：初出茅庐的学习者，通过模仿专家的判断（软标签）快速成长。

## 1.3 为什么“软标签”更有用？

传统训练使用 **硬标签**（one-hot 向量，如 `[0,1,0,...]`），只告诉学生正确答案是什么。  
而教师模型输出的 **软标签**（概率分布，如 `[0.1, 0.7, 0.05,...]`）包含了类别间的相似性信息，例如：
- 一张猫的图片，教师可能输出：猫 0.7，狗 0.2，汽车 0.01 …
  这暗示“猫和狗在特征上更接近”。

学生通过学习这些软标签，能捕捉到更丰富的结构信息，从而提升泛化能力。

## 1.4 温度参数的作用

为了让软标签的分布更加平滑（即“暗知识”更明显），蒸馏引入了 **温度 T**：

$$
q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

- T=1：标准 Softmax
- T>1：分布更平滑，小概率类别获得更多权重
- T<1：分布更极端，接近硬标签

在训练时，教师和学生都使用相同的 T 生成软标签；推理时 T=1 恢复标准 Softmax。

## 1.5 损失函数构成

蒸馏的总损失通常为两部分：

1. **软标签损失**（蒸馏损失）：  
$$
   mathcal{L}_{\text{soft}} = T^2 \cdot \text{KL}(p_{\text{teacher}} \| p_{\text{student}})
 $$
   其中 (p_{\text{teacher}}\) 和 \(p_{\text{student}}\) 是教师和学生在温度 T 下的 softmax 输出。

2. **硬标签损失**：  
   $$
   \mathcal{L}_{\text{hard}} = \text{CrossEntropy}(y_{\text{true}}, p_{\text{student}})
   $$

最终损失：  
$$
\mathcal{L} = \alpha \mathcal{L}_{\text{soft}} + (1-\alpha) \mathcal{L}_{\text{hard}}
$$

- \(\alpha\) 平衡两种损失，通常取 0.5~0.9。

## 1.6 蒸馏的优势与局限

| 优势 | 局限 |
|------|------|
| 学生模型小，推理快 | 需要训练教师模型，增加前期成本 |
| 性能提升显著 | 教师与学生结构需匹配（通常同类架构） |
| 可与其他压缩方法结合 | 对超参数（T、α）敏感 |

## 1.7 本笔记系列概览

- 02_theory：深入推导公式与数学原理  
- 03_experiments：CIFAR-10 实验设置与代码解析  
- 04_analysis：实验结果讨论与超参数调优  
- 05_summary：总结与延伸思考

## 参考文献

- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. *arXiv:1503.02531*.
- [PyTorch Knowledge Distillation Tutorial](https://pytorch.org/tutorials/intermediate/knowledge_distillation.html)
