本项目是对知识蒸馏（Knowledge Distillation）的完整学习实践，基于 **CIFAR-10** 数据集，使用 ResNet-34 作为教师模型，一个轻量级 CNN 作为学生模型，通过对比实验展示蒸馏如何提升小模型性能。包含详细代码、实验结果分析和学习笔记。

##  内容概览

- **理论介绍**：蒸馏的动机、软标签与温度参数、损失函数构成。
- **实验代码**：教师预训练、学生单独训练、蒸馏训练（软+硬标签混合）的完整实现。
- **结果分析**：准确率对比、超参数影响（温度 T、权重 α）的简要讨论。
- **学习笔记**：分章节记录个人对蒸馏的理解与思考。

##  快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/yourusername/knowledge-distillation-tutorial.git
cd knowledge-distillation-tutorial
```
### 2. 安装依赖
推荐使用 Python 3.8+ 和 PyTorch 2.0+，可通过 pip 一键安装：

```bash
pip install -r requirements.txt
```
或使用 conda：

```bash
conda env create -f environment.yaml
conda activate kd_env
```
### 3. 运行实验
Jupyter Notebook 方式：

```bash
jupyter notebook notebooks/distillation_tutorial.ipynb
```
按顺序执行所有单元格即可复现实验。

脚本方式（可选）：

```bash
python scripts/train_teacher.py    # 训练教师模型
python scripts/train_student.py    # 训练学生模型（单独与蒸馏）
```
### 4.实验结果
模型	测试准确率
Teacher (ResNet-34)	82.3%
Student alone	70.1%
Student + Distillation	74.5%
提升幅度：+4.4% （具体数值以实际运行结果为准）

注：由于随机初始化，每次运行结果可能略有浮动。

### 5.核心概念
硬标签损失：标准交叉熵损失，让学生直接拟合真实标签。

软标签损失：KL 散度，让学生模仿教师输出的概率分布（通常带温度参数 T）。

混合损失：α * soft_loss + (1-α) * hard_loss，平衡模仿教师与拟合真实标签。

详细理论推导见 notes/ 目录。

### 6.依赖环境
主要依赖：
```
Python 3.8+

PyTorch >= 2.0.0

torchvision >= 0.15.0

matplotlib >= 3.5.0

numpy >= 1.21.0
```
完整列表见 requirements.txt。

### 7.仓库结构
```text
.
├── README.md
├── requirements.txt
├── environment.yaml          # conda 环境配置
├── notebooks/
│   └── distillation_tutorial.ipynb   # 主实验代码
├── scripts/                  # 可选，可将核心代码拆分为 .py
│   ├── train_teacher.py
│   ├── train_student.py
│   └── utils.py
├── results/                  # 实验结果
│   ├── logs/
│   ├── figures/
│   └── metrics.csv
├── notes/                    # 学习笔记
│   ├── 01_intro.md
│   ├── 02_theory.md
│   ├── 03_experiments.md
│   ├── 04_analysis.md
│   └── 05_summary.md
└── assets/                   # 图片、公式等资源
```
### 8.学习笔记
个人对知识蒸馏的理解记录在 notes/ 目录下，包括：
```
引言与背景

理论基础与公式推导

实验设置与代码详解

结果分析与超参数讨论

总结与心得

欢迎交流探讨！
```
### 9.许可证
本项目采用 MIT 许可证，详情见 LICENSE 文件。

### 10.参考文献
```
Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. arXiv preprint arXiv:1503.02531.

PyTorch 官方教程：Knowledge Distillation Tutorial
```
