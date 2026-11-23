# Jev-Semantic MIDAS

[English](README.md)

如何在不引入前视偏差、也不过度放大文本噪声的前提下，把央行政策文本信号融入混频通胀状态空间预测？

本仓库实现 *Jev-Semantic MIDAS: Loss-Calibrated FOMC Signals for Inflation Nowcasting*（IEEE BigData 2026 录用）的核心状态空间滤波与校准模块。代码对 FOMC 发布时间执行严格时点对齐，将文本信号投影到语义基上输入 HF-MIDAS 状态空间更新，并直接依据历史预测损失校准信号强度，而不是依赖孤立的情感打分。

- `src/semantic_midas/asof.py`：严格发布日对齐，杜绝前视泄漏。
- `src/semantic_midas/basis.py` 与 `semantics.py`：语义基构造与文本信号投影。
- `src/semantic_midas/filter.py`：混频状态空间滤波更新。
- `src/semantic_midas/calibration.py` 与 `theory.py`：损失驱动的信号缩放校准与理论误差分解。

## 快速开始

```bash
pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -v
```

`run_demo.py` 构造合成季度通胀目标与发布时点对齐的语义冲击，运行 HF-MIDAS 滤波并对比校准前后的即时预测误差。

## 范围说明

本仓库为合成输入上的公开 MVP，完整 FOMC 文本抽取与 60 季度实证复现脚本按需提供。
