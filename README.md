# BLOCC - BiLevel Optimization problems with Coupled Constraints

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/liuyuan999/Penalty_Based_Lagrangian_Bilevel_Tianyi_Chen-s_Lab/blob/main/LICENSE) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)  [![Arxiv link](https://img.shields.io/badge/math.OC-2406.10148-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2406.10148)

This repository provides the implementation of the BLOCC algorithm for bilevel optimization problems with coupled constraints, as presented in our NeurIPS 2024 paper:

> Liuyuan Jiang, Quan Xiao, Victor M. Tenorio, Fernando Real-Rojas, Antonio G. Marques and Tianyi Chen, "A Primal-Dual-Assisted Penalty Approach to Bilevel Optimization with Coupled Constraints", arXiv preprint arXiv:2406.10148, 2024.

The paper presents an effective way to deal with Bilevel Optimization problems with coupled constraints by means of a (fully) first-order algorithm, which makes it scalable and efficient.

# Code repository organization

The experiments are run over Jupyter Python notebooks (toy example and hyperparameter optimization) or MATLAB code. In all cases, the provided code generates the figures presented in the paper. More specifically, the code repository contains the following elements:

- `toy_example.ipynb` is a Jupyter notebook containing 2 toy examples to show that the BLOCC Algorithm can effectively work in a simplified setting. This corresponds to the experiments presented in section 4.1 of the paper.
- `hyperparam_opt` folder contains experiments about the SVM hyperparameter optimization problem. An SVM hyperparameter optimization problem is well-known for testing the performance of BLO algorithms, such as LV-HBA [[1]](#1) and GAM [[2]](#2). This corresponds to the experiments presented in section 4.2 of the paper. This folder includes the following contents organized for reproducibility of the experiments:
  - `algorithms` folder contains the code for the three methods compared in the corresponding Python files. For the BLOCC method we provide 2 implementations: one using the library CVXPY [[3]](#3) to solve the lower-level problem (file `blocc_cvxpy.py`) and another one using the proposed MaxMin solver (see Algorithm 2 in the paper, file `blocc.py`).
  - `datasets` folder includes the diabetes and fourclass datasets.
  - `SVM_Tests.ipynb` is a Jupyter notebook that runs the experiments and creates the figures used in the paper.
- `Transportation` folder contains the experiments in MATLAB code about the network design problem in two synthetic networks (a simple network with 3 nodes and a more realistic one with 9) and a real-world network of 26 nodes the city of Seville, Spain. In the network design problem, the operator's profit is modeled as the upper-level objective, influenced by passenger behavior in the lower-level problem. This corresponds to the experiments presented in section 4.3 of the paper.

# Citation

If you find our work interesting, please consider citing this paper:

```
@article{jiang2024primaldual,
  title={A Primal-Dual-Assisted Penalty Approach to Bilevel Optimization with Coupled Constraints}, 
  author={Liuyuan Jiang and Quan Xiao and Victor M. Tenorio and Fernando Real-Rojas and Antonio G. Marques and Tianyi Chen},
  journal={arXiv preprint arXiv:2406.10148},
  year={2024}
}
```

# References

<a id="1">[1]</a> Wei Yao, Chengming Yu, Shangzhi Zeng, and Jin Zhang. "Constrained bi-level optimization: Proximal lagrangian value function approach and hessian-free algorithm". arXiv preprint arXiv:2401.16164, 2024. [Code Link](https://github.com/SUSTech-Optimization/LV-HBA)

<a id="2">[2]</a> Siyuan Xu and Minghui Zhu. "Efficient gradient approximation method for constrained bilevel optimization". In _Proceedings of the AAAI Conference on Artificial Intelligence_, 2023. [Code Link](https://github.com/xsy786912649/Efficient-gradient-approximation-method-for-constrained-bilevel-optimization-problem)
