# BayesFlow 2: Multi-Backend Amortized Bayesian Inference in Python

This repository presents the replication material for the [JSS](https://www.jstatsoft.org/) submission titled

> [**BayesFlow 2: Multi-Backend Amortized Bayesian Inference in Python**](https://arxiv.org/abs/2602.07098)

The code was tested using BayesFlow
[v2.0.10 (f9a7f2f)](https://github.com/bayesflow-org/bayesflow/releases/tag/v2.0.10)
with the JAX backend.

## Installation

The simplest way is to install all dependencies from the `pyproject.toml` using `uv`:

```bash
uv venv
uv sync
```

## Running the Case Study

You can run the full case study using one of the following:

```bash
uv run case-study.py
```

```bash
python case-study.py
```

Figures are generated in the [`figures`](figures/) directory.
A log file of all output is further saved as 'case-study.out'.

## Reproducibility

Note that despite all efforts to ensure reproducibility,
small differences in the results may occur due to differences in software versions and hardware.

## Citation

If you find this work or the [corresponding paper](https://arxiv.org/abs/2602.07098) useful, please consider citing the following:

```
@article{kuhmichel2026bayesflow,
  title={{BayesFlow} 2: Multi-backend amortized {B}ayesian inference in Python},
  author={Kühmichel, Lars and Huang, Jerry M and Pratz, Valentin and Arruda, Jonas and Olischläger, Hans and Habermann, Daniel and Kucharsky, Simon and Elsemüller, Lasse and Mishra, Aayush and Bracher, Niels and Jedhoff, Svenja and Schmitt, Marvin and Bürkner, Paul-Christian and Radev, Stefan T},
  journal={arXiv preprint arXiv:2602.07098},
  year={2026}
}
```
