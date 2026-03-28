# Robust Low-Thrust Rendezvous with MTF-Adaptive Homotopy

This repository contains MATLAB code for a robust low-thrust terminal rendezvous guidance framework that tightly couples state-estimation confidence to a homotopy-regularized indirect optimal control solver.

The method is developed for uncooperative spacecraft rendezvous scenarios and is motivated by ENVISAT-class debris-removal and servicing problems. The implementation compares open-loop energy-optimal and fuel-optimal references against closed-loop guidance architectures under degraded sensing.

## Main idea

Minimum-fuel low-thrust guidance naturally leads to bang-bang control, which can become brittle when sensing quality degrades. This repository implements a closed-loop strategy that:

- estimates the translational relative state with a linear Kalman filter,
- uses measurement-side Multiple Tuning Factors (MTF) covariance inflation to reduce trust in suspicious measurements,
- forms a confidence score from the innovation statistics,
- maps that score to the homotopy parameter of an indirect solver,
- and re-solves the guidance problem in receding-horizon fashion.

As sensing degrades, the controller automatically becomes smoother and more conservative. As confidence recovers, the controller moves back toward the fuel-optimal bang-bang regime.

## What is included

The main script computes:

- open-loop energy-optimal reference trajectory,
- open-loop fuel-optimal reference trajectory via homotopy continuation,
- closed-loop plain KF + fixed bang-bang control,
- closed-loop MTF KF + fixed bang-bang control,
- closed-loop MTF KF + adaptive homotopy control,
- quantitative metrics such as cumulative delta-v, terminal miss distance, and solve-time statistics,
- and summary plots for control effort, adaptation score, homotopy parameter, trajectory comparison, and estimation error.

## Scenario

The current benchmark uses:

- Clohessy-Wiltshire relative dynamics,
- an 800 s terminal rendezvous horizon,
- bounded control acceleration,
- position-only measurements,
- injected sensor degradation through covariance inflation and temporary bias,
- and receding-horizon indirect re-solving.

This study is connected to a Blender-derived synthetic proximity-operations environment used in prior relative-navigation work.

## File structure

Example structure:

```text
.
├── main_closed_loop_case.m
├── adaptive_only_ablation.m
├── Figures/
│   ├── aas1.png
│   ├── aas2.png
│   ├── aas3.png
│   └── aas4.png
└── README.md
