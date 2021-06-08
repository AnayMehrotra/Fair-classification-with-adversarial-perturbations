# Code for "Fair Classification with Adversarial Perturbations"

This repository contains the code for reproducing the simulations in our paper "Fair Classification with Adversarial Perturbations."

## Running simulations
1. `simulation-with-adversarial-perturbations-all-algorithms.ipynb` contains code to reproduce the simulations in Figure 1 and Figure 7 for all algorithms except WGN+dro and WGN+SW; `WGN+dro.ipynb` and `WGN+SW.ipynb` contain code to reproduce the simulations in Figure 1 and Figure 7 for WGN+dro and WGN+SW (respectively).
2. `simulation-with-sythentic-data.ipynb` contains code to reproduce the simulation in Table 1.
3. `simulation-with-stochastic-perturbations-all-algorithms.ipynb` contains the code to reproduce the simulations in Figure 5.


## Acknowledgements
This repository uses code from the following repositories 
1. [AIF360](https://github.com/Trusted-AI/AIF360),
2. [noise_fairlearn](https://github.com/AIasd/noise_fairlearn),
3. [equalized_odds_under_perturbation](https://github.com/matthklein/equalized_odds_under_perturbation),
4. [robust-fairness-code](https://github.com/wenshuoguo/robust-fairness-code), and
5. [Noisy-Fair-Classification](https://github.com/vijaykeswani/Noisy-Fair-Classification).
