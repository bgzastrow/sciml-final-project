Final project for George Biros' Fall 2024 Scientific Machine Learning course. Reproduces the methodology and results of Maes et al.'s 2024 MACE paper.

Create the necessary environment with conda: `conda create --file sciml.yaml`

[Link to powerpoint presentation](https://utexas-my.sharepoint.com/:p:/g/personal/jskeens1_austin_utexas_edu/Ee1_6IZxipdMlmS7NLQug5EBATZghg6B8v0EAlRV8G52Qg?e=mth3j7)

[Link to word document with notes on implementation challenges](https://utexas-my.sharepoint.com/:w:/g/personal/jskeens1_austin_utexas_edu/EYJisQPsRRxLl8d-vFYR64ABq29jrNaJqGgMl2_2cAk3Wg?e=jK2m1i)

Run for the case L96 (Lorenz 96 model) with an example configuration file, `./input/L96/L96_example.in`:
`run.py L96 L96_example.in`

or for the CSE case (AGB circumstellar envelope model), `./input/CSE/CSE_example.in`:
`run.py CSE CSE_example.in`

The philosophy here is that we should be able to switch between the models on the fly. I have created L96 and CSE versions of a few files -- `input.py`, `dataset.py`, `test.py` and placed them in L96 and CSE folders within `./src/mace/`. All of the adaptations of these files to the L96 problem are still very much a work in progress.

