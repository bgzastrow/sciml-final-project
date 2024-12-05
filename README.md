Final project for George Biros' Fall 2024 Scientific Machine Learning course. Reproduces the methodology and results of Maes et al.'s 2024 MACE paper.

Create the necessary environment with conda: `conda create --file sciml.yaml`

[Link to powerpoint presentation](https://utexas-my.sharepoint.com/:p:/g/personal/jskeens1_austin_utexas_edu/Ee1_6IZxipdMlmS7NLQug5EBATZghg6B8v0EAlRV8G52Qg?e=mth3j7)

[Link to word document with notes on implementation challenges](https://utexas-my.sharepoint.com/:w:/g/personal/jskeens1_austin_utexas_edu/EYJisQPsRRxLl8d-vFYR64ABq29jrNaJqGgMl2_2cAk3Wg?e=jK2m1i)

Run for the case L96 (Lorenz 96 model) with an example configuration file, `./input/L96/L96_example.in`:
`run.py L96 L96_example`

or for the CSE case (AGB circumstellar envelope model), `./input/CSE/CSE_example.in`:
`run.py CSE CSE_example`

The philosophy here is that we should be able to switch between the models on the fly. I have created L96 and CSE versions of a few files -- `input.py`, `dataset.py`, `test.py` and placed them in L96 and CSE folders within `./src/mace/`. All of the adaptations of these files to the L96 problem are still very much a work in progress.

NB: you will have to be on Python <3.12:
```
Traceback (most recent call last):
  File "/home/jskeens/sciml-final-project/run.py", line 62, in <module>
    model = mace.Solver(n_dim=input_data.n_dim, p_dim=4,z_dim = input_data.z_dim,
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jskeens/sciml-final-project/src/mace/mace.py", line 104, in __init__
    self.jit_solver = torch.compile(self.adjoint)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jskeens/.conda/envs/sciml/lib/python3.12/site-packages/torch/__init__.py", line 1868, in compile
    raise RuntimeError("Dynamo is not supported on Python 3.12+")
RuntimeError: Dynamo is not supported on Python 3.12+
```

