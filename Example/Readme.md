# This is an example of how to use the simulation and inference tools included in this package.

##  Importing the necessary libraries:

```python
import sabs_pkpd
```

Note : numpy, matplotlib.pyplot are imported in the __init__.py file

## Setting up the parameters for simulation and fitting
```python
sabs_pkpd.constants.n = 2
sabs_pkpd.constants.s = sabs_pkpd.run_model.set_myokit_simulation('./tests/test resources/pints_problem_def_test.mmt')
```
Then we set up the constants n and s:
  - n is the number of parameters fitted
  - s is the myokit model loaded for myokit.simulation 

## Loading experimental data and setting up the instructions for fitting
```python
fit_param_annot = ['constants.unknown_cst', 'constants.unknown_cst2']
exp_cond_annot = 'constants.T'
model_output_annot = 'comp1.y'
sabs_pkpd.constants.data_exp = sabs_pkpd.load_data.load_data_file('./tests/test resources/load_data_test.csv')
sabs_pkpd.constants.data_exp.Add_fitting_instructions(fit_param_annot, exp_cond_annot, model_output_annot)
```


```python
initial_point = [0.5, 0.5]
boundaries_low = [0, 0]
boundaries_high = [1, 1]
```

