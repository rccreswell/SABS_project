
# Loading CellML files to run simulations with published models

## Getting the CellML model to your machine

If you are brave enough, you can start writing the CellML model from scratch on your own. This is a whole journey, for which you will need to get to know how to write CellML code (have a look at CellmL tutorials: https://www.cellml.org/getting-started/tutorials). Otherwise, you can also download the published models in the CellML format directly, from the repo: https://models.physiomeproject.org/cellml. You just need to browse to the desired model and download it to your machine.

## Loading the CellML model with the package

Here, we will use the example of Grandi, Pasqualini, Bers 2010 model for cardiac cells (https://models.physiomeproject.org/workspace/grandi_pasqualini_bers_2010/file/8a2c891da5f2f1b90d9c465b7bc63f652999d096/grandi_pasqualini_bers_2010.cellml).

Start by importing the needed librairies:

```python
import sabs_pkpd
import matplotlib.pyplot as plt
```

Then browse to the CellML file for the model you just downloaded. You also need to enter a path to where to save the MMT conversion of the model, which will be useful for two reasons. The first reason is that this avoids you to redo the conversion of your model from CellML to MMT (which is anyway needed to run simulations) everytime you want to change anything in your model. The second reason is that changing anything in your model will be more convenient in the MMT format. So by saving the MMT model, you can easily update your model and reload it using the function for loading MMT models presented in another example.

```python
cellml = 'Examples/Example loading CellML model/grandi_pasqualini_bers_2010.cellml'
mmt = 'Examples/Example loading CellML model/grandi_pasqualini_bers_2010.mmt'

simulation = sabs_pkpd.load_model.load_model_from_cellml(cellml, mmt)
``` 

That's it ! You have loaded the model and are ready to run simulations with it. It is loaded to the variable ```simulation``` which is a Myokit object.

## Run a quick simulation with your freshly loaded model

Now that you have your MMT model loaded in ```simulation```, you can run it. The simplest ever 


