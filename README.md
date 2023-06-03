[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11122274&assignment_repo_type=AssignmentRepo)
# XNAP-Segons la AI, quina edat aparentes?
En aquest projecte s'implementa varios models amb diferents datasets amb l'objectiu de predir l'edat d'una persona a partir d'una imatge. En aquest projecte s'ha treballat amb 3 datasets diferents:
- Appa_real: 7.5k imatges
- CACD (Cross Age Celebrity Dataset): 160k imatges de 2k celebrities diferents
- AFAD (Asiatic Face Age Dataset): 160k imatges de persones asiàtiques
Per cada dataset hi ha el seu folder personalitzat. Dins de cada folder hi trobem l'arxiu 'lectura_' + nom del dataset, 'model_' + nom del dataset, 'train_' + nom del dataset i 'main_' + nom del dataset.       
  *En el cas del dataset CACD també hi trobem el 'preprocessing_cacd'.*

En aquest treball s'han fet diferents entrenament i s'han dut a terme diferents proves ajustant constantment els hiperparàmetres, les arquitecures dels models i els datasets per tal d'intentar obtenir un rendiment óptim.

## Estructura del codi
En aquest repositori hi trobem 
You must create as many folders as you consider. You can use the proposed structure or replace it by the one in the base code that you use as starting point. Do not forget to add Markdown files as needed to explain well the code and how to use it.

## Example Code
The given code is a simple CNN example training on the MNIST dataset. It shows how to set up the [Weights & Biases](https://wandb.ai/site)  package to monitor how your network is learning, or not.

Before running the code you have to create a local environment with conda and activate it. The provided [environment.yml](https://github.com/DCC-UAB/XNAP-Project/environment.yml) file has all the required dependencies. Run the following command: ``conda env create --file environment.yml `` to create a conda environment with all the required dependencies and then activate it:
```
conda activate xnap-example
```

To run the example code:
```
python main.py
```



## Contributors

Gabriel Gausachs Fernández de los Ronderos      1604373@uab.cat

Arnau Gómez                              1601488@uab.cat


Xarxes Neuronals i Aprenentatge Profund
Grau de Data Engineering, 
UAB, 2023
