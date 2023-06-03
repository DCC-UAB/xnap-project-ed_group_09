[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11122274&assignment_repo_type=AssignmentRepo)
# XNAP-Segons la AI, quina edat aparentes?
En aquest projecte s'implementa varios models amb diferents datasets amb l'objectiu de predir l'edat d'una persona a partir d'una imatge. En aquest projecte s'ha treballat amb 3 datasets diferents:
- Appa_real: 7.5k imatges
- CACD (Cross Age Celebrity Dataset): 160k imatges de 2k celebrities diferents
- AFAD (Asiatic Face Age Dataset): 160k imatges de persones asiàtiques

Per cada dataset hi ha el seu folder personalitzat. Dins de cada folder hi trobem l'arxiu 'lectura_' + nom del dataset + '.py', 'model_' + nom del dataset + '.py', 'train_' + nom del dataset + '.py', 'main_' + nom del dataset + '.py' i els arxius '.csv' que s'utilitzen per llegir les dades. En el cas del Appa_real, els arxius '.csv' no esta dins d'aquest repositori. Més endavant, s'explica com descarregar-se aquests datasets.     
  *En el cas del dataset CACD també hi trobem el 'preprocessing_cacd'.*

En aquest treball s'han fet diferents entrenaments i s'han dut a terme diferents proves ajustant constantment els hiperparàmetres, les arquitecures dels models i els datasets per tal d'intentar obtenir un rendiment óptim. Durant tot el procés, s'ha fet ús del Weight & Biases per tal de fer un seguiment de l'aprenentatge dels nostres models i visualitzar els resultats.

## Estructura del codi
En cada folder hi trobem l'arxiu **'lectura_'+ nom del dataset + '.py'**. En aquest arxiu hi trobem una classe personalitzada per cada dataset on llegeix les imatges i les edats. En aquesta classe, se li pasa com a paràmetre el path del directori on es troben les imatges, el path d'on esta el csv amb el nom de l'arxiu de cada imatge i la edat corresponent i les transformacions en el cas que es vulguin aplicar, sinó no cal pasar-ho com a paràmetre.

També hi trobem el **'model_' + nom del dataset + '.py'**. En aquest arxiu python hi trobem la funció que crea el model. En cada folder hi trobem una configuració diferent ja que està personalitzat per cada dataset. En tots tres a la funció se li pasa el tipus de transfer learning (finetunning o feature extraction). S'utilitza transfer learning en cada un amb l'arquitectura resnet34 pre-entrenada.

En cada folder veiem un **'train_' + nom del dataset + 'py.'** En aquest arxiu està la funcio train. Se li pasa com a paràmetres el model creat, els dataloaders, la loss function, l'optimizer, el número de époques que es vol entrenar el model, el nom del projecte wandb i el nom de l'execució i per últim el device. Aquesta funció entrena el model amb el dataloader del train, l'evalua amb el dataloader del validation i retorna el state del millor model trobat i les losses.

Hi ha un arxiu **'main_' + nom del dataset + 'py.'** en cada folder. En aquest arxiu, s'inicia sessió al wandb, es crea un projecte si es vol crear, es crida a la lectura del dataset ('lectura_' + nom del dataset + 'py.') i es creen els dataloaders, es crea el model i es defineix la loss function i l'optimizer. Un cop tot definit i creat, es crida al train perque el model s'entreni i es guarda el state del millor model trobat, retornat per la funció train, a un arxiu '.pth'.

## Execucions

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
