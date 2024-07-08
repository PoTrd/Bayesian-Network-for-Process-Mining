# Fork of ``Preserving Complex Object-Centric Graph Structures to Improve Machine Learning Tasks in Process Mining``

### Jan Niklas Adams, Gyunam Park, Wil van der Aalst forked by Paul Taurand

The experiments are based on the Python library [ocpa](https://ocpa.readthedocs.io)

_____________

### Instructions

Use venv to create the environment.  

``python -m venv /path/to/new/virtual/environment``.  

Activate the environment   

``/path/to/new/virtual/environment/.venv/Script/activate``  

Install all the dependencies from ``requirements.txt``  

``pip install -r requirements.txt ``

Go into the repository directory and unzip ``BPI2017-Final.zip`` before run

``python main_BN.py``

This will run the code for prediction with bayesian network. The output is the file ``results_BN\metrics_BN.csv``. 
To vizualize the curve on the graph of the initial projet, you need to run the notebook ``Visualization_BN.ipynb`` the documention is in.

_____________


### Notes 

- The original code is used to work with conda
- All files followed by ``_BN`` are files modified by the Fork.  
- In main.py, you'll find the if statement for Bayesian network prediction. However, the code fails to make all predictions.
- A change has been made in the gnn_utils.py file between lines 101 and 240. A library problem was preventing main_BN.py, which was the aim of this project, from working properly. 
