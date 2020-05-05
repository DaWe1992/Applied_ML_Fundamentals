# Python Code 🐍
Machine learning algorithms (probably all algorithms) are easier to understand, if you see them implemented. Here you find Python implementations for some of the algorithms (which are not part of the assignments).
Play around with the hyper-parameters of the algorithms and try different data sets in order to get a better feeling for how the algorithms work. Also, debug through the code line for line and check what each line does.

> **The algorithms were implemented for educational purposes only, and should therefore not be used in production environments!**

## Which Algorithms are included?
The following algorithms are implemented (some are missing, since they are part of the exercises):

1. **Classification** ([folder clf](https://github.com/DaWe1992/Applied_ML_Fundamentals/tree/master/06_python/clf))
	* Decision trees
	* Iterative reweighted least squares (IRLS)
	* k-nearest neighbors (kNN)
	* Linear discriminant analysis
	* Logistic regression
	* Multi-layer perceptron (MLP)
	* Perceptron
	* Support vector machine (SVM)
2. **Regression** ([folder reg](https://github.com/DaWe1992/Applied_ML_Fundamentals/tree/master/06_python/reg))
	* Bayesian linear regression
	* Gaussian process regression
	* Kernel (ridge) regression
	* k-nearest neighbors regression
	* LOWESS
	* Support vector regression (SVR)
3. **Reinforcement learning** ([folder rl](https://github.com/DaWe1992/Applied_ML_Fundamentals/tree/master/06_python/rl))
	* Policy iteration
	* Value iteration
	* Q-learning
	* REINFORCE
4. **Unsupervised learning** ([folder unsupervised](https://github.com/DaWe1992/Applied_ML_Fundamentals/tree/master/06_python/unsupervised))
	* Apriori (association rules)
	* Expectation-maximization (EM) for Gaussian mixtures
	* Dimensionality reduction ([folder decomposition](https://github.com/DaWe1992/Applied_ML_Fundamentals/tree/master/06_python/unsupervised/decomposition))
		* Auto-encoders
		* Independent component analysis (ICA)
		* Non-negative matrix factorization
		* Kernel PCA
		* Principal component analysis (PCA)
		* t-SNE
	* Clustering ([folder clustering](https://github.com/DaWe1992/Applied_ML_Fundamentals/tree/master/06_python/unsupervised/clustering))
		* Affinity propagation
		* Agglomerative clustering
		* DBSCAN
		* k-Medoids
		* Mean-shift clustering
		* OPTICS
		* Spectral clustering
5. **Sequence learning** ([folder seq](https://github.com/DaWe1992/Applied_ML_Fundamentals/tree/master/06_python/seq))
	* Hidden Markov models (HMM)
6. **Optimization** ([folder optim](https://github.com/DaWe1992/Applied_ML_Fundamentals/tree/master/06_python/optim))
	* Nelder-Mead optimization
	* Particle swarm optimization
	* Simulated annealing
	
Please find further exemplary implementations in the folder [misc](https://github.com/DaWe1992/Applied_ML_Fundamentals/tree/master/06_python/misc).

## Installation
It is good practice to create a new virtual environment for each of your projects. Please create a new environment e.g. using Anaconda (Navigator)
or use the `virtualenv` [package](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
You may want to call the virtual environment `ml_lecture`.

Below you find an example using the `virtualenv` package (on Linux):

**Step 1)** Create the virtual environment:

```shell
$ pip install virtualenv # install virtualenv package
$ mkdir project_folder
$ cd project_folder
$ virtualenv ml_lecture # creates a folder in the current directory containing the virtual env
```

**Step 2)** Activate the virtual environment `ml_lecture`:

```shell
$ source ml_lecture/bin/activate
```

**Step 3)** Install packages using `pip`:

Finally, install the required packages listed in `requirements.txt` which can be found in this directory.
You can do this by executing the following command:

```shell
$ pip install -r requirements.txt
```

Now you should be ready to go!

## How to run the Code?
Please use the script located in `main.py` to execute the code for the algorithms. The file contains functions for
classification, regression, reinforcement learning and unsupervised learning. Simply uncomment the parts which you want to use. Also, this repository contains classes for data creation, evaluation and plotting.
Please refer to the folder [utils](https://github.com/DaWe1992/Applied_ML_Fundamentals/tree/master/06_python/utils) for this.

The file `main.py` has the following structure:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:06:24 2019
@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# numpy
# -----------------------------------------------------------------------------
import numpy as np

# data creation and plotting
# -----------------------------------------------------------------------------
from utils.data_creator import DataCreator
from utils.plotter import Plotter

# classifiers
# -----------------------------------------------------------------------------
from clf.knn import kNN
from clf.svm import SVM
from clf.lda import LDA
from clf.logistic_regression import LogisticRegression, LogRegOneVsOne
from clf.irls import IRLS
from clf.mlp_torch import MLP
from clf.decision_tree import DecisionTree

# more imports...


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def classification():
    """
    Classification.
    """
    # create data
    X, y = DataCreator().make_classification(name="linear", n_classes=2)
    
    # train logistic regression classifier
    clf = LogisticRegression(poly=True)
    clf.fit(X, y, batch_size=X.shape[0])
    
    # plot boundary
    Plotter(X, y).plot_boundary(clf, step_size=0.005)
    
    # evaluation
    evaluator = Evaluator()
    acc = evaluator.accuracy(clf, X, y)
    print("Accuracy: {} %".format(acc))
    evaluator.conf_mat(clf, X, y)
    
    
def regression():
    """
    Regression.
    """
    # create data
    X, y = DataCreator().make_regression(name="sine")
    
    # train kernel ridge regressor
    reg = KernelRegression()
    reg.fit(X, y, kernel="gaussian")
    
    # plot boundary
    Plotter(X, y).plot_regression(reg, n_points=500)


def unsupervised_learning():
    """
    Unsupervised learning.
    """
    # ...
    
    
def reinforcement_learning():
    """
    Reinforcement learning.
    """
    # ...


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Main function.
    """
    classification()
    regression()
    unsupervised_learning()
    reinforcement_learning()
```

## Image Gallery

### Classification

<img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/06_python/z_img/boundary.png" width="450px" height="300px"> <img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/06_python/z_img/svm.png" width="450px" height="300px">

### Regression

<img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/06_python/z_img/gaussian_process.png" width="450px" height="300px"> <img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/06_python/z_img/knn_regression.png" width="450px" height="300px">
<p align="center">
	<img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/06_python/z_img/grad_desc.gif" width="300px" height="300px">
</p>

### Clustering

<img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/06_python/z_img/optics.png" width="450px" height="300px"> <img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/06_python/z_img/affinity_propagation.gif" width="450px" height="300px">

### Optimization

<img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/06_python/z_img/nelder_mead.gif" width="300px" height="300px"> <img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/06_python/z_img/particle_swarm.gif" width="300px" height="300px"> <img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/06_python/z_img/simulated_annealing.png" width="300px" height="300px">

<sub>© 2020 Daniel Wehner</sub>
