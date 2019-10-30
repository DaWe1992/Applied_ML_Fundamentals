# Applied Machine Learning Fundamentals (DHBW Lecture)

```
'We are drowning in information and starving for knowledge.' – John Naisbitt
```

Machine learning is a subfield of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience **without being explicitly programmed**. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves. Machine learning has become an integral part of many modern applications.

<p align="center">
	<img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/03_tex_files/03_img/data_science.png" width=400px>
</p>

This lecture is supposed to give a general introduction into state-of-the-art machine learning algorithms and their applications.

## Contents <img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/03_tex_files/03_img/toc.png" width="20px" height="20px">

1. **Introduction to machine learning** ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/01_intro_ml.pdf))
    * Motivation and applications
    * Terminology
    * Key challenges in ML: Generalization, feature engineering, model selection, ...
2. **Mathematical foundations**
	* Statistics
	* Linear algebra
	* Optimization
3. **Bayesian decision theory** ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/03_decision_theory.pdf))
    * Bayes optimal classifier
    * Naive Bayes
4. **Probability density estimation** ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/04_density_estimation.pdf))
    * Parametric models
	* Non-parametric models
	* (Mixture models)
5. **Supervised learning**
    * Regression ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/05_regression.pdf))
      * Linear regression
	  * Probabilistic regression
      * Basis functions: Radial basis functions, polynomial basis functions
    * Classification I
      * k-nearest neighbors
      * Logistic regression ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/07_logistic_regression.pdf))
      * Decision trees and ensemble methods ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/08_decision_trees.pdf))
	* Classification II
      * Deep learning
        * Perceptrons
        * Multi-layer-perceptrons and back-propagation
        * Deep learning application: Natural language processing (word embeddings, text classification, sentiment analysis)
      * Support vector machines ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/11_svm.pdf))
6. **Evaluation of ML models** ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/09_evaluation.pdf))
	* Out-of-sample testing and cross validation
	* Confusion matrices
	* Evaluation metrics: Precision, recall, F1 score, ROC, accuracy
	* Cost-sensitive evaluation
	* Model selection: Grid search, random search
6. **Unsupervised learning**
    * Clustering
      * k-Means
      * Hierarchical clustering (divisive and agglomerative)
    * Principal component analysis ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/13_pca.pdf))
7. **Reinforcement learning**
    * Markov decision processes
    * Dynamic programming: Policy iteration, value iteration
    * Q-learning, Q-networks
8. **Lecture summary**

## Assignments <img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/03_tex_files/03_img/assignments.png" width="25px" height="25px">
The assignments have to be solved in groups comprising 3 to 4 students. We provide starter code and task descriptions in this repository (folder [exercises](https://github.com/DaWe1992/Applied_ML_Fundamentals/tree/master/02_exercises)). Solutions will be presented and discussed in the next class. Please submit your solutions via Moodle. We will correct your homework and give it back to you. Correct solutions are rewarded with a bonus for the exam or project. **The solutions have to be your own work. If you plagiarize, you will lose all bonus points!**

## Exam <img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/03_tex_files/03_img/exam.png" width="30px" height="25px">
The exam will take 60 minutes. You are allowed to bring a calculator to the exam. We do not ask you for any derivations, rather we want to test whether you understand the general concepts. Make sure that you can answer the self-test questions we provide for each topic. Some of the slides give you hints. A slide marked with <img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/03_tex_files/03_img/scream.png" width="25px" height="25px"> provides in-depth information which you do not have to know by heart (think of it as additional material for the sake of completeness). On the other hand, very important content is indicated by the symbol <img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/03_tex_files/03_img/important.png" width="25px" height="25px"> Make sure you understand it!

## Python code <img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/03_tex_files/03_img/python.png" width="30px" height="30px">
You can find additional implementations for some of the algorithms - which are not part of the assignments ;) - in the folder [python](https://github.com/DaWe1992/Applied_ML_Fundamentals/tree/master/04_python). Play around with the hyper-parameters and try different data sets in order to get a better feeling for how the algorithms work.

## Literature <img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/03_tex_files/03_img/literature.png" width="25px" height="25px">
You do not need to buy any books for the lecture. Most resources are available online. Please find a list below:

| Title                                    	     | Author(s)                    | Publisher 				| Link                                                         			|
|------------------------------------------	     |-----------------------------	|-----------				|--------------------------------------------------------------			|
| Deep Learning                            	     | Goodfellow et al. (2016)		| MIT Press     			| [click here](https://www.deeplearningbook.org/) 		        		|
| Elements of statistical Learning               | Hastie et al. (2008) 		| Springer      			| [click here](https://web.stanford.edu/~hastie/Papers/ESLII.pdf)		|
| Machine Learning                               | Mitchell (1997)              | McGraw-Hill   			| [click here](http://profsite.um.ac.ir/~monsefi/machine-learning/pdf/Machine-Learning-Tom-Mitchell.pdf)																																							|
| Machine Learning - A probabilistic perspective | Murphy (2012)				| MIT Press     			| [click here](https://doc.lagout.org/science/Artificial%20Intelligence/Machine%20learning/Machine%20Learning_%20A%20Probabilistic%20Perspective%20%5BMurphy%202012-08-24%5D.pdf)																			|
| Mathematics for Machine Learning               | Deisenroth et al. (2019)     | Cambridge Univ. Press		| [click here](https://mml-book.github.io/)								|
| Pattern Recognition and Machine Learning 	     | Bishop (2006)   			    | Springer  				| [click here](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) 																									|
| Reinforcement Learning - An introduction       | Sutton et al. (2014)         | MIT Press     			| [click here](http://incompleteideas.net/book/bookdraft2017nov5.pdf)   |

## Bugs and Errors <img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/03_tex_files/03_img/bug.png" width="25px" height="30px">
Help us improve the lecture. Feel free to file an issue if you spot any errors in the slides, exercises or code.
Thank you very much. *Please do not open issues for questions concerning the content!*
