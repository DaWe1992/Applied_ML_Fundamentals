# Artificial Intelligence and Machine Learning (Lecture) 🤖

```
'We are drowning in information and starving for knowledge.' – John Naisbitt
```

Machine learning and data science represent a subfield of artificial intelligence (AI) that provides systems the ability to **automatically learn and improve** from experience
**without being explicitly programmed**. Machine learning focuses on the development of computer programs which can access data and use it to learn for themselves.
A machine learning algorithm learns by building a mathematical / statistical model from the data. This model can then be used for inference and decision making. 
Machine learning has become an integral part of many modern applications. In general, data science is a **cross-topic discipline** which comprises computer science, math / statistics as well as
domain and business knowledge:

<p align="center">
	<img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/img/data_science.png" width=400px>
</p>

The lecture *'Artificial Intelligence and Machine Learning'* is supposed to provide in-depth knowledge about state-of-the-art machine learning algorithms and their applications.
This Readme file provides you with all necessary information about the lecture. It is structured as follows:

1. 📜 Lecture contents
2. ✒️ Exercises
3. 📝 Exam
4. 🐍 Programming tasks (bonus points for the exam)
5. 📚 Literature and recommended reading
6. 🐞 Bugs and errors

## Lecture Contents 📜
The following topics and algorithms will be covered in the lecture:

<details>
<summary>1. Introduction to machine learning</summary>

* Link: [click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/01_intro_ml.pdf)
* Content:
	* Motivation and basic terminology
	* Problem types in machine learning
	* Key challenges in machine learning:
		* Generalization
		* Feature engineering
		* Performance measurement
		* Model selection
		* Computation
	* Applications
</details>

<details>
<summary>2. Optimization techniques</summary>

* Link: [click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/02_optimization.pdf)
* Content:
	* Important concepts and definitions
		* Gradients
		* Hessian matrix
		* Taylor expansion
		* Convex sets and convex functions
	* Unconstrained optimization
	* Constrained optimization
		* Karush-Kuhn-Tucker (KKT) conditions
		* Lagrange function
		* Lagrange duality
	* Numeric optimization
		* Gradient descent (with momentum)
		* Newton's method
</details>

<details>
<summary>3. Bayesian decision theory</summary>

* Link: [click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/03_decision_theory.pdf)
* Content:
	* Bayes optimal classifiers
	* Error minimization vs. risk minimization
    * Multinomial and Gaussian naive Bayes
	* Probability density estimation and maximum likelihood estimation
	* Generative and discriminative models
	* Exponential family distributions
</details>

<details>
<summary>4. Non-parametric density estimation and expectation-maximization (EM)</summary>

* Link: [click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/04_nde_em.pdf)
* Content:
	* Histograms
	* Kernel density estimation
	* k-nearest neighbors
	* Expectation-maximization (EM) algorithm for Gaussian mixture models
	* BIC and AIC
</details>

<details>
<summary>5. Probabilistic graphical models</summary>

* Link: [click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/05_pgm.pdf)
* Content:
	* Bayesian networks
	* Inference and sampling in graphical models
	* Hidden Markov models (HMMs) and the Viterbi algorithm
</details>

<details>
<summary>6. Linear regression</summary>

* Link: [click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/06_regression.pdf)
* Content:
	* Normal equations and gradient descent for linear regression
	* Probabilistic regression
	* Basis function regression (polynomial basis functions, radial basis functions)
	* Regularization techniques
</details>

<details>
<summary>7. Logistic regression</summary>

* Link: [click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/07_logistic_regression.pdf)
* Content:
	* Why you should not use linear regression for classification
	* Derivation of the logistic regression model
	* Logistic regression with basis functions
	* Regularization techniques
	* Dealing with multi-class problems:
		* Softmax regression
		* One-vs-one and one-vs-rest
</details>

<details>
<summary>8. Deep learning</summary>

* Link: [click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/08_deep_learning.pdf)
* Content:
	* Biological motivation of deep learning
	* Rosenblatt's perceptron
	* Network architectures
	* Multi-layer-perceptrons (MLPs) and the backpropagation algorithm
	* Extensions and improvements
		* Activation functions
		* Parameter initialization techniques
		* Optimization algorithms for deep learning models (AdaGrad, RMSProp)
	* Introduction to convolutional neural networks (CNNs)
</details>

<details>
<summary>9. Evaluation of machine learning models</summary>

* Link: [click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/09_evaluation.pdf)
* Content:
	* Out-of-sample testing and cross validation
	* Confusion matrices
	* Evaluation metrics: Precision, recall, F1-score, ROC, accuracy, RMSE, MAE
	* Model selection: Grid search, random search, early stopping
	* Bias-variance decomposition
</details>

<details>
<summary>10. Decision trees and ensemble methods</summary>

* Link: [click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/10_decision_trees.pdf)
* Content:
	* The ID3 algorithm
	* Extensions and variants:
		* Impurity measures
		* Dealing with numeric attributes
		* Regression trees
	* Ensemble methods:
		* Bagging
		* Random forests
		* ExtraTrees
</details>

<details>
<summary>11. Support vector machines (SVMs)</summary>

* Link: [click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/11_svm.pdf)
* Content:
	* Hard-margin SVMs (primal and dual formulation)
	* The kernel concept
	* Soft-margin SVMs
	* Sequential minimal optimization (SMO)
</details>

<details>
<summary>12. Clustering algorithms</summary>

* Link: [click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/12_clustering.pdf)
* Content:
	* KMeans
	* Hierarchical clustering
	* DBSCAN
	* Mean-shift clustering
</details>

<details>
<summary>13. Dimensionality reduction: Principal component analysis (PCA)</summary>

* Link: [click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/13_pca.pdf)
* Content:
	* Why dimensionality reduction?
	* Maximum variance formulation of PCA
	* Properties of covariance matrices
	* The PCA algorithm
	* PCA applications
</details>

<details>
<summary>14. Introduction to reinforcement learning</summary>

* Link: [click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/14_rl.pdf)
* Content:
	* What is reinforcement learning?
	* Key challenges in reinforcement learning
	* Dynamic programming:
		* Value iteration
		* Policy iteration
		* Q-learning
		* SARSA
	* Exploitation versus exploration
	* Non-deterministic rewards and actions
	* Temporal difference learning
	* Deep reinforcement learning
</details>

<details>
<summary>15. Advanced regression techniques</summary>

* Link: [click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/15_advanced_regression.pdf)
* Content:
	* MLE and MAP regression
	* Full Bayesian regression
	* Kernel regression
	* Gaussian process regression
</details>

Please refer to the official [DHBW module catalogue](https://www.dhbw.de/fileadmin/user/public/SP/MA/Data_Science_und_Kuenstliche_Intelligenz/Business_Management.pdf) for further details.

## Exercises ✒️
An exercise sheet is provided for (almost) all lecture units. Most of the time, the exercises are a compilation of old exam questions.
However, the exercises also include programming tasks and questions which would not be suitable for an exam (due to time constraints).
But the programming tasks can be used to collect bonus points for the exam (see description below).

The solutions will be provided via the Moodle forum after two weeks. **It is highly recommended to solve the exercises on your own!**
Do not wait for the solutions to be uploaded.

| Number        | Title                                               | Link 🔗                                                                                                 	 |
|---------------|-----------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| **Sheet 1:**  | Numeric Optimization Techniques                     | [Download](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/02_exercises/exercise1.pdf)   |
| **Sheet 2:**  | Decision Theory and Probability Density Estimation  | [Download](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/02_exercises/exercise2.pdf)   |
| **Sheet 3:**  | k-nearest Neighbors                                 | [Download](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/02_exercises/exercise3.pdf)   |
| **Sheet 4:**  | Linear Regression                                   | [Download](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/02_exercises/exercise4.pdf)   |
| **Sheet 5:**  | Logistic Regression                                 | [Download](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/02_exercises/exercise5.pdf)   |
| **Sheet 6:**  | Neural Networks / Deep Learning                     | [Download](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/02_exercises/exercise6.pdf)   |
| **Sheet 7:**  | Evaluation of Machine Learning Models               | [Download](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/02_exercises/exercise7.pdf)   |
| **Sheet 8:**  | Decision Trees and Ensemble Methods                 | [Download](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/02_exercises/exercise8.pdf)   |
| **Sheet 9:**  | Support Vector Machines                             | [Download](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/02_exercises/exercise9.pdf)   |
| **Sheet 10:** | Clustering                                          | [Download](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/02_exercises/exercise10.pdf)  |
| **Sheet 11:** | Principal Component Analysis                        | [Download](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/02_exercises/exercise11.pdf)  |

## Exam 📝
The exam is going to take 120 minutes. The maximum attainable score will be 120 points, so you have one minute per point.
**Important:** Keep your answers short and simple in order not to lose too much valuable time.

The exam questions will be given in German, but you may answer them in either English or German (you are also allowed to mix the languages if you like).
Please do not translate domain specific technical terms in order to avoid confusion. Please answer all questions (except for multiple choice questions) on the concept paper which is handed out during the exam.

**Exam preparation:**
* You will not be asked for lengthy derivations. Instead, I want to test whether you understand the general concepts.
* Any content not discussed in the lecture **will not be part** of the exam. A list of relevant topics will be shared at the end of the lecture.
* The exam will contain a mix of multiple choice questions, short answer questions and calculations.
* Make sure you can answer the self-test questions provided for each topic. You can find those at the end of each slide deck. **There won't be sample solutions for those questions!**
* Solve the exercises and work through the solutions if necessary! The solutions will be uploaded after two weeks.
* Some of the slides give you important hints (upper left corner):
	*  A slide marked with symbol (1) provides in-depth information which you do not have to know by heart. Think of it as additional material for the sake of completeness. However, do not ignore the content completely during the exam preparation. The content may still be relevant, but won't be a focus topic.
	*  Symbol (2) indicates very important content. Make sure you understand it!
* Have a look at the [old exam](https://github.com/DaWe1992/Applied_ML_Fundamentals/tree/master/03_exam) to familiarize yourself with the exam format.
* The last lecture slot is reserved for exam preparation and additional questions.

Symbol (1):

<img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/img/scream.png" width="60px" height="60px">

Symbol (2):

<img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/img/important.png" width="60px" height="60px">

**Auxiliary material for the exam:**
* Non-programmable pocket calculator
* Two-sided **hand-written** cheat sheet (you may note whatever you want). Hand-written means pen and paper (not on a tablet or computer!)

## Programming Tasks (Bonus Points for the Exam) 🐍
Almost every exercise sheet contains at least one programming task. You can collect bonus points for the exam by working on one of these tasks.

**How it works:**
* Form groups of 2 to 3 students.
* At the end of each lecture slot one group will be chosen to work on the next programming task. Each group will be given a separate programming task.
* The group solves the programming task and presents the results in the next lecture slot.
* The usage of advanced machine learning libraries (sklearn, etc.) is strictly forbidden. The goal is to implement the algorithms in Numpy from scratch.
* The code has to be shared with me.

**Grading:**
* A group can achieve 10 points per group member (2 students = 20 points, 3 students = 30 points).
* The group will be given a total number of points.
* The group members can distribute these points between them.
* No member can achieve more than 15 points.


## Literature and recommended Reading 📚
You do not need to buy any books for the lecture, most resources are available online. <br />
Please find a curated list below:

| Title                                      	       | Author(s)                    | View online 🔗                                                         																												     |
|------------------------------------------------------|------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Convex Optimization** 							   | Boyd/Vandenberghe (2004)     | [click here](https://web.stanford.edu/~boyd/cvxbook/)
| **Deep Learning**                            	       | Goodfellow et al. (2016)	  | [click here](https://www.deeplearningbook.org/) 		        																												         |
| **Elements of statistical Learning**                 | Hastie et al. (2008) 		  | [click here](https://www.sas.upenn.edu/~fdiebold/NoHesitations/BookAdvanced.pdf)																														     |
| **Gaussian Processes for Machine Learning**		   | Rasmussen/Williams (2006)	  | [click here](https://gaussianprocess.org/gpml/chapters/RW.pdf)																														     |
| **Machine Learning**                                 | Mitchell (1997)              | [click here](https://www.cs.cmu.edu/~tom/files/MachineLearningTomMitchell.pdf)																		                     |
| **Machine Learning - A probabilistic Perspective**   | Murphy (2012)				  | [click here](https://doc.lagout.org/science/Artificial%20Intelligence/Machine%20learning/Machine%20Learning_%20A%20Probabilistic%20Perspective%20%5BMurphy%202012-08-24%5D.pdf)		     |
| **Mathematics for Machine Learning**                 | Deisenroth et al. (2019)     | [click here](https://mml-book.github.io/)																																			     |
| **Pattern Recognition and Machine Learning** 	       | Bishop (2006)   			  | [click here](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) 							     |
| **Probabilistic Graphical Models** 			       | Koller et al. (2009)		  | [click here](https://github.com/Zhenye-Na/machine-learning-uiuc/blob/master/docs/Probabilistic%20Graphical%20Models%20-%20Principles%20and%20Techniques.pdf)                             |
| **Reinforcement Learning - An Introduction**         | Sutton et al. (2014)         | [click here](http://incompleteideas.net/book/bookdraft2017nov5.pdf)   																										        	 |
| **Speech and Language Processing** 				   | Jurafsky/Martin (2006)       | [click here](https://pages.ucsd.edu/~bakovic/compphon/Jurafsky,%20Martin.-Speech%20and%20Language%20Processing_%20An%20Introduction%20to%20Natural%20Language%20Processing%20(2007).pdf) |
| **The Matrix Cookbook**                              | Petersen et al. (2012)       | [click here](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)                                                                                                                 |

🔗 **YouTube resources:**
* [Machine learning lecture](https://www.youtube.com/watch?v=jGwO_UgTS7I&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU) by Andrew Ng, Stanford University (new version)
* [Machine learning lecture](https://www.youtube.com/watch?v=UzxYlbK2c7E&list=PLA89DCFA6ADACE599) by Andrew Ng, Stanford University (old version)
* [Support vector machines](https://www.youtube.com/watch?v=_PwhiWxHK8o) by Patrick Winston, MIT
* [Linear algebra](https://www.youtube.com/watch?v=ZK3O402wf1c&list=PL49CF3715CB9EF31D&index=1) by Gilbert Strang, MIT
* [Matrix methods in data analysis, signal processing, and machine learning](https://www.youtube.com/watch?v=Cx5Z-OslNWE&list=PLUl4u3cNGP63oMNUHXqIUcrkS2PivhN3k) by Gilbert Strang, MIT
* [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) (3BlueOneBrown)
* [Hidden Markov models](https://www.youtube.com/watch?v=fX5bYmnHqqE)
* [Viterbi algorithm](https://www.youtube.com/watch?v=IqXdjdOgXPM)

🔗 **Interesting papers:**
* [Playing atari with deep reinforcement learning](https://arxiv.org/abs/1312.5602) (Mnih et al., 2013)
* [Efficient estimation of word representations in vector space](https://arxiv.org/abs/1301.3781) (Mikolov et al., 2013)
* [Fast Training of Support Vector Machines using Sequential Minimal Optimization](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/smo-book.pdf) (Platt, 1998)
* [An Implementation of the Mean Shift Algorithm](https://www.ipol.im/pub/art/2019/255/article_lr.pdf) (Demirovic, 2019)

🔗 **Miscellaneous:**
* [CS229 Lecture Notes](https://cs229.stanford.edu/lectures-spring2022/main_notes.pdf) (Andrew Ng)
* [The simplified SMO Algorithm](https://cs229.stanford.edu/materials/smo.pdf) (CS229 Stanford)

## Bugs and Errors 🐞
Help me improve the lecture. Please feel free to file an issue in case you spot any errors or issues.
Thank you very much in advance! **Please do not open issues for questions concerning the content!** Either use the Moodle forum or send me an e-mail for that ([daniel.wehner@sap.com](mailto:daniel.wehner@sap.com)).

<sub>© 2025 Daniel Wehner, M.Sc.</sub>
