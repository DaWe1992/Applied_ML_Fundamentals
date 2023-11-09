# Applied Machine Learning Fundamentals (Lecture) 🤖

```
'We are drowning in information and starving for knowledge.' – John Naisbitt
```

Machine learning / data science is a subfield of artificial intelligence (AI) that provides systems the ability to **automatically learn and improve** from experience
**without being explicitly programmed**. Machine learning focuses on the development of computer programs which can access data and use it to learn for themselves.
A machine learning algorithm learns by building a mathematical / statistical model from the data. This model can then be used for inference and decision making. 
Machine learning has become an integral part of many modern applications. It is a **cross-topic discipline** which comprises computer science, math / statistics as well as
domain and business knowledge:

<p align="center">
	<img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/img/data_science.png" width=400px>
</p>

The lecture *'Applied Machine Learning Fundamentals'* is supposed to give a general introduction into state-of-the-art machine learning algorithms and their applications.
This Readme file provides you with all necessary information. It is structured as follows:

1. 📜 Lecture contents
2. ✒️ Exercises
3. 📝 Exam
4. 🐍 Python code
5. 📚 Literature and recommended reading
6. 🐞 Bugs and errors

## Lecture Contents 📜

The following topics / algorithms will be covered in the lecture:

1. **Introduction to machine learning** ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/01_intro_ml.pdf))
    * Motivation and applications
    * Terminology
    * Key challenges in ML: Generalization, feature engineering, model selection, ...
2. **Mathematical foundations** ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/02_math.pdf))
	* Linear algebra
	* Statistics
	* Optimization
3. **Bayesian decision theory** ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/03_decision_theory.pdf))
    * Bayes optimal classifiers
	* Error minimization vs. risk minimization
    * Multinomial and Gaussian naive Bayes
	* Probability density estimation and maximum likelihood estimation 
4. **Supervised learning**
    * Regression ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/04_regression.pdf))
		* Linear regression
		* Probabilistic regression
		* Basis function regression
    * Classification I
		* k-nearest neighbors ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/05_knn.pdf))
		* Logistic regression ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/06_logistic_regression.pdf))
		* Decision trees and ensemble methods ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/07_decision_trees.pdf))
	* Classification II: Deep learning ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/09_deep_learning.pdf))
		* Perceptrons
		* Multi-layer-perceptrons and back-propagation
		* Further network architectures (CNNs, RNNs)
5. **Evaluation of ML models** ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/08_evaluation.pdf))
	* Out-of-sample testing and cross validation
	* Confusion matrices
	* Evaluation metrics: Precision, recall, F1 score, ROC, accuracy, RMSE, MAE
	* Model selection: Grid search, random search, early stopping
	* Bias-variance decomposition
6. **Unsupervised learning**
    * Clustering ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/10_clustering.pdf))
		* k-Means
		* Hierarchical clustering
		* DBSCAN
    * Principal component analysis ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/11_pca.pdf))
7. **Lecture summary**

## Exercises ✒️
An exercise sheet is provided for each lecture unit. Most of the time, the exercises are a compilation of old exam questions.
However, the exercises also include programming tasks and questions which would not be suitable for an exam (due to time constraints).

The solutions will be provided via the Moodle forum after two weeks. **It is highly recommended to solve the exercises on your own!**
Do not wait for the solutions to be uploaded.

| Number        | Title                                                              | Link 🔗                                                                                                     |
|---------------|-----------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| **Sheet 1:**  | Numeric Optimization Techniques                     | [Download](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/02_exercises/exercise1.pdf)   |
| **Sheet 2:**  | Decision Theory and Probability Density Estimation  |  [Download](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/02_exercises/exercise2.pdf)  |
| **Sheet 3:**  | Linear Regression                                   | [Download](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/02_exercises/exercise3.pdf)   |
| **Sheet 4:**  | Logistic Regression                                 | [Download](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/02_exercises/exercise4.pdf)   |
| **Sheet 5:**  | k-nearest Neighbors                                 | [Download](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/02_exercises/exercise5.pdf)   |
| **Sheet 6:**  | Decision Trees and Ensemble Methods                 | [Download](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/02_exercises/exercise6.pdf)   |
| **Sheet 7:**  | Neural Networks / Deep Learning                     | [Download](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/02_exercises/exercise7.pdf)   |
| **Sheet 8:**  | Evaluation of Machine Learning Models               | [Download](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/02_exercises/exercise8.pdf)   |
| **Sheet 9:**  | Clustering                                          | [Download](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/02_exercises/exercise9.pdf)   |
| **Sheet 10:** | Principal Component Analysis                        | [Download](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/02_exercises/exercise10.pdf)  |

## Exam 📝
The exam is going to take 60 minutes. The maximum attainable score will be 60 points, so you have one minute per point.
**Important:** Keep your answers short and simple in order not to lose too much valuable time.

The exam questions will be given in German, but you may answer them in either English or German (you are also allowed to mix the languages).
Please do not translate domain specific technical terms in order to avoid confusion. Please answer all questions on the task sheets (you may also write on the empty back-sides).

**Exam preparation:**
* You will not be asked for any derivations, rather I want to test whether you understand the general concepts.
* Any content not discussed in the lecture **will not be part** of the exam.
* The exam will contain a mix of multiple choice questions, short answer questions and calculations.
* Make sure you can answer the self-test questions provided for each topic. **There won't be sample solutions for those questions!**
* Solve the exercises and work through the solutions if necessary!
* Some of the slides give you important hints (upper left corner):
	*  A slide marked with symbol (1) provides in-depth information which you do not have to know by heart (think of it as additional material for the sake of completeness).
	*  Symbol (2) indicates very important content. Make sure you understand it!
* Have a look at the [old exams](https://github.com/DaWe1992/Applied_ML_Fundamentals/tree/master/03_exam) to familiarize yourself with the exam format.
* The last lecture slot is reserved for exam preparation and additional questions.

Symbol (1):

<img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/img/scream.png" width="60px" height="60px">

Symbol (2):

<img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/img/important.png" width="60px" height="60px">

**Auxiliary material for the exam:**
* Non-programmable pocket calculator
* Two-sided **hand-written** cheat sheet (you may note whatever you want). Hand-written means pen and paper (not on tablet!)

**Exam grading**
Since the lecture *Applied Machine Learning Fundamentals* is part of a bigger module (*Machine Learning Fundamentals, W3WI_DS304)*, it is not graded individually.
Instead, the score you achieved in the exam (at most 60 points) will be added to the points you receive in the second element of the module, the *Data Exploration Project* in the 4th semester
which is also worth 60 points at maximum. **Your performance in both elements combined will determine your eventual grade.**

Please refer to the official [DHBW data science module catalogue](https://www.dhbw.de/fileadmin/user/public/SP/MA/Wirtschaftsinformatik/Data_Science.pdf) for further details.

## Python Code 🐍
Machine learning algorithms are easier to understand, if you see them implemented.
Please find Python implementations for some of the algorithms in this [repository](https://github.com/DaWe1992/Applied_ML_Algorithms).

Play around with the hyper-parameters of the algorithms and try different data sets in order to get a better feeling for how the algorithms work.
Also, debug through the code line by line and check what each line does.
Please find further instructions in the Readme there.

## Literature and recommended Reading 📚
You do not need to buy any books for the lecture, most resources are available online. <br />
Please find a curated list below:

| Title                                      	       | Author(s)                    | View online 🔗                                                         																												|
|------------------------------------------------------|------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Deep Learning**                            	       | Goodfellow et al. (2016)	  | [click here](https://www.deeplearningbook.org/) 		        																												    |
| **Elements of statistical Learning**                 | Hastie et al. (2008) 		  | [click here](https://web.stanford.edu/~hastie/Papers/ESLII.pdf)																														|
| **Machine Learning**                                 | Mitchell (1997)              | [click here](https://www.cin.ufpe.br/~cavmj/Machine%20-%20Learning%20-%20Tom%20Mitchell.pdf)																		                |
| **Machine Learning - A probabilistic Perspective**   | Murphy (2012)				  | [click here](https://doc.lagout.org/science/Artificial%20Intelligence/Machine%20learning/Machine%20Learning_%20A%20Probabilistic%20Perspective%20%5BMurphy%202012-08-24%5D.pdf)		|
| **Mathematics for Machine Learning**                 | Deisenroth et al. (2019)     | [click here](https://mml-book.github.io/)																																			|
| **Pattern Recognition and Machine Learning** 	       | Bishop (2006)   			  | [click here](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) 							|
| **Probabilistic Graphical Models** 			       | Koller et al. (2009)		  | [click here](https://github.com/Zhenye-Na/machine-learning-uiuc/blob/master/docs/Probabilistic%20Graphical%20Models%20-%20Principles%20and%20Techniques.pdf)                        |
| **Reinforcement Learning - An Introduction**         | Sutton et al. (2014)         | [click here](http://incompleteideas.net/book/bookdraft2017nov5.pdf)   																												|
| **The Matrix Cookbook**                              | Petersen et al. (2012)       | [click here](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)                                                                                                            |

🔗 **YouTube resources:**
* [Machine learning lecture](https://www.youtube.com/watch?v=UzxYlbK2c7E&list=PLA89DCFA6ADACE599) by Andrew Ng, Stanford University
* [Support vector machines](https://www.youtube.com/watch?v=_PwhiWxHK8o) by Patrick Winston, MIT
* [Linear algebra](https://www.youtube.com/watch?v=ZK3O402wf1c&list=PL49CF3715CB9EF31D&index=1) by Gilbert Strang, MIT
* [Matrix methods in data analysis, signal processing, and machine learning](https://www.youtube.com/watch?v=Cx5Z-OslNWE&list=PLUl4u3cNGP63oMNUHXqIUcrkS2PivhN3k) by Gilbert Strang, MIT
* [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) (3BlueOneBrown)

🔗 **Interesting papers:**
* [Playing atari with deep reinforcement learning](https://arxiv.org/abs/1312.5602) (Mnih et al., 2013)
* [Efficient estimation of word representations in vector space](https://arxiv.org/abs/1301.3781) (Mikolov et al., 2013)

## Bugs and Errors 🐞
Help me improve the lecture. Please feel free to file an issue in case you spot any errors.
Thank you very much in advance! **Please do not open issues for questions concerning the content!** Either use the Moodle forum or send me an e-mail for that ([daniel.wehner@sap.com](mailto:daniel.wehner@sap.com)).

<sub>© 2023 Daniel Wehner, M.Sc.</sub>
