# 📔 Applied Machine Learning Fundamentals (Lecture)

```
'We are drowning in information and starving for knowledge.' – John Naisbitt
```

Machine learning / data science is a subfield of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience
**without being explicitly programmed**. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.
Machine learning has become an integral part of many modern applications. It is a cross-topic discipline which comprises computer science, math / statistics as well as
domain and business knowledge:

<p align="center">
	<img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/03_tex_files/03_img/data_science.png" width=400px>
</p>

The lecture 'Applied Machine Learning Fundamentals' is supposed to give a general introduction into state-of-the-art machine learning algorithms and their applications.

## Contents 📜

We are going to cover the following topics:

1. **Introduction to machine learning** ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/01_intro_ml.pdf))
    * Motivation and applications
    * Terminology
    * Key challenges in ML: Generalization, feature engineering, model selection, ...
2. **Mathematical foundations** ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/02_math.pdf))
	* Linear algebra
	* Statistics
	* Optimization
3. **Bayesian decision theory** ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/03_decision_theory.pdf))
    * Bayes optimal classifier
    * Naive Bayes
	* Risk minimization
4. **Probability density estimation** ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/04_density_estimation.pdf))
    * Parametric models
    * Non-parametric models
    * Gaussian mixture models and expectation maximization
5. **Supervised learning**
    * Regression ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/05_regression.pdf))
		* Linear regression
		* Probabilistic regression
		* Basis functions: Radial basis functions, polynomial basis functions
    * Classification I
		* k-nearest neighbors ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/06_knn.pdf))
		* Logistic regression ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/07_logistic_regression.pdf))
		* Decision trees and ensemble methods ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/08_decision_trees.pdf))
	* Classification II: Deep learning ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/10_deep_learning.pdf))
		* Perceptrons
		* Multi-layer-perceptrons and back-propagation
		* Deep learning application: NLP (word embeddings, text classification, sentiment analysis)
6. **Evaluation of ML models** ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/09_evaluation.pdf))
	* Out-of-sample testing and cross validation
	* Confusion matrices
	* Evaluation metrics: Precision, recall, F1 score, ROC, accuracy
	* Cost-sensitive evaluation
	* Model selection: Grid search, random search
7. **Unsupervised learning**
    * Clustering ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/12_clustering.pdf))
		* k-Means
		* Hierarchical clustering (divisive and agglomerative)
    * Principal component analysis ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/13_pca.pdf))
8. **Lecture summary**

A list of abbreviations, symbols and mathematical notation used in the context of the slides can be found [here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/symbols.pdf).
Please find additional material below.

## Assignments ✒️
The assignments are voluntary. All students who choose to participate have to form groups comprising three to four students (not more and not less).
The groups do not have to be static, you may form new groups for each assignment.
The task descriptions, starter code and data sets for the assignments can be found in the folder [02_exercises](https://github.com/DaWe1992/Applied_ML_Fundamentals/tree/master/02_exercises).
You have two weeks to answer the questions and to submit your work. The solutions are going to be presented and discussed after the submission deadline.
Sample solutions will **not** be uploaded. However, you are free to share correct solutions with your colleagues **after they have been graded**.

**Formal requirements for submissions:**
* Please submit your solutions via Moodle (as a *.zip* file) as well as in printed form. The *.zip* file must contain one *.pdf* file for the pen-and-paper tasks as well as one *.py* file per programming task. Only pen-and-paper tasks have to be printed, you do not have to print the source code.
* Only one member of the group has to submit the solutions. Please make sure to specify the matriculation numbers (not the names!) of all group members so that all participants receive the points they deserve!
* Please refrain from submitting hand-written solutions or images of solutions (*.png* / *.jpg* files). Rather use proper type-setting software like LaTeX or other comparable programs.
If you choose to use LaTeX, you may want to use the template files located [here](https://github.com/DaWe1992/Applied_ML_Fundamentals/tree/master/03_tex_files/02_exercises) for your answers.
* Code assignments have to be done in Python. Please submit *.py* files (no jupyter notebooks).
* The following packages are allowed for code submissions: *numpy*, *pandas* and *scipy*. Please ask **beforehand**, if you want to use a specific package not mentioned here.
* Do not use already implemented models (e.g. from *scikit-learn*).

> **Please make sure to fulfill the above mentioned formal requirements. Otherwise, you may risk to lose points. Submissions which severely violate the specifications might not get any points at all!**

**Grading details**

Your homework is going to be corrected and given back to you. Correct solutions are rewarded with a bonus for the exam which amounts to at most ten percent of the exam,
if all solutions submitted by you are correct (this corresponds to at most six points in the exam). It is still possible to achieve full points in the exam, even if you choose not to participate in the assignments (it is additional).
Below you find the [function](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/03_tex_files/03_img/bonus_point_function.png) which is used to compute the bonus as well as a legend which explains what the components mean.
Please note that this is not a linear function.

<p align="center">
	<img src="https://render.githubusercontent.com/render/math?math=b(a) = \text{min}\left(B, \left\lceil \frac{B}{A^2} \cdot a^2 \right\rceil \right)" width=300px height=130px>
</p>

| Parameter 													                        			|	Explanation										|	Value 		|
|---------------------------------------------------------------------------------------------------|---------------------------------------------------|---------------|
| <img src="https://render.githubusercontent.com/render/math?math=b" width=15px height=15px>		|	bonus points attained for the exam				| 	up to you	|
| <img src="https://render.githubusercontent.com/render/math?math=B" width=15px height=15px>		|	maximum attainable bonus points for the exam	|	6			|
| <img src="https://render.githubusercontent.com/render/math?math=A" width=15px height=15px>		|	maximum attainable points in the assignments	|	40 			|
| <img src="https://render.githubusercontent.com/render/math?math=a" width=15px height=15px>		|	score achieved in the assignments				|	up to you	|

**Please note:** You have to pass the exam **without the bonus points!** This means that it is not possible to turn a failing grade (= 5.0) into a passing grade (<= 4.0).
The bonus points will be taken into account in case you have to repeat the exam (i.e. they do not expire if you fail the first attempt).
Unsurprisingly, the solutions have to be your own work. **If you plagiarize, you will lose all bonus points!**

## Exam 📝
The exam will take 60 minutes. The maximum attainable score will be 60 points, so you have one minute per point.
**Important:** Keep your answers short and simple in order not to lose too much valuable time.
The exam questions will be given in English, but you may answer them in either English or German (you are also allowed to mix the languages).
Please do not translate domain specific technical terms in order to avoid confusion. Please answer all questions on the task sheets.

**Exam preparation:**
* You will not be asked for any derivations, rather we want to test whether you understand the general concepts.
* The exam will contain multiple choice questions, short answer questions and calculations.
* Make sure you can answer the self-test questions provided for each topic.
* Some of the slides give you important hints (upper left corner):
	*  A slide marked with symbol (1) provides in-depth information which you do not have to know by heart (think of it as additional material for the sake of completeness).
	*  Symbol (2) indicates very important content. Make sure you understand it!
* Work through the [list of 150+ terms](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/04_exam/terms_you_have_to_know.pdf) which you should be able to explain.

Symbol (1):

<img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/03_tex_files/03_img/scream.png" width="60px" height="60px">

Symbol (2):

<img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/03_tex_files/03_img/important.png" width="60px" height="60px">

**Auxiliary material for the exam:**
* Non-programmable pocket calculator
* Two-sided **hand-written** cheat sheet (you may note whatever you want)

## Python Code 🐍
Machine learning algorithms (probably all algorithms) are easier to understand, if you see them implemented. Please find Python implementations for some of the algorithms (which are not part of the assignments) in the folder
[06_python](https://github.com/DaWe1992/Applied_ML_Fundamentals/tree/master/06_python).
Play around with the hyper-parameters of the algorithms and try different data sets in order to get a better feeling for how the algorithms work. Also, debug through the code line for line and check what each line does.

## Literature and recommended Reading 📚
You do not need to buy any books for the lecture, most resources are available online. Please find a curated list below:

| Title                                    	     | Author(s)                    | Publisher 				| View online                                                         																														|
|------------------------------------------------|------------------------------|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Deep Learning                            	     | Goodfellow et al. (2016)		| MIT Press     			| [click here](https://www.deeplearningbook.org/) 		        																															|
| Elements of statistical Learning               | Hastie et al. (2008) 		| Springer      			| [click here](https://web.stanford.edu/~hastie/Papers/ESLII.pdf)																															|
| Machine Learning                               | Mitchell (1997)              | McGraw-Hill   			| [click here](https://www.cs.ubbcluj.ro/~gabis/ml/ml-books/McGrawHill%20-%20Machine%20Learning%20-Tom%20Mitchell.pdf)																		|
| Machine Learning - A probabilistic perspective | Murphy (2012)				| MIT Press     			| [click here](https://doc.lagout.org/science/Artificial%20Intelligence/Machine%20learning/Machine%20Learning_%20A%20Probabilistic%20Perspective%20%5BMurphy%202012-08-24%5D.pdf)			|
| Mathematics for Machine Learning               | Deisenroth et al. (2019)     | Cambridge Univ. Press		| [click here](https://mml-book.github.io/)																																					|
| Pattern Recognition and Machine Learning 	     | Bishop (2006)   			    | Springer  				| [click here](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) 								|
| Reinforcement Learning - An introduction       | Sutton et al. (2014)         | MIT Press     			| [click here](http://incompleteideas.net/book/bookdraft2017nov5.pdf)   																													|
| Probabilistic Graphical Models 				 | Koller et al. (2009)			| MIT Press                 | [click here](https://github.com/Zhenye-Na/machine-learning-uiuc/blob/master/docs/Probabilistic%20Graphical%20Models%20-%20Principles%20and%20Techniques.pdf)                              |

**YouTube resources:**
* [Machine learning lecture](https://www.youtube.com/watch?v=UzxYlbK2c7E&list=PLA89DCFA6ADACE599) by Andrew Ng, Stanford University
* [Support vector machines](https://www.youtube.com/watch?v=_PwhiWxHK8o) by Patrick Winston, MIT
* [Linear algebra](https://www.youtube.com/watch?v=ZK3O402wf1c&list=PL49CF3715CB9EF31D&index=1) by Gilbert Strang, MIT
* [Matrix methods in data analysis, signal processing, and machine learning](https://www.youtube.com/watch?v=Cx5Z-OslNWE&list=PLUl4u3cNGP63oMNUHXqIUcrkS2PivhN3k) by Gilbert Strang, MIT

**Interesting papers:**
* [Playing atari with deep reinforcement learning](https://arxiv.org/abs/1312.5602) (Mnih et al., 2013)
* [Efficient estimation of word representations in vector space](https://arxiv.org/abs/1301.3781) (Mikolov et al., 2013)

In general, we recommend the [Coursera machine learning course](https://de.coursera.org/learn/machine-learning#about) by Andrew Ng.
You can get a nice certificate if you want (around $60):

<img src="https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/03_tex_files/03_img/ml_certificate.png" width="400px" height="300px">

In the exam we will not ask anything which was not discussed in the lecture. Regard the literature as additional resources in case you want to dig deeper into
specific topics. Please give me a hint, if you feel that some important resources are missing. I am happy to add them here.

## Data Exploration Project (4th Semester) 📐
Please find the material for the project in the folder [05_project](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/05_project).

The material contains:
* Organization and goals of the project
* List of topic suggestions
* Submission details
* Grading details

## Additional Material 🎁
Have a look at the following slides (unless stated otherwise, these slides are not relevant for the exam):

1. **Support vector machines** ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/11_svm.pdf))
	* Linear SVMs
	* Non-linear SVMs and the kernel-trick
	* Soft-margin SVMs
2. **Reinforcement learning** ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/14_rl.pdf))
    * Markov decision processes
	* Algorithms:
		* Policy iteration and value iteration
		* Q-learning and Q-networks
		* Policy gradient methods
3. **Probabilistic graphical models** ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/15_pgm.pdf))
	* Bayesian networks (representation and inference)
	* Hidden Markov models and viterbi
4. **Apriori / association rules** ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/16_association_rules.pdf))
5. **Data preprocessing** ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/17_data_preprocessing.pdf))
	* Data mining processes (KDD, CRISP-DM)
	* Data cleaning
	* Data transformation (e. g. normalization, discretization)
	* Data reduction and feature subset selection
	* Data integration
6. **Advanced regression** ([click here](https://github.com/DaWe1992/Applied_ML_Fundamentals/blob/master/01_slides/18_advanced_regression.pdf))
	* Bayesian regression
	* Kernel regression
	* Gaussian process regression
	* Support vector regression
7. **Deep learning II** (not yet available)

## Bugs and Errors 🐞
Help me improve the lecture. Feel free to file an issue, if you spot any errors in the slides, exercises or code.
Thank you very much in advance! **Please do not open issues for questions concerning the content!** Either use the Moodle forum or send me an e-mail for that ([daniel.wehner@sap.com](mailto:daniel.wehner@sap.com)).

<sub>© 2020 Daniel Wehner</sub>
