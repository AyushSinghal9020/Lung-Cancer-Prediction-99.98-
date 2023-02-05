# Model Addtion

* **KNeighborsRegressor** - KNeighbors Regressor is a `supervised` machine learning algorithm for `regression` problems. This is an `instance-based` algorithm that uses 
the k closest data points to make predictions for a given query instance. Query instance prediction is done by `averaging the k nearest neighbor response values`. K is a 
user-tunable hyperparameter to control model complexity. The predictions made by the algorithm are influenced by the choice of k and the distance metric used to compute 
the similarity between instances. KNeighbors Regressor can be used for both `linear` and `nonlinear regression` problems and is generally used for simple low-dimensional 
problems. 
$$D = \sqrt {(x_i - x_j)^2 (y_i - y_j)^2}$$

* **Linear Regression** - Linear regression is a `statistical technique` used to model the relationship between a dependent variable and one or more independent 
variables. The basic idea is to `fit a line` (linear model) to the data that best `predicts` the values of the `dependent variable` based on the `values` of the 
`independent variables`. `Simple linear regression` has only `one independent variable`. `Multiple linear regression` can have `multiple independent variables`. The 
coefficients are estimated using a method called `ordinary least squares (OLS)`. The goal is to find the value of the coefficient that minimizes the sum of the 
squared differences between the predicted and actual values of the dependent variable. Linear regression is commonly used for `predictive analysis` and helps us 
understand the relationships between variables and make predictions based on those relationships. 
$$f(x) = b_0x_0 + b_1x_1 + b_2x_2 b_3x_3 + ... + b_nx_n$$
$$f(b_0 , b_1 , b_2) = \frac {1}{2n} \sum \limits _{i = 1} ^{n}(h(x_i) + y_i)^2$$

* **Logistic Regression** - Logistic regression is a `statistical technique` for analyzing data sets that have one or `more independent variables` that determine the 
outcome. Used for `binary classification` problems where the result is either yes or no, true or false, or 0 or 1. Logistic regression uses the `logistic function`, 
also known as the `sigmoid function`, to model the relationship between the independent and dependent variables. This function maps each real number to a value between 
0 and 1 that can be interpreted as a probability. The predicted probabilities can then be thresholded to produce a binary result. Models are trained on labeled 
datasets by adjusting a `cost function` that measures the difference between predicted probabilities and actual outcomes. 
$$S(x) = \frac {1}{1+e^{-x}}$$
$$J(\theta_0 , \theta_1) = \frac {1}{2m} \sum \limits _{i = 1} ^{m}(h(x_i) + y_i)^2$$

* **Ridge Regreesion** - Ridge regression is a type of `regular linear regression` algorithm that helps prevent `overfitting` by adding a penalty term to the `cost 
function`. This adds a `"boundary"` term to the cost function. It is the sum of the squares of the coefficients multiplied by the `regularization parameter alpha`. 
The purpose of the ridge term is to reduce the coefficient size to reduce the risk of overfitting. In this way, ridge regression helps `balance` the `trade-off` 
between `fitting` the data well and having a simple model that generalizes well to unseen data. The `final prediction` is a `linear combination` of features and 
coefficients learned by the algorithm. The goal is to minimize the sum of squared residuals and ridge terms in the cost function. 
$$y = \frac {x_T y}{x^TX + \lambda I}$$

* **Lasso Regression** - LASSO regression is a `linear regression` technique that adds a `regularization` term to the loss function in order to reduce model complexity 
and avoid `overfitting`. The `regularization term` is a penalty on feature coefficient size and is controlled by the `hyperparameter alpha`. In lasso regression, the 
penalty term has an L1 norm (also called "lasso"), hence the name lasso regression. The L1 norm penalty has the effect of shrinking the coefficients of less important 
features to zero, making feature selection more efficient and reducing the number of features in the model. This produces a sparse model where only a subset of features 
are used in the final prediction. LASSO regression can be used for both linear and nonlinear regression and is especially useful in scenarios where the number of 
features is large compared to the number of samples. 
$$L_1 = \lambda \sum_{i=1}^p |\beta_i|$$
$$Least Square = \sum_{i=1}^n (y_i - \beta_0 - \sum_{j=1}^p x_{ij} \beta_j)^2$$
$$Combined Objective Function = \lambda \sum_{i=1}^p |\beta_i| + \sum_{i=1}^n (y_i - \beta_0 - \sum_{j=1}^p x_{ij} \beta_j)^2$$

* **Support Vector Machine** - A support vector machine (SVM) is a supervised learning algorithm used for classification and regression analysis. It tries to find a 
hyperplane in a high-dimensional space that maximally separates the data into different classes. Hyperplanes are chosen to maximize the margin, defined as the distance 
between the closest data points of different classes and the hyperplane. The closest data points, called support vectors, have the most influence on the hyperplane 
position. SVM can also handle nonlinear classification problems by transforming the data into a higher dimensional space and finding linear bounds in that space. SVMs 
have proven effective in various applications such as image classification, text classification, and bioinformatics. 
$$Hyperplace = w^Tx + b = 0 $$
$$Distance = \frac{|w^Tx + b|}{||w||} $$
$$Margin Maximization = \frac{2}{||w||} = y_i(w^Tx_i + b) \geq 1, i = 1, 2, ..., n $$

* **Gaussian Naive Bayes** - Gaussian Naive Bayes (Gaussian NB) is a probabilistic classification algorithm based on Bayes' theorem that assumes independence between 
each pair of features. Gaussian NB models the distribution of each feature as a Gaussian (normal) distribution and computes probabilities for each class given a 
feature vector. Then classify the input feature vector into the most probable class. Gaussian NB is a fast and simple algorithm that works well on high-dimensional 
datasets and is especially useful for text classification problems. Despite its simplicity and strong assumptions, Gaussian NB is very effective in practice and is 
often used as a base classifier for machine learning problems. 

$$Probablity Density Function = f(x) = \frac {1}{\sqrt {2*\pie*\sigma^2}} e^{\frac {-(x - \mu)^2}{2 * \sigma^2}}$$
$$Bayes Theorem = P(C | X) = \frac {P(X | C) P(C)}{P(X)}$$

* **Random Forest Classfier** - Random Forest Classifier (RFC) is a machine learning algorithm that uses an ensemble of decision trees to classify data into different 
classes. Train multiple decision trees on different subsets of the training data obtained by random sampling with replacement and combine their predictions to produce 
a final prediction. In RFC, two key concepts of his, bagging (bootstrap aggregation) and feature importance, are used to reduce overfitting and improve model accuracy. 
It is a robust and flexible algorithm that works well with a large number of functions and is widely used for both classification and regression problems. 
$$Gini(S) = 1 - ∑ (p_i)^2$$
$$Entropy(S) = - ∑ p_i log2(p_i)$$
