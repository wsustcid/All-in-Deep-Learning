<!--
 * @Author: Shuai Wang
 * @Github: https://github.com/wsustcid
 * @Version: 1.0.0
 * @Date: 2020-05-06 22:16:28
 * @LastEditTime: 2020-05-08 11:49:02
 -->

# Overview

## How to Choose
How to Choose estimator
https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

# Preprocessing
## StandardScaler
### Introduction
Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual features do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).

Standardize features by removing the mean and scaling to unit variance. The standard score of a sample x is calculated as:
$$
z = (x - u) / s
$$
where u is the mean of the training samples or zero if with_mean=False, and s is the standard deviation of the training samples or one if with_std=False.

- **Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set**. Mean and standard deviation are then stored to be used on later data using transform.
- For instance many elements used in the objective function of a learning algorithm (such as the **RBF kernel** of Support Vector Machines or the **L1 and L2 regularizers** of linear models) **assume that all features are centered around 0 and have variance in the same order**. If a feature has a variance that is orders of magnitude larger that others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected. (应该是一些需要迭代求解的算法(算法中有max_iter参数的)，归一化后方便迭代学习；有显式解的算法不受影响； 我初始化权重矩阵都是一个尺度的，那么你某个特征尺度大，就会导致这个特征在目标方程中占比大，也等价于其对应权重大，因此会主导优化的方向)
- 总的来说，特征归一化有两个作用：一是消除特征之间单位和尺度差应的影响，使得对每维特征同等看待；二是原始特征的尺度差异，会导致损失函数等高线是椭圆，下降走zigzag，归一化后接近圆形，梯度下降振荡较小；
- 更详细的分析参考：<https://www.cnblogs.com/shine-lee/p/11779514.html>

### Usage
```python
class sklearn.preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True
```
**Methods:**
- fit(self, X[, y]): Compute the mean and std to be used for later scaling.(fit完之后才能transform)
- fit_transform(self, X[, y]): Fit to data, then transform it.
- get_params(self[, deep]): Get parameters for this estimator.
- inverse_transform(self, X[, copy]): Scale back the data to the original representation
- partial_fit(self, X[, y]): Online computation of mean and std on X for later scaling.
- set_params(self, \*\*params): Set the parameters of this estimator.
- transform(self, X[, copy]): Perform standardization by centering and scaling


## [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
### Introduction
Pipeline of transforms with a final estimator: Sequentially apply **a list of transforms and a final estimator**. Intermediate steps of the pipeline must be ‘transforms’, that is, they must implement fit and transform methods. The final estimator only needs to implement fit. The transformers in the pipeline can be cached using memory argument.

The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters. For this, it enables setting parameters of the various steps using their names and the parameter name separated by a ‘__’, as in the example below. A step’s estimator may be replaced entirely by setting the parameter with its name to another estimator, or a transformer removed by setting it to ‘passthrough’ or None.

### Usage
```python
class sklearn.pipeline.Pipeline(steps, memory=None, verbose=False)
```
- steps: list; List of (name, transform) tuples (implementing fit/transform) that are chained, in the order in which they are chained, with the last object an estimator.
**Methods:**
- fit(self, X[, y]): Fit the model

# Regression Models
## [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
### Introduction
Ordinary least squares Linear Regression.
$$
min_w ||Xw-y||_2^2
$$
LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.

### Usage
```python
class sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
```

**Methods**
  - fit(self, X, y[, sample_weight]): Fit linear model.
    - X: array-like, sparse matrix of shape (n_samples, n_features) Training data
    - y: array-like of shape (n_samples,) or (n_samples, n_targets) Target values. Will be cast to X’s dtype if necessary
    - sample_weight: array-like of shape (n_samples,), default=None.Individual weights for each sample
    - Returns: self,returns an instance of self.
  - get_params(self[, deep]): Get parameters for this estimator.
  - predict(self, X): Predict using the linear model.
  - score(self, X, y[, sample_weight]): Return the coefficient of determination R^2 of the prediction.
  - set_params(self, \*\*params): Set the parameters of this estimator.

## [Polynomial Regression](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
### Introduction
Generate polynomial and interaction features. Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. For example, if an input sample is two dimensional and of the form [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].
$$
(x_0+x_1+x_2)^2 + (x_0+x_1+x_2)^1 + (x_0+x_1+x_2)^0
$$
Note:
Be aware that the number of features in the output array scales polynomially in the number of features of the input array, and exponentially in the degree. High degrees can cause overfitting.

### Usage
```python
class sklearn.preprocessing.PolynomialFeatures(degree=2, interaction_only=False, include_bias=True, order='C')[source]
```
- degree：控制多项式的度
- interaction_only： 默认为False，如果指定为True，那么就不会有特征自己和自己结合的项
- include_bias：默认为True。如果为True的话，那么就会有上面的 1那一项
**Methods**
  - fit(self, X[, y]): Compute number of output features. return self
  - fit_transform(self, X[, y]): Fits transformer to X and y with optional parameters fit_params and returns a transformed version of X
  - get_feature_names(self[, input_features]): Return feature names for output features
  - get_params(self[, deep]): Get parameters for this estimator.
  - set_params(self, \*\*params): Set the parameters of this estimator.
  - transform(self, X): Transform data to polynomial features

## [Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
### Introduction
Linear least squares with l2 regularization. Minimizes the objective function:
$$
||y - Xw||^2_2 + \alpha * ||w||^2_2
$$
- This model solves a regression model where the loss function is the linear least squares function and regularization is given by the l2-norm. Also known
- as Ridge Regression or Tikhonov regularization. 
This estimator has built-in support for multi-variate regression (i.e., when y is a 2d-array of shape (n_samples, n_targets)).

### Usage
```python
class sklearn.linear_model.Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None)
```
- alpha: Regularization strength; must be a positive float.
- fit_interceptbool: default=True, Whether to calculate the intercept for this model.
- max_iterint: default=None, Maximum number of iterations for conjugate gradient solver. For ‘sparse_cg’ and ‘lsqr’ solvers, the default value is determined by scipy.sparse.linalg. For ‘sag’ solver, the default value is 1000.
- tol: float,default=1e-3,Precision of the solution.

**Methods**
  - fit(self, X, y[, sample_weight]): Fit Ridge regression model.
  - get_params(self[, deep]): Get parameters for this estimator.
  - predict(self, X): Predict using the linear model.
  - score(self, X, y[, sample_weight]): Return the coefficient of determination R^2 of the prediction.
  - set_params(self, \*\*params): Set the parameters of this estimator.

## [Lasso Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
### Introduction
Linear Model trained with L1 prior as regularizer (aka the Lasso). The optimization objective for Lasso is:
$$
(1 / (2 * n_{samples})) * ||y - Xw||^2_2 + \alpha * ||w||_1
$$

### Usage
```python
class sklearn.linear_model.Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
```
- 
**Methods**
  - fit(self, X, y[, check_input])
  - get_params(self[, deep])
  - path(X, y[, l1_ratio, eps, n_alphas, …])
  - predict(self, X)
  - score(self, X, y[, sample_weight])
  - set_params(self, \*\*params)
  
## [Support Vector Regression](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
### Introduction
The method of Support Vector Classification can be extended to solve regression problems. This method is called Support Vector Regression.

The model produced by support vector classification (as described above) depends only on a subset of the training data, because the cost function for building the model does not care about training points that lie beyond the margin. Analogously, the model produced by Support Vector Regression depends only on a subset of the training data, because the cost function for building the model ignores any training data close to the model prediction.

For more details: <https://scikit-learn.org/stable/modules/svm.html>
### Usage
Epsilon-Support Vector Regression.
```python
class sklearn.svm.SVR(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
```
The free parameters in the model are C and epsilon.
- kernel: string, optional (default=’rbf’); Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable.
- degree: int, optional (default=3); Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
- gamma: {‘scale’, ‘auto’} or float, optional (default=’scale’); Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. 
  - if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
  - if ‘auto’, uses 1 / n_features.
- coef0: float, optional (default=0.0); Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
- tol: float, optional (default=1e-3); Tolerance for stopping criterion.
- C: float, optional (default=1.0); Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.
- epsilon: float, optional (default=0.1); Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.
- shrinking: boolean, optional (default=True); Whether to use the shrinking heuristic.
- cache_size: float, optional; Specify the size of the kernel cache (in MB).
- verbose: bool, default: False; Enable verbose output.
- max_iter: int, optional (default=-1); Hard limit on iterations within solver, or -1 for no limit.

**Methods**
  - fit(self, X, y[, check_input])
  - get_params(self[, deep])
  - predict(self, X)
  - score(self, X, y[, sample_weight])
  - set_params(self, \*\*params)

## [Decision Tree Regression](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
### Introduction
A decision tree regressor.
https://scikit-learn.org/stable/modules/tree.html#tree

### Usage
```python
class sklearn.tree.DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort='deprecated', ccp_alpha=0.0)
```
- criterion: {“mse”, “friedman_mse”, “mae”}, default=”mse”； The function to measure the quality of a split. 
- splitter: {“best”, “random”}, default=”best”; The strategy used to choose the split at each node. 
- max_depth: int, default=None; The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
- min_samples_split: int or float, default=2; The minimum number of samples required to split an internal node:
- min_samples_leaf: int or float, default=1; The minimum number of samples required to be at a leaf node. 
- min_weight_fraction_leaf: float, default=0.0; The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
- max_features: int, float or {“auto”, “sqrt”, “log2”}, default=None; The number of features to consider when looking for the best split:
- max_leaf_nodes: int, default=None; Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. 
**Methods**
  - apply(self, X[, check_input]): Return the index of the leaf that each sample is predicted as.
  - cost_complexity_pruning_path(self, X, y[, …]): Compute the pruning path during Minimal Cost-Complexity Pruning.
  - decision_path(self, X[, check_input]): Return the decision path in the tree.
  - fit(self, X, y[, sample_weight, …]): Build a decision tree regressor from the training set (X, y).
  - get_depth(self): Return the depth of the decision tree.
  - get_n_leaves(self): Return the number of leaves of the decision tree
  - get_params(self[, deep]): Get parameters for this estimator.
  - predict(self, X[, check_input]): Predict class or regression value for X
  - score(self, X, y[, sample_weight]): Return the coefficient of determination R^2 of the prediction.
  - set_params(self, \*\*params): Set the parameters of this estimator.

## [Random Forest Regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
### Introduction
A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default).
https://scikit-learn.org/stable/modules/ensemble.html#forest
$$

$$

### Usage
```python
class sklearn.ensemble.RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)
```
- n_estimators: integer, optional (default=10): The number of trees in the forest.
**Methods**
  - apply(self, X[, check_input]): Return the index of the leaf that each sample is predicted as.
  - decision_path(self, X[, check_input]): Return the decision path in the tree.
  - fit(self, X, y[, sample_weight, …]): Build a decision tree regressor from the training set (X, y).
  - get_params(self[, deep]): Get parameters for this estimator.
  - predict(self, X[, check_input]): Predict class or regression value for X
  - score(self, X, y[, sample_weight]): Return the coefficient of determination R^2 of the prediction.
  - set_params(self, \*\*params): Set the parameters of this estimator.





# Evaluation
## [Cross-Validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)
### Introduction
Evaluate a score by cross-validation
### Usage
```python
sklearn.model_selection.cross_val_score(estimator, X, y=None, groups=None, scoring=None, cv=None, n_jobs=None, verbose=0, fit_params=None, pre_dispatch='2*n_jobs', error_score=nan)
```
- scoring: string, callable or None, optional, default: None; A string (see model evaluation documentation) or a scorer callable object / function with signature scorer(estimator, X, y) which should return only a single value. If None, the estimator’s default scorer (if available) is used.
- Return: Array of scores of the estimator for each run of the cross validation.


## Metrics
### R-squared Score
The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares ((y_true - y_pred) ** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum(). The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.

## Grid Search
```python

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)
    cv_sets.get_n_splits(X)

    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor(random_state=0)

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth':range(1,11)}

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search cv object --> GridSearchCV()
    # Make sure to include the right parameters in the object:
    # (estimator, param_grid, scoring, cv) which have values 'regressor', 'params', 'scoring_fnc', and 'cv_sets' respectively.
    grid = GridSearchCV(regressor, params, scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.best_estimator_.get_params()['max_depth']))
print("Best Score is {:.2f}".format(reg.best_score_))

```