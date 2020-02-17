# KNN을 통한 Parameter Tuning

## Assignment 1 <a id="Assignment-1"></a>

KNN으로 HyperParameter 이해하기

## Load Dataset <a id="Load-Dataset"></a>

### **Import packages**

```python
# data
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore") 

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from pandas.plotting import parallel_coordinates

# preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# grid search
from sklearn.model_selection import GridSearchCV

# evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import *
```

### **Load iris data**

```python
from sklearn.datasets import load_iris
# sklearn에 내장되어있는 iris 데이터를 사용
```

```python
iris = load_iris()
```

```python
print(iris.DESCR)
# iris dataset 정보를 알 수 있다
```

* feature에는 sepal length, sepal width, petal length, petal width가 있다.
* target은 3개의 class가 있으며 각각 Iris-Setosa, Iris-Versicolour, Iris-Virginica, 즉, 붖꽃의 종류이다.
* Setosa, Versicolour, Virginica가 각각 0, 1, 2로 분류 되어있다.
* 총 150개의 instance가 존재한다.

```text
Iris Plants Database
====================

Notes
-----
Data Set Characteristics:
    :Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, predictive attributes and the class
    :Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        - class:
                - Iris-Setosa
                - Iris-Versicolour
                - Iris-Virginica
    :Summary Statistics:

    ============== ==== ==== ======= ===== ====================
                    Min  Max   Mean    SD   Class Correlation
    ============== ==== ==== ======= ===== ====================
    sepal length:   4.3  7.9   5.84   0.83    0.7826
    sepal width:    2.0  4.4   3.05   0.43   -0.4194
    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    petal width:    0.1  2.5   1.20  0.76     0.9565  (high!)
    ============== ==== ==== ======= ===== ====================

    :Missing Attribute Values: None
    :Class Distribution: 33.3% for each of 3 classes.
    :Creator: R.A. Fisher
    :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
    :Date: July, 1988

This is a copy of UCI ML iris datasets.
http://archive.ics.uci.edu/ml/datasets/Iris

The famous Iris database, first used by Sir R.A Fisher

This is perhaps the best known database to be found in the
pattern recognition literature.  Fisher's paper is a classic in the field and
is referenced frequently to this day.  (See Duda & Hart, for example.)  The
data set contains 3 classes of 50 instances each, where each class refers to a
type of iris plant.  One class is linearly separable from the other 2; the
latter are NOT linearly separable from each other.

References
----------
   - Fisher,R.A. "The use of multiple measurements in taxonomic problems"
     Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
     Mathematical Statistics" (John Wiley, NY, 1950).
   - Duda,R.O., & Hart,P.E. (1973) Pattern Classification and Scene Analysis.
     (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
   - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
     Structure and Classification Rule for Recognition in Partially Exposed
     Environments".  IEEE Transactions on Pattern Analysis and Machine
     Intelligence, Vol. PAMI-2, No. 1, 67-71.
   - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
     on Information Theory, May 1972, 431-433.
   - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
     conceptual clustering system finds 3 classes in the data.
   - Many, many more ...

```

### **Make DataFrame**

```python
# feature와 target를 하나의 DataFrame으로 만들고 각각의 column명을 붙여주었다.
df = pd.DataFrame(iris.data, columns = iris.feature_names)
y = pd.Series(iris.target, dtype="category")
y = y.cat.rename_categories(iris.target_names)
df['species'] = y
```

```text
df.head()
```

|  | sepal length \(cm\) | sepal width \(cm\) | petal length \(cm\) | petal width \(cm\) | species |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 5.1 | 3.5 | 1.4 | 0.2 | setosa |
| 1 | 4.9 | 3.0 | 1.4 | 0.2 | setosa |
| 2 | 4.7 | 3.2 | 1.3 | 0.2 | setosa |
| 3 | 4.6 | 3.1 | 1.5 | 0.2 | setosa |
| 4 | 5.0 | 3.6 | 1.4 | 0.2 | setosa |

## **EDA**

```text
df.describe()
```

|  | sepal length \(cm\) | sepal width \(cm\) | petal length \(cm\) | petal width \(cm\) |
| :--- | :--- | :--- | :--- | :--- |
| count | 150.000000 | 150.000000 | 150.000000 | 150.000000 |
| mean | 5.843333 | 3.054000 | 3.758667 | 1.198667 |
| std | 0.828066 | 0.433594 | 1.764420 | 0.763161 |
| min | 4.300000 | 2.000000 | 1.000000 | 0.100000 |
| 25% | 5.100000 | 2.800000 | 1.600000 | 0.300000 |
| 50% | 5.800000 | 3.000000 | 4.350000 | 1.300000 |
| 75% | 6.400000 | 3.300000 | 5.100000 | 1.800000 |
| max | 7.900000 | 4.400000 | 6.900000 | 2.500000 |

```text
df.groupby('species').size()
```

* 각 Class 별로 data가 50개씩 존재한다.

```text
species
setosa        50
versicolor    50
virginica     50
dtype: int64
```

### **Pairplot**

```python
sns.pairplot(df, hue="species")
plt.show()
```

![](../.gitbook/assets/image%20%2884%29.png)

* petal length 만으로 0과 1종을 완전히 구분할 수 있다
* petal length와 petal width 두가지로 나누면 1과 2종도 구분해 낼 수 있을 것으로 보인다

### **Distplot**

```python
sns.distplot(df[df.species != "setosa"]["petal length (cm)"], hist=True, rug=True, label="setosa")
sns.distplot(df[df.species == "setosa"]["petal length (cm)"], hist=True, rug=True, label="others")
plt.legend()
plt.show()
```

![](../.gitbook/assets/image%20%2833%29.png)

* 위의 분포를 보면 petal length 하나의 변수만으로 setosa와 다른 종들을 쉽게 분류해낼 수 있을 것으로 보인다.

```python
sns.distplot(df[df.species == "virginica"]["petal length (cm)"], hist=True, rug=True, label="virginica")
sns.distplot(df[df.species == "versicolor"]["petal length (cm)"], hist=True, rug=True, label="versicolor")
plt.legend()
plt.show()
```

![](../.gitbook/assets/image%20%2824%29.png)



* 반면 virginica와 versicolor는 petal length 만으로는 완전히 분류해내기 어려워보인다.

### **Parallel coordinates plot**

```python
parallel_coordinates(df, "species")
plt.xlabel('Features', fontsize=15)
plt.ylabel('Features values', fontsize=15)
plt.legend(loc=1, prop={'size': 15}, frameon=True, shadow=True, facecolor="white", edgecolor="black")
plt.show()
```

![](../.gitbook/assets/image%20%2814%29.png)

* 각 feature들로 얼마나 붓꽃 종류를 구분할 수 있는지 한눈에 알 수 있다.
* 앞에서 언급했듯이 petal length와 petal width로 0과 1를 구분해낼 수 있다

### **Boxplot¶**

```python
df.boxplot()
plt.show()

```

![](../.gitbook/assets/image%20%2847%29.png)

* 각 데이터의 분포를 살펴보았다.
* sepal\_width 데이터에서 outlier가 있는 것을 알 수 있었다.

## Preprocessing

* KNN을 사용하기에 앞서 거리를 구하는 것이 분류의 핵심이므로 Scaling을 해주기로 하였다.
* 그 전에 train data와 test data를 나누어 주기로 하였다.

### **Split data**

```python
# train과 test data를 0.75 : 0.25 비율로 나눈다.
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], random_state=3)
```

### **Scaling**

```python
# Standard Scaler
ss = StandardScaler() # Scaling
X_train_s = pd.DataFrame(ss.fit_transform(X_train), columns = X_train.columns)
X_test_s = pd.DataFrame(ss.transform(X_test), columns = X_test.columns)
X_train_s.head()
```

|  | sepal length \(cm\) | sepal width \(cm\) | petal length \(cm\) | petal width \(cm\) |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 0.701282 | -0.850872 | 0.852239 | 0.902670 |
| 1 | 0.444603 | -2.033225 | 0.390008 | 0.369166 |
| 2 | 0.701282 | -0.614401 | 1.025575 | 1.169422 |
| 3 | -0.068753 | -0.850872 | 0.043334 | -0.030962 |
| 4 | -1.608823 | 1.277364 | -1.632254 | -1.364723 |

```python
# Minmax Scaler
ms = MinMaxScaler()
X_train_m = pd.DataFrame(ms.fit_transform(X_train), columns = X_train.columns)
X_test_m = pd.DataFrame(ms.transform(X_test), columns = X_test.columns)
X_train_m.head()
```

|  | sepal length \(cm\) | sepal width \(cm\) | petal length \(cm\) | petal width \(cm\) |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 0.583333 | 0.318182 | 0.754386 | 0.750000 |
| 1 | 0.527778 | 0.090909 | 0.614035 | 0.583333 |
| 2 | 0.583333 | 0.363636 | 0.807018 | 0.833333 |
| 3 | 0.416667 | 0.318182 | 0.508772 | 0.458333 |
| 4 | 0.083333 | 0.727273 | 0.000000 | 0.041667 |

## Modeling

### **Print metrics function**

```python
def print_metrics(model, X_train):
    scores = cross_val_score(model, X_train, y_train, cv=10)
    print('*** Cross val score *** \n   {}'.format(scores))
    print('\n*** Mean Accuracy *** \n   {:.7f}'.format(scores.mean()))
    # print('\n*** Confusion Matrix *** \n', confusion_matrix(y_train, model.predict(X_train)))
```

* 한번에 validation용 metrics를 출력할 수 있는 함수를 생성하였다.

### **Simple Model**

```python
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print_metrics(knn, X_train)
```

```text
*** Cross val score *** 
   [0.91666667 1.         0.91666667 1.         1.         1.
 1.         1.         1.         0.88888889]

*** Mean Accuracy *** 
   0.9722222
```

### **Standard Scaled Model**

```python
knn_s = KNeighborsClassifier()
knn_s.fit(X_train_s, y_train)
print_metrics(knn_s, X_train_s)
```

```text
*** Cross val score *** 
   [0.83333333 1.         0.91666667 1.         0.91666667 1.
 1.         0.90909091 0.8        0.88888889]

*** Mean Accuracy *** 
   0.9264646
```

### **MinMax Scaled Model**

```python
knn_m = KNeighborsClassifier()
knn_m.fit(X_train_m, y_train)
print_metrics(knn_m, X_train_m)
```

```text
*** Cross val score *** 
   [0.83333333 1.         0.91666667 1.         0.91666667 1.
 1.         1.         0.9        0.88888889]

*** Mean Accuracy *** 
   0.9455556
```

## Hyperparameter Tuning

### **Parameters**

* **n\_neighbors**: 검색할 이웃의 수로 default 값은 5이다.
* **Metric**: 거리 측정 방식을 변경하는 매개변수로 default 값은 minkowsi이다
* **Weights**: 예측에 사용하는 가중치로 uniform 은 각 이웃에 동일한 가중치를 , ‘distance’는 가까운 이웃이 멀리 있는 이웃보다 더욱 큰 영향을 미친다.

### **Grid Search CV**

```text
grid_params = {
    'n_neighbors' : list(range(1,20)),
    'weights' : ["uniform", "distance"],
    'metric' : ['euclidean', 'manhattan', 'minkowski']
}
```

### **1. No Scaled**

```python
gs = GridSearchCV(knn, grid_params, cv=10)
gs.fit(X_train, y_train)
print("Best Parameters : ", gs.best_params_)
print("Best Score : ", gs.best_score_)
print("Best Test Score : ", gs.score(X_test, y_test))
```

```text
Best Parameters :  {'metric': 'euclidean', 'n_neighbors': 5, 'weights': 'uniform'}
Best Score :  0.9732142857142857
Best Test Score :  0.9473684210526315
```

### **2. Standard Scaled**

```python
gs_s = GridSearchCV(knn_s, grid_params, cv=10)
gs_s.fit(X_train_s, y_train)
print("Best Parameters : ", gs_s.best_params_)
print("Best Score : ", gs_s.best_score_)
print("Best Test Score : ", gs_s.score(X_test_s, y_test))
```

```text
Best Parameters :  {'metric': 'euclidean', 'n_neighbors': 16, 'weights': 'distance'}
Best Score :  0.9732142857142857
Best Test Score :  0.9473684210526315
```

### **3. MinMax Scaled**

```python
gs_m = GridSearchCV(knn_m, grid_params, cv=10)
gs_m.fit(X_train_m, y_train)
print("Best Parameters : ", gs_m.best_params_)
print("Best Score : ", gs_m.best_score_)
print("Best Test Score : ", gs_m.score(X_test_m, y_test))
```

* Score의 결과가 가장 좋은 MinMax Scaled Data의 {'metric': 'euclidean', 'n\_neighbors': 9, 'weights': 'uniform'} parameters를 선택하기로 하였다.

```text
Best Parameters :  {'metric': 'euclidean', 'n_neighbors': 9, 'weights': 'uniform'}
Best Score :  0.9732142857142857
Best Test Score :  0.9736842105263158
```

### **Final Model**

```python
knn_m = KNeighborsClassifier(metric = 'euclidean', n_neighbors = 9, weights = 'uniform')
knn_m.fit(X_train_m, y_train)
print_metrics(knn_m, X_train_m)
```

```text
*** Cross val score *** 
   [0.91666667 1.         1.         1.         0.91666667 1.
 1.         1.         1.         0.88888889]

*** Mean Accuracy *** 
   0.9722222
```

## Evaluation

```python
def print_test_metrics(model, X_test):
    print('*** Test Accuracy *** \n   {}'.format(model.score(X_test, y_test)))
    print('\n*** Confusion Matrix *** \n', confusion_matrix(y_test, model.predict(X_test)))
```

```text
print_test_metrics(knn_m, X_test_m)
```

```text
*** Test Accuracy *** 
   0.9736842105263158

*** Confusion Matrix *** 
 [[15  0  0]
 [ 0 11  1]
 [ 0  0 11]]
```

* 하나의 test instance를 제외하고 모두 분류해냈다.
* 역시 setosa는 완전히 분류해냈으나 나머지 두 종을 분류해내기 힘들었던 것 같다.

