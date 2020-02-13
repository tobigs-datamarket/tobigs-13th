# Python을 이용한 클러스터링 \(3\)

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

![](../.gitbook/assets/image%20%2875%29.png)

* petal length 만으로 0과 1종을 완전히 구분할 수 있다
* petal length와 petal width 두가지로 나누면 1과 2종도 구분해 낼 수 있을 것으로 보인다

### **Distplot**

```python
sns.distplot(df[df.species != "setosa"]["petal length (cm)"], hist=True, rug=True, label="setosa")
sns.distplot(df[df.species == "setosa"]["petal length (cm)"], hist=True, rug=True, label="others")
plt.legend()
plt.show()
```

![](../.gitbook/assets/image%20%2831%29.png)

* 위의 분포를 보면 petal length 하나의 변수만으로 setosa와 다른 종들을 쉽게 분류해낼 수 있을 것으로 보인다.

```python
sns.distplot(df[df.species == "virginica"]["petal length (cm)"], hist=True, rug=True, label="virginica")
sns.distplot(df[df.species == "versicolor"]["petal length (cm)"], hist=True, rug=True, label="versicolor")
plt.legend()
plt.show()
```

![](../.gitbook/assets/image%20%2823%29.png)

* 반면 virginica와 versicolor는 petal length 만으로는 완전히 분류해내기 어려워보인다.



