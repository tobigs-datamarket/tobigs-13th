# Navie Bayes 이해와 실습 \(Sklearn,Numpy\)

## Assignment <a id="Assignment"></a>

* 중간 중간 assignment라고 되어있고 ''' ? ''' 이 있을거에요 그 부분을 채워주시면 됩니다.

{% hint style="info" %}
**우수과제 선정이유**

나이브베이즈 관련 함수들의 인자와 반환값 형태 및 성질들을 꼼꼼히 살펴보았으며 군더더기없이 과제를 수행하였습니다**.**
{% endhint %}

## Assignment 1 : Gaussian Naive Bayes Classification 해보기 <a id="Assignment-1-:-Gaussian-Naive-Bayes-Classification-&#xD574;&#xBCF4;&#xAE30;"></a>

* sklearn에 Gaussian Naive Bayes Classification 클래스 함수가 이미 있습니다
* 그것을 활용하여 간단하게 예측만 하시면 됩니다
* 필요 함수 링크를 주석으로 처리하여 첨부했으니 보시고 사용해주세요

```python
import pandas as pd
import numpy as np
```

```python
from sklearn.datasets import load_iris
```

* sklearn에 내장되어있는 붓꽃 데이터를 사용할 겁니다

```python
iris = load_iris()
```

* 붓꽃데이터를 불러옵니다

```text
print(iris.DESCR)
```

```text
_iris_dataset:

Iris plants dataset
--------------------

**Data Set Characteristics:**

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
    petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
    ============== ==== ==== ======= ===== ====================

    :Missing Attribute Values: None
    :Class Distribution: 33.3% for each of 3 classes.
    :Creator: R.A. Fisher
    :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
    :Date: July, 1988

The famous Iris database, first used by Sir R.A. Fisher. The dataset is taken
from Fisher's paper. Note that it's the same as in R, but not as in the UCI
Machine Learning Repository, which has two wrong data points.

This is perhaps the best known database to be found in the
pattern recognition literature.  Fisher's paper is a classic in the field and
is referenced frequently to this day.  (See Duda & Hart, for example.)  The
data set contains 3 classes of 50 instances each, where each class refers to a
type of iris plant.  One class is linearly separable from the other 2; the
latter are NOT linearly separable from each other.

.. topic:: References

   - Fisher, R.A. "The use of multiple measurements in taxonomic problems"
     Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
     Mathematical Statistics" (John Wiley, NY, 1950).
   - Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis.
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

* 설명변수 x가 꽃받침 길이, 꽃받침 폭 , 꽃잎 길이 , 꽃잎 폭 임을 알 수 있습니다
* sepal length , sepal width , petal length , petal width
* 타겟변수 y는 붓꽃의 품종으로 총 3가지의 종류가 있는 걸 알 수 있습니다
* Iris-Setosa , Iris-Versicolour , Iris-Virginica 가 0 1 2 로 분류되어 있습니다

```python
X = pd.DataFrame(iris.data)
y = pd.DataFrame(iris.target)
```

* 설명변수 x와 타겟변수 y를 판다스의 데이터프레임 형태로 만듭니다.

```text
X
```

|  | 0 | 1 | 2 | 3 |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 5.1 | 3.5 | 1.4 | 0.2 |
| 1 | 4.9 | 3.0 | 1.4 | 0.2 |
| 2 | 4.7 | 3.2 | 1.3 | 0.2 |
| 3 | 4.6 | 3.1 | 1.5 | 0.2 |
| 4 | 5.0 | 3.6 | 1.4 | 0.2 |
| ... | ... | ... | ... | ... |
| 145 | 6.7 | 3.0 | 5.2 | 2.3 |
| 146 | 6.3 | 2.5 | 5.0 | 1.9 |
| 147 | 6.5 | 3.0 | 5.2 | 2.0 |
| 148 | 6.2 | 3.4 | 5.4 | 2.3 |
| 149 | 5.9 | 3.0 | 5.1 | 1.8 |

```text
y
```

|  | 0 |
| :--- | :--- |
| 0 | 0 |
| 1 | 0 |
| 2 | 0 |
| 3 | 0 |
| 4 | 0 |
| ... | ... |
| 145 | 2 |
| 146 | 2 |
| 147 | 2 |
| 148 | 2 |
| 149 | 2 |

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
```

* 가우시안 나이브 베이즈 함수를 불러옵니다

## 1-1\) assignment <a id="1-1)-assignment"></a>

* train set과 test set을 80 대 20의 비율로 나누어 주세요
* [https://scikit-learn.org/stable/modules/generated/sklearn.model\_selection.train\_test\_split.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
* 해당 함수를 사용하시고 사용 후에 주석으로 해당 함수에 어떤 인자들이 있는지 설명해주세요

```python
"""
train_test_split(*arrays, **options)

(1) Parameter

arrays : sequence of indexables with same length / shape[0]
    분할시킬 데이터를 입력 (input format : List, Numpy array, Pandas dataframe 등..)

test_size : float, int or None, optional (default=None)
    input format: float -> test dataset의 비율(0.0~1)
                 int -> test dataset의 개수 (default train_size = 0.25)

train_size :  float, int or None, optional (default=None)
    input format: float -> train dataset의 비율(0.0~1)
                 int -> train dataset의 개수 (default train_size = 1-test_size의 나머지 = 0.75)

random_state : int, RandomState instance or None, optional (default=None) 
   input format: int -> random number generator의 시드값
                 RandomState  -> random number generator 
                 (default random_state = np.random가 제공하는 random number generato)
       

shuffle : boolean, optional (default=True)
    split전 shuffle의 여부

stratify : array-like or None (default=None)
    test,train data들을 input data의 class비율에 맞게 split할 것인지 여부
    
(shuffle = False이면 stratify = None이어야 한다)



(2) Return
splittinglist, length=2 * len(arrays)

    X_train, X_test, Y_train, Y_test : 
        arrays에 data와 label을 둘 다 넣었을 경우의 반환. data와 class의 순서쌍은 유지된다.

    X_train, X_test : 
        arrays에 label 없이 data만 넣었을 경우의 반환. class 값을 포함하여 하나의 data로 반환
"""
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
# data : X   label : y, test_set의 비율:train_ser의 비율=0.2:0.8(1-0.2)
```

## 1-2\) assignment <a id="1-2)-assignment"></a>

* 가우시안 나이브 베이즈 함수를 사용하여 학습을 시킨 후 score 값을 계산하여 주세요
* 모두 GaussianNB 클래스 안에 메서드로 들어있습니다
* [https://scikit-learn.org/stable/modules/generated/sklearn.naive\_bayes.GaussianNB.html](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
* 해당 함수를 사용하시고 사용 후에 주석으로 해당 함수에 어떤 인자들이 있는지 설명해주세요

![image.png](data:image/png;base64,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)

```python
"""
GaussianNB(priors=None, var_smoothing=1e-09)

(1) Parameter

prioprs : array-like, shape (n_classes,)
    사전확률

var_smoothing : float, optional (default=1e-9)
    분산 극단적인 값으로 가는것을 방지하기 위한 예외처리

(2) Attributes

class_count_ : array, shape (n_classes,)
    종속변수 Y의(class) 값이 특정한 클래스인 표본 데이터의 수(training sample 수)

class_prior_ : array, shape (n_classes,)
    종속변수 Y의(class) 무조건부 확률분포  P(Y)  
    
classes_ : array, shape (n_classes,)
    종속변수Y(class)의 label
    

sigma_array, shape (n_classes, n_features)
    정규분포의 분산  σ2

theta_array, shape (n_classes, n_features)
    정규분호의 기댓값 μ
    
    
"""

"""
fit(self, X, y, sample_weight=None)

(1) Parameter

X : array-like, shape (n_samples, n_features)
Training vectors, where n_samples is the number of samples and n_features is the number of features.

y : array-like, shape (n_samples,)
Target values.

sample_weightarray-like, shape (n_samples,), optional (default=None)
Weights applied to individual samples (1. for unweighted).

New in version 0.17: Gaussian Naive Bayes supports fitting with sample_weight.

(2) Return

self : object

"""

"""
predict(self, X)

(1) Parameter

X : array-like of shape (n_samples, n_features)


(2) Retrun

C: ndarray of shape (n_samples,)
    X에 대해 예측한 target vlaue값들
"""


gnb = GaussianNB()
y_pred = gnb.fit(X_train,y_train).predict(X_test)
gnb.score(X_test, y_test)
#score = 1
```

```text
1.0
```

## Assignment 2 : Naive Bayes Classification 해보기 <a id="Assignment-2-:-Naive-Bayes-Classification-&#xD574;&#xBCF4;&#xAE30;"></a>

* 제가 임의로 만든 데이터 셋입니다
* spam 메세지에 gamble money hi라는 단어의 유무를 기준으로 0과 1을 주었고 spam 메세지인지 아닌지를 spam에 0과1로 정해주었습니다
* 설명변수는 gamble, money, spam 이고 종속변수는 spam입니다\(data가 세개~ sam일 확률 구하자룽\)

```python
gamble_spam = {'gamble' : [1,0,1,0,1,0,0,0,1,0,0,0,1,1,1,1,0,1,0,0,0,0,0,0,0,
                           1,0,1,1,0,1,0,1,0,1,1,1,1,0,0,0,1,0,1,0,1,0,1,0,1,
                           0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,1,1,0,1,1,1,1,1,0,0,
                           1,0,0,1,0,0,0,1,1,0,1,0,1,1,0,0,0,1,0,1,1,1,0,1,1],
               'money' : [1,1,1,0,1,0,0,0,1,0,0,0,1,0,1,1,0,1,1,0,1,1,1,1,1,
                          0,0,0,1,1,1,0,0,0,1,1,0,0,0,1,0,1,1,0,1,0,0,1,0,1,
                          1,0,1,1,0,1,0,1,0,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,
                          1,1,0,1,0,1,1,0,0,1,0,1,1,1,1,0,0,1,0,0,1,0,0,1,0],
               'hi' : [0,1,0,1,0,1,0,0,1,0,0,0,1,0,1,1,1,0,1,1,1,0,0,0,0,
                       1,0,0,1,0,0,0,1,0,1,1,1,0,1,1,1,0,0,1,0,1,0,1,1,0,
                       1,0,0,0,1,1,1,1,0,1,0,1,1,0,0,1,1,1,1,0,0,0,0,0,0,
                       1,1,0,0,0,1,1,0,1,0,1,1,0,0,0,0,0,0,1,0,0,0,1,0,0],
                'spam' : [1,0,1,0,1,0,0,0,1,0,0,0,1,1,1,1,0,1,1,0,0,0,1,1,0,
                          1,0,1,1,0,0,1,1,0,0,0,0,1,1,0,1,0,0,1,0,1,0,1,0,1,
                          0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,
                          1,0,0,1,0,0,0,1,1,0,1,0,1,1,0,0,0,1,0,1,1,1,0,1,1]}
```

* 해당 딕셔너리 데이터를 판다스 데이터 프레임으로 변경하여줍니다

```python
df  = pd.DataFrame(gamble_spam, columns = ['gamble', 'money', 'hi', 'spam'])
df
```

|  | gamble | money | hi | spam |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 1 | 1 | 0 | 1 |
| 1 | 0 | 1 | 1 | 0 |
| 2 | 1 | 1 | 0 | 1 |
| 3 | 0 | 0 | 1 | 0 |
| 4 | 1 | 1 | 0 | 1 |
| ... | ... | ... | ... | ... |
| 95 | 1 | 1 | 0 | 1 |
| 96 | 1 | 0 | 0 | 1 |
| 97 | 0 | 0 | 1 | 0 |
| 98 | 1 | 1 | 0 | 1 |
| 99 | 1 | 0 | 0 | 1 |

## 2-1\) assignment <a id="2-1)-assignment"></a>

* 해당 판다스 데이터프레임 형식을 numpy array형식으로 변환해 주세요
* [https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.as\_matrix.html](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.as_matrix.html)

```python
spam_data = df.to_numpy()
spam_data
#pandas.DataFrame.to_numpy(self, dtype=None, copy=False) → numpy.ndarra
```

```text
array([[1, 1, 0, 1],
       [0, 1, 1, 0],
       [1, 1, 0, 1],
       [0, 0, 1, 0],
       [1, 1, 0, 1],
       [0, 0, 1, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [1, 0, 0, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1],
       [0, 0, 1, 0],
       [1, 1, 0, 1],
       [0, 1, 1, 1],
       [0, 0, 1, 0],
       [0, 1, 1, 0],
       [0, 1, 0, 0],
       [0, 1, 0, 1],
       [0, 1, 0, 1],
       [0, 1, 0, 0],
       [1, 0, 1, 1],
       [0, 0, 0, 0],
       [1, 0, 0, 1],
       [1, 1, 1, 1],
       [0, 1, 0, 0],
       [1, 1, 0, 0],
       [0, 0, 0, 1],
       [1, 0, 1, 1],
       [0, 0, 0, 0],
       [1, 1, 1, 0],
       [1, 1, 1, 0],
       [1, 0, 1, 0],
       [1, 0, 0, 1],
       [0, 0, 1, 1],
       [0, 1, 1, 0],
       [0, 0, 1, 1],
       [1, 1, 0, 0],
       [0, 1, 0, 0],
       [1, 0, 1, 1],
       [0, 1, 0, 0],
       [1, 0, 1, 1],
       [0, 0, 0, 0],
       [1, 1, 1, 1],
       [0, 0, 1, 0],
       [1, 1, 0, 1],
       [0, 1, 1, 0],
       [0, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 1, 0],
       [0, 1, 1, 0],
       [0, 0, 1, 0],
       [0, 1, 1, 0],
       [1, 0, 0, 0],
       [0, 1, 1, 0],
       [0, 1, 0, 0],
       [1, 0, 1, 0],
       [0, 0, 1, 0],
       [0, 0, 0, 0],
       [1, 1, 0, 0],
       [1, 1, 1, 0],
       [1, 0, 1, 1],
       [0, 0, 1, 1],
       [1, 0, 1, 1],
       [1, 1, 0, 1],
       [1, 1, 0, 1],
       [1, 1, 0, 0],
       [1, 1, 0, 0],
       [0, 1, 0, 0],
       [0, 1, 0, 0],
       [1, 1, 1, 1],
       [0, 1, 1, 0],
       [0, 0, 0, 0],
       [1, 1, 0, 1],
       [0, 0, 0, 0],
       [0, 1, 1, 0],
       [0, 1, 1, 0],
       [1, 0, 0, 1],
       [1, 0, 1, 1],
       [0, 1, 0, 0],
       [1, 0, 1, 1],
       [0, 1, 1, 0],
       [1, 1, 0, 1],
       [1, 1, 0, 1],
       [0, 1, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [1, 1, 0, 1],
       [0, 0, 1, 0],
       [1, 0, 0, 1],
       [1, 1, 0, 1],
       [1, 0, 0, 1],
       [0, 0, 1, 0],
       [1, 1, 0, 1],
       [1, 0, 0, 1]], dtype=int64)
```

## 2-2\) assignment <a id="2-2)-assignment"></a>

* P\(spam=1\), P\(spam=0\)
* P\(gamble=1\|spam=1\), P\(money=1\|spam=1\), P\(hi=1\|spam=1\)
* P\(gamble=1\|spam=0\), P\(money=1\|spam=0\), P\(hi=1\|spam=0\)
* 위의 확률들을 구하여 주세요
* 먼저 P\(spam\)의 확률 부터 구해주세요
* P\(spam=1\)인 경우만 제가 완성을 해두었습니다 참고하시면 금방 금방 채우실거에요 :\)

```python
p_spam = sum(spam_data[:,3]==1)/len(spam_data) # P(spam=1)
p_spam_not = 1- p_spam # P(spam=0)

#베르누이 분포를 따르기 때문에 해당 상황이 일어날 확률은 1-(반대 상황이 일어날 확률)

"""
other solution:
p_spam_not = sum(spam_data[:,3]==0)/len(spam_data)

"""
p_spam_not
```

* gamble , money , hi의 조건부 확률을 다 구해주세요
* P\(gamble=0\|spam=1\) = 1 - P\(gamble=1\|spam=1\) 의 형태로 구하면 되므로 굳이 따로 구하지 않습니다
* 각 값이 어떤 조건부 확률인지 이름만으로는 알기 어려울거같아 바로 옆에 주석을 달아놨습니다 참고하세요
* 위랑 마찮가지로 제일 위에 껀 제가 해놨어요 참고해서 한번 구해보세요

```text
0.5800000000000001
```

```python
p_gamble_spam = sum((spam_data[:, 0] == 1) & (spam_data[:, 3] == 1)) / sum(spam_data[:, 3] == 1) 
# P(gamble=1|spam=1) = P(gambled=1∩spam=1)/P(spam=1)
p_gamble_spam_not = sum((spam_data[:, 0] == 1) & (spam_data[:, 3] == 0)) / sum(spam_data[:, 3] == 0) 
# P(gamble=1|spam=0) = P(gambled=1∩spam=0)/P(spam=0)

p_money_spam = sum((spam_data[:, 1] == 1) & (spam_data[:, 3] == 1)) / sum(spam_data[:, 3] == 1)  
# P(money=1|spam=1) = P(money=1∩spam=1)/P(spam=1)
p_money_spam_not = sum((spam_data[:, 1] == 1) & (spam_data[:, 3] == 0)) / sum(spam_data[:, 3] == 0) 
# P(money=1|spam=0) = P(money=1∩spam=0)/P(spam=0)

p_hi_spam = sum((spam_data[:, 2] == 1) & (spam_data[:, 3] == 1)) / sum(spam_data[:, 3] == 1) 
# P(hi=1|spam=1) = P(hi=1∩spam=1)/P(spam=1)
p_hi_spam_not = sum((spam_data[:, 2] == 1) & (spam_data[:, 3] == 0)) / sum(spam_data[:, 3] == 0) 
# P(hi=1|spam=0) = P(hi=1∩spam=0)/P(spam=0)
```

```text
p_hi_spam_not
```

```text
0.4482758620689655
```

* 이제 P\(_\|spam=1\)값 리스트와 P\(_\|spam=0\)값 리스트를 생성해줍니다

```python
proba = [p_gamble_spam,p_money_spam,p_hi_spam]
proba_not = [p_gamble_spam_not,p_money_spam_not,p_hi_spam_not]
proba
```

```text
[0.8333333333333334, 0.5476190476190477, 0.4523809523809524]
```

* 요건 테스트 셋이에요
* 예를 들어 \[0,1,0\]인 경우 gamble=0,money=1,hi=0인 경우에 spam인지 아닌지 확률을 계산해 달라는 의미 입니다
* 설명변수가 3개 밖에 안되기때문에 \[0,0,0\] ~ \[1,1,1\] 8가지 모든 경우에 대해 확률 P\(\*\|spam=1\)를 구할 거에요

```python
test = [[i,j,k] for i in range(2) for j in range(2) for k in range(2)]
```

```text
test
```

```text
[[0, 0, 0],
 [0, 0, 1],
 [0, 1, 0],
 [0, 1, 1],
 [1, 0, 0],
 [1, 0, 1],
 [1, 1, 0],
 [1, 1, 1]]
```

## 2-3\) assignment <a id="2-3)-assignment"></a>

* 조건부 확률을 구하는 함수를 구해주세요
* x는 해당 독립변수가 0인지 1인지를 받는 인자이구요
* p는 해당독립변수가 1일때의 조건부 확률이 들어갑니다
* P\(X=x\|Y=1\) = x_P\(X=1\|Y=1\)+\(1-x\)_P\(X=0\|Y=1\)을 응용하세요

```python
def con_proba(x,p):
    #money인경우 x=1 == money=1 p==P(money=1|spam=1)/x=0 == money=0 p==P(money=0|spam=1)
    return (x*p+(1-x)*p) #P(X=x|Y=1) = xP(X=1|Y=1)+(1-x)P(X=0|Y=1)
```

* test경우에 대해 각 확률을 반환해주는 함수를 생성해주세요

```python
def process(p_spam,p_spam_not,test,proba,proba_not):
    result = []
    for i in range(8):
        a = p_spam
        b = p_spam_not
        for j in range(3):
            a = a*con_proba(test[i][j],proba[j] if test[i][j] == 1 else (1-proba[j]))
            b = b*con_proba(test[i][j],proba_not[j] if test[i][j] == 1 else (1-proba_not[j]))
        summation = a+b
        result.append([a/summation,b/summation])
    return result
```

* 결과 입니다 다음과 같은 값들이 똑같이 나오면 과제 성공이에요
* 왼쪽이 spam 메세지일 확률, 오른쪽이 spam 메세지가 아닐 확률입니다
* gamble money hi라는 단어가 들어가면 들어갈수록 spam메세지인걸 알수가 있네요
* spam 메세지일 확률이 0.5를 넘기는 아래에서 6,7,8행의 경우가 spam 메세지로 분류가 되겠네요
* 즉 gamble이라는 단어와 money 혹은 hi라는 단어가 하나라도 같이 있으면 spam메세지가 되나봐요.

```text
process(p_spam,p_spam_not,test,proba,proba_not)
```

성공!

```text
[[0.12561158047604412, 0.874388419523956],
 [0.12744440801721746, 0.8725555919827825],
 [0.1315383089295994, 0.8684616910704007],
 [0.13344441688939654, 0.8665555831106033],
 [0.7542408952456083, 0.24575910475439158],
 [0.7573019784074859, 0.24269802159251408],
 [0.7639150506833852, 0.23608494931661478],
 [0.7668928774284339, 0.23310712257156604]]
```

  


  


