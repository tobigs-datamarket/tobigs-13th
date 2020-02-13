---
description: Numpy를 이용한 구현
---

# Python을 이용한 경사하강법, 로지스틱회귀 구현\(2\)

## 데이터 불러오기

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
data = pd.read_csv('assignment2.csv')
data.head()
```

|  | Label | bias | experience | salary |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 1 | 1 | 0.7 | 48000 |
| 1 | 0 | 1 | 1.9 | 48000 |
| 2 | 1 | 1 | 2.5 | 60000 |
| 3 | 0 | 1 | 4.2 | 63000 |
| 4 | 0 | 1 | 6.0 | 76000 |

### Train Test 데이터 나누기

```python
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, 1:], data.iloc[:, 0], random_state = 0)
```

```python
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```

```text
((150, 3), (50, 3), (150,), (50,))
```

## 데이터 스케일링

experience와 salary를 스케일링한다.

```python
X_train.head()
```

|  | bias | experience | salary |
| :--- | :--- | :--- | :--- |
| 71 | 1 | 5.3 | 48000 |
| 124 | 1 | 8.1 | 66000 |
| 184 | 1 | 3.9 | 60000 |
| 97 | 1 | 0.2 | 45000 |
| 149 | 1 | 1.1 | 66000 |

```python
scaler = StandardScaler()
bias_train = X_train["bias"]
bias_train = bias_train.reset_index()["bias"]
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
X_train["bias"] = bias_train
X_train.head()
```

|  | bias | experience | salary |
| :--- | :--- | :--- | :--- |
| 0 | 1 | 0.187893 | -1.143335 |
| 1 | 1 | 1.185555 | 0.043974 |
| 2 | 1 | -0.310938 | -0.351795 |
| 3 | 1 | -1.629277 | -1.341220 |
| 4 | 1 | -1.308600 | 0.043974 |

```python
y_train = y_train.reset_index()["Label"]
y_test = y_test.reset_index()["Label"]
```

```python
bias_test = X_test["bias"]
bias_test = bias_test.reset_index()["bias"]
X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
X_test["bias"] = bias_test
X_test.head()
```

|  | bias | experience | salary |
| :--- | :--- | :--- | :--- |
| 0 | 1 | -1.344231 | -0.615642 |
| 1 | 1 | 0.508570 | 0.307821 |
| 2 | 1 | -0.310938 | 0.571667 |
| 3 | 1 | 1.363709 | 1.956862 |
| 4 | 1 | -0.987923 | -0.747565 |

## 1. sigmoid

```python
import random
```

```python
X_train = np.array(X_train[["bias", "experience", "salary"]])
```

```python
beta = np.array([random.random(), random.random(), random.random()]) # 임의의 beta값 생성
beta
```

```text
array([0.90546443, 0.75530667, 0.65834986])
```

![](../.gitbook/assets/image%20%2860%29.png)

```python
def sigmoid(x, beta) :
    multiplier = 0
    for i in range(x.size):
        multiplier += x[i]*beta[i]
    p = 1.0/(1.0+np.exp(-multiplier))
    return p
sigmoid(X_train[0], beta)
```

```text
0.5731382785117154
```

## 2. log likelihood

![](../.gitbook/assets/image%20%2815%29.png)

```python
#개별 likelihood, 각각의 x입력값에 대한 p의 값 산정
def lg_likelihood_i(x, y, beta, j) :
    p_hat = 0
    p = sigmoid(x[j], beta)
    p_hat += y[j]*np.log(p) + (1-y[j])*np.log(1-p)
    return p_hat
lg_likelihood_i(X_train, y_train, beta, 0)
```

```text
-0.5566282676261158
```

```python
def lg_likelihood(x, y, beta) :
    log_p_hat = 0
    for i in range(y.size) :
        log_p_hat += lg_likelihood_i(x, y, beta, i) # log p 의 추정값에 계속 더해준다.

    return log_p_hat
lg_likelihood(X_train, y_train, beta)
```

```text
-168.57600337087965
```

## 3. gradient Ascent

get\_gradients는 cost function\(log likelihood\)상에서 각각의 beta 계수들로 편미분했을 때, 각각의 기울기를 구하는 함수이다.

```python
# gradients 한 번 구하기
def get_gradients(x, y, beta):
    gradients = []

    for i in range(x[0].size) :
        gradient = 0                                  # 각 계수별 기울기
        for j in range(y.size) :
            p = sigmoid(x[j], beta)
            gradient += (y[j] - p)*x[j][i]            # 개별 데이터 x에 대한 값을 합산

        gradient = gradient/y.size                    # 전체 n 값으로 나누기
        gradients.append(gradient)

    gradients = np.array(gradients)

    return gradients

gradients = np.array(get_gradients(X_train, y_train, beta))
gradients
```

```text
array([-0.37681215, -0.09655044, -0.3003301 ])
```

step은 구한 기울기를 바탕으로 다음 학습을 진행할 지점을 지정하는 함수이다.

```python
def step(beta, gradients, stepsize=np.array([0.01,0.01,0.01])) : #stepsize:학습률, 기본값은 0.01
    beta = beta + stepsize*gradients
    return beta
```

```python
step(beta, gradients)
```

```text
array([0.90169631, 0.75434116, 0.65534656])
```

```python
#max_cycle:최대 학습 횟수
#tolerance:이 값보다 step의 변화율이 낮으면 학습을 종료함
#theta_0:학습 이전의 계수
#theta:학습 이후의 계수

def gradientAscent(x, y, beta, max_cycle = 200000, tolerance = 0.000001, stepsize=np.array([0.01,0.01,0.01])) :
    theta_0 = beta
    i = 0
    cost = lg_likelihood(x, y, theta_0)/y.size
    gradients = np.array([])
    while i < max_cycle:
        gradients = get_gradients(x, y, theta_0)
        theta = step(theta_0, gradients, stepsize)
        temp = theta_0 - theta
        theta_0 = theta

        if i % 1000 == 0:
            print(gradients)
            #print(theta_0)
            #print(theta)
            #print(np.abs(temp.sum()))
        if np.abs(temp.sum()) < tolerance :
            print("stop")
            break
        i += 1
    return theta_0
```

```python
beta.sum()
```

```text
2.3191209550302005
```

## 4. Fitting

cf.\) Step Size를 고르는 기법으로는 다음 세가지 방법이 있다.

* Fixed step size
* Backtracking line search
* Exact line search

참고 : [https://wikidocs.net/18088](https://wikidocs.net/18088)

```python
# 학습률 0.01은 진행속도가 느려 학습결과에 큰 차이가 없는 0.1로 설정함.
beta = gradientAscent(X_train, y_train, beta, stepsize=np.array([0.1,0.1,0.1]))
beta
```

```text
[-0.37681215 -0.09655044 -0.3003301 ]
[-0.0030989   0.0103664  -0.00977676]
[-0.00119877  0.0039189  -0.00364469]
[-0.00056101  0.00182317 -0.00168622]
[-0.00028209  0.0009145  -0.00084362]
[-0.00014655  0.00047454 -0.00043719]
[-7.73735736e-05  2.50391335e-04 -2.30529092e-04]
[-4.11904585e-05  1.33256706e-04 -1.22642622e-04]
[-2.20238194e-05  7.12383899e-05 -6.55517795e-05]
stop





array([-1.86332543,  4.25483493, -4.02439239])
```

```python
lg_likelihood(X_train, y_train, beta) # 수렴한 우도
```

```text
-44.73076829152757
```

## 5. 예측

```python
X_test = np.array(X_test[["bias", "experience", "salary"]])
```

```python
Label_predict = []
for i in range(y_test.size) :
    p = sigmoid(X_test[i], beta)  # 학습한 beta 값으로 p를 추정한다.
    if p > 0.5 :
        Label_predict.append(1) # p값이 0.5보다 크면 1로 분류한다.
    else :
        Label_predict.append(0)
Label_predict = np.array(Label_predict)
Label_predict
```

```text
array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
       1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0])
```

## 6. confusion\_matrix

```python
from sklearn.metrics import *
tn, fp, fn, tp = confusion_matrix(y_test, Label_predict).ravel()
confusion_matrix(y_test, Label_predict)
```

```text
array([[38,  2],
       [ 1,  9]], dtype=int64)
```

```python
#Accuracy
Accuracy = (tp+tn)/(tp+fn+fp+tn)
Accuracy
```

```text
0.94
```

데이터 출처:  Data Science from Scratch: First Principles with Python \(2015\)

