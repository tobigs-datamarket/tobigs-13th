---
description: 구현 시 외부 라이브러리 사용 없이 구현
---

# Python을 이용한 경사하강법, 로지스틱회귀 구현\(1\)

&lt;로지스틱 회귀 주요 포인트&gt;

* 로지스틱 회귀\(Logistic Regression\): 범주형 종속 변수를 예측하는 목적. 회귀계수 추 시 최대우도추정법을 사용한다.
* 최대우도추정법\(Maximum Likelihood Estimation\): 확률분포의 모수\(parameter\)를 추정하는 하나의 방법. 확률변수의 우도\(Likelihood\)를 최대화하는 모수의 값을 모수 추정값으로 사용하는 방법
* 로그우도\(Log Likelihood\): 계산의 편의성을 위해 우도에 로그를 취하여 사용. 로그함수는 단조 증가함수이므로, 함수의 최적점을 왜곡시키지 않는다.
* 경사하강법\(Gradient Descent Algorithm\): 목적함수\(Objective Function\)의 값을 최소화하는 모수를 찾는 하나의 방법. 임의의 시작점에서 목적함수의 편미분 벡터\(Gradient; 쉽게 기울기라고 생각하자\)를 계산하여 가장 큰 그래디언트 값의 방향으로 움직이면서 목적함수의 값을 최소화함.

&lt; 과제 수행 시 주요 고려 사항&gt;

 1\) 로지스틱 회귀 모형의 Objective Function을 구현.  
 2\) Gradient Descent 알고리즘을 구현하고, 1\)을 이에 적용.

In \[1\]:

```text
import math, random
from functools import partial,reduce
from assignment2 import *
import pandas as pd
from sklearn.model_selection import train_test_split
```

In \[2\]:

```text
"""
data 설명
1) Label: 유료 계정 등록 여부(target)
2) bias: 회귀 모형에서의 상수항을 위한 term (추정 시 포함하지 않아도 ok)
3) experience: 근속연수
4) salary: 연봉

어떤 사용자가 유료 계정을 등록할지(Label == 1)에 대한 예측을 로지스틱 회귀 모형으로 진행합니다.
"""
```

Out\[2\]:

```text
'\ndata 설명\n1) Label: 유료 계정 등록 여부(target)\n2) bias: 회귀 모형에서의 상수항을 위한 term (추정 시 포함하지 않아도 ok)\n3) experience: 근속연수\n4) salary: 연봉\n\n어떤 사용자가 유료 계정을 등록할지(Label == 1)에 대한 예측을 로지스틱 회귀 모형으로 진행합니다.\n'
```

In \[3\]:

```text
data = pd.read_csv('assignment_2.csv')
```

Out\[4\]:

|  | Label | bias | experience | salary |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 1 | 1 | 0.7 | 48000 |
| 1 | 0 | 1 | 1.9 | 48000 |
| 2 | 1 | 1 | 2.5 | 60000 |
| 3 | 0 | 1 | 4.2 | 63000 |
| 4 | 0 | 1 | 6.0 | 76000 |
| 5 | 0 | 1 | 6.5 | 69000 |
| 6 | 0 | 1 | 7.5 | 76000 |
| 7 | 0 | 1 | 8.1 | 88000 |
| 8 | 1 | 1 | 8.7 | 83000 |
| 9 | 1 | 1 | 10.0 | 83000 |
| 10 | 0 | 1 | 0.8 | 43000 |
| 11 | 0 | 1 | 1.8 | 60000 |
| 12 | 1 | 1 | 10.0 | 79000 |
| 13 | 0 | 1 | 6.1 | 76000 |
| 14 | 0 | 1 | 1.4 | 50000 |
| 15 | 0 | 1 | 9.1 | 92000 |
| 16 | 0 | 1 | 5.8 | 75000 |
| 17 | 0 | 1 | 5.2 | 69000 |
| 18 | 0 | 1 | 1.0 | 56000 |
| 19 | 0 | 1 | 6.0 | 67000 |
| 20 | 0 | 1 | 4.9 | 74000 |
| 21 | 1 | 1 | 6.4 | 63000 |
| 22 | 0 | 1 | 6.2 | 82000 |
| 23 | 0 | 1 | 3.3 | 58000 |
| 24 | 1 | 1 | 9.3 | 90000 |
| 25 | 1 | 1 | 5.5 | 57000 |
| 26 | 0 | 1 | 9.1 | 102000 |
| 27 | 0 | 1 | 2.4 | 54000 |
| 28 | 1 | 1 | 8.2 | 65000 |
| 29 | 0 | 1 | 5.3 | 82000 |
| ... | ... | ... | ... | ... |
| 170 | 0 | 1 | 6.2 | 70000 |
| 171 | 1 | 1 | 6.6 | 56000 |
| 172 | 0 | 1 | 6.3 | 76000 |
| 173 | 0 | 1 | 6.5 | 78000 |
| 174 | 0 | 1 | 5.1 | 59000 |
| 175 | 1 | 1 | 9.5 | 74000 |
| 176 | 0 | 1 | 4.5 | 64000 |
| 177 | 0 | 1 | 2.0 | 54000 |
| 178 | 0 | 1 | 1.0 | 52000 |
| 179 | 0 | 1 | 4.0 | 69000 |
| 180 | 0 | 1 | 6.5 | 76000 |
| 181 | 0 | 1 | 3.0 | 60000 |
| 182 | 0 | 1 | 4.5 | 63000 |
| 183 | 0 | 1 | 7.8 | 70000 |
| 184 | 1 | 1 | 3.9 | 60000 |
| 185 | 0 | 1 | 0.8 | 51000 |
| 186 | 0 | 1 | 4.2 | 78000 |
| 187 | 0 | 1 | 1.1 | 54000 |
| 188 | 0 | 1 | 6.2 | 60000 |
| 189 | 0 | 1 | 2.9 | 59000 |
| 190 | 0 | 1 | 2.1 | 52000 |
| 191 | 0 | 1 | 8.2 | 87000 |
| 192 | 0 | 1 | 4.8 | 73000 |
| 193 | 1 | 1 | 2.2 | 42000 |
| 194 | 0 | 1 | 9.1 | 98000 |
| 195 | 0 | 1 | 6.5 | 84000 |
| 196 | 0 | 1 | 6.9 | 73000 |
| 197 | 0 | 1 | 5.1 | 72000 |
| 198 | 1 | 1 | 9.1 | 69000 |
| 199 | 1 | 1 | 9.8 | 79000 |

200 rows × 4 columns

In \[5\]:

```text
def step(v, direction, step_size):
    """
    한 지점에서 step size만큼 이동하는 step 함수를 구현하세요.
    v와 direction은 벡터.
    """
    # v : 모델에서 현재 parameter vector
    # direction : objective function을 parameter vector에 대해 편미분한 gradient vector
    # new_parameter = old_parameter - learning_rate * gradient 방식으로 parameter update
    return [i-step_size*j for i, j in zip(v, direction)]
```

In \[6\]:

```text
def safe(f) :
    """
    f에 대한 예외처리를 위한 함수(f가 infinite일 때)
    """
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')
    return safe_f
```

In \[7\]:

```text
def minimize_bgd(target_fn, gradient_fn, theta_0, tolerance = 0.00001): # bgd: batch gradient descent
    """
    목적함수를 최소화시키는 theta를 경사 하강법을 사용해서 찾는다.
    """
    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001] # 여러가지 step sizes에 대해서 테스트
    
    theta = theta_0 # 시작점 설정
    target_fn = safe(target_fn) # 함수가 infinite일 때의 오류를 처리할 수 있는 target_fn으로 변환
    value = target_fn(theta) # 현재 model 에서의 target function value
    
    while True:
        gradient = gradient_fn(theta) # 현재 theta에서의 gradient (목적함수를 theta에서 미분한 값)
        next_thetas =  [step(theta, gradient, step_size) for step_size in step_sizes] # 다양한 learning_rate에 대해서 다음 thetas를 구함


        obj = next_thetas 
        key = target_fn # 최소화하는 함수 = 목적함수
        next_theta = min(obj, key = key) # 위에서 구한 next_thetas 중 목적함수를 가장 최소화하는 theta 하나만 구함
        next_value = target_fn(next_theta) # new_theta에서의 목적함수 값
        
        # print(f'value={value}, next_value={next_value} ')
        # tolerance만큼 수렴하면 멈춤
        temp = abs(value-next_value) # old_theta에서의 목적함수의 값과 new_theta에서의 목적함수의 값의 차이
        if temp < tolerance: # tolerance보다 작으면 stop
            return theta
        else: # tolerance보다 크면 다시 theta와 value를 update
            theta = next_theta
            value = next_value
        
```

In \[8\]:

```text
def stochastic():
    """
    sgd 구현 (추가적인 부분이니 필수는 아닙니다.)
    random sampling 하는 부분(함수로 따로 구현하셔도 ok) + gd 부분
    
    """
```

### 1. 로지스틱 함수[¶]() <a id="1.-&#xB85C;&#xC9C0;&#xC2A4;&#xD2F1;-&#xD568;&#xC218;"></a>

해당 함수는 1/\(1+exp\[-\(ax+b\)\]로 표현되었음을 기억합시다.

In \[9\]:

```text
def logistic(x):
    try:
        return 1.0 / (1 + math.exp(-x)) # 시그모이드 함수 
    except:
        return 1e-8 
```

In \[10\]:

```text
def softmax():
    """
    softmax 구현
    """
    return None
```

### 2. Likelihood 구현[¶]() <a id="2.-Likelihood-&#xAD6C;&#xD604;"></a>

그냥 Likelihood function 대신, log likelihood function을 이용해서 구현하세요.

In \[11\]:

```text
def logistic_log_likelihood_i(x_i, y_i, beta): # 개별 데이터포인트에 대한 likelihood 값
    """
    해당 함수에 대한 설명을 작성하고,
    리턴문을 채우세요.
    """
    if y_i == 1:
        # target이 1일 때
        return math.log(logistic(dot(x_i, beta))) 
    else:
        # target이 0일 때
        return math.log(1-logistic(dot(x_i, beta))) 
```

In \[12\]:

```text
def logistic_log_likelihood(X, y, beta): # 전체 데이터에 대한 likelihood
    """
    함수의 인자를 채워넣고,
    zip 함수를 이용하여 return 문을 완성하세요.
    """
    
    # 각 데이터에서 구한 log_likelihood의 합
    log_likelihood =  [logistic_log_likelihood_i(x_i, y_i, beta) for x_i, y_i in zip(X, y)]
    return sum( log_likelihood ) 
```

### 3. Gradient for Log Reg[¶]() <a id="3.-Gradient-for-Log-Reg"></a>

아래 3가지 함수에 대해 해당 함수의 인자와 기능을 자세히 설명하세요.

In \[13\]:

```text
# def vector_add(v, w):
#     return [v_i + w_i for v_i, w_i in zip(v,w)]
```

In \[14\]:

```text
# x_i : 하나의 데이터 (한 개의 row)
# y_i : target value
# beta : parameter
# j번째 beta에 대해서 편미분 하라는 의미
# -> 즉, log likelihood 값을 j번째 beta에 대해서 편미분한 값
def logistic_log_partial_ij(x_i, y_i, beta, j):
    return (y_i - logistic(dot(x_i, beta))) * x_i[j]

# 하나의 데이터에 대해서 log likelihood 값을 계산한 후
# 모든 beta에 대해서 편미분해서 나온 gradient vector를 return
def logistic_log_gradient_i(x_i, y_i, beta):
    return [logistic_log_partial_ij(x_i, y_i, beta, j) for j, _ in enumerate(beta)]


# 모든 데이터에 대해서 log likelihood 값을 계산한 후
# 모든 beta에 대해서 편미분해서 나온 gradient vector를 return
# [logistic_log_gradient_i(x_i, y_i, beta) for x_i, y_i in zip(x, y)]
# => 리스트 속 리스트(하나의 데이터에 대해서 나온 gradient)
def logistic_log_gradient(x, y, beta):
    return reduce(vector_add, [logistic_log_gradient_i(x_i, y_i, beta) for x_i, y_i in zip(x, y)])
```

In \[15\]:

```text
reduce(vector_add, [[1, 2, 3], [4, 5, 6]])
```

### 4. Model Fitting[¶]() <a id="4.-Model-Fitting"></a>

위에서 구현한 log likelihood를 이용하여 Model을 Fitting 시켜보세요.  
 앞서 우리는 log likelihood를 maximize하는 방향으로 회귀계수를 추정한다고 배웠습니다.  
 Gradient Descent는 경사 "하강법"으로 최솟값을 찾는 데에 사용되는 알고리즘입니다.  
 따라서 log likelihood를 적절히 변형을 해야 Gradient Descent 코드를 적용할 수 있습니다.  
 log likelihood 변형 함수는 assignment2.py에 구현되어있으니, None값만 채워주시면 됩니다.

Out\[16\]:

|  | Label | bias | experience | salary |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 1 | 1 | 0.7 | 48000 |
| 1 | 0 | 1 | 1.9 | 48000 |
| 2 | 1 | 1 | 2.5 | 60000 |
| 3 | 0 | 1 | 4.2 | 63000 |
| 4 | 0 | 1 | 6.0 | 76000 |
| 5 | 0 | 1 | 6.5 | 69000 |
| 6 | 0 | 1 | 7.5 | 76000 |
| 7 | 0 | 1 | 8.1 | 88000 |
| 8 | 1 | 1 | 8.7 | 83000 |
| 9 | 1 | 1 | 10.0 | 83000 |
| 10 | 0 | 1 | 0.8 | 43000 |
| 11 | 0 | 1 | 1.8 | 60000 |
| 12 | 1 | 1 | 10.0 | 79000 |
| 13 | 0 | 1 | 6.1 | 76000 |
| 14 | 0 | 1 | 1.4 | 50000 |
| 15 | 0 | 1 | 9.1 | 92000 |
| 16 | 0 | 1 | 5.8 | 75000 |
| 17 | 0 | 1 | 5.2 | 69000 |
| 18 | 0 | 1 | 1.0 | 56000 |
| 19 | 0 | 1 | 6.0 | 67000 |
| 20 | 0 | 1 | 4.9 | 74000 |
| 21 | 1 | 1 | 6.4 | 63000 |
| 22 | 0 | 1 | 6.2 | 82000 |
| 23 | 0 | 1 | 3.3 | 58000 |
| 24 | 1 | 1 | 9.3 | 90000 |
| 25 | 1 | 1 | 5.5 | 57000 |
| 26 | 0 | 1 | 9.1 | 102000 |
| 27 | 0 | 1 | 2.4 | 54000 |
| 28 | 1 | 1 | 8.2 | 65000 |
| 29 | 0 | 1 | 5.3 | 82000 |
| ... | ... | ... | ... | ... |
| 170 | 0 | 1 | 6.2 | 70000 |
| 171 | 1 | 1 | 6.6 | 56000 |
| 172 | 0 | 1 | 6.3 | 76000 |
| 173 | 0 | 1 | 6.5 | 78000 |
| 174 | 0 | 1 | 5.1 | 59000 |
| 175 | 1 | 1 | 9.5 | 74000 |
| 176 | 0 | 1 | 4.5 | 64000 |
| 177 | 0 | 1 | 2.0 | 54000 |
| 178 | 0 | 1 | 1.0 | 52000 |
| 179 | 0 | 1 | 4.0 | 69000 |
| 180 | 0 | 1 | 6.5 | 76000 |
| 181 | 0 | 1 | 3.0 | 60000 |
| 182 | 0 | 1 | 4.5 | 63000 |
| 183 | 0 | 1 | 7.8 | 70000 |
| 184 | 1 | 1 | 3.9 | 60000 |
| 185 | 0 | 1 | 0.8 | 51000 |
| 186 | 0 | 1 | 4.2 | 78000 |
| 187 | 0 | 1 | 1.1 | 54000 |
| 188 | 0 | 1 | 6.2 | 60000 |
| 189 | 0 | 1 | 2.9 | 59000 |
| 190 | 0 | 1 | 2.1 | 52000 |
| 191 | 0 | 1 | 8.2 | 87000 |
| 192 | 0 | 1 | 4.8 | 73000 |
| 193 | 1 | 1 | 2.2 | 42000 |
| 194 | 0 | 1 | 9.1 | 98000 |
| 195 | 0 | 1 | 6.5 | 84000 |
| 196 | 0 | 1 | 6.9 | 73000 |
| 197 | 0 | 1 | 5.1 | 72000 |
| 198 | 1 | 1 | 9.1 | 69000 |
| 199 | 1 | 1 | 9.8 | 79000 |

200 rows × 4 columns

In \[17\]:

```text
X = data.drop('Label', axis = 1) # feature
y = data['Label'].values # target
```

In \[59\]:

```text
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(y, palette='Set2')
plt.title('Target Distirubtion', fontsize=15)

print(pd.Series(y).value_counts()/len(y))

# 0인 target 0.74% (148건)
# 1인 target 0.26% (52건)
```

```text
0    0.74
1    0.26
dtype: float64
```

![](data:image/png;base64,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)

In \[18\]:

```text
from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler() # standard scaling
X = standard_scaler.fit_transform(X)
```

In \[19\]:

```text
from sklearn.model_selection import train_test_split
import random
```

In \[20\]:

```text
random.seed(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
```

In \[21\]:

```text
# 아래에 Model Fitting 진행
from functools import partial # partial을 이용해 fn과 gradient_fn 구현

# 위에서 짠 코드는 목적함수를 최소화하는 방향으로 학습을 진행하는데,
# log likelihood를 목적함수로 사용할 경우 최대회되는 방향으로 학습이 이뤄져야 한다.
# 따라서 neg를 이용해 negative log likelihood를 목적함수로 지정
fn = neg(partial(logistic_log_likelihood, X_train, y_train))
gradient_fn = neg_all(partial(logistic_log_gradient, X_train, y_train))

beta_0 = [random.random() for _ in range(3)] # 임의의 시작점


# 경사 하강법으로 최적화
beta_hat = minimize_bgd(fn, gradient_fn, beta_0)
```

Out\[22\]:

```text
[0.8444218515250481, 2.6123852042859523, -2.538819721385238]
```

## sklearn과 비교[¶]() <a id="sklearn&#xACFC;-&#xBE44;&#xAD50;"></a>

우리가 구현한 함수와 sklearn의 LogisticRegression을 비교함.

In \[23\]:

```text
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
```

In \[24\]:

```text
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
sk_predict_value = lr_clf.predict(X_test)
sk_predict_proba = lr_clf.predict_proba(X_test)[:, 1]
```

In \[63\]:

```text
# 여러가지 평가지표 값을 반환해주는 함수 작성
def get_score(y_true, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    print('accuracy\t> {}'.format( round(accuracy, 4)))
    print('recall\t\t> {}'.format( round(recall, 4)))
    print('precision\t> {}'.format( round(precision, 4)))
    print('auc\t\t> {}'.format( round(auc, 4)))
    
    return (accuracy, recall, precision, auc)
    
```

In \[44\]:

```text
# threshold를 변화해가며 어떤 threshold에서 평가지표가 좋은 지 확인해보고자 한다.
def change_threshold(X, beta, threshold):
    values = X.dot(beta)
    predict_proba = [logistic(value) for value in values]
    predict_value = [1 if proba>=threshold else 0 for proba in predict_proba]
    return predict_value, predict_proba
```

In \[45\]:

```text
# 구현한 gradient descent의 성능
import numpy as np
thresholds = np.arange(0.2, 0.9, 0.1)
for threshold in thresholds:
    predict_value, predict_proba = change_threshold(X_test, beta_hat, threshold)
    print(f'threshold = {threshold}')
    get_score(y_test, predict_value, predict_proba)
    print('='*50)
    
# threshold는 0.2부터 0.8까지 0.1 간격으로 변화해보았다.
# 여기서는 accuracy를 주된 평가지표로 사용하도록 하겠다. 
# (유료계정을 등록할 지, 하지 않을지 둘 다 예측해야 하는 문제라고 생각했기 때문에)
# threshold를 0.7, 0.8로 설정했을 때 accuracy가 0.90으로 가장 높았으며
# 다른 평가지표들도 threshold가 0.7, 0.8일 때 전반적으로 고르게 높게 나왔다.
# 우리는 0.7~0.8 근처의 값에서 좀 더 세밀하게 threshold를 조절해보려고 한다.
```

```text
threshold = 0.2
accuracy	> 0.5152
recall		> 1.0
precision	> 0.3191
auc		> 0.9634
==================================================
threshold = 0.30000000000000004
accuracy	> 0.6061
recall		> 1.0
precision	> 0.3659
auc		> 0.9634
==================================================
threshold = 0.4000000000000001
accuracy	> 0.697
recall		> 1.0
precision	> 0.4286
auc		> 0.9634
==================================================
threshold = 0.5000000000000001
accuracy	> 0.803
recall		> 1.0
precision	> 0.5357
auc		> 0.9634
==================================================
threshold = 0.6000000000000001
accuracy	> 0.8333
recall		> 0.9333
precision	> 0.5833
auc		> 0.9634
==================================================
threshold = 0.7000000000000002
accuracy	> 0.9091
recall		> 0.9333
precision	> 0.7368
auc		> 0.9634
==================================================
threshold = 0.8000000000000003
accuracy	> 0.9091
recall		> 0.8
precision	> 0.8
auc		> 0.9634
==================================================
```

Out\[93\]:

```text
dict_values([0.9090909090909091, 0.9090909090909091, 0.9242424242424242, 0.9242424242424242, 0.9393939393939394, 0.9242424242424242, 0.9242424242424242, 0.9242424242424242, 0.9242424242424242, 0.9090909090909091, 0.9090909090909091, 0.9090909090909091, 0.8939393939393939, 0.8939393939393939, 0.9090909090909091, 0.8787878787878788, 0.8636363636363636, 0.8181818181818182, 0.8181818181818182, 0.8333333333333334, 0.8333333333333334])
```

In \[96\]:

```text
thresholds = np.arange(0.7, 0.9, 0.01)
accuracy_dict = {}
for threshold in thresholds:
    predict_value, predict_proba = change_threshold(X_test, beta_hat, threshold)
    print(f'threshold = {threshold}')
    accuracy, _, _, _ = get_score(y_test, predict_value, predict_proba)
    accuracy_dict[threshold] = accuracy
    print('='*50)

plt.figure(figsize=(13, 6))
sns.lineplot(x=thresholds, y=list(accuracy_dict.values()))
plt.axvline(x=0.74, ymin=0, ymax=1, ls='--', c='red')
plt.axhline(y=accuracy_dict[0.74], xmin=0, xmax=1, ls='--', c='green')
plt.xticks(ticks=thresholds)
plt.show()

# 구간을 좀 더 세밀하게 하여 thresholds를 조정해본 결과
# threhsold가 0.74일 때 accuracy가 0.9394로 가장 높게 나왔다.
# 이 때의 다른 평가지표들도 다른 threshold에 비해 고르게 좋은 결과가 나왔다.
```

```text
threshold = 0.7
accuracy	> 0.9091
recall		> 0.9333
precision	> 0.7368
auc		> 0.9634
==================================================
threshold = 0.71
accuracy	> 0.9091
recall		> 0.9333
precision	> 0.7368
auc		> 0.9634
==================================================
threshold = 0.72
accuracy	> 0.9242
recall		> 0.9333
precision	> 0.7778
auc		> 0.9634
==================================================
threshold = 0.73
accuracy	> 0.9242
recall		> 0.9333
precision	> 0.7778
auc		> 0.9634
==================================================
threshold = 0.74
accuracy	> 0.9394
recall		> 0.9333
precision	> 0.8235
auc		> 0.9634
==================================================
threshold = 0.75
accuracy	> 0.9242
recall		> 0.8667
precision	> 0.8125
auc		> 0.9634
==================================================
threshold = 0.76
accuracy	> 0.9242
recall		> 0.8667
precision	> 0.8125
auc		> 0.9634
==================================================
threshold = 0.77
accuracy	> 0.9242
recall		> 0.8667
precision	> 0.8125
auc		> 0.9634
==================================================
threshold = 0.78
accuracy	> 0.9242
recall		> 0.8667
precision	> 0.8125
auc		> 0.9634
==================================================
threshold = 0.79
accuracy	> 0.9091
recall		> 0.8
precision	> 0.8
auc		> 0.9634
==================================================
threshold = 0.8
accuracy	> 0.9091
recall		> 0.8
precision	> 0.8
auc		> 0.9634
==================================================
threshold = 0.81
accuracy	> 0.9091
recall		> 0.7333
precision	> 0.8462
auc		> 0.9634
==================================================
threshold = 0.8200000000000001
accuracy	> 0.8939
recall		> 0.6667
precision	> 0.8333
auc		> 0.9634
==================================================
threshold = 0.8300000000000001
accuracy	> 0.8939
recall		> 0.6667
precision	> 0.8333
auc		> 0.9634
==================================================
threshold = 0.8400000000000001
accuracy	> 0.9091
recall		> 0.6667
precision	> 0.9091
auc		> 0.9634
==================================================
threshold = 0.8500000000000001
accuracy	> 0.8788
recall		> 0.5333
precision	> 0.8889
auc		> 0.9634
==================================================
threshold = 0.8600000000000001
accuracy	> 0.8636
recall		> 0.4667
precision	> 0.875
auc		> 0.9634
==================================================
threshold = 0.8700000000000001
accuracy	> 0.8182
recall		> 0.2667
precision	> 0.8
auc		> 0.9634
==================================================
threshold = 0.8800000000000001
accuracy	> 0.8182
recall		> 0.2667
precision	> 0.8
auc		> 0.9634
==================================================
threshold = 0.8900000000000001
accuracy	> 0.8333
recall		> 0.2667
precision	> 1.0
auc		> 0.9634
==================================================
threshold = 0.9000000000000001
accuracy	> 0.8333
recall		> 0.2667
precision	> 1.0
auc		> 0.9634
==================================================
```

![](data:image/png;base64,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)

### sklearn의 LogisticRegression에서 threshold 조정[¶]() <a id="sklearn&#xC758;-LogisticRegression&#xC5D0;&#xC11C;-threshold-&#xC870;&#xC815;"></a>

In \[74\]:

```text
# sklearn의 LogisticRegression의 성능
get_score(y_test, sk_predict_value, sk_predict_proba)
```

```text
accuracy	> 0.8939
recall		> 0.6667
precision	> 0.8333
auc		> 0.9595
```

Out\[74\]:

```text
(0.8939393939393939,
 0.6666666666666666,
 0.8333333333333334,
 0.9594771241830066)
```

In \[76\]:

```text
thresholds = np.arange(0.2, 0.9, 0.1)
for threshold in thresholds:
    predict_value = [1 if proba>=threshold else 0 for proba in sk_predict_proba]
    print(f'threshold = {threshold}')
    get_score(y_test, predict_value, sk_predict_proba)
    print('='*50)
    
# 위에서와 동일하게 threshold는 0.2부터 0.8까지 0.1 간격으로 변화해보았다.
# sklearn의 Logistic에서는 우리가 구현한 함수와는 다르게 threshold가 0.4일 때 accuracy가 0.9242로 가장 높았다.
# (우리가 구현한 함수에서는 threshold를 0.7, 0.8로 설정했을 때 accuracy가 0.90으로 가장 높았다)
# 이번에는 0.4 근처의 값에서 threshold를 좀 더 세밀하게 조정해보도록 하겠다.
```

```text
threshold = 0.2
accuracy	> 0.803
recall		> 1.0
precision	> 0.5357
auc		> 0.9595
==================================================
threshold = 0.30000000000000004
accuracy	> 0.8788
recall		> 0.9333
precision	> 0.6667
auc		> 0.9595
==================================================
threshold = 0.4000000000000001
accuracy	> 0.9242
recall		> 0.8667
precision	> 0.8125
auc		> 0.9595
==================================================
threshold = 0.5000000000000001
accuracy	> 0.8939
recall		> 0.6667
precision	> 0.8333
auc		> 0.9595
==================================================
threshold = 0.6000000000000001
accuracy	> 0.8182
recall		> 0.2667
precision	> 0.8
auc		> 0.9595
==================================================
threshold = 0.7000000000000002
accuracy	> 0.8182
recall		> 0.2
precision	> 1.0
auc		> 0.9595
==================================================
threshold = 0.8000000000000003
accuracy	> 0.8182
recall		> 0.2
precision	> 1.0
auc		> 0.9595
==================================================
```

In \[107\]:

```text
thresholds = np.arange(0.3, 0.5, 0.01)
accuracy_dict = {}
for threshold in thresholds:
    predict_value = [1 if proba>=threshold else 0 for proba in sk_predict_proba]
    print(f'threshold = {threshold}')
    accuracy, _, _, _ = get_score(y_test, predict_value, sk_predict_proba)
    accuracy_dict[round(threshold, 3)] = accuracy
    print('='*50)

plt.figure(figsize=(13, 6))
sns.lineplot(x=thresholds, y=list(accuracy_dict.values()))
plt.axvline(x=0.39, ymin=0, ymax=1, ls='--', c='red')
plt.axhline(y=accuracy_dict[0.39], xmin=0, xmax=1, ls='--', c='green')
plt.xticks(ticks=thresholds)
plt.show()

# 구간을 좀 더 세밀하게 하여 thresholds를 조정해본 결과
# threhsold가 0.39일 때 accuracy가 0.9394로 가장 높게 나왔다.
# 이 때의 다른 평가지표들도 다른 threshold에 비해 고르게 좋은 결과가 나왔다.
```

```text
threshold = 0.3
accuracy	> 0.8788
recall		> 0.9333
precision	> 0.6667
auc		> 0.9595
==================================================
threshold = 0.31
accuracy	> 0.8939
recall		> 0.9333
precision	> 0.7
auc		> 0.9595
==================================================
threshold = 0.32
accuracy	> 0.9091
recall		> 0.9333
precision	> 0.7368
auc		> 0.9595
==================================================
threshold = 0.33
accuracy	> 0.9091
recall		> 0.9333
precision	> 0.7368
auc		> 0.9595
==================================================
threshold = 0.34
accuracy	> 0.9091
recall		> 0.9333
precision	> 0.7368
auc		> 0.9595
==================================================
threshold = 0.35000000000000003
accuracy	> 0.9091
recall		> 0.9333
precision	> 0.7368
auc		> 0.9595
==================================================
threshold = 0.36000000000000004
accuracy	> 0.9242
recall		> 0.9333
precision	> 0.7778
auc		> 0.9595
==================================================
threshold = 0.37000000000000005
accuracy	> 0.9242
recall		> 0.9333
precision	> 0.7778
auc		> 0.9595
==================================================
threshold = 0.38000000000000006
accuracy	> 0.9242
recall		> 0.9333
precision	> 0.7778
auc		> 0.9595
==================================================
threshold = 0.39000000000000007
accuracy	> 0.9394
recall		> 0.9333
precision	> 0.8235
auc		> 0.9595
==================================================
threshold = 0.4000000000000001
accuracy	> 0.9242
recall		> 0.8667
precision	> 0.8125
auc		> 0.9595
==================================================
threshold = 0.4100000000000001
accuracy	> 0.9242
recall		> 0.8667
precision	> 0.8125
auc		> 0.9595
==================================================
threshold = 0.4200000000000001
accuracy	> 0.9242
recall		> 0.8667
precision	> 0.8125
auc		> 0.9595
==================================================
threshold = 0.4300000000000001
accuracy	> 0.9242
recall		> 0.8667
precision	> 0.8125
auc		> 0.9595
==================================================
threshold = 0.4400000000000001
accuracy	> 0.9242
recall		> 0.8667
precision	> 0.8125
auc		> 0.9595
==================================================
threshold = 0.4500000000000001
accuracy	> 0.9091
recall		> 0.8
precision	> 0.8
auc		> 0.9595
==================================================
threshold = 0.46000000000000013
accuracy	> 0.9091
recall		> 0.8
precision	> 0.8
auc		> 0.9595
==================================================
threshold = 0.47000000000000014
accuracy	> 0.9091
recall		> 0.7333
precision	> 0.8462
auc		> 0.9595
==================================================
threshold = 0.48000000000000015
accuracy	> 0.9091
recall		> 0.7333
precision	> 0.8462
auc		> 0.9595
==================================================
threshold = 0.49000000000000016
accuracy	> 0.9091
recall		> 0.7333
precision	> 0.8462
auc		> 0.9595
==================================================
```

![](data:image/png;base64,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)

In \[ \]:

```text
# threshold를 조정했을 때 나온 best 결과는 아래와 같다.

# 구현함 함수
# threshold = 0.74
# accuracy > 0.9394
# recall > 0.9333
# precision > 0.8235
# auc > 0.9634

# sklearn LogisticRegression
# threshold = 0.39
# accuracy > 0.9394
# recall > 0.9333
# precision > 0.8235
# auc > 0.9595

# accuracy를 기준으로 best 결과를 뽑아본 결과 threshold에는 차이가 있었으나
# 전반적인 평가지표에는 큰 차이가 없었다. (accuracy, recall, precision이 소수점 넷째자리에서 동일하다! 신기하다)
```

