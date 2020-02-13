# Python을 이용한 SVM



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, KFold
```

In \[2\]:

```python
# Anomaly detection(사기감지 데이터) 로드
data = pd.read_csv('creditcard.csv')
print(data.columns)
```

```text
Index(['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
       'Class'],
      dtype='object')
```

In \[3\]:

```python
X = np.array(data.loc[:, data.columns != 'Class'].values)
y = np.array(data.Class)
```

In \[4\]:

```python
# Train, Test 셋 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.95, random_state=0)
```

In \[5\]:

```python
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```

Out\[5\]:

```text
((14240, 30), (270566, 30), (14240,), (270566,))
```

In \[6\]:

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

In \[7\]:

```python
X_test
```

Out\[7\]:

```text
array([[ 0.64698237, -0.17516838,  0.67679542, ...,  0.28359773,
         0.53175665, -0.21321296],
       [ 1.30680833, -0.18907225,  0.59652865, ...,  0.19973923,
         0.57913126, -0.38379471],
       [ 1.20674431, -0.85567843, -1.55048369, ...,  0.75445011,
         1.43893504,  0.03803843],
       ...,
       [ 1.19643125, -0.17022017,  0.67311331, ..., -1.53817237,
        -0.40964519, -0.34175497],
       [ 0.80608919,  1.0705119 , -0.04942977, ..., -0.12699757,
        -0.23548545, -0.38819161],
       [-0.30377277,  0.5088799 , -0.01341662, ...,  0.14715975,
         0.02807013, -0.21074532]])
```

In \[8\]:

```python
y_train.sum()
```

Out\[8\]:

```text
26
```

In \[9\]:

```python
from sklearn.metrics.pairwise import rbf_kernel
```

In \[10\]:

```python
import timeit
start = timeit.default_timer()

svc=SVC(kernel='linear', C = 100)
svc.fit(X_train, y_train)

stop = timeit.default_timer()
print(stop - start)
```

```text
8.0188527
```

In \[11\]:

```python
y_pred = svc.predict(X_test)
metrics.confusion_matrix(y_test, y_pred)
```

Out\[11\]:

```text
array([[270037,     63],
       [   113,    353]], dtype=int64)
```

In \[12\]:

```python
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))
print('Recall Score:')
print(metrics.recall_score(y_test, y_pred))
print('Precision Score:')
print(metrics.precision_score(y_test, y_pred))
```

```text
Accuracy Score:
0.9993495117642276
Recall Score:
0.7575107296137339
Precision Score:
0.8485576923076923
```

## Weight <a id="Weight"></a>

svc의 class\_weight를 통해 적게 관찰되는 경우에 weight를 준다

In \[13\]:

```python
svc_weight=SVC(kernel='linear', C = 100, class_weight='balanced')
svc_weight.fit(X_train,y_train)
```

Out\[13\]:

```text
SVC(C=100, break_ties=False, cache_size=200, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
```

In \[14\]:

```python
y_pred_weight = svc_weight.predict(X_test)
metrics.confusion_matrix(y_test, y_pred_weight)
```

Out\[14\]:

```text
array([[269766,    334],
       [   129,    337]], dtype=int64)
```

In \[15\]:

```python
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred_weight))
print('Recall Score:')
print(metrics.recall_score(y_test, y_pred_weight))
print('Precision Score:')
print(metrics.precision_score(y_test, y_pred_weight))
```

```text
Accuracy Score:
0.9982887724252123
Recall Score:
0.723175965665236
Precision Score:
0.5022354694485842
```

## RandomUnderSampler <a id="RandomUnderSampler"></a>

RandomUnderSampler를 통해 0, 1 사이의 대칭을 맞춰준다.In \[16\]:

```python
# Train, Test 셋 나누기
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size = 0.05, random_state=0)
```

In \[17\]:

```python
from imblearn.under_sampling import RandomUnderSampler

Xresampled, yresampled = RandomUnderSampler(random_state=0).fit_sample(X_train2, y_train2)
```

In \[18\]:

```python
Xresampled.shape, X_test2.shape, yresampled.shape, y_test2.shape
```

Out\[18\]:

```text
((920, 30), (14241, 30), (920,), (14241,))
```

In \[19\]:

```python
Xresampled = scaler.transform(Xresampled)
X_test2 = scaler.transform(X_test2)
```

In \[20\]:

```python
svc_rus=SVC(kernel='linear', C = 100)
svc_rus.fit(Xresampled, yresampled)
```

Out\[20\]:

```text
SVC(C=100, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
```

In \[21\]:

```python
y_pred_rus = svc_rus.predict(X_test2)
metrics.confusion_matrix(y_test2, y_pred_rus)
```

Out\[21\]:

```text
array([[13582,   627],
       [    2,    30]], dtype=int64)
```

In \[22\]:

```python
print('Accuracy Score:')
print(metrics.accuracy_score(y_test2, y_pred_rus))
print('Recall Score:')
print(metrics.recall_score(y_test2, y_pred_rus))
print('Precision Score:')
print(metrics.precision_score(y_test2, y_pred_rus))
```

```text
Accuracy Score:
0.9558317533881048
Recall Score:
0.9375
Precision Score:
0.045662100456621
```

In \[23\]:

```python
svc_rus_weight=SVC(kernel='linear', C = 100, class_weight='balanced')
svc_rus_weight.fit(Xresampled, yresampled)
```

Out\[23\]:

```text
SVC(C=100, break_ties=False, cache_size=200, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
```

Undersampling으로 인해 weight의 여부가 결과에 변화를 주지 못했다.In \[24\]:

```python
y_pred_rus_wieght = svc_rus_weight.predict(X_test2)
metrics.confusion_matrix(y_test2, y_pred_rus_wieght)
```

Out\[24\]:

```text
array([[13582,   627],
       [    2,    30]], dtype=int64)
```

In \[25\]:

```python
print('Accuracy Score:')
print(metrics.accuracy_score(y_test2, y_pred_rus_wieght))
print('Recall Score:')
print(metrics.recall_score(y_test2, y_pred_rus_wieght))
print('Precision Score:')
print(metrics.precision_score(y_test2, y_pred_rus_wieght))
```

```text
Accuracy Score:
0.9558317533881048
Recall Score:
0.9375
Precision Score:
0.045662100456621
```

In \[26\]:

```python
values = [0.001, 0.01, 0.1, 1, 10, 100]
param_grid = [{'kernel':['rbf'], 'C':values, 'gamma':values},
               {'kernel':['linear'], 'C':values}]
gs = GridSearchCV(SVC(), param_grid, cv= KFold(n_splits=5), scoring='accuracy', verbose = 3)
gs.fit(Xresampled, yresampled)
```

```text
Fitting 5 folds for each of 42 candidates, totalling 210 fits
[CV] C=0.001, gamma=0.001, kernel=rbf ................................
[CV] .... C=0.001, gamma=0.001, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.001, gamma=0.001, kernel=rbf ................................
[CV] .... C=0.001, gamma=0.001, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.001, gamma=0.001, kernel=rbf ................................
[CV] .... C=0.001, gamma=0.001, kernel=rbf, score=0.755, total=   0.0s
[CV] C=0.001, gamma=0.001, kernel=rbf ................................
[CV] .... C=0.001, gamma=0.001, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.001, gamma=0.001, kernel=rbf ................................
[CV] .... C=0.001, gamma=0.001, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.001, gamma=0.01, kernel=rbf .................................
[CV] ..... C=0.001, gamma=0.01, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.001, gamma=0.01, kernel=rbf .................................
[CV] ..... C=0.001, gamma=0.01, kernel=rbf, score=0.000, total=   0.0s
```

```text
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
[Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    0.0s remaining:    0.0s
```

```text
[CV] C=0.001, gamma=0.01, kernel=rbf .................................
[CV] ..... C=0.001, gamma=0.01, kernel=rbf, score=0.913, total=   0.0s
[CV] C=0.001, gamma=0.01, kernel=rbf .................................
[CV] ..... C=0.001, gamma=0.01, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.001, gamma=0.01, kernel=rbf .................................
[CV] ..... C=0.001, gamma=0.01, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.001, gamma=0.1, kernel=rbf ..................................
[CV] ...... C=0.001, gamma=0.1, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.001, gamma=0.1, kernel=rbf ..................................
[CV] ...... C=0.001, gamma=0.1, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.001, gamma=0.1, kernel=rbf ..................................
[CV] ...... C=0.001, gamma=0.1, kernel=rbf, score=0.641, total=   0.0s
[CV] C=0.001, gamma=0.1, kernel=rbf ..................................
[CV] ...... C=0.001, gamma=0.1, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.001, gamma=0.1, kernel=rbf ..................................
[CV] ...... C=0.001, gamma=0.1, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.001, gamma=1, kernel=rbf ....................................
[CV] ........ C=0.001, gamma=1, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.001, gamma=1, kernel=rbf ....................................
[CV] ........ C=0.001, gamma=1, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.001, gamma=1, kernel=rbf ....................................
[CV] ........ C=0.001, gamma=1, kernel=rbf, score=0.500, total=   0.0s
[CV] C=0.001, gamma=1, kernel=rbf ....................................
[CV] ........ C=0.001, gamma=1, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.001, gamma=1, kernel=rbf ....................................
[CV] ........ C=0.001, gamma=1, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.001, gamma=10, kernel=rbf ...................................
[CV] ....... C=0.001, gamma=10, kernel=rbf, score=0.000, total=   0.1s
[CV] C=0.001, gamma=10, kernel=rbf ...................................
[CV] ....... C=0.001, gamma=10, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.001, gamma=10, kernel=rbf ...................................
[CV] ....... C=0.001, gamma=10, kernel=rbf, score=0.500, total=   0.1s
[CV] C=0.001, gamma=10, kernel=rbf ...................................
[CV] ....... C=0.001, gamma=10, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.001, gamma=10, kernel=rbf ...................................
[CV] ....... C=0.001, gamma=10, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.001, gamma=100, kernel=rbf ..................................
[CV] ...... C=0.001, gamma=100, kernel=rbf, score=0.000, total=   0.1s
[CV] C=0.001, gamma=100, kernel=rbf ..................................
[CV] ...... C=0.001, gamma=100, kernel=rbf, score=0.000, total=   0.1s
[CV] C=0.001, gamma=100, kernel=rbf ..................................
[CV] ...... C=0.001, gamma=100, kernel=rbf, score=0.500, total=   0.1s
[CV] C=0.001, gamma=100, kernel=rbf ..................................
[CV] ...... C=0.001, gamma=100, kernel=rbf, score=0.000, total=   0.1s
[CV] C=0.001, gamma=100, kernel=rbf ..................................
[CV] ...... C=0.001, gamma=100, kernel=rbf, score=0.000, total=   0.1s
[CV] C=0.01, gamma=0.001, kernel=rbf .................................
[CV] ..... C=0.01, gamma=0.001, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.01, gamma=0.001, kernel=rbf .................................
[CV] ..... C=0.01, gamma=0.001, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.01, gamma=0.001, kernel=rbf .................................
[CV] ..... C=0.01, gamma=0.001, kernel=rbf, score=0.783, total=   0.0s
[CV] C=0.01, gamma=0.001, kernel=rbf .................................
[CV] ..... C=0.01, gamma=0.001, kernel=rbf, score=0.364, total=   0.0s
[CV] C=0.01, gamma=0.001, kernel=rbf .................................
[CV] ..... C=0.01, gamma=0.001, kernel=rbf, score=0.364, total=   0.0s
[CV] C=0.01, gamma=0.01, kernel=rbf ..................................
[CV] ...... C=0.01, gamma=0.01, kernel=rbf, score=0.935, total=   0.0s
[CV] C=0.01, gamma=0.01, kernel=rbf ..................................
[CV] ...... C=0.01, gamma=0.01, kernel=rbf, score=0.913, total=   0.0s
[CV] C=0.01, gamma=0.01, kernel=rbf ..................................
[CV] ...... C=0.01, gamma=0.01, kernel=rbf, score=0.924, total=   0.0s
[CV] C=0.01, gamma=0.01, kernel=rbf ..................................
[CV] ...... C=0.01, gamma=0.01, kernel=rbf, score=0.821, total=   0.0s
[CV] C=0.01, gamma=0.01, kernel=rbf ..................................
[CV] ...... C=0.01, gamma=0.01, kernel=rbf, score=0.821, total=   0.0s
[CV] C=0.01, gamma=0.1, kernel=rbf ...................................
[CV] ....... C=0.01, gamma=0.1, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.01, gamma=0.1, kernel=rbf ...................................
[CV] ....... C=0.01, gamma=0.1, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.01, gamma=0.1, kernel=rbf ...................................
[CV] ....... C=0.01, gamma=0.1, kernel=rbf, score=0.641, total=   0.0s
[CV] C=0.01, gamma=0.1, kernel=rbf ...................................
[CV] ....... C=0.01, gamma=0.1, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.01, gamma=0.1, kernel=rbf ...................................
[CV] ....... C=0.01, gamma=0.1, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.01, gamma=1, kernel=rbf .....................................
[CV] ......... C=0.01, gamma=1, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.01, gamma=1, kernel=rbf .....................................
[CV] ......... C=0.01, gamma=1, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.01, gamma=1, kernel=rbf .....................................
[CV] ......... C=0.01, gamma=1, kernel=rbf, score=0.500, total=   0.0s
[CV] C=0.01, gamma=1, kernel=rbf .....................................
[CV] ......... C=0.01, gamma=1, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.01, gamma=1, kernel=rbf .....................................
[CV] ......... C=0.01, gamma=1, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.01, gamma=10, kernel=rbf ....................................
[CV] ........ C=0.01, gamma=10, kernel=rbf, score=0.000, total=   0.1s
[CV] C=0.01, gamma=10, kernel=rbf ....................................
[CV] ........ C=0.01, gamma=10, kernel=rbf, score=0.000, total=   0.1s
[CV] C=0.01, gamma=10, kernel=rbf ....................................
[CV] ........ C=0.01, gamma=10, kernel=rbf, score=0.500, total=   0.1s
[CV] C=0.01, gamma=10, kernel=rbf ....................................
[CV] ........ C=0.01, gamma=10, kernel=rbf, score=0.000, total=   0.1s
[CV] C=0.01, gamma=10, kernel=rbf ....................................
[CV] ........ C=0.01, gamma=10, kernel=rbf, score=0.000, total=   0.1s
[CV] C=0.01, gamma=100, kernel=rbf ...................................
[CV] ....... C=0.01, gamma=100, kernel=rbf, score=0.000, total=   0.1s
[CV] C=0.01, gamma=100, kernel=rbf ...................................
[CV] ....... C=0.01, gamma=100, kernel=rbf, score=0.000, total=   0.1s
[CV] C=0.01, gamma=100, kernel=rbf ...................................
[CV] ....... C=0.01, gamma=100, kernel=rbf, score=0.500, total=   0.1s
[CV] C=0.01, gamma=100, kernel=rbf ...................................
[CV] ....... C=0.01, gamma=100, kernel=rbf, score=0.000, total=   0.1s
[CV] C=0.01, gamma=100, kernel=rbf ...................................
[CV] ....... C=0.01, gamma=100, kernel=rbf, score=0.000, total=   0.1s
[CV] C=0.1, gamma=0.001, kernel=rbf ..................................
[CV] ...... C=0.1, gamma=0.001, kernel=rbf, score=1.000, total=   0.0s
[CV] C=0.1, gamma=0.001, kernel=rbf ..................................
[CV] ...... C=0.1, gamma=0.001, kernel=rbf, score=1.000, total=   0.0s
[CV] C=0.1, gamma=0.001, kernel=rbf ..................................
[CV] ...... C=0.1, gamma=0.001, kernel=rbf, score=0.918, total=   0.0s
[CV] C=0.1, gamma=0.001, kernel=rbf ..................................
[CV] ...... C=0.1, gamma=0.001, kernel=rbf, score=0.788, total=   0.0s
[CV] C=0.1, gamma=0.001, kernel=rbf ..................................
[CV] ...... C=0.1, gamma=0.001, kernel=rbf, score=0.799, total=   0.0s
[CV] C=0.1, gamma=0.01, kernel=rbf ...................................
[CV] ....... C=0.1, gamma=0.01, kernel=rbf, score=0.962, total=   0.0s
[CV] C=0.1, gamma=0.01, kernel=rbf ...................................
[CV] ....... C=0.1, gamma=0.01, kernel=rbf, score=0.978, total=   0.0s
[CV] C=0.1, gamma=0.01, kernel=rbf ...................................
[CV] ....... C=0.1, gamma=0.01, kernel=rbf, score=0.929, total=   0.0s
[CV] C=0.1, gamma=0.01, kernel=rbf ...................................
[CV] ....... C=0.1, gamma=0.01, kernel=rbf, score=0.864, total=   0.0s
[CV] C=0.1, gamma=0.01, kernel=rbf ...................................
[CV] ....... C=0.1, gamma=0.01, kernel=rbf, score=0.870, total=   0.0s
[CV] C=0.1, gamma=0.1, kernel=rbf ....................................
[CV] ........ C=0.1, gamma=0.1, kernel=rbf, score=0.451, total=   0.0s
[CV] C=0.1, gamma=0.1, kernel=rbf ....................................
[CV] ........ C=0.1, gamma=0.1, kernel=rbf, score=0.424, total=   0.0s
[CV] C=0.1, gamma=0.1, kernel=rbf ....................................
[CV] ........ C=0.1, gamma=0.1, kernel=rbf, score=0.728, total=   0.0s
[CV] C=0.1, gamma=0.1, kernel=rbf ....................................
[CV] ........ C=0.1, gamma=0.1, kernel=rbf, score=0.016, total=   0.0s
[CV] C=0.1, gamma=0.1, kernel=rbf ....................................
[CV] ........ C=0.1, gamma=0.1, kernel=rbf, score=0.016, total=   0.0s
[CV] C=0.1, gamma=1, kernel=rbf ......................................
[CV] .......... C=0.1, gamma=1, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.1, gamma=1, kernel=rbf ......................................
[CV] .......... C=0.1, gamma=1, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.1, gamma=1, kernel=rbf ......................................
[CV] .......... C=0.1, gamma=1, kernel=rbf, score=0.500, total=   0.0s
[CV] C=0.1, gamma=1, kernel=rbf ......................................
[CV] .......... C=0.1, gamma=1, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.1, gamma=1, kernel=rbf ......................................
[CV] .......... C=0.1, gamma=1, kernel=rbf, score=0.000, total=   0.0s
[CV] C=0.1, gamma=10, kernel=rbf .....................................
[CV] ......... C=0.1, gamma=10, kernel=rbf, score=0.000, total=   0.1s
[CV] C=0.1, gamma=10, kernel=rbf .....................................
[CV] ......... C=0.1, gamma=10, kernel=rbf, score=0.000, total=   0.1s
[CV] C=0.1, gamma=10, kernel=rbf .....................................
[CV] ......... C=0.1, gamma=10, kernel=rbf, score=0.500, total=   0.1s
[CV] C=0.1, gamma=10, kernel=rbf .....................................
[CV] ......... C=0.1, gamma=10, kernel=rbf, score=0.000, total=   0.1s
[CV] C=0.1, gamma=10, kernel=rbf .....................................
[CV] ......... C=0.1, gamma=10, kernel=rbf, score=0.000, total=   0.1s
[CV] C=0.1, gamma=100, kernel=rbf ....................................
[CV] ........ C=0.1, gamma=100, kernel=rbf, score=0.000, total=   0.1s
[CV] C=0.1, gamma=100, kernel=rbf ....................................
[CV] ........ C=0.1, gamma=100, kernel=rbf, score=0.000, total=   0.1s
[CV] C=0.1, gamma=100, kernel=rbf ....................................
[CV] ........ C=0.1, gamma=100, kernel=rbf, score=0.500, total=   0.1s
[CV] C=0.1, gamma=100, kernel=rbf ....................................
[CV] ........ C=0.1, gamma=100, kernel=rbf, score=0.000, total=   0.1s
[CV] C=0.1, gamma=100, kernel=rbf ....................................
[CV] ........ C=0.1, gamma=100, kernel=rbf, score=0.000, total=   0.1s
[CV] C=1, gamma=0.001, kernel=rbf ....................................
[CV] ........ C=1, gamma=0.001, kernel=rbf, score=0.989, total=   0.0s
[CV] C=1, gamma=0.001, kernel=rbf ....................................
[CV] ........ C=1, gamma=0.001, kernel=rbf, score=0.995, total=   0.0s
[CV] C=1, gamma=0.001, kernel=rbf ....................................
[CV] ........ C=1, gamma=0.001, kernel=rbf, score=0.929, total=   0.0s
[CV] C=1, gamma=0.001, kernel=rbf ....................................
[CV] ........ C=1, gamma=0.001, kernel=rbf, score=0.848, total=   0.0s
[CV] C=1, gamma=0.001, kernel=rbf ....................................
[CV] ........ C=1, gamma=0.001, kernel=rbf, score=0.848, total=   0.0s
[CV] C=1, gamma=0.01, kernel=rbf .....................................
[CV] ......... C=1, gamma=0.01, kernel=rbf, score=0.967, total=   0.0s
[CV] C=1, gamma=0.01, kernel=rbf .....................................
[CV] ......... C=1, gamma=0.01, kernel=rbf, score=0.984, total=   0.0s
[CV] C=1, gamma=0.01, kernel=rbf .....................................
[CV] ......... C=1, gamma=0.01, kernel=rbf, score=0.935, total=   0.0s
[CV] C=1, gamma=0.01, kernel=rbf .....................................
[CV] ......... C=1, gamma=0.01, kernel=rbf, score=0.891, total=   0.0s
[CV] C=1, gamma=0.01, kernel=rbf .....................................
[CV] ......... C=1, gamma=0.01, kernel=rbf, score=0.897, total=   0.0s
[CV] C=1, gamma=0.1, kernel=rbf ......................................
[CV] .......... C=1, gamma=0.1, kernel=rbf, score=0.783, total=   0.0s
[CV] C=1, gamma=0.1, kernel=rbf ......................................
[CV] .......... C=1, gamma=0.1, kernel=rbf, score=0.810, total=   0.0s
[CV] C=1, gamma=0.1, kernel=rbf ......................................
[CV] .......... C=1, gamma=0.1, kernel=rbf, score=0.913, total=   0.0s
[CV] C=1, gamma=0.1, kernel=rbf ......................................
[CV] .......... C=1, gamma=0.1, kernel=rbf, score=0.940, total=   0.0s
[CV] C=1, gamma=0.1, kernel=rbf ......................................
[CV] .......... C=1, gamma=0.1, kernel=rbf, score=0.935, total=   0.0s
[CV] C=1, gamma=1, kernel=rbf ........................................
[CV] ............ C=1, gamma=1, kernel=rbf, score=0.049, total=   0.0s
[CV] C=1, gamma=1, kernel=rbf ........................................
[CV] ............ C=1, gamma=1, kernel=rbf, score=0.038, total=   0.0s
[CV] C=1, gamma=1, kernel=rbf ........................................
[CV] ............ C=1, gamma=1, kernel=rbf, score=0.598, total=   0.0s
[CV] C=1, gamma=1, kernel=rbf ........................................
[CV] ............ C=1, gamma=1, kernel=rbf, score=0.098, total=   0.0s
[CV] C=1, gamma=1, kernel=rbf ........................................
[CV] ............ C=1, gamma=1, kernel=rbf, score=0.103, total=   0.0s
[CV] C=1, gamma=10, kernel=rbf .......................................
[CV] ........... C=1, gamma=10, kernel=rbf, score=0.005, total=   0.1s
[CV] C=1, gamma=10, kernel=rbf .......................................
[CV] ........... C=1, gamma=10, kernel=rbf, score=0.000, total=   0.1s
[CV] C=1, gamma=10, kernel=rbf .......................................
[CV] ........... C=1, gamma=10, kernel=rbf, score=0.527, total=   0.1s
[CV] C=1, gamma=10, kernel=rbf .......................................
[CV] ........... C=1, gamma=10, kernel=rbf, score=0.054, total=   0.1s
[CV] C=1, gamma=10, kernel=rbf .......................................
[CV] ........... C=1, gamma=10, kernel=rbf, score=0.038, total=   0.1s
[CV] C=1, gamma=100, kernel=rbf ......................................
[CV] .......... C=1, gamma=100, kernel=rbf, score=0.000, total=   0.1s
[CV] C=1, gamma=100, kernel=rbf ......................................
[CV] .......... C=1, gamma=100, kernel=rbf, score=0.000, total=   0.1s
[CV] C=1, gamma=100, kernel=rbf ......................................
[CV] .......... C=1, gamma=100, kernel=rbf, score=0.516, total=   0.1s
[CV] C=1, gamma=100, kernel=rbf ......................................
[CV] .......... C=1, gamma=100, kernel=rbf, score=0.054, total=   0.1s
[CV] C=1, gamma=100, kernel=rbf ......................................
[CV] .......... C=1, gamma=100, kernel=rbf, score=0.038, total=   0.1s
[CV] C=10, gamma=0.001, kernel=rbf ...................................
[CV] ....... C=10, gamma=0.001, kernel=rbf, score=0.978, total=   0.0s
[CV] C=10, gamma=0.001, kernel=rbf ...................................
[CV] ....... C=10, gamma=0.001, kernel=rbf, score=1.000, total=   0.0s
[CV] C=10, gamma=0.001, kernel=rbf ...................................
[CV] ....... C=10, gamma=0.001, kernel=rbf, score=0.951, total=   0.0s
[CV] C=10, gamma=0.001, kernel=rbf ...................................
[CV] ....... C=10, gamma=0.001, kernel=rbf, score=0.891, total=   0.0s
[CV] C=10, gamma=0.001, kernel=rbf ...................................
[CV] ....... C=10, gamma=0.001, kernel=rbf, score=0.886, total=   0.0s
[CV] C=10, gamma=0.01, kernel=rbf ....................................
[CV] ........ C=10, gamma=0.01, kernel=rbf, score=0.951, total=   0.0s
[CV] C=10, gamma=0.01, kernel=rbf ....................................
[CV] ........ C=10, gamma=0.01, kernel=rbf, score=0.940, total=   0.0s
[CV] C=10, gamma=0.01, kernel=rbf ....................................
[CV] ........ C=10, gamma=0.01, kernel=rbf, score=0.951, total=   0.0s
[CV] C=10, gamma=0.01, kernel=rbf ....................................
[CV] ........ C=10, gamma=0.01, kernel=rbf, score=0.902, total=   0.0s
[CV] C=10, gamma=0.01, kernel=rbf ....................................
[CV] ........ C=10, gamma=0.01, kernel=rbf, score=0.908, total=   0.0s
[CV] C=10, gamma=0.1, kernel=rbf .....................................
[CV] ......... C=10, gamma=0.1, kernel=rbf, score=0.750, total=   0.0s
[CV] C=10, gamma=0.1, kernel=rbf .....................................
[CV] ......... C=10, gamma=0.1, kernel=rbf, score=0.783, total=   0.0s
[CV] C=10, gamma=0.1, kernel=rbf .....................................
[CV] ......... C=10, gamma=0.1, kernel=rbf, score=0.918, total=   0.0s
[CV] C=10, gamma=0.1, kernel=rbf .....................................
[CV] ......... C=10, gamma=0.1, kernel=rbf, score=0.940, total=   0.0s
[CV] C=10, gamma=0.1, kernel=rbf .....................................
[CV] ......... C=10, gamma=0.1, kernel=rbf, score=0.940, total=   0.0s
[CV] C=10, gamma=1, kernel=rbf .......................................
[CV] ........... C=10, gamma=1, kernel=rbf, score=0.060, total=   0.0s
[CV] C=10, gamma=1, kernel=rbf .......................................
[CV] ........... C=10, gamma=1, kernel=rbf, score=0.054, total=   0.0s
[CV] C=10, gamma=1, kernel=rbf .......................................
[CV] ........... C=10, gamma=1, kernel=rbf, score=0.603, total=   0.0s
[CV] C=10, gamma=1, kernel=rbf .......................................
[CV] ........... C=10, gamma=1, kernel=rbf, score=0.114, total=   0.0s
[CV] C=10, gamma=1, kernel=rbf .......................................
[CV] ........... C=10, gamma=1, kernel=rbf, score=0.125, total=   0.0s
[CV] C=10, gamma=10, kernel=rbf ......................................
[CV] .......... C=10, gamma=10, kernel=rbf, score=0.005, total=   0.1s
[CV] C=10, gamma=10, kernel=rbf ......................................
[CV] .......... C=10, gamma=10, kernel=rbf, score=0.000, total=   0.1s
[CV] C=10, gamma=10, kernel=rbf ......................................
[CV] .......... C=10, gamma=10, kernel=rbf, score=0.527, total=   0.1s
[CV] C=10, gamma=10, kernel=rbf ......................................
[CV] .......... C=10, gamma=10, kernel=rbf, score=0.060, total=   0.1s
[CV] C=10, gamma=10, kernel=rbf ......................................
[CV] .......... C=10, gamma=10, kernel=rbf, score=0.038, total=   0.1s
[CV] C=10, gamma=100, kernel=rbf .....................................
[CV] ......... C=10, gamma=100, kernel=rbf, score=0.000, total=   0.1s
[CV] C=10, gamma=100, kernel=rbf .....................................
[CV] ......... C=10, gamma=100, kernel=rbf, score=0.000, total=   0.1s
[CV] C=10, gamma=100, kernel=rbf .....................................
[CV] ......... C=10, gamma=100, kernel=rbf, score=0.516, total=   0.1s
[CV] C=10, gamma=100, kernel=rbf .....................................
[CV] ......... C=10, gamma=100, kernel=rbf, score=0.054, total=   0.1s
[CV] C=10, gamma=100, kernel=rbf .....................................
[CV] ......... C=10, gamma=100, kernel=rbf, score=0.038, total=   0.1s
[CV] C=100, gamma=0.001, kernel=rbf ..................................
[CV] ...... C=100, gamma=0.001, kernel=rbf, score=0.962, total=   0.0s
[CV] C=100, gamma=0.001, kernel=rbf ..................................
[CV] ...... C=100, gamma=0.001, kernel=rbf, score=0.973, total=   0.0s
[CV] C=100, gamma=0.001, kernel=rbf ..................................
[CV] ...... C=100, gamma=0.001, kernel=rbf, score=0.957, total=   0.0s
[CV] C=100, gamma=0.001, kernel=rbf ..................................
[CV] ...... C=100, gamma=0.001, kernel=rbf, score=0.897, total=   0.0s
[CV] C=100, gamma=0.001, kernel=rbf ..................................
[CV] ...... C=100, gamma=0.001, kernel=rbf, score=0.913, total=   0.0s
[CV] C=100, gamma=0.01, kernel=rbf ...................................
[CV] ....... C=100, gamma=0.01, kernel=rbf, score=0.908, total=   0.0s
[CV] C=100, gamma=0.01, kernel=rbf ...................................
[CV] ....... C=100, gamma=0.01, kernel=rbf, score=0.886, total=   0.0s
[CV] C=100, gamma=0.01, kernel=rbf ...................................
[CV] ....... C=100, gamma=0.01, kernel=rbf, score=0.946, total=   0.0s
[CV] C=100, gamma=0.01, kernel=rbf ...................................
[CV] ....... C=100, gamma=0.01, kernel=rbf, score=0.935, total=   0.0s
[CV] C=100, gamma=0.01, kernel=rbf ...................................
[CV] ....... C=100, gamma=0.01, kernel=rbf, score=0.913, total=   0.0s
[CV] C=100, gamma=0.1, kernel=rbf ....................................
[CV] ........ C=100, gamma=0.1, kernel=rbf, score=0.750, total=   0.0s
[CV] C=100, gamma=0.1, kernel=rbf ....................................
[CV] ........ C=100, gamma=0.1, kernel=rbf, score=0.783, total=   0.0s
[CV] C=100, gamma=0.1, kernel=rbf ....................................
[CV] ........ C=100, gamma=0.1, kernel=rbf, score=0.924, total=   0.0s
[CV] C=100, gamma=0.1, kernel=rbf ....................................
[CV] ........ C=100, gamma=0.1, kernel=rbf, score=0.940, total=   0.0s
[CV] C=100, gamma=0.1, kernel=rbf ....................................
[CV] ........ C=100, gamma=0.1, kernel=rbf, score=0.940, total=   0.0s
[CV] C=100, gamma=1, kernel=rbf ......................................
[CV] .......... C=100, gamma=1, kernel=rbf, score=0.060, total=   0.0s
[CV] C=100, gamma=1, kernel=rbf ......................................
[CV] .......... C=100, gamma=1, kernel=rbf, score=0.054, total=   0.0s
[CV] C=100, gamma=1, kernel=rbf ......................................
[CV] .......... C=100, gamma=1, kernel=rbf, score=0.603, total=   0.0s
[CV] C=100, gamma=1, kernel=rbf ......................................
[CV] .......... C=100, gamma=1, kernel=rbf, score=0.114, total=   0.0s
[CV] C=100, gamma=1, kernel=rbf ......................................
[CV] .......... C=100, gamma=1, kernel=rbf, score=0.125, total=   0.0s
[CV] C=100, gamma=10, kernel=rbf .....................................
[CV] ......... C=100, gamma=10, kernel=rbf, score=0.005, total=   0.1s
[CV] C=100, gamma=10, kernel=rbf .....................................
[CV] ......... C=100, gamma=10, kernel=rbf, score=0.000, total=   0.1s
[CV] C=100, gamma=10, kernel=rbf .....................................
[CV] ......... C=100, gamma=10, kernel=rbf, score=0.527, total=   0.1s
[CV] C=100, gamma=10, kernel=rbf .....................................
[CV] ......... C=100, gamma=10, kernel=rbf, score=0.060, total=   0.1s
[CV] C=100, gamma=10, kernel=rbf .....................................
[CV] ......... C=100, gamma=10, kernel=rbf, score=0.038, total=   0.1s
[CV] C=100, gamma=100, kernel=rbf ....................................
[CV] ........ C=100, gamma=100, kernel=rbf, score=0.000, total=   0.1s
[CV] C=100, gamma=100, kernel=rbf ....................................
[CV] ........ C=100, gamma=100, kernel=rbf, score=0.000, total=   0.1s
[CV] C=100, gamma=100, kernel=rbf ....................................
[CV] ........ C=100, gamma=100, kernel=rbf, score=0.516, total=   0.1s
[CV] C=100, gamma=100, kernel=rbf ....................................
[CV] ........ C=100, gamma=100, kernel=rbf, score=0.054, total=   0.1s
[CV] C=100, gamma=100, kernel=rbf ....................................
[CV] ........ C=100, gamma=100, kernel=rbf, score=0.038, total=   0.1s
[CV] C=0.001, kernel=linear ..........................................
[CV] .............. C=0.001, kernel=linear, score=0.995, total=   0.0s
[CV] C=0.001, kernel=linear ..........................................
[CV] .............. C=0.001, kernel=linear, score=1.000, total=   0.0s
[CV] C=0.001, kernel=linear ..........................................
[CV] .............. C=0.001, kernel=linear, score=0.929, total=   0.0s
[CV] C=0.001, kernel=linear ..........................................
[CV] .............. C=0.001, kernel=linear, score=0.815, total=   0.0s
[CV] C=0.001, kernel=linear ..........................................
[CV] .............. C=0.001, kernel=linear, score=0.826, total=   0.0s
[CV] C=0.01, kernel=linear ...........................................
[CV] ............... C=0.01, kernel=linear, score=0.978, total=   0.0s
[CV] C=0.01, kernel=linear ...........................................
[CV] ............... C=0.01, kernel=linear, score=1.000, total=   0.0s
[CV] C=0.01, kernel=linear ...........................................
[CV] ............... C=0.01, kernel=linear, score=0.946, total=   0.0s
[CV] C=0.01, kernel=linear ...........................................
[CV] ............... C=0.01, kernel=linear, score=0.875, total=   0.0s
[CV] C=0.01, kernel=linear ...........................................
[CV] ............... C=0.01, kernel=linear, score=0.875, total=   0.0s
[CV] C=0.1, kernel=linear ............................................
[CV] ................ C=0.1, kernel=linear, score=0.978, total=   0.0s
[CV] C=0.1, kernel=linear ............................................
[CV] ................ C=0.1, kernel=linear, score=0.962, total=   0.0s
[CV] C=0.1, kernel=linear ............................................
[CV] ................ C=0.1, kernel=linear, score=0.957, total=   0.0s
[CV] C=0.1, kernel=linear ............................................
[CV] ................ C=0.1, kernel=linear, score=0.897, total=   0.0s
[CV] C=0.1, kernel=linear ............................................
[CV] ................ C=0.1, kernel=linear, score=0.897, total=   0.0s
[CV] C=1, kernel=linear ..............................................
[CV] .................. C=1, kernel=linear, score=0.957, total=   0.0s
[CV] C=1, kernel=linear ..............................................
[CV] .................. C=1, kernel=linear, score=0.951, total=   0.0s
[CV] C=1, kernel=linear ..............................................
[CV] .................. C=1, kernel=linear, score=0.957, total=   0.0s
[CV] C=1, kernel=linear ..............................................
[CV] .................. C=1, kernel=linear, score=0.897, total=   0.0s
[CV] C=1, kernel=linear ..............................................
[CV] .................. C=1, kernel=linear, score=0.886, total=   0.0s
[CV] C=10, kernel=linear .............................................
[CV] ................. C=10, kernel=linear, score=0.962, total=   0.1s
[CV] C=10, kernel=linear .............................................
[CV] ................. C=10, kernel=linear, score=0.929, total=   0.1s
[CV] C=10, kernel=linear .............................................
[CV] ................. C=10, kernel=linear, score=0.957, total=   0.1s
[CV] C=10, kernel=linear .............................................
[CV] ................. C=10, kernel=linear, score=0.902, total=   0.1s
[CV] C=10, kernel=linear .............................................
[CV] ................. C=10, kernel=linear, score=0.886, total=   0.1s
[CV] C=100, kernel=linear ............................................
[CV] ................ C=100, kernel=linear, score=0.967, total=   0.5s
[CV] C=100, kernel=linear ............................................
[CV] ................ C=100, kernel=linear, score=0.935, total=   0.7s
[CV] C=100, kernel=linear ............................................
[CV] ................ C=100, kernel=linear, score=0.951, total=   0.6s
[CV] C=100, kernel=linear ............................................
[CV] ................ C=100, kernel=linear, score=0.902, total=   1.0s
[CV] C=100, kernel=linear ............................................
[CV] ................ C=100, kernel=linear, score=0.902, total=   0.6s
```

```text
[Parallel(n_jobs=1)]: Done 210 out of 210 | elapsed:   10.5s finished
```

Out\[26\]:

```text
GridSearchCV(cv=KFold(n_splits=5, random_state=None, shuffle=False),
             error_score=nan,
             estimator=SVC(C=1.0, break_ties=False, cache_size=200,
                           class_weight=None, coef0=0.0,
                           decision_function_shape='ovr', degree=3,
                           gamma='scale', kernel='rbf', max_iter=-1,
                           probability=False, random_state=None, shrinking=True,
                           tol=0.001, verbose=False),
             iid='deprecated', n_jobs=None,
             param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10, 100],
                          'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
                          'kernel': ['rbf']},
                         {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                          'kernel': ['linear']}],
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring='accuracy', verbose=3)
```

In \[27\]:

```python
print('optimal parameter ==> {}'.format(gs.best_params_))
print('optimal parameter의 점수 ==> {:.3f}'.format(gs.best_score_))
print('optimal parameter로 일반화 점수 ==> {:.3f}'.format(gs.score(X_test, y_test)))
```

```text
optimal parameter ==> {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}
optimal parameter의 점수 ==> 0.941
optimal parameter로 일반화 점수 ==> 0.973
```

In \[28\]:

```python
svc_optimal=SVC(kernel='rbf', C = 10, gamma = 0.001)
svc_optimal.fit(Xresampled, yresampled)
```

Out\[28\]:

```text
SVC(C=10, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
```

In \[29\]:

```python
y_pred_optimal = svc_optimal.predict(X_test2)
metrics.confusion_matrix(y_test2, y_pred_optimal)
```

Out\[29\]:

```text
array([[13834,   375],
       [    2,    30]], dtype=int64)
```

In \[30\]:

```python
print('Accuracy Score:')
print(metrics.accuracy_score(y_test2, y_pred_optimal))
print('Recall Score:')
print(metrics.recall_score(y_test2, y_pred_optimal))
print('Precision Score:')
print(metrics.precision_score(y_test2, y_pred_optimal))
```

```text
Accuracy Score:
0.9735271399480374
Recall Score:
0.9375
Precision Score:
0.07407407407407407
```

결론  
모델 중 Recall Score가 가장 높으면서 다른 Score가 가장 높은 모델은 svc\_optimal 였다.

Recall Score가 중요한 이유  
기계가 실제 사기인 사건을 사기가 아니라고 판단했을 경우가 더 치명적이기 때문이다.

