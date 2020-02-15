# Python을 이용한 SVM \(2\)

## **Import packages**

In \[1\]:

```python
# data
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# model
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *

import warnings
warnings.filterwarnings(action='ignore')
```

## **Load iris data**

In \[2\]:

```python
iris =  sns.load_dataset('iris') 
X= iris.iloc[:,:4] #학습할데이터
y = iris.iloc[:,-1] #타겟
y
```

Out\[2\]:

```text
0         setosa
1         setosa
2         setosa
3         setosa
4         setosa
         ...    
145    virginica
146    virginica
147    virginica
148    virginica
149    virginica
Name: species, Length: 150, dtype: object
```

## **Split Data & Scaling**

In \[3\]:

```python
from sklearn.model_selection import train_test_split #테스트/트레인 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=48)
```

In \[4\]:

```python
scaler = StandardScaler() #scaling
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## Implement One vs Rest Multiclass SVM

* One vs Rest SVM 을 Class로 구현하였습니다.

In \[5\]:

```python
class OneVsRestSVM:
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.clfs = []
        self.y_pred = []
    
    # y를 onehot encoding하는 과정
    def one_vs_rest_labels(self, y_train):
        y_train = pd.get_dummies(y_train)
        return y_train
    
    # encoding된 y를 가져와서 class 개수 만큼의 classifier에 각각 돌리는 과정
    def fit(self, X_train, y_train, C=5, gamma=5):
        # y encoding
        y_encoded = self.one_vs_rest_labels(y_train)
        
        for i in range(self.n_classes):
            clf = SVC(kernel='rbf', C=C, gamma=gamma)
            clf.fit(X_train, y_encoded.iloc[:,i])
            self.clfs.append(clf)

    # 각각의 classifier에서 나온 결과를 바탕으로 투표를 진행하는 과정
    def predict(self, X_test):
        vote = np.zeros((len(X_test), 3), dtype=int)
        size = X_test.shape[0]
        
        for i in range(size):
            # 해당 class에 속하는 샘플을 +1 만큼 투표를, 나머지 샘플에 -1 만큼 투표를 진행한다.
            if self.clfs[0].predict(X_test)[i] == 1:
                vote[i][0] += 1
                vote[i][1] -= 1
                vote[i][2] -= 1
            elif self.clfs[1].predict(X_test)[i] == 1:
                vote[i][0] -= 1
                vote[i][1] += 1
                vote[i][2] -= 1
            elif self.clfs[2].predict(X_test)[i] == 1:
                vote[i][0] -= 1
                vote[i][1] -= 1
                vote[i][2] += 1
    
            # 투표한 값 중 가장 큰 값의 인덱스를 test label에 넣는다
            self.y_pred.append(np.argmax(vote[i]))
            
            # 경우의 수
            # 1. 한 분류기의 투표 결과가 양수이고 나머지는 음수인 경우
            # 2. 세 분류기의 투표 결과가 모두 0으로 같은 경우
            # 3. 두 분류기의 투표 결과가 양수로 같은 경우
            
            # 2번째, 동점일 경우 decision_function의 값이 가장 큰 경우를 test label에 넣는다
            if (np.sign(self.clfs[0].decision_function(X_test)[i]) == np.sign(self.clfs[1].decision_function(X_test)[i])) \
            and (np.sign(self.clfs[1].decision_function(X_test)[i]) == np.sign(self.clfs[2].decision_function(X_test)[i])):
                self.y_pred[i] = np.argmax([self.clfs[0].decision_function(X_test)[i], self.clfs[1].decision_function(X_test)[i], self.clfs[2].decision_function(X_test)[i]])
        
            # 3번째, 두 분류기의 투표 결과가 양수로 같을 경우 decision_function의 값이 가장 큰 경우를 test label에 넣는다
            elif (vote[i][0] == vote[i][1]) and vote[i][0] > 0 and vote[i][1] > 0:
                self.y_pred[i] = np.argmax([self.clfs[0].decision_function(X_test)[i], self.clfs[1].decision_function(X_test)[i]])
            elif (vote[i][0] == vote[i][2]) and vote[i][0] > 0 and vote[i][2] > 0:
                self.y_pred[i] = np.argmax([self.clfs[0].decision_function(X_test)[i], self.clfs[2].decision_function(X_test)[i]])
            elif (vote[i][1] == vote[i][2]) and vote[i][1] > 0 and vote[i][2] > 0:
                self.y_pred[i] = np.argmax([self.clfs[1].decision_function(X_test)[i], self.clfs[2].decision_function(X_test)[i]])

        # test를 진행하기 위해 0,1,2로 되어있던 데이터를 다시 문자 label로 변환
        self.y_pred = pd.DataFrame(self.y_pred).replace({0:'setosa', 1:'versicolor', 2:'virginica'})
        return self.y_pred
    
    # accuracy 확인
    def evaluate(self, y_test):
        print('Accuacy : {: .5f}'.format(accuracy_score(y_test, self.y_pred)))
```

In \[6\]:

```python
onevsrest = OneVsRestSVM()
onevsrest.fit(X_train, y_train)
```

In \[7\]:

```python
y_pred_rest = onevsrest.predict(X_test)
y_pred_rest
```

Out\[7\]:

|  | 0 |
| :--- | :--- |
| 0 | versicolor |
| 1 | versicolor |
| 2 | versicolor |
| 3 | virginica |
| 4 | virginica |
| 5 | virginica |
| 6 | setosa |
| 7 | virginica |
| 8 | setosa |
| 9 | versicolor |
| 10 | virginica |
| 11 | setosa |
| 12 | setosa |
| 13 | virginica |
| 14 | versicolor |
| 15 | versicolor |
| 16 | setosa |
| 17 | versicolor |
| 18 | virginica |
| 19 | virginica |
| 20 | setosa |
| 21 | virginica |
| 22 | versicolor |
| 23 | versicolor |
| 24 | virginica |
| 25 | setosa |
| 26 | setosa |
| 27 | virginica |
| 28 | virginica |
| 29 | versicolor |

In \[8\]:

```python
onevsrest.evaluate(y_test)
```

```text
Accuacy :  0.86667
```

* 사실상 one vs rest svm의 경우 투표의 개념을 도입할 필요 없이 decision function 값만 비교해주어도 된다!
* 이를 이용해서 다시 간단하게 구현해보자

In \[9\]:

```python
class OneVsRestSVM:
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.clfs = []
        self.y_pred = []
    
    # y를 onehot encoding하는 과정
    def one_vs_rest_labels(self, y_train):
        y_train = pd.get_dummies(y_train)
        return y_train
    
    # encoding된 y를 가져와서 class 개수 만큼의 classifier에 각각 돌리는 과정
    def fit(self, X_train, y_train, C=5, gamma=5):
        # y encoding
        y_encoded = self.one_vs_rest_labels(y_train)
        
        for i in range(self.n_classes):
            clf = SVC(kernel='rbf', C=C, gamma=gamma)
            clf.fit(X_train, y_encoded.iloc[:,i])
            self.clfs.append(clf)

    # 각각의 classifier에서 나온 결과를 decision function으로 비교
    def predict(self, X_test):
        vote = np.zeros((len(X_test), 3), dtype=int)
        size = X_test.shape[0]
        
        for i in range(size):
            self.y_pred.append(np.argmax([self.clfs[0].decision_function(X_test)[i], self.clfs[1].decision_function(X_test)[i], self.clfs[2].decision_function(X_test)[i]]))
        
        # test를 진행하기 위해 0,1,2로 되어있던 데이터를 다시 문자 label로 변환
        self.y_pred = pd.DataFrame(self.y_pred).replace({0:'setosa', 1:'versicolor', 2:'virginica'})
        return self.y_pred
    
    # accuracy 확인
    def evaluate(self, y_test):
        print('Accuacy : {: .5f}'.format(accuracy_score(y_test, self.y_pred)))
```

In \[10\]:

```python
onevsrest = OneVsRestSVM()
onevsrest.fit(X_train, y_train)
y_pred_rest = onevsrest.predict(X_test)
onevsrest.evaluate(y_test)
```

```text
Accuacy :  0.86667
```

#### Implement One vs One Multiclass SVM <a id="Implement-One-vs-One-Multiclass-SVM"></a>

* One vs One SVM 을 Class로 구현하였습니다.

In \[11\]:

```python
class OneVsOneSVM:
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.clfs = []
        self.y_pred = []
    
    # c1 class와 c2 class의 label 만드는 과정
    def one_vs_one_labels(self, c1, c2, y_train):
        size = y_train.shape[0]
        y = np.zeros(size)
        # one vs one을 학습시키기 위해 c1 class인지 c2 class인지를 구분해야 하므로
        # class가 c1인 경우 1, c2인 경우 -1을 넣은 새로운 label을 생성한다.
        for i in range(size):
            if y_train[i] == c1:
                y[i] = 1
            else:
                y[i] = -1
        return y
    
    # one vs one label을 적용해 두 class의 데이터만 가져오는 과정
    def one_vs_one_data(self, c1, c2, X_train, y_train):
        y_train = pd.DataFrame(y_train).replace({'setosa':0, 'versicolor':1, 'virginica':2}).values.flatten()
        
        # 해당 class의 index를 가져온다.
        index_c1 = (y_train == c1)
        index_c2 = (y_train == c2)
        
        # c1 class인지 c2 class인지를 비교해야 하므로
        # 해당 두 class에 속하는 데이터만 가져온다.
        y_train_c = np.concatenate((y_train[index_c1], y_train[index_c2]))
        y_train_c = self.one_vs_one_labels(c1, c2, y_train_c)
        X_train_c = np.vstack((X_train[index_c1], X_train[index_c2]))
        
        return y_train_c, X_train_c
    
    # class들의 조합 개수만큼의 classifier를 만들고 fitting 시키는 과정
    def fit(self, X_train, y_train, C=5, gamma=5):
        # class가 m개 라면 m * (m-1) / 2 개의 classifer가 필요하다.
        for c1 in range(self.n_classes):
            for c2 in range(c1+1, self.n_classes):
                data_c = self.one_vs_one_data(c1, c2, X_train, y_train)
                y_c = data_c[0].reshape(-1,1)
                X_c = data_c[1]
                
                clf = SVC(kernel='rbf', C=C, gamma=gamma)
                clf.fit(X_c, y_c)
                self.clfs.append([clf, c1, c2])
    
    # 각각의 classifier에서 나온 결과를 바탕으로 투표를 진행하는 과정
    def predict(self, X_test):
        vote = np.zeros((len(X_test), 3), dtype=int)
        size = X_test.shape[0]
        
        for i in range(size):
            x = X_test[i, :].reshape(-1, 4)
            for j in range(len(self.clfs)):
                clf, c1, c2 = self.clfs[j]
                pred = clf.predict(x)
                
                # x를 class c1으로 분류하면 class c1에 +1점
                # c2로 분류하면 class c2에 +1점을 준다.
                if pred == 1:
                    vote[i][c1] += 1
                else:
                    vote[i][c2] += 1
                    
            # 투표한 값 중 가장 큰 값의 인덱스를 test label에 넣는다
            self.y_pred.append(np.argmax(vote[i]))
            
            # 경우의 수
            # 1. 한 분류기의 투표 결과가 제일 높은 경우
            # 2. 세 분류기의 투표 결과가 모두 같은 경우
            # 3. 두 분류기의 투표 결과가 같고 나머지 한 분류기는 다른 경우
            
            # 2번째, 모두 동점일 경우 decision_function의 값이 가장 큰 경우를 test label에 넣는다
            if (vote[i][0] == vote[i][1]) and (vote[i][1] == vote[i][2]):
                self.y_pred[i] = np.argmax([self.clfs[0].decision_function(X_test)[i], self.clfs[1].decision_function(X_test)[i], self.clfs[2].decision_function(X_test)[i]])
            
            # 3번째, 두 분류기의 투표 결과가 양수로 같은 경우 decision_function이 값이 큰 경우를 test label에 넣는다
            elif (vote[i][0] == vote[i][1]) and vote[i][0] > 0 and vote[i][1] > 0:
                self.y_pred[i] = np.argmax([self.clfs[0].decision_function(X_test)[i], self.clfs[1].decision_function(X_test)[i]])
            elif (vote[i][0] == vote[i][2]) and vote[i][0] > 0 and vote[i][2] > 0:
                self.y_pred[i] = np.argmax([self.clfs[0].decision_function(X_test)[i], self.clfs[2].decision_function(X_test)[i]])
            elif (vote[i][1] == vote[i][2]) and vote[i][1] > 0 and vote[i][2] > 0:
                self.y_pred[i] = np.argmax([self.clfs[1].decision_function(X_test)[i], self.clfs[2].decision_function(X_test)[i]])

        # test를 진행하기 위해 0,1,2로 되어있던 데이터를 다시 문자 label로 변환
        self.y_pred = pd.DataFrame(self.y_pred).replace({0:'setosa', 1:'versicolor', 2:'virginica'})
        return self.y_pred
    
    # accuracy 확인
    def evaluate(self, y_test):
        print('Accuacy : {: .5f}'.format(accuracy_score(y_test, self.y_pred)))
```

In \[12\]:

```python
onevsone = OneVsOneSVM()
onevsone.fit(X_train, y_train)
```

In \[13\]:

```python
y_pred_one = onevsone.predict(X_test)
y_pred_one
```

Out\[13\]:

|  | 0 |
| :--- | :--- |
| 0 | versicolor |
| 1 | versicolor |
| 2 | versicolor |
| 3 | virginica |
| 4 | virginica |
| 5 | virginica |
| 6 | setosa |
| 7 | virginica |
| 8 | setosa |
| 9 | versicolor |
| 10 | virginica |
| 11 | setosa |
| 12 | setosa |
| 13 | virginica |
| 14 | versicolor |
| 15 | versicolor |
| 16 | setosa |
| 17 | versicolor |
| 18 | virginica |
| 19 | virginica |
| 20 | setosa |
| 21 | virginica |
| 22 | versicolor |
| 23 | versicolor |
| 24 | virginica |
| 25 | setosa |
| 26 | setosa |
| 27 | virginica |
| 28 | virginica |
| 29 | versicolor |

In \[14\]:

```python
onevsone.evaluate(y_test)
```

```text
Accuacy :  0.86667
```

## **Compare Result**

In \[16\]:

```python
# 그냥 원래 라이브러리가 제공하는 멀티클래스 SVM과 여러분이 구현한 multiclass SVM결과를 한 번 비교해주세요
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y, test_size=0.2, random_state=48)

scaler = StandardScaler() #scaling
X_train_2 = scaler.fit_transform(X_train_2)
X_test_2 = scaler.transform(X_test_2)

svm_4 = SVC(kernel ='rbf', C = 5, gamma = 5)
svm_4.fit(X_train_2, y_train_2)
y_pred = svm_4.predict(X_test_2)

accuracy_score(y_test_2,y_pred)
```

Out\[16\]:

```text
0.8666666666666667
```

* One vs Rest Accuracy: 0.86667
* One vs One Accuracy: 0.86667
* Library Accuracy: 0.86667
  * 모두 같은 성능을 보이고 있다!

