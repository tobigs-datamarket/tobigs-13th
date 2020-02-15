# Python을 이용한 Decision Tree \(4\)

```python
import pandas as pd 
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/AugustLONG/ML01/master/01decisiontree/AllElectronics.csv')
df.drop("RID",axis=1, inplace = True) #RID는 그냥 Index라서 삭제
```

In \[2\]:

```text
df
```

Out\[2\]:

|  | age | income | student | credit\_rating | class\_buys\_computer |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | youth | high | no | fair | no |
| 1 | youth | high | no | excellent | no |
| 2 | middle\_aged | high | no | fair | yes |
| 3 | senior | medium | no | fair | yes |
| 4 | senior | low | yes | fair | yes |
| 5 | senior | low | yes | excellent | no |
| 6 | middle\_aged | low | yes | excellent | yes |
| 7 | youth | medium | no | fair | no |
| 8 | youth | low | yes | fair | yes |
| 9 | senior | medium | yes | fair | yes |
| 10 | youth | medium | yes | excellent | yes |
| 11 | middle\_aged | medium | no | excellent | yes |
| 12 | middle\_aged | high | yes | fair | yes |
| 13 | senior | medium | no | excellent | no |

### 함수 만들기 <a id="&#xD568;&#xC218;-&#xB9CC;&#xB4E4;&#xAE30;"></a>

In \[3\]:

```python
import math
def getEntropy(df, feature) :
    ''' 
    채우세요!
    데이터프레임 df에서 특정 feature에 대한 엔트로피를 구하는 함수입니다.
    '''
    unique = list(df[feature].unique()) # df[feature]에 존재하는 값들을 받아내기 위해
    #unique를 사용해서 그걸 리스트로 받았고
    entropy = 0
    for i in range(len(unique)):
        p = len(df[df[feature].str.contains(unique[i])])/len(df[feature])
        #각각yes일때와 no일때를 unique로 받은 리스트를 통해 p를 구했습니다
        entropy = entropy - p*math.log(p,2)
        #엔트로피 식을 이용해 for문 돌리면서 계속 -로 빼가는 방식으로 함수를 만들었습니다
    return(entropy)
```

In \[4\]:

```python
getEntropy(df, "class_buys_computer")
```

Out\[4\]:

```text
0.9402859586706309
```

In \[5\]:

```python
def getGainA(df, feature) :
    info_D = getEntropy(df, feature) # 목표변수 Feature에 대한 Info(Entropy)를 구한다.
    columns = list(df.loc[:, df.columns != feature])
    # 목표변수를 제외한 나머지 설명변수들을 리스트 형태로 저장한다.
    ''' 
    채우세요!
    데이터프레임 df에서 feature을 목표변수로 삼고 나머지 각 변수들의 Gain을 구하는 함수입니다.
    결과는 Key = 변수명, Value = Information Gain 으로 이루어진 Dictionary 여야 합니다.
    미리 주어진 3줄은 문제풀이에 도움을 주기 위해 주어졌으나, 무시하고 각자 원하는 방법으로 풀으셔도 전혀 무방합니다.
    '''
    gains =[]
    
    for i in columns:
        info_Di = 0
        unique = list(df[i].unique()) #각 column에 존재하는 값들을 받기위해 unique함수를 사용했고
        #unique라는 변수로 list를 저장했습니다
        for j in range(len(unique)):
            new = df.loc[df[i] == unique[j]]
            #for문을 돌리면서 그 값을 가지는 row들만 뽑아낸 new변수를 만들었습니다
            info_Di += (len(new[i])/len(df[i])) *(getEntropy(new, feature))
            #위에서 만든 entropy함수를 사용해 각각의 entropy를 구하고 확률에 곱한걸 더해서 info_di값을 구합니다
        
        gain = info_D - info_Di
        #infoD에서 infoDI를 빼서 gain을 얻습니다
        gains.append(gain)
        
        
        
    result = dict(zip(columns,gains)) # 각 변수에 대한 Information Gain 을 Dictionary 형태로 저장한다.
    return(result)
```

![](1.png)In \[6\]:

```python
getGainA(df, "class_buys_computer")
```

Out\[6\]:

```text
{'age': 0.2467498197744391,
 'income': 0.029222565658954647,
 'student': 0.15183550136234136,
 'credit_rating': 0.04812703040826927}
```

### 결과 확인하기 <a id="&#xACB0;&#xACFC;-&#xD655;&#xC778;&#xD558;&#xAE30;"></a>

In \[7\]:

```python
my_dict = getGainA(df, "class_buys_computer")
def f1(x):
    return my_dict[x]
key_max = max(my_dict.keys(), key=f1)
print('정보 획득이 가장 높은 변수는',key_max, "이며 정보 획득량은", my_dict[key_max], "이다.")
```

```text
정보 획득이 가장 높은 변수는 age 이며 정보 획득량은 0.2467498197744391 이다.
```

