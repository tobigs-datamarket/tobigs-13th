# Python을 이용한 Decision Tree \(2\)



```python
import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('https://raw.githubusercontent.com/AugustLONG/ML01/master/01decisiontree/AllElectronics.csv')
df.drop("RID",axis=1, inplace = True) #RID는 그냥 Index라서 삭제
```

In \[117\]:

```text
df
```

Out\[117\]:

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

In \[118\]:

```python
from functools import reduce

def getEntropy(df, feature) :
    D_len = df[feature].count() # 데이터 전체 길이
    # reduce함수를 이용하여 초기값 0에 
    # 각 feature별 count을 엔트로피 식에 대입한 값을 순차적으로 더함
    return reduce(lambda x, y: x+(-(y[1]/D_len) * np.log2(y[1]/D_len)), \
                  df[feature].value_counts().items(), 0)
```

In \[119\]:

```python
getEntropy(df, "class_buys_computer")
```

Out\[119\]:

```text
0.9402859586706311
```

In \[5\]:

```python
# 정답
getEntropy(df, "class_buys_computer")
```

Out\[5\]:

```text
0.9402859586706311
```

In \[159\]:

```python
def get_target_true_count(col, name, target, true_val, df=df):
    """
    df[col]==name인 조건에서 Target이 참인 경우의 갯수를 반환
    """
    return df.groupby([col,target]).size()[name][true_val]

def NoNan(x):
    """
    Nan의 경우 0을 반환
    """
    return np.nan_to_num(x)

def getGainA(df, feature) :
    info_D = getEntropy(df, feature) # 목표변수 Feature에 대한 Info(Entropy)를 구한다.
    columns = list(df.loc[:, df.columns != feature]) # 목표변수를 제외한 나머지 설명변수들을 리스트 형태로 저장한다.
    gains = []
    D_len = df.shape[0] # 전체 길이
    for col in columns:
        info_A = 0
        # Col내 개별 Class 이름(c_name)과 Class별 갯수(c_len)
        for c_name, c_len in df[col].value_counts().items():
            target_true = get_target_true_count(col, c_name, feature, 'yes') 
            prob_t = target_true / c_len
            # Info_A <- |Dj|/|D| *  Entropy(label) | NoNan을 이용해 prob_t가 0인 경우 nan이 나와 생기는 오류 방지
            info_A += (c_len/D_len) * -(NoNan(prob_t*np.log2(prob_t)) + NoNan((1 - prob_t)*np.log2(1 - prob_t)))
        gains.append(info_D - info_A)
    
    result = dict(zip(columns,gains)) # 각 변수에 대한 Information Gain 을 Dictionary 형태로 저장한다.
    return(result)
```

In \[161\]:

```python
df.groupby(['age','class_buys_computer']).size()
```

Out\[161\]:

```text
age          class_buys_computer
middle_aged  yes                    4
senior       no                     2
             yes                    3
youth        no                     3
             yes                    2
dtype: int64
```

In \[160\]:

```python
getGainA(df, "class_buys_computer")
```

Out\[160\]:

```text
{'age': 0.24674981977443933,
 'income': 0.02922256565895487,
 'student': 0.15183550136234159,
 'credit_rating': 0.04812703040826949}
```

정답

```text
{'age': 0.24674981977443933, 
 'income': 0.02922256565895487, 
 'student': 0.15183550136234159, 
 'credit_rating': 0.04812703040826949}
```

### 결과 확인하기 <a id="&#xACB0;&#xACFC;-&#xD655;&#xC778;&#xD558;&#xAE30;"></a>

In \[139\]:

```python
my_dict = getGainA(df, "class_buys_computer")
def f1(x):
    return my_dict[x]
key_max = max(my_dict.keys(), key=f1)
print('정보 획득이 가장 높은 변수는',key_max, "이며 정보 획득량은", my_dict[key_max], "이다.")
```

```text
정보 획득이 가장 높은 변수는 age 이며 정보 획득량은 0.24674981977443933 이다.
```

In \[7\]:

```python
# 정답
my_dict = getGainA(df, "class_buys_computer")
def f1(x):
    return my_dict[x]
key_max = max(my_dict.keys(), key=f1)
print('정보 획득이 가장 높은 변수는',key_max, "이며 정보 획득량은", my_dict[key_max], "이다.")
```

```text
정보 획득이 가장 높은 변수는 age 이며 정보 획득량은 0.24674981977443933 이다.
```

