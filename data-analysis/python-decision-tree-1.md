# Python을 이용한 Decision Tree \(1\)

## DT Assignment1 <a id="DT-Assignment1"></a>

**주의사항** 

* 본인이 구현한 함수임을 증명하기 위해 주석 꼼꼼히 달아주세요. 
* 이 데이터셋 뿐만 아니라 변수의 class가 더 많은 데이터에도 상관없이 적용 가능하도록 함수를 구현해 주세요.

  변수의 class가 3개를 넘는 경우 모든 이진분류 경우의 수를 따져 보아야 합니다.

  Hint\) itertools 라이브러리의 combination 함수 & isin 함수 등이 활용될 수 있으며 이 밖에도 본인의 방법대로 마음껏 구현해주세요.

* 함수에 들어가는 변수나 flow 등은 본인이 변경해도 무관하며 결과만 똑같이 나오면 됩니다. 

**우수과제 선정이유** 

함수마다 잘 작동하는지 확인하는 과정을 거치는 것이 인상적이였습니다. 

또한 1번 과제의 3번 문제를 해결할 때 데이터 프레임을 나누는 함수를 만들고 문제에서 요구하는 것보다 한층 더 깊게 살펴보는 모습이 굉장히 좋았습니다.

## Data Loading <a id="Data-Loading"></a>

In \[1\]:

```python
import pandas as pd 
import numpy as np
```

In \[2\]:

```python
pd_data = pd.read_csv('https://raw.githubusercontent.com/AugustLONG/ML01/master/01decisiontree/AllElectronics.csv')
pd_data.drop("RID",axis=1, inplace = True) #RID는 그냥 순서라서 삭제
```

## 1. Gini 계수를 구하는 함수 만들기 <a id="1.-Gini-&#xACC4;&#xC218;&#xB97C;-&#xAD6C;&#xD558;&#xB294;-&#xD568;&#xC218;-&#xB9CC;&#xB4E4;&#xAE30;"></a>

* Input: df\(데이터\), label\(타겟변수명\)
* 해당 결과는 아래와 같이 나와야 합니다.

In \[3\]:

```python
from functools import reduce
```

In \[4\]:

```python
def get_gini(df, label):
    D_len = df[label].count() # 데이터 전체 길이
    # 각 클래스별 Count를 담은 Generator 생성
    count_arr = (value for key, value in df[label].value_counts().items())
    # reduce를 이용해 초기값 1에서 각 클래스 (count / D_len)^2 빼기
    return reduce(lambda x, y: x - (y/D_len)**2 ,count_arr,1)
```

In \[5\]:

```python
pd_data['class_buys_computer']
```

Out\[5\]:

```text
0      no
1      no
2     yes
3     yes
4     yes
5      no
6     yes
7      no
8     yes
9     yes
10    yes
11    yes
12    yes
13     no
Name: class_buys_computer, dtype: object
```

In \[6\]:

```python
get_gini(pd_data,'class_buys_computer')
```

Out\[6\]:

```text
0.4591836734693877
```

In \[35\]:

```python
# 정답
get_gini(pd_data,'class_buys_computer')
```

Out\[35\]:

```text
0.4591836734693877
```

## 2. Feature의 Class를 이진 분류로 만들기 <a id="2.-Feature&#xC758;-Class&#xB97C;-&#xC774;&#xC9C4;-&#xBD84;&#xB958;&#xB85C;-&#xB9CC;&#xB4E4;&#xAE30;"></a>

### ex\) {A,B,C} -&gt; \({A}, {B,C}\), \({B}, {A,C}\), \({C}, {A,B}\) <a id="ex)-{A,B,C}--&gt;-({A},-{B,C}),-({B},-{A,C}),-({C},-{A,B})"></a>

* CART 알고리즘은 이진분류만 가능합니다. 수업때 설명했던데로 변수가 3개라면 3가지 경우의 수에 대해 지니계수를 구해야합니다.
* 만약 변수가 4개라면, 총 7가지 경우의 수가 가능합니다. \(A, BCD\) \(B, ACD\) \(C, ABD\) \(D, ABC\) \(AB, CD\) \(AC, BD\) \(AD, BC\)
* Input: df\(데이터\), attribute\(Gini index를 구하고자 하는 변수명\)
* 해당 결과는 아래와 같이 나와야 합니다.

In \[7\]:

```python
import itertools 
```

In \[48\]:

```python
import itertools # 변수의 모든 클래시 조합을 얻기 위해 itertools 불러오기

def get_binary_split(df, attribute):
    attr_unique = df[attribute].unique()
    # 이중 For loop List Comprehension
    result = [
            list(item) 
            for i in range(1, len(attr_unique)) # 1부터 변수의 클래스 갯수-1 까지 Iteration
            for item in itertools.combinations(attr_unique, i) # i를 길이로 하는 조합 생성
        ]
    return result
```

In \[61\]:

```python
# 검증을 위한 테스트데이터 제작
df = pd.DataFrame([1,2,3,4,2,1,3], columns=['d'])
print(df['d'].unique())
a = get_binary_split(df,'d')
```

```text
[1 2 3 4]
```

In \[59\]:

```python
# get_binary_split 검증, 짝을 찾아 전체 클래스가 나오는지 확인
for i in range(len(a) // 2):
    b = a[i] + a[len(a)-i-1]
    b.sort()
    print(a[i], a[len(a)-i-1], '=>', b)
```

```text
[1] [2, 3, 4] => [1, 2, 3, 4]
[2] [1, 3, 4] => [1, 2, 3, 4]
[3] [1, 2, 4] => [1, 2, 3, 4]
[4] [1, 2, 3] => [1, 2, 3, 4]
[1, 2] [3, 4] => [1, 2, 3, 4]
[1, 3] [2, 4] => [1, 2, 3, 4]
[1, 4] [2, 3] => [1, 2, 3, 4]
```

In \[60\]:

```python
# 정답
get_binary_split(pd_data, "age")
```

Out\[60\]:

```text
[['youth'],
 ['middle_aged'],
 ['senior'],
 ['youth', 'middle_aged'],
 ['youth', 'senior'],
 ['middle_aged', 'senior']]
```

In \[37\]:

```python
# 정답
get_binary_split(pd_data, "age")
```

Out\[37\]:

```text
[['youth'],
 ['middle_aged'],
 ['senior'],
 ['youth', 'middle_aged'],
 ['youth', 'senior'],
 ['middle_aged', 'senior']]
```

## 3. 다음은 모든 이진분류의 경우의 Gini index를 구하는 함수 만들기 <a id="3.-&#xB2E4;&#xC74C;&#xC740;-&#xBAA8;&#xB4E0;-&#xC774;&#xC9C4;&#xBD84;&#xB958;&#xC758;-&#xACBD;&#xC6B0;&#xC758;-Gini-index&#xB97C;-&#xAD6C;&#xD558;&#xB294;-&#xD568;&#xC218;-&#xB9CC;&#xB4E4;&#xAE30;"></a>

* 위에서 완성한 두 함수를 사용하여 만들어주세요!
* 해당 결과는 아래와 같이 나와야 합니다.
* 결과로 나온 Dictionary의 Key 값은 해당 class 들로 이루어진 tuple 형태로 들어가 있습니다.

In \[88\]:

```python
def get_attribute_gini_index(df, attribute, label):
    result = {}
    keys = get_binary_split(df, attribute)
    D_len = df[attribute].shape[0]
    for key in keys:
        t_index = df[attribute].map(lambda x: x in key) # Split한 클래스들에 속하는 df Index 추출
        Dj_len = sum(t_index) # Sum으로 True갯수 계산
        # Gini 식 계산,  ~index를 통해 False_index로 전환
        gini = (Dj_len / D_len) * get_gini(df[t_index], label) + ((D_len - Dj_len) / D_len) * get_gini(df[~t_index], label)
        result[tuple(key)] = gini
    return result
```

In \[89\]:

```python
get_attribute_gini_index(pd_data, "age", "class_buys_computer")
```

Out\[89\]:

```text
{('youth',): 0.3936507936507937,
 ('middle_aged',): 0.35714285714285715,
 ('senior',): 0.4571428571428572,
 ('youth', 'middle_aged'): 0.4571428571428572,
 ('youth', 'senior'): 0.35714285714285715,
 ('middle_aged', 'senior'): 0.3936507936507937}
```

In \[39\]:

```python
# 정답
get_attribute_gini_index(pd_data, "age", "class_buys_computer")
```

Out\[39\]:

```text
{('youth',): 0.3936507936507936,
 ('middle_aged',): 0.35714285714285715,
 ('senior',): 0.45714285714285713,
 ('youth', 'middle_aged'): 0.45714285714285713,
 ('youth', 'senior'): 0.35714285714285715,
 ('middle_aged', 'senior'): 0.3936507936507936}
```

여기서 가장 작은 Gini index값을 가지는 class를 기준으로 split해야겠죠?

결과를 확인해보도록 하겠습니다.

In \[90\]:

```python
my_dict = get_attribute_gini_index(pd_data, "age", "class_buys_computer")
key_min = min(my_dict.keys(), key=(lambda k: my_dict[k]))
print('Min -',key_min, ":", my_dict[key_min])
```

```text
Min - ('middle_aged',) : 0.35714285714285715
```

In \[40\]:

```python
# 정답
my_dict = get_attribute_gini_index(pd_data, "age", "class_buys_computer")
key_min = min(my_dict.keys(), key=(lambda k: my_dict[k]))
print('Min -',key_min, ":", my_dict[key_min])
```

```text
Min - ('middle_aged',) : 0.35714285714285715
```

## 다음의 문제를 위에서 작성한 함수를 통해 구한 값으로 보여주세요! <a id="&#xB2E4;&#xC74C;&#xC758;-&#xBB38;&#xC81C;&#xB97C;-&#xC704;&#xC5D0;&#xC11C;-&#xC791;&#xC131;&#xD55C;-&#xD568;&#xC218;&#xB97C;-&#xD1B5;&#xD574;-&#xAD6C;&#xD55C;-&#xAC12;&#xC73C;&#xB85C;-&#xBCF4;&#xC5EC;&#xC8FC;&#xC138;&#xC694;!"></a>

### 문제1\) 변수 ‘income’의 이진분류 결과를 보여주세요. <a id="&#xBB38;&#xC81C;1)-&#xBCC0;&#xC218;-&#x2018;income&#x2019;&#xC758;-&#xC774;&#xC9C4;&#xBD84;&#xB958;-&#xACB0;&#xACFC;&#xB97C;-&#xBCF4;&#xC5EC;&#xC8FC;&#xC138;&#xC694;."></a>

### 문제2\) 분류를 하는 데 가장 중요한 변수를 선정하고, 해당 변수의 Gini index를 제시해주세요. <a id="&#xBB38;&#xC81C;2)-&#xBD84;&#xB958;&#xB97C;-&#xD558;&#xB294;-&#xB370;-&#xAC00;&#xC7A5;-&#xC911;&#xC694;&#xD55C;-&#xBCC0;&#xC218;&#xB97C;-&#xC120;&#xC815;&#xD558;&#xACE0;,-&#xD574;&#xB2F9;-&#xBCC0;&#xC218;&#xC758;-Gini-index&#xB97C;-&#xC81C;&#xC2DC;&#xD574;&#xC8FC;&#xC138;&#xC694;."></a>

### 문제3\) 문제 2에서 제시한 feature로 DataFrame을 split한 후 나눠진 2개의 DataFrame에서 각각 다음으로 중요한 변수를 선정하고 해당 변수의 Gini index를 제시해주세요. <a id="&#xBB38;&#xC81C;3)-&#xBB38;&#xC81C;-2&#xC5D0;&#xC11C;-&#xC81C;&#xC2DC;&#xD55C;-feature&#xB85C;-DataFrame&#xC744;-split&#xD55C;-&#xD6C4;-&#xB098;&#xB220;&#xC9C4;-2&#xAC1C;&#xC758;-DataFrame&#xC5D0;&#xC11C;-&#xAC01;&#xAC01;---&#xB2E4;&#xC74C;&#xC73C;&#xB85C;-&#xC911;&#xC694;&#xD55C;-&#xBCC0;&#xC218;&#xB97C;-&#xC120;&#xC815;&#xD558;&#xACE0;-&#xD574;&#xB2F9;-&#xBCC0;&#xC218;&#xC758;-Gini-index&#xB97C;-&#xC81C;&#xC2DC;&#xD574;&#xC8FC;&#xC138;&#xC694;."></a>

In \[91\]:

```python
##문제1 답안
get_binary_split(pd_data, 'income')
```

Out\[91\]:

```text
[['high'],
 ['medium'],
 ['low'],
 ['high', 'medium'],
 ['high', 'low'],
 ['medium', 'low']]
```

In \[94\]:

```python
pd_data.columns[:-1]
```

Out\[94\]:

```text
Index(['age', 'income', 'student', 'credit_rating'], dtype='object')
```

In \[116\]:

```python
##문제2 답안
target = "class_buys_computer"

def get_important_feature(df, target):
    cols = df.columns[df.columns != target]
    results = []
    for col_name in cols:
        my_dict = get_attribute_gini_index(df, col_name, target)
        if my_dict:
            min_key = min(my_dict.keys(), key=(lambda k: my_dict[k]))
            print(f"{col_name}) Gini Index: {my_dict[min_key]} {min_key}")
            results.append((my_dict[min_key], col_name, min_key))
    results.sort()
    return results[0]

print('최적의 값:', get_important_feature(pd_data, "class_buys_computer"))
```

```text
age) Gini Index: 0.35714285714285715 ('middle_aged',)
income) Gini Index: 0.4428571428571429 ('high',)
student) Gini Index: 0.3673469387755103 ('no',)
credit_rating) Gini Index: 0.42857142857142855 ('fair',)
최적의 값: (0.35714285714285715, 'age', ('middle_aged',))
```

Age의 Gini Index가 0.35로 가장 적기때문에 가장 중요한 변수이며 그중에서도 'middle\_aged'가 Split의 기준이 될 것이다.In \[4\]:

```text
##문제3 답안
```

In \[141\]:

```python
# Split 함수 생성
def split_by_vals(attr, vals, df=pd_data):
    t_index = pd_data[attr].map(lambda x: x in vals)
    # Index 에 따라 DF 분리
    return df[t_index], df[~t_index]
```

In \[125\]:

```python
# 기준에 따라 데이터프레임 2개 생성
df_split_t, df_split_f = split_by_vals('age',('middle_aged',))
```

In \[126\]:

```python
df_split_t
```

Out\[126\]:

|  | age | income | student | credit\_rating | class\_buys\_computer |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2 | middle\_aged | high | no | fair | yes |
| 6 | middle\_aged | low | yes | excellent | yes |
| 11 | middle\_aged | medium | no | excellent | yes |
| 12 | middle\_aged | high | yes | fair | yes |

In \[127\]:

```python
get_important_feature(df_split_t, 'class_buys_computer')
```

```text
income) Gini Index: 0.0 ('high',)
student) Gini Index: 0.0 ('no',)
credit_rating) Gini Index: 0.0 ('fair',)
```

Out\[127\]:

```text
(0.0, 'credit_rating', ('fair',))
```

위에서 만든 함수를 이용해 구한 결과 모두 0이 나와 최적의 상태라 볼 수 있음In \[137\]:

```python
# 'age',('middle_aged',)) False 그룹
df_split_f 
```

Out\[137\]:

|  | age | income | student | credit\_rating | class\_buys\_computer |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | youth | high | no | fair | no |
| 1 | youth | high | no | excellent | no |
| 3 | senior | medium | no | fair | yes |
| 4 | senior | low | yes | fair | yes |
| 5 | senior | low | yes | excellent | no |
| 7 | youth | medium | no | fair | no |
| 8 | youth | low | yes | fair | yes |
| 9 | senior | medium | yes | fair | yes |
| 10 | youth | medium | yes | excellent | yes |
| 13 | senior | medium | no | excellent | no |

In \[138\]:

```python
gini, s_attr, s_vals = get_important_feature(df_split_f, 'class_buys_computer')
```

```text
age) Gini Index: 0.48 ('youth',)
income) Gini Index: 0.375 ('high',)
student) Gini Index: 0.31999999999999984 ('no',)
credit_rating) Gini Index: 0.4166666666666667 ('fair',)
```

In \[139\]:

```python
print(gini, s_attr, s_vals)
```

```text
0.31999999999999984 student ('no',)
```

Student 가 no인지 아닌지에 대한 지니계수가 0.319로 최소임으로 이에 따른 분류가 최적이라는 것을 알 수 있다.In \[142\]:

```python
# df_split_f 가지에서 한 번 더 분류 시도
df_split2_t, df_split2_f = split_by_vals(s_attr, s_vals, df=df_split_f)
```

```text
/Users/josang-yeon/tobigs/lib/python3.7/site-packages/ipykernel_launcher.py:5: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
  """
```

In \[143\]:

```python
df_split2_t
```

Out\[143\]:

|  | age | income | student | credit\_rating | class\_buys\_computer |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | youth | high | no | fair | no |
| 1 | youth | high | no | excellent | no |
| 3 | senior | medium | no | fair | yes |
| 7 | youth | medium | no | fair | no |
| 13 | senior | medium | no | excellent | no |

In \[144\]:

```python
get_important_feature(df_split2_t, 'class_buys_computer')
```

```text
age) Gini Index: 0.2 ('youth',)
income) Gini Index: 0.26666666666666666 ('high',)
credit_rating) Gini Index: 0.26666666666666666 ('fair',)
```

Out\[144\]:

```text
(0.2, 'age', ('youth',))
```

2번째 좌측\(True\) 가지에선 Age - youth 조합이 지니계수 0.2로 최적의 변수임을 알 수 있다.In \[145\]:

```python
df_split2_f
```

{% file src="../.gitbook/assets/1-\_-eda\_-.ipynb" %}

Out\[145\]:

|  | age | income | student | credit\_rating | class\_buys\_computer |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 4 | senior | low | yes | fair | yes |
| 5 | senior | low | yes | excellent | no |
| 8 | youth | low | yes | fair | yes |
| 9 | senior | medium | yes | fair | yes |
| 10 | youth | medium | yes | excellent | yes |

In \[146\]:

```python
get_important_feature(df_split2_f, 'class_buys_computer')
```

```text
age) Gini Index: 0.26666666666666666 ('senior',)
income) Gini Index: 0.26666666666666666 ('low',)
credit_rating) Gini Index: 0.2 ('fair',)
```

Out\[146\]:

```text
(0.2, 'credit_rating', ('fair',))
```

2번째 우측\(False\) 가지에선 'credit\_rating', \('fair',\) 조합이 지니계수 0.2로 최적의 변수임을 알 수 있다.

