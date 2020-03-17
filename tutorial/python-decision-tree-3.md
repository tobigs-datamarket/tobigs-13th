# Python을 이용한 Decision Tree \(3\)

## DT Assignment1 <a id="DT-Assignment1"></a>

**주의사항** 

* 본인이 구현한 함수임을 증명하기 위해 주석 꼼꼼히 달아주세요. 
  * 이 데이터셋 뿐만 아니라 변수의 class가 더 많은 데이터에도 상관없이 적용 가능하도록 함수를 구현해 주세요.

    변수의 class가 3개를 넘는 경우 모든 이진분류 경우의 수를 따져 보아야 합니다.

    Hint\) itertools 라이브러리의 combination 함수 & isin 함수 등이 활용될 수 있으며 이 밖에도 본인의 방법대로 마음껏 구현해주세요.
* 함수에 들어가는 변수나 flow 등은 본인이 변경해도 무관하며 결과만 똑같이 나오면 됩니다. 

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

In \[3\]:

```python
pd_data
```

Out\[3\]:

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

![](2.png)

## 1. Gini 계수를 구하는 함수 만들기 <a id="1.-Gini-&#xACC4;&#xC218;&#xB97C;-&#xAD6C;&#xD558;&#xB294;-&#xD568;&#xC218;-&#xB9CC;&#xB4E4;&#xAE30;"></a>

* Input: df\(데이터\), label\(타겟변수명\)
* 해당 결과는 아래와 같이 나와야 합니다.

In \[4\]:

```python
def get_gini(df, label):
    gini = 1 - (len(df.loc[df[label] == 'yes'])/len(df))**2 - (len(df.loc[df[label] =='no'])/len(df))**2 
    return gini
#타겟변수에 대한 값이 yes일때와 no일대 각각 그 확률값을 제고배서 빼주는 방식이므로
#loc을 통해 그 값만 가지는 row들만 추출해서 그 개수를 새는 식으로 함수를 짰습니다
```

In \[5\]:

```python
get_gini(pd_data,'class_buys_computer')
```

Out\[5\]:

```text
0.4591836734693877
```

## 2. Feature의 Class를 이진 분류로 만들기 <a id="2.-Feature&#xC758;-Class&#xB97C;-&#xC774;&#xC9C4;-&#xBD84;&#xB958;&#xB85C;-&#xB9CC;&#xB4E4;&#xAE30;"></a>

### ex\) {A,B,C} -&gt; \({A}, {B,C}\), \({B}, {A,C}\), \({C}, {A,B}\) <a id="ex)-{A,B,C}--&gt;-({A},-{B,C}),-({B},-{A,C}),-({C},-{A,B})"></a>

* Input: df\(데이터\), attribute\(Gini index를 구하고자 하는 변수명\)
* 해당 결과는 아래와 같이 나와야 합니다.

In \[6\]:

```python
import itertools # 변수의 모든 클래시 조합을 얻기 위해 itertools 불러오기
def get_binary_split(df, attribute):
    
    '''
        이 부분을 채워주세요
                           '''
    
    a = list(df[attribute].unique())
    #타겟변수에 대한 값들이 어떤게 있는지 리스트로 만들기 위해 unique를 사용했습니다
    unique =[]
    for i in range(len(a)):
        unique.append([a[i]])
        #각각의 리스트 속 값들을 []또 리스트 안에 넣기 위한 코드입니다

    binary = list(itertools.combinations(a,2))
    #itertools의 combinations을 사용해서 이진분류로 만들었고
        
    for i in range(len(binary)):
        binary[i] = list(map(str,binary[i]))
    #각각의 값들을 또 list로 만들었습니다
    
    result = unique+list(binary)
    #두개를 합쳐서 이진 분류로 완성
    
    return result 
```

In \[7\]:

```python
get_binary_split(pd_data, "age")
```

Out\[7\]:

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

In \[8\]:

```python
def get_attribute_gini_index(df, attribute, label):
    
    binary_split = get_binary_split(df,attribute)
    #위에서 구한 함수를 사용해서 split한 변수들을 list로 받습니다
    
    #(멘토링에서 도움 받음.)

    result  = {}
    for i in binary_split:
        unique = list(df[attribute].unique())
        #unique함수를 이용해서 각 타겟변수에 대한 값들을 unique로 받습니다
        for j in range(len(i)):
            if j == 0:
                #binary_split에서 'youth'하나만 고려하는 것처럼 그런 경우를 먼저 
                #그에 해당하는 row들만 뽑아냅니다
                new1 = df.loc[df[attribute] == i[j]]
            else:
                #여러개를 동시에 봐야하는 경우에는 두 경우 모두를 각각 concat으로 더해
                #그 경우의 row들만 뽑아냅니다
                new1 = pd.concat([new1,df.loc[df[attribute]==i[j]]], axis=0)
        
        gini1 = (len(new1[attribute])/len(df[attribute]))* get_gini(new1,label)
        #그래서 앞서 구한 새로운 new data를 통해 gini계수를 구하고 거기에 di/d 확률을 곱해줍니다
        
        # 그다음은 앞서 나눈 것의 나머지 부분을 확인해 gini계수를 구해야하는데
        #위에 방식과 똑같이 진행하면 된다
        #앞서 확인한 값제외한 걸로 확인을 해야하므로 그걸 확인시켜주기 위해 그걸 지우고!
        for k in i:
            unique.remove(k)
        
        for s in range(len(unique)):
            if s == 0:
                new2 = df.loc[df[attribute] == unique[s]]
            else:
                new2 = pd.concat([new2,df.loc[df[attribute]==unique[s]]], axis = 0)
        gini2 = (len(new2[attribute])/len(df[attribute])) *get_gini(new2,label)
        #똑같은 방식으로 두번째 gini계수를 구해준다
        
        #앞서 구한 두 gini계수를 더해주면 된다
        gini = gini1 + gini2
        
        #결과로 나온 dictionary의 key값은 해당 class들로 이루어진 tuple형태로 되어있으므로 tuple()을 취해준다
        result[tuple(i)] = gini
        
    return result                    
                
                
```

In \[9\]:

```python
get_attribute_gini_index(pd_data, "age", "class_buys_computer")
```

Out\[9\]:

```text
{('youth',): 0.3936507936507937,
 ('middle_aged',): 0.35714285714285715,
 ('senior',): 0.4571428571428572,
 ('youth', 'middle_aged'): 0.4571428571428572,
 ('youth', 'senior'): 0.35714285714285715,
 ('middle_aged', 'senior'): 0.3936507936507937}
```

여기서 가장 작은 Gini index값을 가지는 class를 기준으로 split해야겠죠?

결과를 확인해보도록 하겠습니다.In \[10\]:

```python
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

In \[11\]:

```python
##문제1 답안

get_binary_split(pd_data, "income")
```

Out\[11\]:

```text
[['high'],
 ['medium'],
 ['low'],
 ['high', 'medium'],
 ['high', 'low'],
 ['medium', 'low']]
```

In \[12\]:

```python
##문제2 답안
#모든 변수들을 모두 gini index를 확인해서 min값을 찾아서 비교했습니다
get_attribute_gini_index(pd_data, "age", "class_buys_computer")
```

Out\[12\]:

```text
{('youth',): 0.3936507936507937,
 ('middle_aged',): 0.35714285714285715,
 ('senior',): 0.4571428571428572,
 ('youth', 'middle_aged'): 0.4571428571428572,
 ('youth', 'senior'): 0.35714285714285715,
 ('middle_aged', 'senior'): 0.3936507936507937}
```

In \[13\]:

```python
my_dict = get_attribute_gini_index(pd_data, "age", "class_buys_computer")
key_min = min(my_dict.keys(), key=(lambda k: my_dict[k]))
print('Min -',key_min, ":", my_dict[key_min])
#0.3571
```

```text
Min - ('middle_aged',) : 0.35714285714285715
```

In \[14\]:

```python
get_attribute_gini_index(pd_data, "income", "class_buys_computer")
```

Out\[14\]:

```text
{('high',): 0.4428571428571429,
 ('medium',): 0.4583333333333333,
 ('low',): 0.45,
 ('high', 'medium'): 0.45,
 ('high', 'low'): 0.4583333333333333,
 ('medium', 'low'): 0.4428571428571429}
```

In \[15\]:

```python
my_dict = get_attribute_gini_index(pd_data, "income", "class_buys_computer")
key_min = min(my_dict.keys(), key=(lambda k: my_dict[k]))
print('Min -',key_min, ":", my_dict[key_min])
#0.4428
```

```text
Min - ('high',) : 0.4428571428571429
```

In \[16\]:

```python
get_attribute_gini_index(pd_data, "student", "class_buys_computer")
```

Out\[16\]:

```text
{('no',): 0.3673469387755103,
 ('yes',): 0.3673469387755103,
 ('no', 'yes'): 0.7040816326530612}
```

In \[17\]:

```python
my_dict = get_attribute_gini_index(pd_data, "student", "class_buys_computer")
key_min = min(my_dict.keys(), key=(lambda k: my_dict[k]))
print('Min -',key_min, ":", my_dict[key_min])
#0.3673
```

```text
Min - ('no',) : 0.3673469387755103
```

In \[18\]:

```python
get_attribute_gini_index(pd_data, "credit_rating", "class_buys_computer")
```

Out\[18\]:

```text
{('fair',): 0.42857142857142855,
 ('excellent',): 0.42857142857142855,
 ('fair', 'excellent'): 0.673469387755102}
```

In \[19\]:

```python
my_dict = get_attribute_gini_index(pd_data, "credit_rating", "class_buys_computer")
key_min = min(my_dict.keys(), key=(lambda k: my_dict[k]))
print('Min -',key_min, ":", my_dict[key_min])
#0.4285
```

```text
Min - ('fair',) : 0.42857142857142855
```

In \[ \]:

```python
#확인결과 가장 작은 gini index는
#min -> middle_agged -> 0.3571 이므로
#가장 중요한 변수 : age임을 알수 있습니다
```

In \[20\]:

```python
##문제3 답안
#문제3) 문제 2에서 제시한 feature로 DataFrame을 split한 후 나눠진 2개의 
#DataFrame에서 각각 다음으로 중요한 변수를 선정하고 해당 변수의 Gini index를 제시해주세요

#앞서 구한 가장 중요한 변수를 기반으로
#age, middle_aged / youth,senior 이렇게 나누어 새로운 데이터를 만들었습니다

df1 = pd_data.loc[pd_data['age'] == 'middle_aged']
youth = pd_data.loc[pd_data['age'] == 'youth'] 
senior = pd_data.loc[pd_data['age'] == 'senior']
df2 = pd.concat([youth,senior],axis=0)    
#df1은 이미 모든 target 변수 값이 yes인 완벽하게 split된 데이터이므로 확인할 필요가 없고
#df2를 사용해서 다음으로 중요한 변수를 선정했습니다
```

In \[21\]:

```python
get_attribute_gini_index(df2, "income", "class_buys_computer")
```

Out\[21\]:

```text
{('high',): 0.375,
 ('medium',): 0.48,
 ('low',): 0.47619047619047616,
 ('high', 'medium'): 0.47619047619047616,
 ('high', 'low'): 0.48,
 ('medium', 'low'): 0.375}
```

In \[22\]:

```python
my_dict = get_attribute_gini_index(df2, "income", "class_buys_computer")
key_min = min(my_dict.keys(), key=(lambda k: my_dict[k]))
print('Min -',key_min, ":", my_dict[key_min])
#0.375
```

```text
Min - ('high',) : 0.375
```

In \[23\]:

```python
get_attribute_gini_index(df2, "student", "class_buys_computer")
```

Out\[23\]:

```text
{('no',): 0.31999999999999984,
 ('yes',): 0.31999999999999984,
 ('no', 'yes'): 0.6599999999999999}
```

In \[24\]:

```python
my_dict = get_attribute_gini_index(df2, "student", "class_buys_computer")
key_min = min(my_dict.keys(), key=(lambda k: my_dict[k]))
print('Min -',key_min, ":", my_dict[key_min])
#0.31944444
```

```text
Min - ('no',) : 0.31999999999999984
```

In \[25\]:

```python
get_attribute_gini_index(df2, "credit_rating", "class_buys_computer")
```

Out\[25\]:

```text
{('fair',): 0.4166666666666667,
 ('excellent',): 0.4166666666666667,
 ('fair', 'excellent'): 0.7666666666666666}
```

In \[26\]:

```python
my_dict = get_attribute_gini_index(df2, "credit_rating", "class_buys_computer")
key_min = min(my_dict.keys(), key=(lambda k: my_dict[k]))
print('Min -',key_min, ":", my_dict[key_min])
#0.416667
```

```text
Min - ('fair',) : 0.4166666666666667
```

In \[28\]:

```python
#확인결과 가장 작은  gini index 갖는 변수 는 student로
#->{('no',): 0.31999999999999984 가장 낮게 나왔습니다

#한번더 gini index를 확인하면
get_attribute_gini_index(df2, "student", "class_buys_computer")
```

Out\[28\]:

```text
{('no',): 0.31999999999999984,
 ('yes',): 0.31999999999999984,
 ('no', 'yes'): 0.6599999999999999}
```

