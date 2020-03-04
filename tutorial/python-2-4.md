# Python을 이용한 차원 축소 실습 \(2\)

## ''' ? ''' 이 있는 부분을 채워주시면 됩니다[¶](https://render.githubusercontent.com/view/ipynb?commit=0b58db1568c04eb17b6c5ca37003747fbd55f281&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f746f626967732d646174616d61726b65742f746f626967732d313372642f306235386462313536386330346562313762366335636133373030333734376662643535663238312f352545432541332542432545432542302541382f357765656b5f5043415f31322545412542382542302545412542392538302545442539412541382545432539442538302e6970796e62&nwo=tobigs-datamarket%2Ftobigs-13rd&path=5%EC%A3%BC%EC%B0%A8%2F5week_PCA_12%EA%B8%B0%EA%B9%80%ED%9A%A8%EC%9D%80.ipynb&repository_id=233872163&repository_type=Repository#'''-?-'''-%EC%9D%B4-%EC%9E%88%EB%8A%94-%EB%B6%80%EB%B6%84%EC%9D%84-%EC%B1%84%EC%9B%8C%EC%A3%BC%EC%8B%9C%EB%A9%B4-%EB%90%A9%EB%8B%88%EB%8B%A4)

나는 내 스타일로 하겠다 하시면 그냥 구현 하셔도 됩니다!!

참고하셔야 하는 함수들은 링크 달아드렸으니 들어가서 확인해보세요

## 1. PCA의 과정

In \[1\]:

```python
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import pandas as pd
import random
#   기본 모듈들을 불러와 줍니다
```

In \[2\]:

```python
x1 = [95, 91, 66, 94, 68, 63, 12, 73, 93, 51, 13, 70, 63, 63, 97, 56, 67, 96, 75, 6]
x2 = [56, 27, 25, 1, 9, 80, 92, 69, 6, 25, 83, 82, 54, 97, 66, 93, 76, 59, 94, 9]
x3 = [57, 34, 9, 79, 4, 77, 100, 42, 6, 96, 61, 66, 9, 25, 84, 46, 16, 63, 53, 30]
#   설명변수 x1, x2, x3의 값이 이렇게 있네요
```

In \[3\]:

```python
X = np.stack((x1, x2, x3), axis=0)
#   설명변수들을 하나의 행렬로 만들어 줍니다
```

In \[4\]:

```python
X = pd.DataFrame(X.T, columns=['x1', 'x2', 'x3'])
```

In \[5\]:

```text
X
```

Out\[5\]:

|  | x1 | x2 | x3 |
| :--- | :--- | :--- | :--- |
| 0 | 95 | 56 | 57 |
| 1 | 91 | 27 | 34 |
| 2 | 66 | 25 | 9 |
| 3 | 94 | 1 | 79 |
| 4 | 68 | 9 | 4 |
| 5 | 63 | 80 | 77 |
| 6 | 12 | 92 | 100 |
| 7 | 73 | 69 | 42 |
| 8 | 93 | 6 | 6 |
| 9 | 51 | 25 | 96 |
| 10 | 13 | 83 | 61 |
| 11 | 70 | 82 | 66 |
| 12 | 63 | 54 | 9 |
| 13 | 63 | 97 | 25 |
| 14 | 97 | 66 | 84 |
| 15 | 56 | 93 | 46 |
| 16 | 67 | 76 | 16 |
| 17 | 96 | 59 | 63 |
| 18 | 75 | 94 | 53 |
| 19 | 6 | 9 | 30 |

In \[6\]:

```python
# OR
X = np.stack((x1, x2, x3), axis=1)
pd.DataFrame(X, columns=['x1', 'x2', 'x3']).head()
```

Out\[6\]:

|  | x1 | x2 | x3 |
| :--- | :--- | :--- | :--- |
| 0 | 95 | 56 | 57 |
| 1 | 91 | 27 | 34 |
| 2 | 66 | 25 | 9 |
| 3 | 94 | 1 | 79 |
| 4 | 68 | 9 | 4 |

1-1\) 먼저 PCA를 시작하기 전에 항상 **데이터를 scaling** 해주어야 해요

In \[7\]:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std  = scaler.fit_transform(X)
```

In \[8\]:

```text
X_std
```

Out\[8\]:

```text
array([[ 1.08573604,  0.02614175,  0.30684189],
       [ 0.93801686, -0.86575334, -0.46445467],
       [ 0.01477192, -0.92726334, -1.30282049],
       [ 1.04880625, -1.66538341,  1.04460382],
       [ 0.08863151, -1.41934339, -1.47049366],
       [-0.09601747,  0.76426183,  0.97753455],
       [-1.97943714,  1.13332186,  1.74883111],
       [ 0.2732805 ,  0.42595679, -0.1961776 ],
       [ 1.01187645, -1.5116084 , -1.40342439],
       [-0.53917504, -0.92726334,  1.61469258],
       [-1.94250735,  0.85652683,  0.44098042],
       [ 0.16249111,  0.82577183,  0.60865359],
       [-0.09601747, -0.03536825, -1.30282049],
       [-0.09601747,  1.28709688, -0.76626636],
       [ 1.15959564,  0.33369178,  1.21227698],
       [-0.35452606,  1.16407687, -0.06203907],
       [ 0.05170172,  0.64124181, -1.06807806],
       [ 1.12266584,  0.11840676,  0.50804969],
       [ 0.3471401 ,  1.19483187,  0.17270336],
       [-2.20101593, -1.41934339, -0.5985932 ]])
```

In \[9\]:

```text
features = X_std.T
```

In \[10\]:

```text
features
```

Out\[10\]:

```text
array([[ 1.08573604,  0.93801686,  0.01477192,  1.04880625,  0.08863151,
        -0.09601747, -1.97943714,  0.2732805 ,  1.01187645, -0.53917504,
        -1.94250735,  0.16249111, -0.09601747, -0.09601747,  1.15959564,
        -0.35452606,  0.05170172,  1.12266584,  0.3471401 , -2.20101593],
       [ 0.02614175, -0.86575334, -0.92726334, -1.66538341, -1.41934339,
         0.76426183,  1.13332186,  0.42595679, -1.5116084 , -0.92726334,
         0.85652683,  0.82577183, -0.03536825,  1.28709688,  0.33369178,
         1.16407687,  0.64124181,  0.11840676,  1.19483187, -1.41934339],
       [ 0.30684189, -0.46445467, -1.30282049,  1.04460382, -1.47049366,
         0.97753455,  1.74883111, -0.1961776 , -1.40342439,  1.61469258,
         0.44098042,  0.60865359, -1.30282049, -0.76626636,  1.21227698,
        -0.06203907, -1.06807806,  0.50804969,  0.17270336, -0.5985932 ]])
```

1-2\) 자 그럼 공분산 행렬을 구해볼게요

```python
# feature 간의 covariance matrix
cov_matrix = np.cov(features)
```

In \[12\]:

```text
cov_matrix
```

Out\[12\]:

```text
array([[ 1.05263158, -0.2037104 , -0.12079228],
       [-0.2037104 ,  1.05263158,  0.3125801 ],
       [-0.12079228,  0.3125801 ,  1.05263158]])
```

1-3\) 이제 고유값과 고유벡터를 구해볼게요

방법은 실습코드에 있어요!!

In \[13\]:

```python
# 공분산 행렬의 eigen value, eigen vector
eigenvalues  = lin.eig(cov_matrix)[0]
eigenvectors = lin.eig(cov_matrix)[1]
```

In \[14\]:

```python
print(eigenvalues)
print(eigenvectors)

# 여기서 eigenvectors는 각 eigen vector가 열벡터로 들어가있는 형태!
# the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
# https://numpy.org/doc/1.18/reference/generated/numpy.linalg.eig.html?highlight=eig#numpy.linalg.eig
```

```text
[1.48756162 0.94435407 0.72597904]
[[ 0.47018528 -0.85137353 -0.23257022]
 [-0.64960236 -0.15545725 -0.74421087]
 [-0.59744671 -0.50099516  0.62614797]]
```

In \[15\]:

```python
# 3*3 영행렬
mat = np.zeros((3, 3))
```

In \[16\]:

```text
mat
```

Out\[16\]:

```text
array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]])
```

In \[17\]:

```python
# symmetric matrix = P*D*P.T로 분해
mat[0][0] = eigenvalues[0]
mat[1][1] = eigenvalues[1]
mat[2][2] = eigenvalues[2]
print(mat)
```

```text
[[1.48756162 0.         0.        ]
 [0.         0.94435407 0.        ]
 [0.         0.         0.72597904]]
```

In \[18\]:

```python
# 혹은 아래와 같이 diagonal matrix를 만들 수 있다
np.diag(eigenvalues)
```

Out\[18\]:

```text
array([[1.48756162, 0.        , 0.        ],
       [0.        , 0.94435407, 0.        ],
       [0.        , 0.        , 0.72597904]])
```

1-4\) 자 이제 고유값 분해를 할 모든 준비가 되었어요 고유값 분해의 곱으로 원래 공분산 행렬을 구해보세요

In \[19\]:

```python
# P*D*P.T
np.dot(np.dot(eigenvectors, mat), eigenvectors.T)
```

Out\[19\]:

```text
array([[ 1.05263158, -0.2037104 , -0.12079228],
       [-0.2037104 ,  1.05263158,  0.3125801 ],
       [-0.12079228,  0.3125801 ,  1.05263158]])
```

In \[20\]:

```python
cov_matrix
```

Out\[20\]:

```text
array([[ 1.05263158, -0.2037104 , -0.12079228],
       [-0.2037104 ,  1.05263158,  0.3125801 ],
       [-0.12079228,  0.3125801 ,  1.05263158]])
```

1-5\) 마지막으로 고유 벡터 축으로 값을 변환해 볼게요

함수로 한번 정의해 보았어요

In \[21\]:

```text
X_std
```

Out\[21\]:

```text
array([[ 1.08573604,  0.02614175,  0.30684189],
       [ 0.93801686, -0.86575334, -0.46445467],
       [ 0.01477192, -0.92726334, -1.30282049],
       [ 1.04880625, -1.66538341,  1.04460382],
       [ 0.08863151, -1.41934339, -1.47049366],
       [-0.09601747,  0.76426183,  0.97753455],
       [-1.97943714,  1.13332186,  1.74883111],
       [ 0.2732805 ,  0.42595679, -0.1961776 ],
       [ 1.01187645, -1.5116084 , -1.40342439],
       [-0.53917504, -0.92726334,  1.61469258],
       [-1.94250735,  0.85652683,  0.44098042],
       [ 0.16249111,  0.82577183,  0.60865359],
       [-0.09601747, -0.03536825, -1.30282049],
       [-0.09601747,  1.28709688, -0.76626636],
       [ 1.15959564,  0.33369178,  1.21227698],
       [-0.35452606,  1.16407687, -0.06203907],
       [ 0.05170172,  0.64124181, -1.06807806],
       [ 1.12266584,  0.11840676,  0.50804969],
       [ 0.3471401 ,  1.19483187,  0.17270336],
       [-2.20101593, -1.41934339, -0.5985932 ]])
```

In \[22\]:

```python
def new_coordinates(X, eigenvectors):
    for i in range(eigenvectors.shape[0]):
        if i == 0:
            new = [X.dot(eigenvectors.T[i])]
        else:
            new = np.concatenate((new, [X.dot(eigenvectors.T[i])]), axis=0)
    return new.T

# 모든 고유 벡터 축으로 데이터를 projection한 값입니다
```

In \[23\]:

```text
X_std
```

Out\[23\]:

```text
array([[ 1.08573604,  0.02614175,  0.30684189],
       [ 0.93801686, -0.86575334, -0.46445467],
       [ 0.01477192, -0.92726334, -1.30282049],
       [ 1.04880625, -1.66538341,  1.04460382],
       [ 0.08863151, -1.41934339, -1.47049366],
       [-0.09601747,  0.76426183,  0.97753455],
       [-1.97943714,  1.13332186,  1.74883111],
       [ 0.2732805 ,  0.42595679, -0.1961776 ],
       [ 1.01187645, -1.5116084 , -1.40342439],
       [-0.53917504, -0.92726334,  1.61469258],
       [-1.94250735,  0.85652683,  0.44098042],
       [ 0.16249111,  0.82577183,  0.60865359],
       [-0.09601747, -0.03536825, -1.30282049],
       [-0.09601747,  1.28709688, -0.76626636],
       [ 1.15959564,  0.33369178,  1.21227698],
       [-0.35452606,  1.16407687, -0.06203907],
       [ 0.05170172,  0.64124181, -1.06807806],
       [ 1.12266584,  0.11840676,  0.50804969],
       [ 0.3471401 ,  1.19483187,  0.17270336],
       [-2.20101593, -1.41934339, -0.5985932 ]])
```

In \[24\]:

```python
new_coordinates(X_std, eigenvectors)

# 새로운 축으로 변환되어 나타난 데이터들입니다
```

Out\[24\]:

```text
array([[ 0.31019368, -1.08215716, -0.07983642],
       [ 1.28092404, -0.43132556,  0.13533091],
       [ 1.38766381,  0.78428014, -0.12911446],
       [ 0.95087515, -1.15737142,  1.6495519 ],
       [ 1.84222365,  0.88189889,  0.11493111],
       [-1.12563709, -0.52680338,  0.06564012],
       [-2.71174416,  0.63290138,  0.71195473],
       [-0.03100441, -0.20059783, -0.50339479],
       [ 2.29618509,  0.07661447,  0.01087174],
       [-0.61585248, -0.205764  ,  1.82651199],
       [-1.73320252,  1.29971699,  0.09045178],
       [-0.82366049, -0.57164535, -0.27123176],
       [ 0.75619512,  0.73995175, -0.76710616],
       [-0.42344386,  0.26555394, -1.41533681],
       [-0.39581307, -1.64646874,  0.24104031],
       [-0.88581498,  0.15195119, -0.82271209],
       [ 0.24587691,  0.39139878, -1.15801831],
       [ 0.14741103, -1.22874561, -0.03110396],
       [-0.7161265 , -0.56781471, -0.86180345],
       [ 0.24475107,  2.39442622,  1.19337361]])
```

## 2. PCA 구현

위의 과정을 이해하셨다면 충분히 하실 수 있을거에요In \[25\]:

```python
from sklearn.preprocessing import StandardScaler

def MYPCA(X, number):
    scaler = StandardScaler()
    x_std = scaler.fit_transform(X)
    features = x_std.T
    cov_matrix = np.cov(features)
    
    eigenvalues  = lin.eig(cov_matrix)[0]
    eigenvectors = lin.eig(cov_matrix)[1]
    
    new_coordinate = new_coordinates(x_std, eigenvectors)
    
    index = eigenvalues.argsort()[::-1] # 내림차순 정렬한 인덱스
    index = list(index)
    
    for i in range(number):
        if i==0:
            new = [new_coordinate[:, index.index(i)]]
        else:
            new = np.concatenate(([new, [new_coordinate[:, index.index(i)]]]), axis=0)
    return new.T
```

In \[26\]:

```python
MYPCA(X,3)

# 새로운 축으로 잘 변환되어서 나타나나요?
# 위에서 했던 PCA랑은 차이가 있을 수 있어요 왜냐하면 위에서는 고유값이 큰 축 순서로 정렬을 안했었거든요
```

Out\[26\]:

```text
array([[ 0.31019368, -1.08215716, -0.07983642],
       [ 1.28092404, -0.43132556,  0.13533091],
       [ 1.38766381,  0.78428014, -0.12911446],
       [ 0.95087515, -1.15737142,  1.6495519 ],
       [ 1.84222365,  0.88189889,  0.11493111],
       [-1.12563709, -0.52680338,  0.06564012],
       [-2.71174416,  0.63290138,  0.71195473],
       [-0.03100441, -0.20059783, -0.50339479],
       [ 2.29618509,  0.07661447,  0.01087174],
       [-0.61585248, -0.205764  ,  1.82651199],
       [-1.73320252,  1.29971699,  0.09045178],
       [-0.82366049, -0.57164535, -0.27123176],
       [ 0.75619512,  0.73995175, -0.76710616],
       [-0.42344386,  0.26555394, -1.41533681],
       [-0.39581307, -1.64646874,  0.24104031],
       [-0.88581498,  0.15195119, -0.82271209],
       [ 0.24587691,  0.39139878, -1.15801831],
       [ 0.14741103, -1.22874561, -0.03110396],
       [-0.7161265 , -0.56781471, -0.86180345],
       [ 0.24475107,  2.39442622,  1.19337361]])
```

## 3. sklearn 비교

In \[27\]:

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
```

In \[28\]:

```python
pca.fit_transform(X_std)[:5]
```

Out\[28\]:

```python
array([[-0.31019368, -1.08215716, -0.07983642],
       [-1.28092404, -0.43132556,  0.13533091],
       [-1.38766381,  0.78428014, -0.12911446],
       [-0.95087515, -1.15737142,  1.6495519 ],
       [-1.84222365,  0.88189889,  0.11493111]])
```

In \[29\]:

```python
MYPCA(X, 3)[:5]
```

Out\[29\]:

```text
array([[ 0.31019368, -1.08215716, -0.07983642],
       [ 1.28092404, -0.43132556,  0.13533091],
       [ 1.38766381,  0.78428014, -0.12911446],
       [ 0.95087515, -1.15737142,  1.6495519 ],
       [ 1.84222365,  0.88189889,  0.11493111]])
```

In \[30\]:

```python
pca = PCA(n_components=2)
pca.fit_transform(X_std)[:5]
```

Out\[30\]:

```text
array([[-0.31019368, -1.08215716],
       [-1.28092404, -0.43132556],
       [-1.38766381,  0.78428014],
       [-0.95087515, -1.15737142],
       [-1.84222365,  0.88189889]])
```

In \[31\]:

```python
MYPCA(X, 2)[:5]
```

Out\[31\]:

```text
array([[ 0.31019368, -1.08215716],
       [ 1.28092404, -0.43132556],
       [ 1.38766381,  0.78428014],
       [ 0.95087515, -1.15737142],
       [ 1.84222365,  0.88189889]])
```

## 4. MNIST data에 적용을 해보!

```python
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from scipy import io
%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D

# mnist 손글씨 데이터를 불러옵니다
```

In \[2\]:

```python
mnist = io.loadmat('mnist-original.mat') 
X = mnist['data'].T
y = mnist['label'].T
```

In \[3\]:

```python
print(X.shape)
print(y.shape)
```

```text
(70000, 784)
(70000, 1)
```

In \[4\]:

```python
np.unique(y)
```

Out\[4\]:

```text
array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
```

In \[5\]:

```python
# data information

# 7만개의 작은 숫자 이미지
# 행 열이 반대로 되어있음 -> 전치
# grayscale 28x28 pixel = 784 feature
# 각 picel은 0~255의 값
# label = 1~10 label이 총 10개인거에 주목하자
```

In \[6\]:

```python
# data를 각 픽셀에 이름붙여 표현
feat_cols = ['pixel'+str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X, columns=feat_cols)
df.head()
```

Out\[6\]:

|  | pixel0 | pixel1 | pixel2 | pixel3 | pixel4 | pixel5 | pixel6 | pixel7 | pixel8 | pixel9 | ... | pixel774 | pixel775 | pixel776 | pixel777 | pixel778 | pixel779 | pixel780 | pixel781 | pixel782 | pixel783 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | ... | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | ... | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | ... | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | ... | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | ... | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

5 rows × 784 columnsIn \[7\]:

```python
# df에 라벨 y를 붙여서 데이터프레임 생성
df['y'] = y
```

In \[8\]:

```text
df.head()
```

Out\[8\]:

|  | pixel0 | pixel1 | pixel2 | pixel3 | pixel4 | pixel5 | pixel6 | pixel7 | pixel8 | pixel9 | ... | pixel775 | pixel776 | pixel777 | pixel778 | pixel779 | pixel780 | pixel781 | pixel782 | pixel783 | y |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | ... | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | ... | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | ... | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | ... | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | ... | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |

5 rows × 785 columns

## 지금까지 배운 여러 머신러닝 기법들이 있을거에요

4-1\) train\_test\_split을 통해 데이터를 0.8 0.2의 비율로 분할 해 주시고요

4-2\) PCA를 이용하여 mnist data를 축소해서 학습을 해주세요 / test error가 제일 작으신 분께 상품을 드리겠습니다 ^0^

특정한 틀 없이 자유롭게 하시면 됩니다!!!!!!!!!

### 1. train test split

In \[9\]:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

In \[10\]:

```python
X_train, X_test, y_train, y_test = train_test_split(df.drop('y', axis=1), df['y'], stratify=df['y'])
```

In \[11\]:

```python
standard_scaler = StandardScaler()
standard_scaler.fit(X_train)
X_scaled_train = standard_scaler.transform(X_train)
X_scaled_test  = standard_scaler.transform(X_test)
```

In \[43\]:

```python
print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)
```

```text
(52500, 784)
(17500, 784)
(52500,)
(17500,)
```

In \[56\]:

```python
print(pd.Series(y_train).value_counts()/len(y_train))
print(pd.Series(y_test).value_counts()/len(y_test))
```

```text
1.0    0.112533
7.0    0.104190
3.0    0.102019
2.0    0.099848
9.0    0.099390
0.0    0.098610
6.0    0.098229
8.0    0.097505
4.0    0.097486
5.0    0.090190
Name: y, dtype: float64
1.0    0.112514
7.0    0.104171
3.0    0.102000
2.0    0.099886
9.0    0.099429
0.0    0.098629
6.0    0.098229
8.0    0.097486
4.0    0.097486
5.0    0.090171
Name: y, dtype: float64
```

### 2. 주성분 개수의 결정

1. elbow point \(곡선의 기울기가 급격히 감소하는 지점\)
2. kaiser's rule \(고유값 1 이상의 주성분들\)
3. 누적설명률이 70%~80% 이상인 지점

In \[13\]:

```python
from sklearn.decomposition import PCA
```

**누적설명률이 70%~80%인 지점**

In \[193\]:

```python
variance_ratio = {}

for i in range(80, 200):
    if(i%10==0):
        print(i)
    pca = PCA(n_components=i)
    pca.fit(X_scaled_train)
    variance_ratio['_'.join(['n', str(i)])] = pca.explained_variance_ratio_.sum()
```

```text
10
20
30
40
50
60
70
80
90
```

In \[247\]:

```python
pca = PCA()
pca.fit(X_scaled_train)
```

Out\[247\]:

```text
PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
    svd_solver='auto', tol=0.0, whiten=False)
```

In \[226\]:

```python
variance_ratio = []
ratio = 0
for i in np.sort(pca.explained_variance_ratio_)[::-1]:
    ratio += i
    variance_ratio.append(ratio)
```

In \[243\]:

```python
plt.figure(figsize=(12, 5))

plt.plot(list(range(50, 500)), variance_ratio[50:500])

plt.axhline(0.7, color='gray', ls='--')
plt.axhline(0.8, color='gray', ls='--')
plt.axhline(0.9, color='gray', ls='--')

plt.axvline(96, color='black', ls='--')
plt.axvline(146, color='black', ls='--')
plt.axvline(230, color='black', ls='--')

plt.title("VARIANCE RATIO (70%~80%)", size=15)
plt.show()

# scaling한 후
# 96개의 주성분을 선택하면 누적설명률이 70%정도
# 146개의 주성분을 선택하면 누적설명률이 80%정도
# 230개 이상의 주성분을 선택하면 누적설명률이 90%이상 된다.
```

![](data:image/png;base64,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%0A)

**elbow point**

In \[248\]:

```python
# eigen value를 내림차순으로 정렬한 뒤, plot을 그려보았다.
plt.figure(figsize=(12, 5))
plt.plot(range(1, X.shape[1]+1), np.sort(pca.explained_variance_)[::-1])
plt.show()
```

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAr8AAAEvCAYAAABMl6kwAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjAsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+17YcXAAAgAElEQVR4nO3de3BcZ5nn8d/TV90syxfFdnyJczGQLEWcIExIYDeEy3pYCjJVMEOWS3Y3lGd3YRYYaiEwU7tQtUwNNQxhpoqh8JBMUlvhthAgFcIlhIRsplgncuIkDrbJhSS+S4ktS7au3f3sH+e01JJlqd3q1pH1fj9VXX3Oe06ffvTakn9+9fZ7zN0FAAAAhCCVdAEAAADAfCH8AgAAIBiEXwAAAASD8AsAAIBgEH4BAAAQDMIvAAAAgpGZzzdbuXKlb9y4cT7fEgAAAAHauXPny+7eObV9XsPvxo0b1d3dPZ9vCQAAgACZ2YvTtTPtAQAAAMEg/AIAACAYhF8AAAAEg/ALAACAYBB+AQAAEAzCLwAAAIJB+AUAAEAwCL8AAAAIBuEXAAAAwVj04fcPL5/St3e8pIHhsaRLAQAAQMKqDr9mljazx83snnj/QjPbYWbPmNn3zCzXuDJr9/hLx/X5Hz2lY6dGky4FAAAACTubkd9PSNpTsf9lSbe4+yZJxyXdVM/C6iVlJkkqljzhSgAAAJC0qsKvma2T9O8kfSveN0nXSfpBfModkq5vRIFzlUpF4bfkhF8AAIDQVTvy+zVJn5FUivdXSOpz90K8f0DS2jrXVhfp8ZHfhAsBAABA4mYNv2b2bkk97r6zsnmaU6cdWjWzbWbWbWbdvb29NZZZu3T8FTLtAQAAANWM/F4j6T1m9oKk7yqa7vA1SR1mlonPWSfp0HQvdvft7t7l7l2dnZ11KPnslOf8Mu0BAAAAs4Zfd/+cu69z942SPiDp1+7+QUkPSHpffNqNkn7SsCrnIM2cXwAAAMTmss7vZyX9hZk9q2gO8K31Kam+WO0BAAAAZZnZT5ng7g9KejDefl7SlvqXVF+s9gAAAICyRX+HN1Z7AAAAQNmiD78pVnsAAABAbNGH3zSrPQAAACC26MMvc34BAABQtvjDL6s9AAAAILbowy/r/AIAAKBs8YdfVnsAAABAbNGHX1Z7AAAAQNmiD7/laQ/OtAcAAIDgLfrwO/6BN8IvAABA8MIJv0x7AAAACN6iD7+s9gAAAICyxR9+We0BAAAAsUUffsurPZSY9gAAABC8xR9+jWkPAAAAiCz68Fue88tqDwAAAFj04Xd85JdpDwAAAMFb9OF3fOSX8AsAABC8xR9+x29ykXAhAAAASNyiD7+s9gAAAICyxR9+We0BAAAAsVnDr5k1mdkjZvaEmT1tZl+M2283sz+Y2a74sbnx5Z49VnsAAABAWaaKc0YkXefuJ80sK+lhM/tZfOy/u/sPGlfe3LHaAwAAAMpmDb/u7pJOxrvZ+HHOJMmJ1R4SLgQAAACJq2rOr5mlzWyXpB5J97n7jvjQl8zsSTO7xczyDatyDuLsy7QHAAAAVBd+3b3o7pslrZO0xcxeK+lzkl4j6Q2Slkv67HSvNbNtZtZtZt29vb11Krt6ZqaUSU74BQAACN5Zrfbg7n2SHpS01d0Pe2RE0j9L2nKG12x39y537+rs7JxzwbVImXGTCwAAAFS12kOnmXXE282S3i5pr5mtidtM0vWSdjey0LlIpYxpDwAAAKhqtYc1ku4ws7SisPx9d7/HzH5tZp2STNIuSf+5gXXOSdqM1R4AAABQ1WoPT0q6Ypr26xpSUQOkU8ZqDwAAAFj8d3iTohUfuMMbAAAAwgi/KSP8AgAAIIzwm2a1BwAAACiQ8MvILwAAAKRAwi8jvwAAAJBCCb+s9gAAAAAFEn5TKW5vDAAAgFDCr3GHNwAAAAQSfpnzCwAAACmQ8MtqDwAAAJACCb+M/AIAAEAKJPymWO0BAAAACiX8Gqs9AAAAIJDwm06x2gMAAAACCb8p5vwCAABAgYTfNKs9AAAAQKGEX0Z+AQAAoEDCbyollVjtAQAAIHhhhF9j2gMAAAACCb+s9gAAAAApkPCbMlOJOb8AAADBCyL8MvILAAAAqYrwa2ZNZvaImT1hZk+b2Rfj9gvNbIeZPWNm3zOzXOPLrU20zm/SVQAAACBp1Yz8jki6zt0vl7RZ0lYzu0rSlyXd4u6bJB2XdFPjypybdIrbGwMAAKCK8OuRk/FuNn64pOsk/SBuv0PS9Q2psA64wxsAAACkKuf8mlnazHZJ6pF0n6TnJPW5eyE+5YCktY0pce5SzPkFAACAqgy/7l50982S1knaIunS6U6b7rVmts3Mus2su7e3t/ZK5yDNag8AAADQWa724O59kh6UdJWkDjPLxIfWSTp0htdsd/cud+/q7OycS601Y7UHAAAASNWt9tBpZh3xdrOkt0vaI+kBSe+LT7tR0k8aVeRcRev8Jl0FAAAAkpaZ/RStkXSHmaUVheXvu/s9ZvY7Sd81s/8l6XFJtzawzjlJmbi9MQAAAGYPv+7+pKQrpml/XtH83wUvnWK1BwAAAARyh7cU4RcAAAAKJPy2ZNMaHC0mXQYAAAASFkT4XdKU1dBYUWPc4xgAACBoQYTf9uZoavPJ4cIsZwIAAGAxCyL8LmnKSpIGCL8AAABBCyT8RiO//cNjCVcCAACAJAURftvjkV/CLwAAQNiCCL/lkV+mPQAAAIQtiPA7PvI7xMgvAABAyIIIv4z8AgAAQCL8AgAAICBBhN9MOqWWXJoPvAEAAAQuiPArRaO/A4RfAACAoAUUfrNMewAAAAhcMOG3OZvW8Fgx6TIAAACQoGDCbyZtKpQ86TIAAACQoGDCbzad0lixlHQZAAAASFBA4dc0VmTkFwAAIGQBhd+UCoz8AgAABC2Y8JtJpTTKyC8AAEDQggm/2bQx8gsAABC4WcOvma03swfMbI+ZPW1mn4jbv2BmB81sV/x4V+PLrR0feAMAAECminMKkj7t7o+Z2RJJO83svvjYLe7+lcaVVz8ZPvAGAAAQvFnDr7sflnQ43h4wsz2S1ja6sHrLMfILAAAQvLOa82tmGyVdIWlH3PRxM3vSzG4zs2V1rq2uuMkFAAAAqg6/ZtYm6YeSPunu/ZK+IeliSZsVjQz/3Rlet83Mus2su7e3tw4l1yabTmmswMgvAABAyKoKv2aWVRR873T3uyTJ3Y+6e9HdS5L+SdKW6V7r7tvdvcvduzo7O+tV91nLplMaKxF+AQAAQlbNag8m6VZJe9z9qxXtaypO+2NJu+tfXv1ES50x7QEAACBk1az2cI2kD0t6ysx2xW2fl3SDmW2W5JJekPRnDamwTjKplAoll7sryvMAAAAITTWrPTwsabq0eG/9y2mcbDr6EsaKrlyG8AsAABCigO7wFn2pLHcGAAAQrmDCbyYOv8z7BQAACFcw4TcXT3sYZeQXAAAgWMGE3/GRX5Y7AwAACFYw4Xd8zm+BaQ8AAAChCij8xqs9MPILAAAQrIDCLx94AwAACF0w4TeTKq/zy8gvAABAqIIJv6zzCwAAgADDL9MeAAAAQhVM+M3EH3grMPILAAAQrGDCb3nkl5tcAAAAhCug8Fse+WXaAwAAQKgCCr984A0AACB0AYXf8k0uGPkFAAAIVUDht3yTC0Z+AQAAQhVM+M0w7QEAACB4wYTf7Pgd3pj2AAAAEKpwwi8jvwAAAMELJvxmWOoMAAAgeMGE31wm+lJHCsWEKwEAAEBSwgm/6ZQyKdPgKOEXAAAgVLOGXzNbb2YPmNkeM3vazD4Rty83s/vM7Jn4eVnjy62dmak1n9GpkULSpQAAACAh1Yz8FiR92t0vlXSVpI+Z2WWSbpZ0v7tvknR/vL+gtebSOjnCyC8AAECoZg2/7n7Y3R+Ltwck7ZG0VtJ7Jd0Rn3aHpOsbVWS9tOYzGhxl5BcAACBUZzXn18w2SrpC0g5Jq9z9sBQFZEnn1bu4emvNZ3SSaQ8AAADBqjr8mlmbpB9K+qS795/F67aZWbeZdff29tZSY9205tPM+QUAAAhYVeHXzLKKgu+d7n5X3HzUzNbEx9dI6pnute6+3d273L2rs7OzHjXXrDWXYbUHAACAgFWz2oNJulXSHnf/asWhuyXdGG/fKOkn9S+vvtqY9gAAABC0TBXnXCPpw5KeMrNdcdvnJf2NpO+b2U2SXpL0/saUWD8tTHsAAAAI2qzh190flmRnOPy2+pbTWK35jE4x7QEAACBYwdzhTZLachmNFkoaK5aSLgUAAAAJCCr8tuSjgW6mPgAAAIQpqPDblk9LElMfAAAAAhVU+G1l5BcAACBoYYXfXBR+We4MAAAgTGGF33jkd3CEaQ8AAAAhCiz8RnN+GfkFAAAIU1jhN8ecXwAAgJCFFX7L0x5GCb8AAAAhCir8tuXLH3hjzi8AAECIggq/TdmUUsa0BwAAgFAFFX7NTK25jE4x7QEAACBIQYVfKZr3y8gvAABAmIILvy35tE4x5xcAACBIwYXftjzTHgAAAEIVXPhtyaWZ9gAAABCo4MJvWz7DUmcAAACBCi78tuYz3OQCAAAgUMGF35Ycqz0AAACEKrjw25ZP6yThFwAAIEjBhd/WfEbDYyUViqWkSwEAAMA8mzX8mtltZtZjZrsr2r5gZgfNbFf8eFdjy6yfJU1ZSWL0FwAAIEDVjPzeLmnrNO23uPvm+HFvfctqnKXNUfjtHyL8AgAAhGbW8OvuD0k6Ng+1zIv2powk6cTQWMKVAAAAYL7NZc7vx83syXhaxLK6VdRg4yO/w4RfAACA0NQafr8h6WJJmyUdlvR3ZzrRzLaZWbeZdff29tb4dvXTHodfRn4BAADCU1P4dfej7l5095Kkf5K0ZYZzt7t7l7t3dXZ21lpn3UzM+SX8AgAAhKam8Gtmayp2/1jS7jOdu9Aw8gsAABCuzGwnmNl3JF0raaWZHZD0PyVda2abJbmkFyT9WQNrrKvWXFrplDHnFwAAIECzhl93v2Ga5lsbUMu8MDO1N2VY6gwAACBAwd3hTYrm/TLtAQAAIDzBhl+mPQAAAIQnyPDbzsgvAABAkIIMvytac+odGEm6DAAAAMyzIMPvqqVN6ukfkbsnXQoAAADmUZDhd3V7k0aLJR0fZOoDAABASIINv5J05MRwwpUAAABgPgUZfs+Lw+/RfsIvAABASIIMv6uXEn4BAABCFGT4PW9JXpJ0hPALAAAQlCDDbzad0sq2HCO/AAAAgQky/ErSqvYmPvAGAAAQmGDD7+r2Jh3t50YXAAAAIQk2/K5a2sS0BwAAgMCEG36XNOmVU6MaKRSTLgUAAADzJNjwu3pptOJDD1MfAAAAghFs+F0V3+iiZ4CpDwAAAKEINvyWb3Rx5AQjvwAAAKEINvyuWhKHXz70BgAAEIxgw29HS1a5TIoVHwAAAAISbPg1M63mRhcAAABBCTb8StKq9jwjvwAAAAGZNfya2W1m1mNmuyvalpvZfWb2TPy8rLFlNsaqdm50AQAAEJJqRn5vl7R1StvNku53902S7o/3zzmr25t0pH9Y7p50KQAAAJgHs4Zfd39I0rEpze+VdEe8fYek6+tc17xYvbRJw2Ml9Q8Vki4FAAAA86DWOb+r3P2wJMXP59WvpPlzUWerJOmJA30JVwIAAID50PAPvJnZNjPrNrPu3t7eRr/dWbn64pVqyaX186ePJF0KAAAA5kGt4feoma2RpPi550wnuvt2d+9y967Ozs4a364xmrJpXfvqTj2w94zlAwAAYBGpNfzeLenGePtGST+pTznz7zWr23X4xLCGx4pJlwIAAIAGq2aps+9I+q2kV5vZATO7SdLfSHqHmT0j6R3x/jlpbUezJOlQ31DClQAAAKDRMrOd4O43nOHQ2+pcSyLWLYvC78G+IV3U2ZZwNQAAAGikoO/wJklry+H3OCO/AAAAi13w4Xd1e5PSKdMBwi8AAMCiF3z4zaRTWt3epIPM+QUAAFj0gg+/knTxeW3ae2Qg6TIAAADQYIRfSZvXLdW+I/0aHOU2xwAAAIsZ4VfS5es7VHJp98H+pEsBAABAAxF+FYVfSfrpk4cSrgQAAACNRPiVtLItrw9fdYHu+O2LeurAiaTLAQAAQIMQfmOffuerlDLpvj1Hky4FAAAADUL4jXW05LR5fYd+s68n6VIAAADQIITfCm+7dJWeOHBCO188lnQpAAAAaADCb4X/cPVGre1o1pd+uifpUgAAANAAhN8KrfmM3t+1To/v79PxU6NJlwMAAIA6I/xO8ZZNK+Uu/fb5V5IuBQAAAHVG+J3i8nUd6mjJ6iu/3KcjJ4aTLgcAAAB1RPidIpNO6Zsfer0O9w3r5ruelLsnXRIAAADqhPA7jTdetEKf2fpqPbivV7/8Hev+AgAALBaE3zP48FUX6JLz2vTln+1VscToLwAAwGJA+D2DTDqlT739VXr+5VO673dHki4HAAAAdUD4ncHW167WxhUt+ut796pvkKXPAAAAznWE3xmkU6av/ulmHeob0ld+uS/pcgAAADBHcwq/ZvaCmT1lZrvMrLteRS0kV25Ypvd3rdf3Hz2gF14+lXQ5AAAAmIN6jPy+1d03u3tXHa61IH38ukvUkk/rP93+qIZGi0mXAwAAgBox7aEKazua9Y///ko9//Ip3fKr3yddDgAAAGo01/Drkn5pZjvNbFs9Clqorr5kpW7YskHbH3pe33jwORWKpaRLAgAAwFnKzPH117j7ITM7T9J9ZrbX3R+qPCEOxdskacOGDXN8u2R94T2XqXdgWF/++V71D4/ps1tfk3RJAAAAOAtzGvl190Pxc4+kH0naMs052929y927Ojs75/J2ictn0vrWjW/Qn3St0zd/85x27e9LuiQAAACchZrDr5m1mtmS8rakd0raXa/CFrK/evdlWtXepE99b5deOTmSdDkAAACo0lxGfldJetjMnpD0iKSfuvvP61PWwtbelNU/3HCFDp8Y0vX/+C96YG9P0iUBAACgCjWHX3d/3t0vjx//yt2/VM/CFro3bFyuOz96lVJm+o+3P6pv73gp6ZIAAAAwC5Y6m4PXX7BMv/qLf6O3bFqpv/rxU/roHY9q54vHki4LAAAAZ0D4naNsOqWvf/BKffQtF+mJAyf0kVsf0Z07XtTwGDfDAAAAWGgIv3XQ3pTV5991qe758zfrVauX6C9/tFvX/u2DenAfc4EBAAAWEsJvHa1qb9Jd/+Vq3fnRN2pJU0Z//u3H9fAzL8vdky4NAAAAIvzWnZnpmktWavtHuuSSPnTrDv3pN/+fDp8YSro0AACA4Nl8jkp2dXV5d3f3vL1f0o6dGtW9Tx3WX9+7RyOFkq5Y36GrL1mpqy9eoc3rO9SUTSddIgAAwKJkZjvdveu0dsJv4z3fe1J3PXZQDz3Tq90HT6jkUjpl+pOu9bp562u0tCWbdIkAAACLCuF3gTgxNKZH/nBMv/l9j77zyH615NJ6+6Wr9KGrNujKDctkZkmXCAAAcM4j/C5ATx86odsefkG/2nNUJ4bG1JpL68oLlum/XnuJtly4XOkUQRgAAKAWhN8F7NRIQT/edVDPHD2pu584pGOnRrWyLa9rLlmh8zua9bq1S/W69R06f2kTI8MAAABVIPyeI06NFPTAvh797Kkj2rW/Tz0DwxorRn9GF6xo0QfesEGXnd+uN2xcppZcJuFqAQAAFibC7zlqeKyofUcG9MSBPv348YN67KU+SZKZ1NmW1/rlLbrqouW6uLNN65a1aP3yZq1a0qQUUyYAAEDACL+LRO/AiPYe6Vf3C8d1qG9Iz/We1OP7+1T5x5hLp7R2WbPWLWvW2o5mdbTk1NGSVUdzVkubs1raktWrVi3RyrZ8cl8IAABAA50p/PJ783NM55K8Opd06i2bOsfbhseKOtg3pAPHh7T/2KD2Hx/UgeNDOnBsUHuPDKhvcHR86kSldcua1bkkr+UtOW25cLletXqJ1i9r1tqOFjXnWIMYAAAsPoTfRaApm9bFnW26uLNt2uPurqGxovoGx9Q3OKbjg6N66uAJPX2oX32Do3rx2KDu39sz6TUrWnNat6xZ65a1xM8T22uXNTPfGAAAnJNIMAEwM7XkMmrJZXR+R7Mk6ZpLVk46p2dgWPuPxSPG449B7Tncr/v2HNVooTTp/OXj4bgiFHdMbLfm+asFAAAWHhIKJEnnLWnSeUua9PoLTj9WKrlePjmi/ceH4ukVEyF575EB/WpPz2nheFlLdjwIr17apJVteXUuyev8pc1avTSvzrYmtTdnWLoNAADMK8IvZpVKmc5rb9J57U16/QXLTjteKrlePjWiA8eHdLBi1PjA8SHtOzqg//vMyzo5UjjtdblMSp1teS1rjT+IFz/a4+eVbXmtWdqkVe1N6ojbm7LMRQYAALUj/GLOUikbHzm+csPp4ViShkaL6h0Y0cG+IfUMDKt3YES9J0fU2z+ivqExnRga05ETAzoxVFD/0JhGi6Vpr5PLpCYF5fHA3JQZD85t+Yyasmk1ZVPKZ9Nqzqa1pCmj9qboWEs+rXyGEA0AQIgIv5gXzbm0Nqxo0YYVLbOeW/6A3isnR3X4xLCO9g/rRByQ+4fH1B9vnxgaU8/AsJ7pGVD/UEH9w2OqduW+bNrUls9oSRyI25oyWhI/l/fLYbkcnNubs2pvjraXNmfVkkszbQMAgHMM4RcLzvgH9JZntH757GG5rFRyDYwUdGqkoOGxoobHShouFDU8WlT/cDSifGq0oMHRok6OFHRyuKCTIwUNDBc0MDymI/3DOtkbtQ8MF844+lyWMsUfJEyrLZ9Ra35iuyWfUWsurdb4uXK/JZdRa758LHpNaz5qy6VTBGoAABqI8ItFI5Wy8WkQ9TBSKMbBOArHJ4bGNBCH6GgEOgrSg6NRiD41UtCp0aKODgxr8OUoYA+OFnVqtFD1iHQmZWrJpdWcSyuXSSmfiQJxa35ywG7JZdScS6spE03vaM6llS+fn0kpl04pny0/R9eIrhc9chXnprkbIAAgIHMKv2a2VdLfS0pL+pa7/01dqgIWgHwmrXxbes53witP4zg1MhGUB0eLUVgeicLxYBycT42PXJc0WixptFDSSCF67csnR/XCK4MajEevh8eK09685GxlUhYF5opgnEtPhON8pjI4nx6uy0E6mzFlUyll0qZMypRJp5RJmbLpirb4eDYdhe5sepq28jUqj1Vcl5FxAMBc1Bx+zSwt6euS3iHpgKRHzexud/9dvYoDFoPKdZal+t5SulAsaaRQ0tBYFIZHC5WhOXouB+iR09ri7WJRI3HYHpkSusvnDAwX9ErcNt31C6X5u016OmVnDNXplClt8XPKlIq3UylT2jSpbdJxM6VTE8cz46+pfP3kc09vq3jYlPec1BadYyalzOJH9PckVW5LRfumM5yTqtyPt1VxzZSmvGbi2tLE61M28RrTxPVNJktp/P2t8rXxfuUx/kMC4Fwyl5HfLZKedffnJcnMvivpvZIIv8A8yaRTyqRTid9UpFhyjRVLGiuWVCi6CiVXoRRtjxVL8fGobazoKpTbStH2WNFVLE0+PlZyFYtRsC63Tb5uvB1fo1B0FT26TvlRKu97NCc82naNFkoqukdt7iqWNL5dKkX1V76+5JXX1Ph1yq+pdlrLYlYZolNxOp4UkDUR6FW5P+WYxSeU83S5rRzuy+eW33O6Y+NRfJprW+W1K9vjY+PXjTesYn/qdSauP+W1U95noo8mapv89dn49sQxm+a88vaUYxXvr1muO7FdcWya655ep016v/HWyW97Wl9MrUWznjdNffHGjOdPU7sqrjHTeVPf84xf9wy1T1fztP19FvWd6Zimu+6kr+P02itrnPHPdtKx2f8cq9GcTevNm1bOfuI8msu/mGsl7a/YPyDpjVNPMrNtkrZJ0oYNG+bwdgAWqmjEMx3sOszuXhGIpUKppFJJ42F8cnieCNFS9FyKX1eKg3TJPX5E13ZF4by8X6o4Z+L8iWMznVNyl6bsl88Zf338NUWnlq8ZvzauJTonrrni/PJ++X18yvmuif8suJ/eXt6Pz4jrOv24a+L6k68z+dqqfO207+PTvs4rrq2pNU69Tim+zqRrx68bv+7EtiqOjfeFKt5nvG2ipikvnXz+lPO84sWn9ceUa0z8WYy/xRlq9ymvnfznNFMtlX/eZ1vf1OtqutpnuAaSt2F5ix76zFuTLmOSuYTf6eL/aX/d3H27pO2S1NXVxV9HAIuOWTxHebwlzP8EAAtR5X9gpJn+czL5vMmvqS7gn204L/8nbmotM15jSpI6839OTv8aNe1503yN07TVKptO1f7iBplL+D0gaX3F/jpJh+ZWDgAAQP1UTjmoaE2kFiwMc4njj0raZGYXmllO0gck3V2fsgAAAID6q3nk190LZvZxSb9Q9Du+29z96bpVBgAAANTZnD4i7u73Srq3TrUAAAAADbXwZiEDAAAADUL4BQAAQDAIvwAAAAgG4RcAAADBIPwCAAAgGIRfAAAABIPwCwAAgGCYz+WGzWf7Zma9kl6ctzecsFLSywm877mMPqsN/VYb+q029NvZo89qQ7/Vhn6rTb367QJ375zaOK/hNylm1u3uXUnXcS6hz2pDv9WGfqsN/Xb26LPa0G+1od9q0+h+Y9oDAAAAgkH4BQAAQDBCCb/bky7gHESf1YZ+qw39Vhv67ezRZ7Wh32pDv9Wmof0WxJxfAAAAQApn5BcAAABY3OHXzLaa2T4ze9bMbk66noXEzG4zsx4z213RttzM7jOzZ+LnZXG7mdk/xP34pJldmVzlyTGz9Wb2gJntMbOnzewTcTv9NgMzazKzR8zsibjfvhi3X2hmO+J++56Z5eL2fLz/bHx8Y5L1J83M0mb2uJndE+/Tb7MwsxfM7Ckz22Vm3XEb36czMLMOM/uBme2Nf8a9iT6bmZm9Ov47Vn70m9kn6bfZmdmn4n8PdpvZd+J/J+btZ9uiDb9mlpb0dUl/JOkySTeY2WXJVrWg3C5p65S2myXd7+6bJN0f70tRH26KH9skfWOealxoCpI+7e6XSrpK0sfiv1P028xGJF3n7pdL2ixpq5ldJenLkm6J++24pJvi82+SdKOebTwAAAQfSURBVNzdL5F0S3xeyD4haU/FPv1Wnbe6++aK5ZL4Pp3Z30v6ubu/RtLliv7O0WczcPd98d+xzZJeL2lQ0o9Ev83IzNZK+m+Sutz9tZLSkj6g+fzZ5u6L8iHpTZJ+UbH/OUmfS7quhfSQtFHS7or9fZLWxNtrJO2Lt78p6Ybpzgv5Ieknkt5Bv51Vn7VIekzSGxUtYJ6J28e/XyX9QtKb4u1MfJ4lXXtC/bVO0T+e10m6R5LRb1X12wuSVk5p4/v0zP3VLukPU/++0Gdn1YfvlPQv9FtVfbVW0n5Jy+OfVfdI+rfz+bNt0Y78aqJzyw7EbTizVe5+WJLi5/PidvpyivjXLldI2iH6bVbxr+53SeqRdJ+k5yT1uXshPqWyb8b7LT5+QtKK+a14wfiapM9IKsX7K0S/VcMl/dLMdprZtriN79Mzu0hSr6R/jqfYfMvMWkWfnY0PSPpOvE2/zcDdD0r6iqSXJB1W9LNqp+bxZ9tiDr82TRtLW9SGvqxgZm2Sfijpk+7eP9Op07QF2W/uXvToV4PrJG2RdOl0p8XP9JskM3u3pB5331nZPM2p9NvprnH3KxX9mvljZvavZziXfotG066U9A13v0LSKU38qn469FmFeG7qeyT9n9lOnaYtuH6L50C/V9KFks6X1Kroe3Wqhv1sW8zh94Ck9RX76yQdSqiWc8VRM1sjSfFzT9xOX8bMLKso+N7p7nfFzfRbldy9T9KDiuZMd5hZJj5U2Tfj/RYfXyrp2PxWuiBcI+k9ZvaCpO8qmvrwNdFvs3L3Q/Fzj6I5mFvE9+lMDkg64O474v0fKArD9Fl1/kjSY+5+NN6n32b2dkl/cPdedx+TdJekqzWPP9sWc/h9VNKm+NODOUW/krg74ZoWursl3Rhv36hoTmu5/SPxJ1WvknSi/CudkJiZSbpV0h53/2rFIfptBmbWaWYd8Xazoh98eyQ9IOl98WlT+63cn++T9GuPJ3uFxN0/5+7r3H2jop9fv3b3D4p+m5GZtZrZkvK2ormYu8X36Rm5+xFJ+83s1XHT2yT9TvRZtW7QxJQHiX6bzUuSrjKzlvjf1fLft/n72Zb0xOcGT6p+l6TfK5pf+JdJ17OQHoq+UQ9LGlP0v6qbFM2huV/SM/Hz8vhcU7RyxnOSnlL0Cc3Ev4YE+uzNin7V8qSkXfHjXfTbrP32OkmPx/22W9L/iNsvkvSIpGcV/bowH7c3xfvPxscvSvprSPoh6VpJ99BvVfXVRZKeiB9Pl3/28306a79tltQdf5/+WNIy+qyqfmuR9IqkpRVt9Nvs/fZFSXvjfxP+t6T8fP5s4w5vAAAACMZinvYAAAAATEL4BQAAQDAIvwAAAAgG4RcAAADBIPwCAAAgGIRfAAAABIPwCwAAgGAQfgEAABCM/w8KmMYaLdVHfQAAAABJRU5ErkJggg==%0A)

```python
# elbow 포인트 지점을 확인하기 위해 0~100 구간을 세밀하게 살펴보도록 한다.
# 확인 결과, 상위 13개의 eigen value를 제외한 나머지 eigen value 사이에는 그다지 큰 차이가 없는 것으로 보인다.

plt.figure(figsize=(12, 5))
plt.plot(range(0, 100), np.sort(pca.explained_variance_)[::-1][0:100])

plt.title('ELBOW POINT', size=15)
plt.axvline(12, ls='--', color='grey')

plt.show()
```

![](data:image/png;base64,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%0A)

**Kaiser's Rule**

In \[266\]:

```python
# 이번에는 kaiser's rule에 따라 고유값 1 이상의 주성분을 찾아보고자 한다.
# 이를 위해 100~200 구간을 세밀하게 살펴보았으며,
# 그 결과 160개의 주성분을 사용했을 때 eigenvalue가 모두 1 이상이었다.
plt.figure(figsize=(12, 5))
plt.plot(range(100, 200), np.sort(pca.explained_variance_)[::-1][100:200])

plt.title("Kaiser's Rule", size=15)
plt.axhline(1, ls='--', color='grey')
plt.axvline(160, ls='--', color='grey')

plt.show()
```

![](data:image/png;base64,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%0A)In \[188\]:

```python
# Kaiser's rule에 따라 160개의 주성분을 선택하여 train을 진행해보도록 하겠다.
# (또한 160개의 주성분을 선택하면 원본 데이터의 변동 중 약 82%정도가 설명가능하다.)
```

In \[14\]:

```python
pca = PCA(n_components=160)
pca.fit(X_scaled_train)
X_PCA_train = pca.transform(X_scaled_train)
X_PCA_test  = pca.transform(X_scaled_test)
```

### 3. Modeling

In \[15\]:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
import time
import warnings
warnings.filterwarnings('ignore')
```

**Random Forest**

* Original Data

In \[290\]:

```python
start = time.time()
rf_clf.fit(X_train, y_train)
end  = time.time()
print(f'time > {end-start}')
```

```text
time > 53.27030920982361
```

In \[291\]:

```python
pred = rf_clf.predict(X_test)
print(accuracy_score(pred, y_test))
```

```text
0.9678857142857142
```

* PCA Data

In \[271\]:

```python
rf_clf = RandomForestClassifier()
```

In \[292\]:

```python
start = time.time()
rf_clf.fit(X_PCA_train, y_train)
end  = time.time()
print(f'time > {end-start}')
```

```text
time > 84.94275045394897
```

In \[294\]:

```python
pred = rf_clf.predict(X_PCA_test)
print(accuracy_score(pred, y_test))
```

```text
0.9408571428571428
```

**Logistic Regression**

* Original Data

In \[295\]:

```python
from sklearn.linear_model import LogisticRegression
```

In \[296\]:

```python
lr_clf = LogisticRegression()
```

In \[307\]:

```python
start = time.time()
lr_clf.fit(X_train, y_train)
end  = time.time()
print(f'time > {end-start}')
```

```text
time > 13.492876052856445
```

In \[308\]:

```python
pred = lr_clf.predict(X_test)
print(accuracy_score(pred, y_test))
```

```text
0.9225714285714286
```

In \[338\]:

```python
param = {
    'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
}

grid = GridSearchCV(lr_clf, param, cv=5, scoring='accuracy', verbose=10)
grid.fit(X_train, y_train)
```

In \[340\]:

```text
grid.best_score_
```

Out\[340\]:

```text
0.918704761904762
```

* PCA Data

In \[320\]:

```python
start = time.time()
lr_clf.fit(X_PCA_train, y_train)
end  = time.time()
print(f'time > {end-start}')
```

```text
time > 5.322706937789917
```

In \[321\]:

```python
# 원본데이터와 accuracy 차이가 크게 나지 않는다!
# 하지만 Random Forest에 비해 accuracy 떨어진다.
pred = lr_clf.predict(X_PCA_test)
print(accuracy_score(pred, y_test))
```

```text
0.9208
```

In \[342\]:

```python
param = {
    'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
}

grid = GridSearchCV(lr_clf, param, cv=5, scoring='accuracy', verbose=10)
grid.fit(X_PCA_train, y_train)

grid.best_score_
```

Out\[342\]:

```text
0.9200571428571429
```

**Decision Tree**

* Original Data

In \[309\]:

```python
from sklearn.tree import DecisionTreeClassifier
```

In \[310\]:

```python
dt_clf = DecisionTreeClassifier()
```

In \[311\]:

```python
start = time.time()
dt_clf.fit(X_train, y_train)
end  = time.time()
print(f'time > {end-start}')
```

```text
time > 24.323933601379395
```

In \[313\]:

```python
pred = dt_clf.predict(X_test)
print(accuracy_score(pred, y_test))
```

```text
0.8713142857142857
```

* PCA data

In \[317\]:

```python
start = time.time()
dt_clf.fit(X_PCA_train, y_train)
end  = time.time()
print(f'time > {end-start}')
```

```text
time > 23.95996594429016
```

In \[318\]:

```python
# PCA한 경우 성능 확연히 떨어진다.
# 또한 Random Forest, Logistic Regression에 비해서 accuracy 너무 낮다.

pred = dt_clf.predict(X_PCA_test)
print(accuracy_score(pred, y_test))
```

```text
0.8281714285714286
```

**SVM**

In \[16\]:

```python
from sklearn.svm import SVC

svm = SVC()
```

In \[429\]:

```python
svm.fit(X_PCA_train, y_train)
```

Out\[429\]:

```text
SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
```

In \[431\]:

```python
pred = svm.predict(X_PCA_test)
accuracy_score(y_test, pred)
```

Out\[431\]:

```text
0.9674857142857143
```

In \[17\]:

```python
param = {
    'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
}

grid = GridSearchCV(svm, param, cv=3, scoring='accuracy', verbose=10, n_jobs=4)
grid.fit(X_PCA_train, y_train)
```

```text
Fitting 3 folds for each of 7 candidates, totalling 21 fits
```

```text
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done   5 tasks      | elapsed: 30.1min
[Parallel(n_jobs=4)]: Done  10 tasks      | elapsed: 36.2min
[Parallel(n_jobs=4)]: Done  17 out of  21 | elapsed: 41.7min remaining:  9.8min
[Parallel(n_jobs=4)]: Done  21 out of  21 | elapsed: 44.6min finished
```

Out\[17\]:

```python
GridSearchCV(cv=3, error_score=nan,
             estimator=SVC(C=1.0, break_ties=False, cache_size=200,
                           class_weight=None, coef0=0.0,
                           decision_function_shape='ovr', degree=3,
                           gamma='scale', kernel='rbf', max_iter=-1,
                           probability=False, random_state=None, shrinking=True,
                           tol=0.001, verbose=False),
             iid='deprecated', n_jobs=4,
             param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring='accuracy', verbose=10)
```

In \[18\]:

```python
# hyper parameter tunning을 통해 accuracy를 높였다.
grid.best_score_
```

Out\[18\]:

```text
0.9722666666666666
```

In \[19\]:

```python
grid.best_params_
```

Out\[19\]:

```text
{'C': 10}
```

**XGBoost**

In \[323\]:

```python
from xgboost import XGBClassifier
```

In \[324\]:

```text
xgb = XGBClassifier()
```

In \[325\]:

```python
start = time.time()
xgb.fit(X_PCA_train, y_train)
end  = time.time()
print(f'time > {end-start}')
```

```text
time > 640.5302169322968
```

In \[326\]:

```python
pred = xgb.predict(X_PCA_test)
print(accuracy_score(pred, y_test))
```

```text
0.9091428571428571
```

**LightGBM**

* Original Data

In \[327\]:

```python
from lightgbm import LGBMClassifier
lgbm = LGBMClassifier()
```

In \[331\]:

```python
start = time.time()
lgbm.fit(X_train, y_train)
end  = time.time()
print(f'time > {end-start}')
```

```text
time > 153.8949658870697
```

In \[332\]:

```python
pred = lgbm.predict(X_test)
print(accuracy_score(pred, y_test))
```

```text
0.9697142857142858
```

* PCA Data

In \[329\]:

```python
start = time.time()
lgbm.fit(X_PCA_train, y_train)
end  = time.time()
print(f'time > {end-start}')
```

```text
time > 48.97201323509216
```

In \[330\]:

```python
pred = lgbm.predict(X_PCA_test)
print(accuracy_score(pred, y_test))
```

```text
0.9494285714285714
```

**Stacking \(CV\)**

* PCA data

In \[432\]:

```python
rf_clf = RandomForestClassifier()
lr_clf = LogisticRegression()
lgbm = LGBMClassifier()
svm = SVC()

final_model = LGBMClassifier()
```

In \[439\]:

```python
def base_model(model, X_train, X_test, y_train, n_split=5):
    
    
#     X_train = X_train.reset_index(drop=True)
#     X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    
    kfold = KFold(n_splits=n_split)
    train_predicted = np.zeros(X_train.shape[0])
    test_predicted  = np.zeros((n_split, X_test.shape[0]))

    for i, (train_index, val_index) in enumerate(kfold.split(X_train)):
        print(f'step > {i} ')
        train_data = X_train[train_index]
        val_data = X_train[val_index]
        train_y = y_train[train_index]
    
        model.fit(train_data, train_y)
        train_predicted[val_index] = model.predict(val_data)
        test_predicted[i] = model.predict(X_test)

    test_predicted = test_predicted.mean(axis=0)
    
    return train_predicted, test_predicted
```

In \[397\]:

```python
rf_train, rf_test = base_model(rf_clf, X_PCA_train, X_PCA_test, y_train)
lr_train, lr_test = base_model(lr_clf, X_PCA_train, X_PCA_test, y_train)
lgbm_train, lgbm_test = base_model(lgbm, X_PCA_train, X_PCA_test, y_train)

stacking_train = np.stack((rf_train, lr_train, lgbm_train)).T
stacking_test  = np.stack((rf_test, lr_test, lgbm_test)).T

# final model로 LGBM 사용
final_model.fit(stacking_train, y_train)
pred = final_model.predict(stacking_test)
print(accuracy_score(y_test, pred)) # 0.9324

# final model로 XGBoost 사용
xgb.fit(stacking_train, y_train)
pred = xgb.predict(stacking_test)
print(accuracy_score(y_test, pred)) # 0.9324

# final model로 Random Forest 사용
rf_clf.fit(stacking_train, y_train)
pred = rf_clf.predict(stacking_test)
print(accuracy_score(y_test, pred)) # 0.9320
```

```text
step > 0 
step > 1 
step > 2 
step > 3 
step > 4 
```

In \[440\]:

```python
PCA_rf_train, PCA_rf_test = base_model(rf_clf, X_PCA_train, X_PCA_test, y_train)
PCA_lgbm_train, PCA_lgbm_test = base_model(lgbm, X_PCA_train, X_PCA_test, y_train)
PCA_svm_train, PCA_svm_test = base_model(svm, X_PCA_train, X_PCA_test, y_train)
```

```text
step > 0 
step > 1 
step > 2 
step > 3 
step > 4 
step > 0 
step > 1 
step > 2 
step > 3 
step > 4 
step > 0 
step > 1 
step > 2 
step > 3 
step > 4 
```

In \[441\]:

```python
PCA_stacking_train = np.stack((PCA_rf_train, PCA_lgbm_train, PCA_svm_train)).T
PCA_stacking_test  = np.stack((PCA_rf_test,  PCA_lgbm_test,  PCA_svm_test)).T
```

In \[443\]:

```python
svm.fit(PCA_stacking_train, y_train)
pred = svm.predict(PCA_stacking_test)
accuracy_score(y_test, pred)
```

Out\[443\]:

```text
0.9583428571428572
```

In \[446\]:

```text
lgbm.fit(PCA_stacking_train, y_train)
pred = lgbm.predict(PCA_stacking_test)
accuracy_score(y_test, pred)
```

Out\[446\]:

```text
0.9577714285714286
```

In \[447\]:

```python
PCA_stacking_train = np.stack((PCA_lgbm_train, PCA_svm_train)).T
PCA_stacking_test  = np.stack((PCA_lgbm_test,  PCA_svm_test)).T

svm.fit(PCA_stacking_train, y_train)
pred = svm.predict(PCA_stacking_test)
accuracy_score(y_test, pred)
```

Out\[447\]:

```text
0.9594285714285714
```

* Original Data

In \[427\]:

```python
rf_train, rf_test = base_model(rf_clf, X_train, X_test, y_train)
lr_train, lr_test = base_model(lr_clf, X_train, X_test, y_train)
lgbm_train, lgbm_test = base_model(lgbm, X_train, X_test, y_train)
```

```text
step > 0 
step > 1 
step > 2 
step > 3 
step > 4 
step > 0 
step > 1 
step > 2 
step > 3 
step > 4 
step > 0 
step > 1 
step > 2 
step > 3 
step > 4 
```

In \[433\]:

```python
stacking_train = np.stack((rf_train, lgbm_train)).T
stacking_test  = np.stack((rf_test, lgbm_test)).T
```

In \[434\]:

```python
# final model로 LGBM 사용
final_model.fit(stacking_train, y_train)
pred = final_model.predict(stacking_test)
print(accuracy_score(y_test, pred))

# 단일모델보다 결과 안좋다
```

```text
0.9558285714285715
```

### 4. Summary

* **PCA Data**
  * SVM - **0.9726**
  * Stacking\(lgbm+svm & svm\) - 0.9594
  * Stacking\(rf+lgbm+svm & svm\) - 0.9583
  * LightGBM - 0.9494
  * Random Forest - 0.9409
  * Stacking\(rf+lr+lgbm & lgbm\) 0.9324
  * Logistic Regression - 0.9208
  * XGBoost -&gt; 0.9091
  * Decision Tree - 0.8282
* **Original Data**
  * LightGBM -&gt; **0.9697**
  * Random Forest- 0.9679
  * Stacking \(rf+lgbm & lgbm\) -&gt; 0.9558
  * Logistic Regression- 0.9226
  * Decision Tree- 0.8713

