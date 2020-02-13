# Python을 이용한 클러스터링 \(3\)

## Load Dataset <a id="Load-Dataset"></a>

### **Import packages**

```python
# data
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore") 

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# model
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth

# grid search
from sklearn.model_selection import GridSearchCV

# evaluation
from sklearn.metrics.cluster import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import *
```

### **Load mall customers data**

```python
df = pd.read_csv('Mall_Customers.csv')
df.head()
```

|  | CustomerID | Gender | Age | Annual Income \(k$\) | Spending Score \(1-100\) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 1 | Male | 19 | 15 | 39 |
| 1 | 2 | Male | 21 | 15 | 81 |
| 2 | 3 | Female | 20 | 16 | 6 |
| 3 | 4 | Female | 23 | 16 | 77 |
| 4 | 5 | Female | 31 | 17 | 40 |

```python
del df["CustomerID"]
```

* ID 값은 clustering을 하는데 있어 필요하지 않아 보이므로 제거하기로 하였다.

```python
df['Gender'].unique()
```

```text
array(['Male', 'Female'], dtype=object)
```

```python
df['Gender'].replace({'Male':1, 'Female':0},inplace=True)
```

* 문자형 데이터를 encoding 하였다.

```text
df.head()
```

|  | Gender | Age | Annual Income \(k$\) | Spending Score \(1-100\) |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 1 | 19 | 15 | 39 |
| 1 | 1 | 21 | 15 | 81 |
| 2 | 0 | 20 | 16 | 6 |
| 3 | 0 | 23 | 16 | 77 |
| 4 | 0 | 31 | 17 | 40 |

```text
df.shape
```

```text
(200, 4)
```

## EDA

### **Describe¶**

```text
df.info()
```

```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200 entries, 0 to 199
Data columns (total 4 columns):
Gender                    200 non-null int64
Age                       200 non-null int64
Annual Income (k$)        200 non-null int64
Spending Score (1-100)    200 non-null int64
dtypes: int64(4)
memory usage: 6.4 KB
```

```text
df.isnull().sum()
```

```text
Gender                    0
Age                       0
Annual Income (k$)        0
Spending Score (1-100)    0
dtype: int64
```

* null 값이 존재하지 않는다 !

```text
df.describe()
```

|  | Gender | Age | Annual Income \(k$\) | Spending Score \(1-100\) |
| :--- | :--- | :--- | :--- | :--- |
| count | 200.000000 | 200.000000 | 200.000000 | 200.000000 |
| mean | 0.440000 | 38.850000 | 60.560000 | 50.200000 |
| std | 0.497633 | 13.969007 | 26.264721 | 25.823522 |
| min | 0.000000 | 18.000000 | 15.000000 | 1.000000 |
| 25% | 0.000000 | 28.750000 | 41.500000 | 34.750000 |
| 50% | 0.000000 | 36.000000 | 61.500000 | 50.000000 |
| 75% | 1.000000 | 49.000000 | 78.000000 | 73.000000 |
| max | 1.000000 | 70.000000 | 137.000000 | 99.000000 |

### **Visualization¶**

```python
sns.countplot('Gender' , data = df)
plt.show()
```

![](../.gitbook/assets/image%20%2833%29.png)

