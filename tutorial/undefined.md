# Python을 이용한 선형회귀분석 구현\(Sklearn,Numpy\) \(1\)

## 요구사

> #### \(1\) 전처리와 시각화”의 데이터로 선형회귀에 필요한 EDA 및 전처리 및 인코딩

* 범주형 변수 인코딩 3개 이상, EDA 3개 이상 
* 1주차 과제 적극적으로 활용 가능 및 자신이 만든 Feature 역시 사용 가능

> #### \(2\) 그 데이터를 다양한 방식으로 선형회귀분석 하기

> #### \(3\) 데이터의 행렬을 통해 구하여 위의 값과 비교

* Sklearn 없이 행렬 연산으로만 구



```python
import pandas as pd
```

```python
#데이터 불러오기 및 첫 데이터 모든 column 보기
data = pd.read_csv('C:/Temp/Auction_master_train.csv')
df=data.copy()  #혹시 몰라서 원데이터는 data란 변수에 남겨놓았다.

pd.set_option('display.max_columns', 500) #데이터 column을 500개 display하라는 함수(모든 column을 보기 위해 임의로 500개로 지정한 것이다.)
```

## 1. Scatter plot EDA

```python
#그래프 그리는데 필요한 패키지 import
import matplotlib.pyplot as plt
import seaborn as sns  #이쁜 sns plot을 위해 import

%matplotlib inline
```

```python
sns.pairplot(data=df, vars=['Total_land_auction_area','Total_building_area',
                            'Total_appraisal_price', 'Hammer_price']) 
plt.show()
```

![](../.gitbook/assets/image%20%2867%29.png)

```python
#outlier처럼 보이는 점 찾기
Index_label = df.query('Total_land_auction_area > 2000').index.tolist() 
print(Index_label)
```

Output : \[1521\]

```python
df.iloc[1521]
```

```text
Auction_key                                                                   10
Auction_class                                                                 강제
Bid_class                                                                     일괄
Claim_price                                                           8955865567
Appraisal_company                                                           대신감정
Appraisal_date                                               2015-05-21 00:00:00
Auction_count                                                                  4
Auction_miscarriage_count                                                      3
Total_land_gross_area                                                          0
Total_land_real_area                                                     2665.84
Total_land_auction_area                                                  2665.84
Total_building_area                                                      4255.07
Total_building_auction_area                                              4255.07
Total_appraisal_price                                                27775000000
Minimum_sales_price                                                  14220800000
First_auction_date                                           2016-04-26 00:00:00
Final_auction_date                                           2016-12-28 00:00:00
Final_result                                                                  낙찰
Creditor                                                                  엘빈종합건설
addr_do                                                                       서울
addr_si                                                                      관악구
addr_dong                                                                    남현동
addr_li                                                                      NaN
addr_san                                                                       N
addr_bunji1                                                                 1079
addr_bunji2                                                                   13
addr_etc                                                   외11 하이파크 101동 1층 101호
Apartment_usage                                                              아파트
Preserve_regist_date                                         1111-11-11 00:00:00
Total_floor                                                                   10
Current_floor                                                                  1
Specific                       *사당초등학교남동측인근\n*부근중소규모아파트단지,다세대주택등공동주택,단독주택,근린생...
Share_auction_YorN                                                             N
road_name                                                                    남현길
road_bunji1                                                                   96
road_bunji2                                                                  NaN
Close_date                                                   2017-04-14 00:00:00
Close_result                                                                  배당
point.y                                                                   37.473
point.x                                                                  126.975
Hammer_price                                                         15151000000
Name: 1521, dtype: object
```

보아하니 area와 price 간에 상관관계가 있음을 알 수 있다.\(영역이 크면 당연히 값이 늘어나긴한다..\) outlier처럼 보이는 값이 존재하는데 이는 1521번째 값임을 알 수 있다. 서울 관악구에 지어진 건물이라는데 값이 너무 큰 것으로 보아 측정에 오류가 있던 것으로 추정된다. 이를 제거하고 출력해본다.

```python
df=df.drop(df.index[1521])
sns.pairplot(data=df, vars=['Total_land_auction_area','Total_building_area',
                            'Total_appraisal_price', 'Hammer_price']) 
plt.show()
```

![](../.gitbook/assets/image%20%2868%29.png)

이제 조금 그래프다운 형태를 띠나 여전히 outlier 의심점이 보인다. 이를 또 찾아보도록 하자.

```python
#두번째 outlier 의심 점 찾기
Index_label = df.query('Total_land_auction_area > 400').index.tolist() 
print(Index_label) 
```

Output : \[1212\]

```python
df.iloc[1212]
```

```text
Auction_key                                                                 1437
Auction_class                                                                 임의
Bid_class                                                                     일괄
Claim_price                                                            600000000
Appraisal_company                                                           영현감정
Appraisal_date                                               2016-07-12 00:00:00
Auction_count                                                                  5
Auction_miscarriage_count                                                      3
Total_land_gross_area                                                          0
Total_land_real_area                                                       603.2
Total_land_auction_area                                                    603.2
Total_building_area                                                      1203.76
Total_building_auction_area                                              1203.76
Total_appraisal_price                                                 5810000000
Minimum_sales_price                                                   2974720000
First_auction_date                                           2016-12-19 00:00:00
Final_auction_date                                           2017-07-17 00:00:00
Final_result                                                                  낙찰
Creditor                                                                 Private
addr_do                                                                       서울
addr_si                                                                      강북구
addr_dong                                                                     번동
addr_li                                                                      NaN
addr_san                                                                       N
addr_bunji1                                                                  460
addr_bunji2                                                                   67
addr_etc                                                           번동모닝힐 1층 101호
Apartment_usage                                                              아파트
Preserve_regist_date                                         1111-11-11 00:00:00
Total_floor                                                                   10
Current_floor                                                                  1
Specific                       *근린생활시설및공동주택(도시형생활주택)\n*수송초등교서측인근\n*주위노변상가,중.소...
Share_auction_YorN                                                             N
road_name                                                                      0
road_bunji1                                                                  NaN
road_bunji2                                                                  NaN
Close_date                                                   2017-09-19 00:00:00
Close_result                                                                  배당
point.y                                                                  37.6373
point.x                                                                  127.031
Hammer_price                                                          3721000000
Name: 1212, dtype: object
```

이는 서울 강북구 번동의 아파트, Total building area는 약 1200평방미터로 나온다. outlier인지 아닌지 판단하기 위하여 주소 '시'의 평균을 보기로 한다.

```python
df.groupby('addr_si').mean()
```

|  Auction\_key | Claim\_price | Auction\_count | Auction\_miscarriage\_count | Total\_land\_gross\_area | Total\_land\_real\_area | Total\_land\_auction\_area | Total\_building\_area | Total\_building\_auction\_area | Total\_appraisal\_price | Minimum\_sales\_price | addr\_bunji1 | addr\_bunji2 | Total\_floor | Current\_floor | road\_bunji1 | road\_bunji2 | point.y | point.x | Hammer\_price |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| addr\_si |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 강남구 | 201.333333 | 8.315065e+08 | 1.794118 | 0.745098 | 41692.277059 | 58.097059 | 54.699020 | 129.649902 | 122.521176 | 1.448500e+09 | 1.210360e+09 | 436.911765 | 9.840909 | 17.303922 | 10.401961 | 129.617647 | 5.000000 | 37.503128 | 127.051402 | 1.399296e+09 |
| 강동구 | 523.512821 | 3.554551e+08 | 1.717949 | 0.717949 | 34967.487179 | 49.724872 | 49.059487 | 100.024359 | 98.572308 | 5.614872e+08 | 4.830585e+08 | 328.743590 | 6.294118 | 15.820513 | 8.358974 | 295.717949 | 10.000000 | 37.545619 | 127.143863 | 5.481503e+08 |
| 강북구 | 1496.714286 | 2.010612e+08 | 1.904762 | 0.857143 | 57663.609524 | 68.720952 | 67.734762 | 136.174762 | 134.985714 | 6.005476e+08 | 4.152057e+08 | 684.904762 | 39.000000 | 17.333333 | 9.571429 | 130.400000 | 3.000000 | 37.630671 | 127.024243 | 4.914191e+08 |
| 강서구 | 1299.871795 | 5.570787e+08 | 1.717949 | 0.679487 | 25424.229487 | 40.545769 | 39.831410 | 76.207821 | 74.926667 | 3.466603e+08 | 2.997796e+08 | 1172.051282 | 19.357143 | 12.794872 | 7.487179 | 89.610390 | 7.000000 | 37.077213 | 127.239927 | 3.400114e+08 |
| 관악구 | 194.857143 | 2.063611e+08 | 2.238095 | 1.095238 | 34628.861667 | 37.690476 | 36.850238 | 87.553095 | 85.854048 | 4.074048e+08 | 3.333106e+08 | 1332.642857 | 32.263158 | 15.547619 | 6.619048 | 150.428571 | 12.500000 | 37.477864 | 126.938094 | 3.820510e+08 |
| 광진구 | 518.521739 | 3.073419e+08 | 1.956522 | 0.956522 | 15470.369565 | 48.255217 | 48.255217 | 99.774348 | 99.774348 | 6.286522e+08 | 5.062035e+08 | 468.130435 | 54.000000 | 14.478261 | 7.086957 | 151.363636 | 2.500000 | 37.542878 | 127.090860 | 5.896738e+08 |
| 구로구 | 1092.596491 | 2.181164e+08 | 1.894737 | 0.877193 | 21730.326316 | 34.275789 | 32.581053 | 83.836316 | 80.343684 | 3.572682e+08 | 2.989674e+08 | 533.357143 | 53.655172 | 16.000000 | 9.035088 | 88.854545 | 9.666667 | 37.496498 | 126.871492 | 3.407228e+08 |
| 금정구 | 2068.827586 | 9.750046e+07 | 1.689655 | 0.620690 | 18515.200000 | 37.806897 | 36.791724 | 88.660000 | 87.241379 | 2.360552e+08 | 2.072538e+08 | 518.379310 | 21.578947 | 15.172414 | 7.793103 | 89.827586 | 4.500000 | 35.237879 | 129.089680 | 2.417927e+08 |
| 금천구 | 1086.200000 | 1.496667e+08 | 1.920000 | 0.920000 | 40223.364000 | 42.963200 | 42.963200 | 93.883600 | 93.883600 | 3.483200e+08 | 2.883920e+08 | 860.520000 | 11.533333 | 14.720000 | 7.920000 | 283.291667 | NaN | 37.455508 | 126.903505 | 3.283875e+08 |
| 기장군 | 2577.130435 | 1.575648e+08 | 2.086957 | 0.956522 | 36607.760870 | 50.761739 | 49.879130 | 88.088261 | 86.249565 | 2.406522e+08 | 1.993061e+08 | 652.652174 | 10.666667 | 16.478261 | 7.173913 | 50.130435 | NaN | 35.299639 | 129.190949 | 2.194524e+08 |
| 남구 | 2557.642857 | 1.665850e+08 | 1.517857 | 0.517857 | 73763.860714 | 40.572143 | 38.901607 | 96.198393 | 93.201429 | 2.739312e+08 | 2.458873e+08 | 459.642857 | 20.627907 | 20.250000 | 10.875000 | 75.285714 | 9.000000 | 35.128166 | 129.096394 | 2.747884e+08 |
| 노원구 | 1517.821705 | 1.957424e+08 | 1.658915 | 0.604651 | 45881.200000 | 38.480078 | 37.894496 | 72.954884 | 71.245659 | 3.298279e+08 | 2.865651e+08 | 568.203125 | 6.360000 | 14.573643 | 7.015504 | 177.558140 | 13.000000 | 37.646962 | 127.067697 | 3.247497e+08 |
| 도봉구 | 1567.970588 | 2.048912e+08 | 1.823529 | 0.794118 | 31795.495588 | 35.976912 | 34.422353 | 85.573235 | 82.335294 | 3.362794e+08 | 2.795235e+08 | 489.676471 | 15.083333 | 15.250000 | 7.161765 | 188.955224 | 14.230769 | 37.657717 | 127.038664 | 3.191248e+08 |
| 동구 | 2144.133333 | 9.165923e+07 | 1.933333 | 0.933333 | 4592.873333 | 30.084667 | 30.084667 | 75.495333 | 74.282000 | 1.702667e+08 | 1.400955e+08 | 816.000000 | 18.500000 | 13.600000 | 6.600000 | 57.733333 | 1.000000 | 35.129565 | 129.048839 | 1.613853e+08 |
| 동대문구 | 1537.528302 | 2.109945e+08 | 1.716981 | 0.716981 | 30361.903774 | 40.425660 | 39.531698 | 95.107170 | 92.544528 | 4.639358e+08 | 3.991042e+08 | 394.094340 | 9.789474 | 18.245283 | 9.547170 | 81.679245 | 7.750000 | 37.579911 | 127.062228 | 4.528658e+08 |
| 동래구 | 1980.431373 | 5.206533e+08 | 3.137255 | 2.000000 | 11937.541176 | 28.005294 | 27.438824 | 85.868431 | 84.916667 | 2.429295e+08 | 1.873124e+08 | 436.000000 | 44.100000 | 16.294118 | 9.372549 | 55.833333 | NaN | 35.205389 | 129.081877 | 2.095611e+08 |
| 동작구 | 235.054054 | 4.359083e+08 | 1.702703 | 0.675676 | 118613.708108 | 38.807568 | 37.560000 | 103.611622 | 100.875676 | 5.890676e+08 | 5.060314e+08 | 500.459459 | 39.625000 | 20.108108 | 10.459459 | 58.638889 | 32.000000 | 37.497308 | 126.948634 | 5.695185e+08 |
| 마포구 | 760.915493 | 2.046736e+08 | 1.549296 | 0.521127 | 26272.074648 | 36.223099 | 35.249577 | 88.201972 | 86.416338 | 5.604366e+08 | 4.892924e+08 | 441.070423 | 33.205882 | 12.676056 | 6.647887 | 150.788732 | 11.500000 | 37.553838 | 126.940071 | 5.501279e+08 |
| 부산진구 | 2127.357143 | 1.672483e+08 | 1.654762 | 0.642857 | 18001.575000 | 26.345833 | 25.768452 | 83.771310 | 81.627857 | 2.547976e+08 | 2.221148e+08 | 483.880952 | 13.034483 | 23.988095 | 12.250000 | 157.178571 | 6.000000 | 35.161446 | 129.049310 | 2.483445e+08 |
| 북구 | 2173.471698 | 1.520414e+08 | 1.603774 | 0.566038 | 65846.461509 | 37.301887 | 35.884340 | 76.286226 | 73.421698 | 2.178868e+08 | 1.953328e+08 | 953.377358 | 5.379310 | 18.981132 | 10.018868 | 116.924528 | 19.500000 | 35.221694 | 129.017753 | 2.161426e+08 |
| 사상구 | 2225.982759 | 1.293041e+08 | 1.931034 | 0.896552 | 31633.053448 | 37.450690 | 37.040172 | 80.144138 | 79.411897 | 1.926724e+08 | 1.586710e+08 | 432.137931 | 13.894737 | 19.051724 | 11.672414 | 124.775862 | 22.307692 | 35.149124 | 128.991819 | 1.842684e+08 |
| 사하구 | 2221.541667 | 1.124876e+08 | 1.979167 | 0.916667 | 41849.603125 | 41.296979 | 41.040208 | 83.443542 | 82.758125 | 1.804333e+08 | 1.478367e+08 | 532.281250 | 12.263158 | 17.656250 | 8.093750 | 145.895833 | 3.000000 | 35.085195 | 128.979449 | 1.666656e+08 |
| 서구 | 2283.823529 | 2.008066e+08 | 2.058824 | 1.000000 | 6988.811765 | 40.782941 | 40.782941 | 91.310588 | 91.310588 | 2.219412e+08 | 1.806075e+08 | 241.647059 | 19.615385 | 14.588235 | 6.235294 | 94.000000 | 12.000000 | 35.090394 | 129.018130 | 2.065203e+08 |
| 서대문구 | 783.378378 | 2.217121e+08 | 1.891892 | 0.837838 | 17057.721622 | 42.075676 | 41.206486 | 98.344595 | 96.632432 | 4.082446e+08 | 3.324879e+08 | 317.891892 | 65.166667 | 13.675676 | 6.297297 | 99.081081 | 13.111111 | 37.581790 | 126.939526 | 3.798891e+08 |
| 서초구 | 183.356164 | 9.050558e+08 | 1.876712 | 0.835616 | 21191.920548 | 62.369041 | 60.195890 | 144.772192 | 140.198082 | 1.208144e+09 | 1.004236e+09 | 960.563380 | 15.673077 | 14.219178 | 7.342466 | 76.232877 | 14.125000 | 37.488934 | 127.014325 | 1.135984e+09 |
| 성동구 | 506.827586 | 4.266443e+08 | 1.448276 | 0.413793 | 30121.396552 | 40.775517 | 40.775517 | 92.398276 | 92.398276 | 5.505517e+08 | 5.026897e+08 | 444.827586 | 95.000000 | 17.620690 | 7.862069 | 123.896552 | 31.000000 | 37.552214 | 127.037199 | 5.849723e+08 |
| 성북구 | 1427.822222 | 2.124852e+08 | 1.777778 | 0.733333 | 34397.335556 | 37.246667 | 35.955111 | 93.304000 | 89.957111 | 4.093556e+08 | 3.484569e+08 | 472.711111 | 7.416667 | 18.355556 | 9.288889 | 182.533333 | 15.000000 | 37.599696 | 127.021874 | 3.994318e+08 |
| 송파구 | 518.883333 | 4.967428e+08 | 1.700000 | 0.616667 | 72965.065000 | 51.162500 | 48.619167 | 112.228167 | 107.484333 | 8.050500e+08 | 7.095367e+08 | 170.950000 | 8.000000 | 17.766667 | 8.833333 | 123.583333 | 9.500000 | 37.504760 | 127.116930 | 8.148680e+08 |
| 수영구 | 2548.515152 | 2.344929e+08 | 1.787879 | 0.636364 | 12873.636364 | 32.580909 | 32.580909 | 91.670909 | 91.670909 | 3.202695e+08 | 2.765762e+08 | 407.878788 | 14.730769 | 18.212121 | 8.818182 | 170.818182 | 8.857143 | 35.159782 | 129.115140 | 3.155445e+08 |
| 양천구 | 1085.941176 | 1.290913e+09 | 1.720588 | 0.676471 | 46748.208824 | 46.806176 | 46.076176 | 93.092059 | 91.155000 | 5.676368e+08 | 5.060369e+08 | 782.088235 | 6.370370 | 15.205882 | 7.602941 | 108.191176 | 22.400000 | 37.524088 | 126.860318 | 5.725753e+08 |
| 연제구 | 2086.967742 | 1.679917e+08 | 1.483871 | 0.451613 | 20028.709677 | 33.820323 | 32.742903 | 95.305806 | 93.627097 | 2.735710e+08 | 2.523039e+08 | 981.935484 | 20.678571 | 18.838710 | 9.516129 | 71.870968 | 10.000000 | 35.183980 | 129.086470 | 2.779615e+08 |
| 영도구 | 2152.344828 | 1.567142e+08 | 2.241379 | 1.206897 | 21137.058621 | 35.642759 | 34.477931 | 82.673103 | 80.311379 | 1.987931e+08 | 1.508135e+08 | 398.241379 | 56.823529 | 14.724138 | 8.172414 | 121.448276 | NaN | 35.083964 | 129.059641 | 1.737148e+08 |
| 영등포구 | 1086.000000 | 1.324188e+09 | 1.872340 | 0.808511 | 23452.095745 | 47.427660 | 45.588085 | 116.111277 | 112.061064 | 7.053809e+08 | 5.929188e+08 | 597.829787 | 3.636364 | 19.468085 | 8.851064 | 134.851064 | 8.500000 | 37.518085 | 126.907750 | 6.793869e+08 |
| 용산구 | 771.555556 | 4.795674e+08 | 1.703704 | 0.703704 | 20592.629630 | 46.136296 | 45.047778 | 121.418889 | 115.571481 | 9.344074e+08 | 8.203644e+08 | 267.333333 | 88.214286 | 18.037037 | 10.148148 | 86.925926 | 20.000000 | 37.528343 | 126.973528 | 9.347509e+08 |
| 은평구 | 759.490566 | 1.892046e+08 | 2.000000 | 0.924528 | 13850.000000 | 34.239623 | 34.239623 | 82.244151 | 82.244151 | 3.618302e+08 | 2.934833e+08 | 302.433962 | 20.640000 | 12.056604 | 5.471698 | 131.000000 | 14.437500 | 37.605664 | 126.919464 | 3.362579e+08 |
| 종로구 | 231.705882 | 3.464952e+08 | 2.176471 | 1.176471 | 17176.235294 | 46.985882 | 45.971176 | 111.871176 | 107.962353 | 6.934706e+08 | 5.233671e+08 | 226.294118 | 8.500000 | 12.882353 | 5.941176 | 116.764706 | 10.500000 | 37.586239 | 126.979263 | 5.984927e+08 |
| 중구 | 962.000000 | 2.969734e+08 | 2.366667 | 1.200000 | 35864.533333 | 27.561667 | 26.355333 | 94.842000 | 91.109667 | 5.276333e+08 | 4.018757e+08 | 545.900000 | 45.888889 | 19.600000 | 9.133333 | 102.500000 | NaN | 36.577890 | 127.811950 | 4.544756e+08 |
| 중랑구 | 1544.324324 | 1.735280e+08 | 1.594595 | 0.594595 | 16380.967568 | 29.984054 | 29.421081 | 79.265946 | 78.384865 | 3.370297e+08 | 2.957265e+08 | 618.378378 | 17.250000 | 15.864865 | 7.189189 | 185.216216 | 23.666667 | 37.598995 | 127.089161 | 3.267374e+08 |
| 해운대구 | 2559.247191 | 2.525467e+08 | 1.876404 | 0.820225 | 30408.001124 | 40.169213 | 40.169213 | 97.062809 | 97.062809 | 4.174944e+08 | 3.495367e+08 | 1268.022472 | 49.750000 | 24.617978 | 15.258427 | 80.147727 | 10.000000 | 35.182566 | 129.148052 | 3.918034e+08 |

거의 모든 지역의 경매 지역 면적 평균이 60제곱미터 이하인데 1200 제곱미터 정도로 나온 것을 보아 outlier로 판단해야 할 것 같다. google maps에 \(37.6373, 127.031\)지역 좌표를 찍어보면 건물이 집합된 장소가 나오며 1200제곱미터에 해당하는 면적을 출력해보니 생각보다 크지 않아 아마 건물 한개가 아닌 단지의 면적을 측정한 것이 아닌가싶다. 다른 측정값들과 측정 방법이 다른 것으로 의심되므로 outlier로 지정, 이를 제거하고 이후 분석을 진행한다.

```python
df=df.drop(df.index[1212])
sns.pairplot(data=df, vars=['Total_land_auction_area','Total_building_area',
                            'Total_appraisal_price', 'Hammer_price']) 
plt.show()
```

![](../.gitbook/assets/image%20%2828%29.png)

훨씬 깔끔한 scatter plot이 출력된다.

## 2. Histogram EDA

위의 pairplot을 보면 Price 관련 변수들이 왼쪽으로 치우침을 알 수 있다. 이를 수정하기 위해 변수 변환을 시도하겠다. 우선 다음과 같이 Hammer price에 관한 요약 설명을 보면 범위가 매우 큼을 알 수 있다.

### 표준화

```python
df['Hammer_price'].describe()
#최소가 6303000, 최대가 4863000000로 범위도 크고 값도 큼을 알 수 있으므로 표준화를 하겠다.
```

```text
count    1.931000e+03
mean     4.634065e+08
std      4.403236e+08
min      6.303000e+06
25%      1.972775e+08
50%      3.535000e+08
75%      5.564095e+08
max      4.863000e+09
Name: Hammer_price, dtype: float64
```

```python
#price를 표준화 하는 김에 area도 표준화시켜보도록 한다.
#object가 있는 dataframe이라 전체를 정규화할 수 없어 area와 price 변수를 따로 df2에 저장하였다.

df2=df.iloc[:,9:15]
df2['Hammer_price']=df['Hammer_price']
df2.head()
```

|  | Total\_land\_real\_area | Total\_land\_auction\_area | Total\_building\_area | Total\_building\_auction\_area | Total\_appraisal\_price | Minimum\_sales\_price | Hammer\_price |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 37.35 | 37.35 | 181.77 | 181.77 | 836000000 | 668800000 | 760000000 |
| 1 | 18.76 | 18.76 | 118.38 | 118.38 | 1073000000 | 858400000 | 971889999 |
| 2 | 71.00 | 71.00 | 49.94 | 49.94 | 119000000 | 76160000 | 93399999 |
| 3 | 32.98 | 32.98 | 84.91 | 84.91 | 288400000 | 230720000 | 256899000 |
| 4 | 45.18 | 45.18 | 84.96 | 84.96 | 170000000 | 136000000 | 158660000 |

```python
##정규화##
from sklearn.preprocessing import MinMaxScaler   #MinMaxScaler import
scaler=MinMaxScaler() #scaler라는 변수로 간편하게 설정가능하도록 입력


scaled=scaler.fit_transform(df2)   #정규화
df2_s = pd.DataFrame(scaled, index=df2.index, columns=df2.columns)
df2_s.head()

#https://soo-jjeong.tistory.com/122   ##정규화
#정규화된 결과값(array형태)를 원래의 dataframe 형태로 변환
#https://stackoverflow.com/questions/35723472/how-to-use-sklearn-fit-transform-with-pandas-and-return-dataframe-instead-of-nu
```

|  | Total\_land\_real\_area | Total\_land\_auction\_area | Total\_building\_area | Total\_building\_auction\_area | Total\_appraisal\_price | Minimum\_sales\_price | Hammer\_price |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 0.171330 | 0.171330 | 0.590181 | 0.600960 | 0.149435 | 0.149272 | 0.155187 |
| 1 | 0.086055 | 0.086055 | 0.373151 | 0.389639 | 0.192018 | 0.191862 | 0.198816 |
| 2 | 0.325688 | 0.325688 | 0.138832 | 0.161483 | 0.020611 | 0.016145 | 0.017933 |
| 3 | 0.151284 | 0.151284 | 0.258559 | 0.278061 | 0.051047 | 0.050865 | 0.051598 |
| 4 | 0.207248 | 0.207248 | 0.258730 | 0.278228 | 0.029774 | 0.029587 | 0.031370 |

MinMax Scaler는 \(x - min\)/\(max-min\) 의 값을 나타낸 것으로 데이터를 0과 1사이의 범위로 맞춰 scaling하는 것이다. scaling의 다른 방식인 standard scaler는 평균=0, 분산=1로 scaling 하는 것이며 from sklearn.preprocessing import StandardScaler 로 import 가능하다.

```python
plt.rcParams['figure.figsize'] = (14.0, 8.0)
df2_s.plot.hist(subplots=True, legend=True, layout=(3, 3))

#figure size 조정
#https://harangdev.github.io/applied-data-science-with-python/applied-data-plotting-in-python/3/
```

![](../.gitbook/assets/image%20%2876%29.png)

위와같이 scaling은 히스토그램에 영향이 없다. 변수 변환을 통해 히스토그램을 좀 더 이쁘게 변환해보겠다.

### 변수 변환 - log를 취하여 그래프 변환

```python
df2_s.head()
```

| Total\_land\_real\_area | Total\_land\_auction\_area | Total\_building\_area | Total\_building\_auction\_area | Total\_appraisal\_price | Minimum\_sales\_price | Hammer\_price |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 0.171330 | 0.171330 | 0.590181 | 0.600960 | 0.149435 | 0.149272 | 0.155187 |
| 1 | 0.086055 | 0.086055 | 0.373151 | 0.389639 | 0.192018 | 0.191862 | 0.198816 |
| 2 | 0.325688 | 0.325688 | 0.138832 | 0.161483 | 0.020611 | 0.016145 | 0.017933 |
| 3 | 0.151284 | 0.151284 | 0.258559 | 0.278061 | 0.051047 | 0.050865 | 0.051598 |
| 4 | 0.207248 | 0.207248 | 0.258730 | 0.278228 | 0.029774 | 0.029587 | 0.031370 |

```python
import numpy as np

df2_s['Total_appraisal_price']=np.log(df2_s['Total_appraisal_price'])
df2_s['Minimum_sales_price']=np.log(df2_s['Minimum_sales_price'])
df2_s['Hammer_price']=np.log(df2_s['Hammer_price'])

df2_s.info()
```

```text
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1931 entries, 0 to 1932
Data columns (total 7 columns):
Total_land_real_area           1931 non-null float64
Total_land_auction_area        1931 non-null float64
Total_building_area            1931 non-null float64
Total_building_auction_area    1931 non-null float64
Total_appraisal_price          1931 non-null float64
Minimum_sales_price            1931 non-null float64
Hammer_price                   1931 non-null float64
dtypes: float64(7)
memory usage: 200.7 KB
```

```python
#Inf, -Inf를 NaN으로 처리 후 제거하기
import numpy as np
df2_s=df2_s.replace([-np.inf,np.inf], np.nan)
df2_s=df2_s.dropna(axis=0)
df2_s.info()

#-Inf 값이 나와 제거를 하니 2 줄이 제거됨을 알 수 있다.
```

```text
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1929 entries, 0 to 1932
Data columns (total 7 columns):
Total_land_real_area           1929 non-null float64
Total_land_auction_area        1929 non-null float64
Total_building_area            1929 non-null float64
Total_building_auction_area    1929 non-null float64
Total_appraisal_price          1929 non-null float64
Minimum_sales_price            1929 non-null float64
Hammer_price                   1929 non-null float64
dtypes: float64(7)
memory usage: 120.6 KB
```

```python
df2_s.plot.hist(subplots=True, legend=True, layout=(3, 3), range=(-8,4), bins=20)

#plot.hist의 다른 조건들 참조
#https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html
```

![](../.gitbook/assets/image%20%2869%29.png)

xlabel의 범위가 일치하지 않아 price 그래프는 이쁘게 나왔으나 area 그래프는 한곳에 몰려있어 보임을 알 수 있다. 이는 다음에 나타낼 subplot 개별 지정을 통해 해결할 수 있다.

```python
import matplotlib.pyplot as plt
from matplotlib import gridspec

fig = plt.figure(figsize=(5, 10)) 
plt.tight_layout()
fig.subplots_adjust(bottom=0.5)

f,ax=plt.subplots(6,2) # 5개의 Figure

#각 subplot 크기를 조정하는 방법
gs = gridspec.GridSpec(nrows=2, # row 몇 개 
                       ncols=3, # col 몇 개 
                       height_ratios=[1, 1], 
                       width_ratios=[2,2,2]
                      )

###첫째줄-price plot
ax[0] = plt.subplot(gs[0])
ax[0] = sns.distplot(df2_s['Total_appraisal_price'])

ax[1] = plt.subplot(gs[1])
ax[1] = sns.distplot(df2_s['Minimum_sales_price'])

ax[2] = plt.subplot(gs[2])
ax[2] = sns.distplot(df2_s['Hammer_price'])


##둘째줄 - area plot
ax[3] = plt.subplot(gs[3])
ax[3] = sns.distplot(df2_s['Total_land_auction_area'])

ax[4] = plt.subplot(gs[4])
ax[4] = sns.distplot(df2_s['Total_building_auction_area'])

ax[5] = plt.subplot(gs[5])
ax[5] = sns.distplot(df2_s['Total_building_area'])

plt.tight_layout()
plt.show()
```

![](../.gitbook/assets/image%20%2861%29.png)

이제 그래프들이 대략적인 정규분포 모양을 따름을 알 수 있다. 다만 Total building auction area가 특정 값에 몰려있게 됨을 알 수 있는데 이는 건물이 지어지는 지역에서의 면적의 한계점이 존재한다고 파악할 수 있을 것 같다. Total building auction area와 Total building area는 거의 일치하나 경매에 사용된 빌딩의 면적이 좀 더 특정 값에 몰리는 것을 보아 경매에 선호되는 면적이 있거나, 범위를 내리거나 올려서 특정 값을 맞춘 것이 아닐까싶다.

## 3. 범주형 변수 인코

### 3-1. 날짜 인코딩 - 과제 1 참조

```python
#아까 제외한 행들이 있으므로 index 다시 지정
df=df.reset_index()
del df['index']
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html


#날짜 뒤에 시간은 필요없으므로 제거한다.
df["First_auction_date"]= df["First_auction_date"].str.split(" ", n = 1, expand = True)
df["Final_auction_date"]= df["Final_auction_date"].str.split(" ", n = 1, expand = True)

#https://www.geeksforgeeks.org/python-pandas-split-strings-into-two-list-columns-using-str-split/


#요일 출력을 위해 ' '로 분리하고 datetime.strftime에 차례로 입력하여 요일을 출력해보도록한다.
from datetime import datetime
import numpy as np


#-를 기준으로 split
a_s=df["First_auction_date"].str.split("-",expand = True)    
a_e=df["Final_auction_date"].str.split("-", expand = True)


#split 결과가 object 형태로 반환되어 numeric으로 바꾸기
a_s[0]=pd.to_numeric(a_s[0]) 
a_s[1]=pd.to_numeric(a_s[1])
a_s[2]=pd.to_numeric(a_s[2])

a_e[0]=pd.to_numeric(a_e[0])
a_e[1]=pd.to_numeric(a_e[1])
a_e[2]=pd.to_numeric(a_e[2])


#요일 출력 변수 생성 및 초기값 지정
df['auction_start_weekday']=0
df['auction_end_weekday']=0


#요일 기록
for i in range(0,len(df)):
    df['auction_start_weekday'][i]=datetime(a_s[0][i], a_s[1][i], a_s[2][i]).weekday()
    df['auction_end_weekday'][i]=datetime(a_e[0][i],a_e[1][i],a_e[2][i]).weekday()

    
#https://stackoverflow.com/questions/9847213/how-do-i-get-the-day-of-week-given-a-date   ##monday-sunday로 글자로 기록
#https://docs.python.org/2/library/datetime.html   ##숫자로 기록, 0=Monday
```

```python
df.head() #변수가 잘 생성되었는지 확인
```

|  Auction\_key | Auction\_class | Bid\_class | Claim\_price | Appraisal\_company | Appraisal\_date | Auction\_count | Auction\_miscarriage\_count | Total\_land\_gross\_area | Total\_land\_real\_area | Total\_land\_auction\_area | Total\_building\_area | Total\_building\_auction\_area | Total\_appraisal\_price | Minimum\_sales\_price | First\_auction\_date | Final\_auction\_date | Final\_result | Creditor | addr\_do | addr\_si | addr\_dong | addr\_li | addr\_san | addr\_bunji1 | addr\_bunji2 | addr\_etc | Apartment\_usage | Preserve\_regist\_date | Total\_floor | Current\_floor | Specific | Share\_auction\_YorN | road\_name | road\_bunji1 | road\_bunji2 | Close\_date | Close\_result | point.y | point.x | Hammer\_price | auction\_start\_weekday | auction\_end\_weekday |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 2687 | 임의 | 개별 | 1766037301 | 정명감정 | 2017-07-26 00:00:00 | 2 | 1 | 12592.0 | 37.35 | 37.35 | 181.77 | 181.77 | 836000000 | 668800000 | 2018-02-13 | 2018-03-20 | 낙찰 | 베리타스자산관리대부 | 부산 | 해운대구 | 우동 | NaN | N | 1398.0 | NaN | 해운대엑소디움 5층 101-502호 | 주상복합 | 2009-07-14 00:00:00 | 45 | 5 | NaN | N | 해운대해변로 | 30.0 | NaN | 2018-06-14 00:00:00 | 배당 | 35.162717 | 129.137048 | 760000000 | 1 | 1 |
| 1 | 2577 | 임의 | 일반 | 152946867 | 희감정 | 2016-09-12 00:00:00 | 2 | 1 | 42478.1 | 18.76 | 18.76 | 118.38 | 118.38 | 1073000000 | 858400000 | 2016-12-29 | 2017-02-02 | 낙찰 | 흥국저축은행 | 부산 | 해운대구 | 우동 | NaN | N | 1407.0 | NaN | 해운대두산위브더제니스 103동 51층 5103호 | 아파트 | 2011-12-16 00:00:00 | 70 | 51 | NaN | N | 마린시티2로 | 33.0 | NaN | 2017-03-30 00:00:00 | 배당 | 35.156633 | 129.145068 | 971889999 | 3 | 3 |
| 2 | 2197 | 임의 | 개별 | 11326510 | 혜림감정 | 2016-11-22 00:00:00 | 3 | 2 | 149683.1 | 71.00 | 71.00 | 49.94 | 49.94 | 119000000 | 76160000 | 2017-07-28 | 2017-10-13 | 낙찰 | 국민은행 | 부산 | 사상구 | 모라동 | NaN | N | 552.0 | NaN | 백양그린 206동 14층 1403호 | 아파트 | 1992-07-31 00:00:00 | 15 | 14 | NaN | N | 모라로110번길 | 88.0 | NaN | 2017-12-13 00:00:00 | 배당 | 35.184601 | 128.996765 | 93399999 | 4 | 4 |
| 3 | 2642 | 임의 | 일반 | 183581724 | 신라감정 | 2016-12-13 00:00:00 | 2 | 1 | 24405.0 | 32.98 | 32.98 | 84.91 | 84.91 | 288400000 | 230720000 | 2017-07-20 | 2017-11-02 | 낙찰 | 고려저축은행 | 부산 | 남구 | 대연동 | NaN | N | 243.0 | 23.0 | 대연청구 109동 11층 1102호 | 아파트 | 2001-07-13 00:00:00 | 20 | 11 | NaN | N | 황령대로319번가길 | 110.0 | NaN | 2017-12-27 00:00:00 | 배당 | 35.154180 | 129.089081 | 256899000 | 3 | 3 |
| 4 | 1958 | 강제 | 일반 | 45887671 | 나라감정 | 2016-03-07 00:00:00 | 2 | 1 | 774.0 | 45.18 | 45.18 | 84.96 | 84.96 | 170000000 | 136000000 | 2016-07-06 | 2016-08-03 | 낙찰 | Private | 부산 | 사하구 | 괴정동 | NaN | N | 399.0 | 2.0 | 동조리젠시 7층 703호 | 아파트 | 2001-11-27 00:00:00 | 7 | 7 | NaN | N | 오작로 | 51.0 | NaN | 2016-10-04 00:00:00 | 배당 | 35.099630 | 128.998874 | 158660000 | 2 | 2 |

날짜를 Monday=0에서 Sunday=7으로 인코딩하였다. 추가 분석을 위해 경매 기간 변수도 추가하겠다.



```python
#초기값 지정
df['auction_start_day']=0
df['auction_end_day']=0


for i in range(0,len(df)):
    #날짜 값들을 date형식으로 남기기
    df['auction_start_day'][i]=datetime(a_s[0][i], a_s[1][i], a_s[2][i])
    df['auction_end_day'][i]=datetime(a_e[0][i], a_e[1][i], a_e[2][i])
    

#초기값 지정
df['auction_day_length']=0
for i in range(0,len(df)):
    df['auction_day_length'][i]=(df['auction_end_day'][i]-df['auction_start_day'][i]).days
    

#https://stackoverflow.com/questions/151199/how-to-calculate-number-of-days-between-two-given-dates
```

```python
df.head()
```

|  | Auction\_key | Auction\_class | Bid\_class | Claim\_price | Appraisal\_company | Appraisal\_date | Auction\_count | Auction\_miscarriage\_count | Total\_land\_gross\_area | Total\_land\_real\_area | Total\_land\_auction\_area | Total\_building\_area | Total\_building\_auction\_area | Total\_appraisal\_price | Minimum\_sales\_price | First\_auction\_date | Final\_auction\_date | Final\_result | Creditor | addr\_do | addr\_si | addr\_dong | addr\_li | addr\_san | addr\_bunji1 | addr\_bunji2 | addr\_etc | Apartment\_usage | Preserve\_regist\_date | Total\_floor | Current\_floor | Specific | Share\_auction\_YorN | road\_name | road\_bunji1 | road\_bunji2 | Close\_date | Close\_result | point.y | point.x | Hammer\_price | auction\_start\_weekday | auction\_end\_weekday | auction\_start\_day | auction\_end\_day | auction\_day\_length |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 2687 | 임의 | 개별 | 1766037301 | 정명감정 | 2017-07-26 00:00:00 | 2 | 1 | 12592.0 | 37.35 | 37.35 | 181.77 | 181.77 | 836000000 | 668800000 | 2018-02-13 | 2018-03-20 | 낙찰 | 베리타스자산관리대부 | 부산 | 해운대구 | 우동 | NaN | N | 1398.0 | NaN | 해운대엑소디움 5층 101-502호 | 주상복합 | 2009-07-14 00:00:00 | 45 | 5 | NaN | N | 해운대해변로 | 30.0 | NaN | 2018-06-14 00:00:00 | 배당 | 35.162717 | 129.137048 | 760000000 | 1 | 1 | 2018-02-13 00:00:00 | 2018-03-20 00:00:00 | 35 |
| 1 | 2577 | 임의 | 일반 | 152946867 | 희감정 | 2016-09-12 00:00:00 | 2 | 1 | 42478.1 | 18.76 | 18.76 | 118.38 | 118.38 | 1073000000 | 858400000 | 2016-12-29 | 2017-02-02 | 낙찰 | 흥국저축은행 | 부산 | 해운대구 | 우동 | NaN | N | 1407.0 | NaN | 해운대두산위브더제니스 103동 51층 5103호 | 아파트 | 2011-12-16 00:00:00 | 70 | 51 | NaN | N | 마린시티2로 | 33.0 | NaN | 2017-03-30 00:00:00 | 배당 | 35.156633 | 129.145068 | 971889999 | 3 | 3 | 2016-12-29 00:00:00 | 2017-02-02 00:00:00 | 35 |
| 2 | 2197 | 임의 | 개별 | 11326510 | 혜림감정 | 2016-11-22 00:00:00 | 3 | 2 | 149683.1 | 71.00 | 71.00 | 49.94 | 49.94 | 119000000 | 76160000 | 2017-07-28 | 2017-10-13 | 낙찰 | 국민은행 | 부산 | 사상구 | 모라동 | NaN | N | 552.0 | NaN | 백양그린 206동 14층 1403호 | 아파트 | 1992-07-31 00:00:00 | 15 | 14 | NaN | N | 모라로110번길 | 88.0 | NaN | 2017-12-13 00:00:00 | 배당 | 35.184601 | 128.996765 | 93399999 | 4 | 4 | 2017-07-28 00:00:00 | 2017-10-13 00:00:00 | 77 |
| 3 | 2642 | 임의 | 일반 | 183581724 | 신라감정 | 2016-12-13 00:00:00 | 2 | 1 | 24405.0 | 32.98 | 32.98 | 84.91 | 84.91 | 288400000 | 230720000 | 2017-07-20 | 2017-11-02 | 낙찰 | 고려저축은행 | 부산 | 남구 | 대연동 | NaN | N | 243.0 | 23.0 | 대연청구 109동 11층 1102호 | 아파트 | 2001-07-13 00:00:00 | 20 | 11 | NaN | N | 황령대로319번가길 | 110.0 | NaN | 2017-12-27 00:00:00 | 배당 | 35.154180 | 129.089081 | 256899000 | 3 | 3 | 2017-07-20 00:00:00 | 2017-11-02 00:00:00 | 105 |
| 4 | 1958 | 강제 | 일반 | 45887671 | 나라감정 | 2016-03-07 00:00:00 | 2 | 1 | 774.0 | 45.18 | 45.18 | 84.96 | 84.96 | 170000000 | 136000000 | 2016-07-06 | 2016-08-03 | 낙찰 | Private | 부산 | 사하구 | 괴정동 | NaN | N | 399.0 | 2.0 | 동조리젠시 7층 703호 | 아파트 | 2001-11-27 00:00:00 | 7 | 7 | NaN | N | 오작로 | 51.0 | NaN | 2016-10-04 00:00:00 | 배당 | 35.099630 | 128.998874 | 158660000 | 2 | 2 | 2016-07-06 00:00:00 | 2016-08-03 00:00:00 | 28 |

```python
#이제 분석에 필요없는 변수이므로 제거 (이후 데이터프레임 형성 편의를 위한 과정)
del df['auction_start_day']
del df['auction_end_day']
```

### 3-2 Bid Class 인코딩

```python
#Bid_class를 숫자로 변환하겠다.
df['Bid']=0
for i in range(0,len(df)):
        if df['Bid_class'][i]=='일반':
            df['Bid'][i]=0
        elif df['Bid_class'][i]=='개별':
            df['Bid'][i]=1
        else: df['Bid'][i]=2
```

```python
df.head() #변수가 잘 생성되었는지 확인
```

|  | Auction\_key | Auction\_class | Bid\_class | Claim\_price | Appraisal\_company | Appraisal\_date | Auction\_count | Auction\_miscarriage\_count | Total\_land\_gross\_area | Total\_land\_real\_area | Total\_land\_auction\_area | Total\_building\_area | Total\_building\_auction\_area | Total\_appraisal\_price | Minimum\_sales\_price | First\_auction\_date | Final\_auction\_date | Final\_result | Creditor | addr\_do | addr\_si | addr\_dong | addr\_li | addr\_san | addr\_bunji1 | addr\_bunji2 | addr\_etc | Apartment\_usage | Preserve\_regist\_date | Total\_floor | Current\_floor | Specific | Share\_auction\_YorN | road\_name | road\_bunji1 | road\_bunji2 | Close\_date | Close\_result | point.y | point.x | Hammer\_price | auction\_start\_weekday | auction\_end\_weekday | auction\_day\_length | Bid |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 2687 | 임의 | 개별 | 1766037301 | 정명감정 | 2017-07-26 00:00:00 | 2 | 1 | 12592.0 | 37.35 | 37.35 | 181.77 | 181.77 | 836000000 | 668800000 | 2018-02-13 | 2018-03-20 | 낙찰 | 베리타스자산관리대부 | 부산 | 해운대구 | 우동 | NaN | N | 1398.0 | NaN | 해운대엑소디움 5층 101-502호 | 주상복합 | 2009-07-14 00:00:00 | 45 | 5 | NaN | N | 해운대해변로 | 30.0 | NaN | 2018-06-14 00:00:00 | 배당 | 35.162717 | 129.137048 | 760000000 | 1 | 1 | 35 | 1 |
| 1 | 2577 | 임의 | 일반 | 152946867 | 희감정 | 2016-09-12 00:00:00 | 2 | 1 | 42478.1 | 18.76 | 18.76 | 118.38 | 118.38 | 1073000000 | 858400000 | 2016-12-29 | 2017-02-02 | 낙찰 | 흥국저축은행 | 부산 | 해운대구 | 우동 | NaN | N | 1407.0 | NaN | 해운대두산위브더제니스 103동 51층 5103호 | 아파트 | 2011-12-16 00:00:00 | 70 | 51 | NaN | N | 마린시티2로 | 33.0 | NaN | 2017-03-30 00:00:00 | 배당 | 35.156633 | 129.145068 | 971889999 | 3 | 3 | 35 | 0 |
| 2 | 2197 | 임의 | 개별 | 11326510 | 혜림감정 | 2016-11-22 00:00:00 | 3 | 2 | 149683.1 | 71.00 | 71.00 | 49.94 | 49.94 | 119000000 | 76160000 | 2017-07-28 | 2017-10-13 | 낙찰 | 국민은행 | 부산 | 사상구 | 모라동 | NaN | N | 552.0 | NaN | 백양그린 206동 14층 1403호 | 아파트 | 1992-07-31 00:00:00 | 15 | 14 | NaN | N | 모라로110번길 | 88.0 | NaN | 2017-12-13 00:00:00 | 배당 | 35.184601 | 128.996765 | 93399999 | 4 | 4 | 77 | 1 |
| 3 | 2642 | 임의 | 일반 | 183581724 | 신라감정 | 2016-12-13 00:00:00 | 2 | 1 | 24405.0 | 32.98 | 32.98 | 84.91 | 84.91 | 288400000 | 230720000 | 2017-07-20 | 2017-11-02 | 낙찰 | 고려저축은행 | 부산 | 남구 | 대연동 | NaN | N | 243.0 | 23.0 | 대연청구 109동 11층 1102호 | 아파트 | 2001-07-13 00:00:00 | 20 | 11 | NaN | N | 황령대로319번가길 | 110.0 | NaN | 2017-12-27 00:00:00 | 배당 | 35.154180 | 129.089081 | 256899000 | 3 | 3 | 105 | 0 |
| 4 | 1958 | 강제 | 일반 | 45887671 | 나라감정 | 2016-03-07 00:00:00 | 2 | 1 | 774.0 | 45.18 | 45.18 | 84.96 | 84.96 | 170000000 | 136000000 | 2016-07-06 | 2016-08-03 | 낙찰 | Private | 부산 | 사하구 | 괴정동 | NaN | N | 399.0 | 2.0 | 동조리젠시 7층 703호 | 아파트 | 2001-11-27 00:00:00 | 7 | 7 | NaN | N | 오작로 | 51.0 | NaN | 2016-10-04 00:00:00 | 배당 | 35.099630 | 128.998874 | 158660000 | 2 | 2 | 28 | 0 |

Bid라는 변수에 입찰 구분 중 일반=0, 개별=1, 일괄=2로 인코딩하였다.

### 3-3. 건물\(토지\)의 대표 용도 인코딩

```text
df.groupby('Apartment_usage').count()
```

|  | Auction\_key | Auction\_class | Bid\_class | Claim\_price | Appraisal\_company | Appraisal\_date | Auction\_count | Auction\_miscarriage\_count | Total\_land\_gross\_area | Total\_land\_real\_area | Total\_land\_auction\_area | Total\_building\_area | Total\_building\_auction\_area | Total\_appraisal\_price | Minimum\_sales\_price | First\_auction\_date | Final\_auction\_date | Final\_result | Creditor | addr\_do | addr\_si | addr\_dong | addr\_li | addr\_san | addr\_bunji1 | addr\_bunji2 | addr\_etc | Preserve\_regist\_date | Total\_floor | Current\_floor | Specific | Share\_auction\_YorN | road\_name | road\_bunji1 | road\_bunji2 | Close\_date | Close\_result | point.y | point.x | Hammer\_price | auction\_start\_weekday | auction\_end\_weekday | auction\_day\_length | Bid |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Apartment\_usage |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 아파트 | 1654 | 1654 | 1654 | 1654 | 1654 | 1654 | 1654 | 1654 | 1654 | 1654 | 1654 | 1654 | 1654 | 1654 | 1654 | 1654 | 1654 | 1654 | 1654 | 1654 | 1654 | 1654 | 22 | 1654 | 1651 | 686 | 1654 | 1654 | 1654 | 1654 | 41 | 1654 | 1654 | 1647 | 129 | 1654 | 1654 | 1654 | 1654 | 1654 | 1654 | 1654 | 1654 | 1654 |
| 주상복합 | 277 | 277 | 277 | 277 | 277 | 277 | 277 | 277 | 277 | 277 | 277 | 277 | 277 | 277 | 277 | 277 | 277 | 277 | 277 | 277 | 277 | 277 | 1 | 277 | 276 | 201 | 277 | 277 | 277 | 277 | 21 | 277 | 277 | 261 | 26 | 277 | 277 | 277 | 277 | 277 | 277 | 277 | 277 | 27 |

건물\(토지\)의 대표 용도를 grouping하여 count한 결과 '아파트'와 '주상복합' 두 가지로 나뉨을 알 수 있다. 이를 인코딩하겠다.

```python
#Apartment usage 인코딩
df['Use']=0
for i in range(0,len(df)):
        if df['Apartment_usage'][i]=='아파트':
            df['Use'][i]=0
        else: df['Use'][i]=1
```

```text
df.head()  #변수가 잘 생성되었는지 확인
```

|  | Auction\_key | Auction\_class | Bid\_class | Claim\_price | Appraisal\_company | Appraisal\_date | Auction\_count | Auction\_miscarriage\_count | Total\_land\_gross\_area | Total\_land\_real\_area | Total\_land\_auction\_area | Total\_building\_area | Total\_building\_auction\_area | Total\_appraisal\_price | Minimum\_sales\_price | First\_auction\_date | Final\_auction\_date | Final\_result | Creditor | addr\_do | addr\_si | addr\_dong | addr\_li | addr\_san | addr\_bunji1 | addr\_bunji2 | addr\_etc | Apartment\_usage | Preserve\_regist\_date | Total\_floor | Current\_floor | Specific | Share\_auction\_YorN | road\_name | road\_bunji1 | road\_bunji2 | Close\_date | Close\_result | point.y | point.x | Hammer\_price | auction\_start\_weekday | auction\_end\_weekday | auction\_day\_length | Bid | Use |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 2687 | 임의 | 개별 | 1766037301 | 정명감정 | 2017-07-26 00:00:00 | 2 | 1 | 12592.0 | 37.35 | 37.35 | 181.77 | 181.77 | 836000000 | 668800000 | 2018-02-13 | 2018-03-20 | 낙찰 | 베리타스자산관리대부 | 부산 | 해운대구 | 우동 | NaN | N | 1398.0 | NaN | 해운대엑소디움 5층 101-502호 | 주상복합 | 2009-07-14 00:00:00 | 45 | 5 | NaN | N | 해운대해변로 | 30.0 | NaN | 2018-06-14 00:00:00 | 배당 | 35.162717 | 129.137048 | 760000000 | 1 | 1 | 35 | 1 | 1 |
| 1 | 2577 | 임의 | 일반 | 152946867 | 희감정 | 2016-09-12 00:00:00 | 2 | 1 | 42478.1 | 18.76 | 18.76 | 118.38 | 118.38 | 1073000000 | 858400000 | 2016-12-29 | 2017-02-02 | 낙찰 | 흥국저축은행 | 부산 | 해운대구 | 우동 | NaN | N | 1407.0 | NaN | 해운대두산위브더제니스 103동 51층 5103호 | 아파트 | 2011-12-16 00:00:00 | 70 | 51 | NaN | N | 마린시티2로 | 33.0 | NaN | 2017-03-30 00:00:00 | 배당 | 35.156633 | 129.145068 | 971889999 | 3 | 3 | 35 | 0 | 0 |
| 2 | 2197 | 임의 | 개별 | 11326510 | 혜림감정 | 2016-11-22 00:00:00 | 3 | 2 | 149683.1 | 71.00 | 71.00 | 49.94 | 49.94 | 119000000 | 76160000 | 2017-07-28 | 2017-10-13 | 낙찰 | 국민은행 | 부산 | 사상구 | 모라동 | NaN | N | 552.0 | NaN | 백양그린 206동 14층 1403호 | 아파트 | 1992-07-31 00:00:00 | 15 | 14 | NaN | N | 모라로110번길 | 88.0 | NaN | 2017-12-13 00:00:00 | 배당 | 35.184601 | 128.996765 | 93399999 | 4 | 4 | 77 | 1 | 0 |
| 3 | 2642 | 임의 | 일반 | 183581724 | 신라감정 | 2016-12-13 00:00:00 | 2 | 1 | 24405.0 | 32.98 | 32.98 | 84.91 | 84.91 | 288400000 | 230720000 | 2017-07-20 | 2017-11-02 | 낙찰 | 고려저축은행 | 부산 | 남구 | 대연동 | NaN | N | 243.0 | 23.0 | 대연청구 109동 11층 1102호 | 아파트 | 2001-07-13 00:00:00 | 20 | 11 | NaN | N | 황령대로319번가길 | 110.0 | NaN | 2017-12-27 00:00:00 | 배당 | 35.154180 | 129.089081 | 256899000 | 3 | 3 | 105 | 0 | 0 |
| 4 | 1958 | 강제 | 일반 | 45887671 | 나라감정 | 2016-03-07 00:00:00 | 2 | 1 | 774.0 | 45.18 | 45.18 | 84.96 | 84.96 | 170000000 | 136000000 | 2016-07-06 | 2016-08-03 | 낙찰 | Private | 부산 | 사하구 | 괴정동 | NaN | N | 399.0 | 2.0 | 동조리젠시 7층 703호 | 아파트 | 2001-11-27 00:00:00 | 7 | 7 | NaN | N | 오작로 | 51.0 | NaN | 2016-10-04 00:00:00 | 배당 | 35.099630 | 128.998874 | 158660000 | 2 | 2 | 28 | 0 | 0 |

## 4. 선형 회귀 분석

우선 회귀 분석에 사용할 데이터만 따로 data에 저장하겠다.

```python
data=df2_s.copy()

columns = df.iloc[:,-5:-1]
data = pd.concat([data,columns], axis = 1)  #-4:-1의 범위로 지정해서 마지막 column이 안보여서 아래에 다시 concat 진행

lastcol=df.iloc[:,-1]
data = pd.concat([data,lastcol], axis = 1)

data.info()


#https://stackoverflow.com/questions/33532216/adding-columns-from-one-dataframe-to-another-python-pandas
```

```text
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1933 entries, 0 to 1932
Data columns (total 12 columns):
Total_land_real_area           1929 non-null float64
Total_land_auction_area        1929 non-null float64
Total_building_area            1929 non-null float64
Total_building_auction_area    1929 non-null float64
Total_appraisal_price          1929 non-null float64
Minimum_sales_price            1929 non-null float64
Hammer_price                   1929 non-null float64
auction_start_weekday          1931 non-null float64
auction_end_weekday            1931 non-null float64
auction_day_length             1931 non-null float64
Bid                            1931 non-null float64
Use                            1931 non-null float64
dtypes: float64(12)
memory usage: 196.3 KB
```

```python
#Inf, -Inf를 NaN으로 처리 후 제거하기
import numpy as np
data=data.replace([-np.inf,np.inf], np.nan)
data=data.dropna(axis=0)
data.info()

#6줄이 제거됨
```

```text
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1927 entries, 0 to 1930
Data columns (total 12 columns):
Total_land_real_area           1927 non-null float64
Total_land_auction_area        1927 non-null float64
Total_building_area            1927 non-null float64
Total_building_auction_area    1927 non-null float64
Total_appraisal_price          1927 non-null float64
Minimum_sales_price            1927 non-null float64
Hammer_price                   1927 non-null float64
auction_start_weekday          1927 non-null float64
auction_end_weekday            1927 non-null float64
auction_day_length             1927 non-null float64
Bid                            1927 non-null float64
Use                            1927 non-null float64
dtypes: float64(12)
memory usage: 195.7 KB
```

모델의 적합성 검증을 위하여 train & test data split을 진행해준다.

```python
x = data.drop('Hammer_price', axis=1)
y = data.Hammer_price
```

```python
# train, test data 분할
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)  #test data size는 전체의 20%
```

회귀적합

```python
from sklearn.linear_model import LinearRegression

#모델 불러옴
model = LinearRegression()
#train data에 fit시킴
model.fit(x_train, y_train)
```

```python
#fit된 모델의 R-square
model.score(x_train, y_train)
```

세상에 r square 값이 0.99가 나왔다. 너무 높다.

```python
sns.pairplot(x)
```

![](../.gitbook/assets/image%20%2856%29.png)

r square가 높게 나온 이유는 서로 관련 있는 변수들이 많아서 생긴다고 볼 수도 있다. 이 데이터에서는 price관련된 변수가 3개, area 관련 변수가 4개가 있으며 서로 관련성이 큰데, 위의 pairplot에서 거의 직선으로 보이는 변수들로 얼마나 상관이 있는지 판단 가능하다. 더 정확한 판단을 위하여 VIF 검정을 진행해본다.

```python
#VIF확인하기
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif["features"] = x.columns
vif.sort_values(["VIF Factor"], ascending=[False])
```

|  | VIF Factor | features |
| :--- | :--- | :--- |
| 0 | 463.486981 | Total\_land\_real\_area |
| 1 | 462.807904 | Total\_land\_auction\_area |
| 3 | 449.634387 | Total\_building\_auction\_area |
| 2 | 415.396782 | Total\_building\_area |
| 4 | 305.326164 | Total\_appraisal\_price |
| 5 | 301.975570 | Minimum\_sales\_price |
| 7 | 32.703886 | auction\_end\_weekday |
| 6 | 32.661878 | auction\_start\_weekday |
| 8 | 1.330857 | auction\_day\_length |
| 10 | 1.294741 | Use |
| 9 | 1.184722 | Bid |

...ㅎㅎㅎ 당연하지만 10이 훨씬 넘는다. 참고로 경매가 시작되는 요일과 끝나는 요일은 같다. 그러므로 총 경매 기간 변수를 추가하고 끝나는 요일 변수를 제거하겠다.

```python
data1=data.copy()

del data1['Total_building_auction_area'] #이것의 분포가 더 특정값에 치우쳐져 있었으므로 제거
del data1['Total_land_auction_area'] #위에 제거하는 변수와 관련이 있어보여 real area 말고 auction area delete
del data1['Minimum_sales_price'] #Hammer Price가 Target이므로 Minimum sales price를 제거
del data1['auction_end_weekday'] #경매가 끝나는 요일 제거(경매 시작 요일과 동일)

data1.info()
```

```python
#다시 회귀 모델링
x = data1.drop('Hammer_price', axis=1)
y = data1.Hammer_price

# train, test data 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)  #test data size는 전체의 20%

#모델 불러옴
model = LinearRegression()
#train data에 fit시킴
model.fit(x_train, y_train)

#fit된 모델의 R-square
model.score(x_train, y_train)
```

Output : 

```text
0.9700522755627569
```

잘 나온 것 같다.

```python
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif["features"] = x.columns
vif.sort_values(["VIF Factor"], ascending=[False])
```

|  | VIF Factor | features |
| :--- | :--- | :--- |
| 1 | 6.563827 | Total\_building\_area |
| 0 | 6.314295 | Total\_land\_real\_area |
| 2 | 3.861034 | Total\_appraisal\_price |
| 3 | 2.541604 | auction\_start\_weekday |
| 6 | 1.287799 | Use |
| 4 | 1.250202 | auction\_day\_length |
| 5 | 1.168293 | Bid |

VIF도 확 줄어듦을 알 수 있다. 모두 10 미만이므로 다중공선성이 해결되었다고 볼 수 있다.

이를 바탕으로 분석을 진행하겠다.

### by sklearn

```python
#MSE
import sklearn as sk
sk.metrics.mean_squared_error(y_train, model.predict(x_train))
```

Output : 

```text
0.021966726224034304
```

```python
#beta hat
print(model.intercept_); print(model.coef_)
```

Output : 

```text
0.3850703572670273
[ 1.31329416e-01 -4.14035077e-01  1.06477497e+00 -3.25654091e-03
 -1.80639852e-04 -5.05362126e-02 -2.31552468e-02]
```

```python
#test데이터 R-square
model.score(x_test, y_test)
```

Output : 

```text
0.9676691334932952
```

```python
# 예측 vs. 실제데이터 plot
y_pred = model.predict(x_test) 
plt.plot(y_test, y_pred, '.')

# 예측과 실제가 비슷하면, 라인상에 분포함
x = np.linspace(-8, 1, 100)
y = x
plt.plot(x, y)
plt.show()
```

![](../.gitbook/assets/image%20%2821%29.png)

아주 잘 추정되었고 모델링도 아주 잘된 것으로 보인다. :3

### by calculation of matrix

```python
import numpy as np
from numpy.linalg import inv 

def estimate_beta(x, y):
    y=y_train.values.reshape(-1,1)
    beta_hat=inv(x.T@x)@x.T@y
    return beta_hat

#https://stackoverflow.com/questions/53723928/attributeerror-series-object-has-no-attribute-reshape  ##reshaping
```

```python
#beta hat by matrix calculation
betahat=estimate_beta(x_train,y_train)
betahat
```

Output : 

```text
array([[ 2.15598318e-01],
       [ 6.70786521e-03],
       [ 9.77307539e-01],
       [-7.22238005e-03],
       [-1.66612830e-04],
       [-4.01543060e-02],
       [-1.71396365e-02]])
```

```python
#MSE by matrix calculation
e=y_train.values.reshape(-1,1)-x_train@betahat
mse=e.T@e/(len(x_train)-len(betahat)-1)
print(mse)
```

Output :

```text
          0
0  0.024926
```

### 비교

```python
from sklearn.linear_model import LinearRegression

model2 = LinearRegression(fit_intercept=False)
#train data에 fit시킴

model2.fit(x_train, y_train)
model2.score(x_train, y_train)

#betahat by sklearn without intercept 
print(model2.coef_)
```

Output : 

```text
[ 2.15598318e-01  6.70786521e-03  9.77307539e-01 -7.22238005e-03
 -1.66612830e-04 -4.01543060e-02 -1.71396365e-02]
```

```python
#MSE by sklearn without intercept 
sk.metrics.mean_squared_error(y_train, model2.predict(x_train))
```

Output : 

```text
0.02479706560872116
```

행렬로 계산한 betahat는 sklearn을 통한 계산 결과와 일치하며, MSE도 크게 다르지 않음을 알 수 있다.

