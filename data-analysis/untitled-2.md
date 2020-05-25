---
description: 13기 조혜원
---

# 앙상블

### 요구사항 

* Assignment 1. 캐글 Guide to Ensembling methods 정독 [https://www.kaggle.com/amrmahmoud123/1-guide-to-ensembling-methods\#](https://www.kaggle.com/amrmahmoud123/1-guide-to-ensembling-methods)

### 1\) Error[¶]() <a id="1)-Error"></a>

에러\(오차\): Bias 에러^2 + Variance + irreducible \(줄일 수 없는\) 에러

1\) Bias Error: 예측값-실제값의 차이

2\) Variance: 트레이닝 변수에 얼마나 적합했는지 \(Underfitting-Overfitting\)

+\) Irreducible Error: [https://aoc55.tistory.com/m/22](https://aoc55.tistory.com/m/22)

* reducible error : f hat와 f의 차이에서 발생, 적절한 방법으로 정확도를 높일 수 있음
* irreducible error : f\(x\)와 y의 차이, x로 y을 완전히 결정할 수 없기에 생김, 이 error term은 x에 의존적이지 않지만 y에 유의미한 영향을 미칠 수도 있기 때문에 줄일 수 없음

=&gt; 모델은 Bias-Variance 사이의 절충을 잘 해야함 Ensemble이 이러한 절충적인 분석을 돕는다

### 2\) Ensemble[¶]() <a id="2)-Ensemble"></a>

* Ensemble: 여러 예측 모델의 집단
* Ensemble Learning: 여러 모델을 조합하여 Accuracy를 높이는 기술
* Ensemble Method: Ensemble Learning Algorithm

## 1.Basic Ensemble Techniques[¶]() <a id="1.Basic-Ensemble-Techniques"></a>

### 1\) Max Voting[¶]() <a id="1)-Max-Voting"></a>

최종값 = 여러 분류기 중 가장 많은 분류기가 추출한 결과 값

In \[ \]:

```text
#직접구현

pred1=model1.predict(x_test)
pred2=model2.predict(x_test)
pred3=model3.predict(x_test)

final_pred = np.array([])

for i in range(0,len(x_test)):
    final_pred = np.append(final_pred, mode([pred1[i], pred2[i], pred3[i]]))
    
    #각 모델의 예측값 중 최빈값(mode())을 배열에 저장
```

In \[ \]:

```text
#VotingClassifier 

from sklearn.ensemble import VotingClassifier

model = VotingClassifier(estimators=[('lr', model1), ('dt', model2)], voting='hard')
# lr=logistic regression, dt=decision tree

model.fit(x_train,y_train)
model.score(x_test,y_test)
```

1\) Hard Voting: 단순하게 가장 많은 값을 고름

2\) Soft Voting: 각 예측 결과의 "확률"을 모두 더한 뒤 비교, 위 코드에서 predict=&gt;predict\_proba\(\), voting='hard' =&gt; voting='soft'로 바꿔주면 됨

### 2\) Averaging[¶]() <a id="2)-Averaging"></a>

예측값의 평균을 최종 예측에 사용, 회귀 예측, 분류에서 확률 계산 등에 사용

In \[ \]:

```text
pred1=model1.predict_proba(x_test)
pred2=model2.predict_proba(x_test)
pred3=model3.predict_proba(x_test)

finalpred=(pred1+pred2+pred3)/3
```

### 3\) Weighted Average[¶]() <a id="3)-Weighted-Average"></a>

\(결과값\*weight\)의 평균, 각 모델마다 할당되는 가중치가 다름을 고려해준 것

In \[ \]:

```text
finalpred=(pred1*0.3+pred2*0.3+pred3*0.4)
#0.3, 0.3, 0.4가 각각 가중치에 해당
```

## 2.Advanced Ensemble Techniques[¶]() <a id="2.Advanced-Ensemble-Techniques"></a>

In \[ \]:

```text
def Stacking(model,train,y,test,n_fold):

    folds=StratifiedKFold(n_splits=n_fold,random_state=1)   
    #fold-cross-validation: 부분 집합으로 나누어 학습시킨 후 가장 에러가 작은 모델을 택하는 검증기
    
    test_pred=np.empty((test.shape[0],1),float)
    train_pred=np.empty((0,1),float)
    
    for train_indices,val_indices in folds.split(train,y.values):
        x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]  #트레인셋을 training set/ validation set로 분리
        y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]

        model.fit(X=x_train,y=y_train)                                  # 그 중 training set을 모델에 학습
        train_pred=np.append(train_pred,model.predict(x_val))           # training set의 예측값과 validation set의 예측값을 배열로 생성
        test_pred=np.append(test_pred,model.predict(test))              # 테스트셋의 예측값
    
    return test_pred.reshape(-1,1),train_pred                          #위에서 만든 예측값 배열을 반환 


#모델 1 생성========================================================
model1 = tree.DecisionTreeClassifier(random_state=1)

test_pred1 ,train_pred1=Stacking(model=model1,n_fold=10, train=x_train,test=x_test,y=y_train) 
#위의 함수를 사용해 training set & validation set의 예측값/ 테스트셋의 예측값을 반환

train_pred1=pd.DataFrame(train_pred1)
test_pred1=pd.DataFrame(test_pred1)   #이 예측값들을 데이터프레임으로 저장

#모델 2 생성==========================================================
model2 = KNeighborsClassifier()

test_pred2 ,train_pred2=Stacking(model=model2,n_fold=10,train=x_train,test=x_test,y=y_train)

train_pred2=pd.DataFrame(train_pred2)
test_pred2=pd.DataFrame(test_pred2)

#final predict model===================================================
df = pd.concat([train_pred1, train_pred2], axis=1)
df_test = pd.concat([test_pred1, test_pred2], axis=1)    #위에서 받은 각 모델의 예측값들을 합침

model = LogisticRegression(random_state=1)
model.fit(df,y_train)                                    #df= training set & validation set의 예측값, 즉 예측값으로 새 모델을 피팅
model.score(df_test, y_test)                             #위 모델에 테스트셋의 예측값과 test label 넣어 Accuracy 확인
```

In \[ \]:

```text
model1 = tree.DecisionTreeClassifier()
model1.fit(x_train, y_train)       #training셋으로 피팅
val_pred1=model1.predict(x_val)    #피팅한 모델에 validation셋 예측   
test_pred1=model1.predict(x_test)  #피팅한 모델에 테스트셋 예측  
val_pred1=pd.DataFrame(val_pred1)
test_pred1=pd.DataFrame(test_pred1)

model2 = KNeighborsClassifier()
model2.fit(x_train,y_train)
val_pred2=model2.predict(x_val)
test_pred2=model2.predict(x_test)
val_pred2=pd.DataFrame(val_pred2)
test_pred2=pd.DataFrame(test_pred2)

df_val=pd.concat([x_val, val_pred1,val_pred2],axis=1)      #validation셋과 그 각 모델의 예측값들 저장
df_test=pd.concat([x_test, test_pred1,test_pred2],axis=1)  #테스트셋과 그 각 모델의 예측값들 저장

model = LogisticRegression()
model.fit(df_val,y_val)          #validation 셋과 그 예측값들을 최종 회귀 모델에 피팅 = meta-feature
model.score(df_test,y_test)      #피팅된 모델에 테스트셋을 넣어 검증!
```

### 3\) Bagging[¶]() <a id="3)-Bagging"></a>

competition에서 가장 많이 쓰이는 방법

1. 트레이닝 셋을 분할만 달리하여 여러개의 sample을 만듦
2. 한 모델에 여러개의 sample을 각각 피팅함으로써 여러 분류기 생성
3. 이러한 분류기를 평균화 하여 앙상블 분류기 생성
4. 앙상블 분류기에 테스트 셋을 넣어 예측값 추출
5. =&gt; 분산을 낮추고 정확도를 향상 \(= 과적합 줄이고 예측 안정화\)

이를 이용한 알고리즘으로는 3-1의 Bagging meta-estimator와 3-2의 Random Forest가 있음

### 4\) Boosting[¶]() <a id="4)-Boosting"></a>

약한 모델을 강화하는 알고리즘, 실행한 모델로 다음 모델이 집중할 피쳐를 결정함으로써 순차적으로 훈련시킨다. 잘못 예측한 관측치에 웨이트를 할당함으로써 후속 모델의 정확도를 높여줌

1. 처음에는 모든 관측치에 동일한 웨이트 할당
2. 하위집단을 기반으로 전체 데이터셋을 예측 - 예측과 실제값 비교로 오차 추정
3. 오차가 큰 데이터일수록 더 큰 웨이트 할당
4. 오차가 변하지 않거나 최대 추정 횟수에 도달할 때까지 반복

## 3.Algorithms based on Bagging and Boosting[¶]() <a id="3.Algorithms-based-on-Bagging-and-Boosting"></a>

### 1\) Bagging meta-estimator[¶]() <a id="1)-Bagging-meta-estimator"></a>

분류, 회귀 문제에 모두 사용할 수 있다

1. 기존 Baggig 방법을 그대로 따름
2. 하위집단은 모든 피쳐를 포함
3. 보다 작은 집단에 의해 추정기가 학습됨
4. 각 모델의 예측값이 합쳐져 최종값이 됨

In \[ \]:

```text
final_dt = DecisionTreeClassifier(max_leaf_nodes=10, max_depth=5)    #의사 결정 나무 분류기, 잎노드 최대 10개, 최대 depth 5              
final_bc = BaggingClassifier(base_estimator=final_dt, n_estimators=40, random_state=1, oob_score=True) #배깅 분류기

final_bc.fit(X_train, train_y)
final_preds = final_bc.predict(X_test)


acc_oob = final_bc.oob_score_
print(acc_oob)
```

### 2\) Random Forest[¶]() <a id="2)-Random-Forest"></a>

Bagging meta-estimator의 확장, 기본 추정기는 의사결정나무 Bagging meta-estimator와 달리 피쳐셋으로 각 의사결정나무 노드에서의 최선의 분할을 결정

1. 원래 데이터 셋에서 랜덤으로 하위집단 생성
2. 각 노드가 랜덤 피쳐셋을 고려하여 분할됨
3. 하위집단을 의사결정모델에 피팅, 최종 예측치는 각 예측치의 평균으로 계산

임의로 데이터 포인트와 피쳐를 선택하고 여러개의 트리, 즉 random forest를 형성하는 것

In \[ \]:

```text
#분류기
from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(random_state=1)

#회귀분석
from sklearn.ensemble import RandomForestRegressor
model= RandomForestRegressor()
```

★파라미터 설명★

* n\_estimators : 의사결정나무 개수, 너무 많으면 시간이 오래 걸리고 적으면 정확도가 떨어짐
* criterion: 분할에 사용할 기능을 정의, default=”gini”/ 우리가 배운 entropy도 사용할 수 있음!
* max\_features: 분할할 수 있는 최대 기능 수, 지나치면 오버피팅 적으면 언더피팅 default="auto"
* max\_depth: 의사결정나무 최대 depth
* min\_samples\_split: 잎노드\(자식노드가 없는 노드\)에 필요한 최소 샘플 수, 이보다 적으면 노드 분할 x default=2
* min\_samples\_leaf: 잎노드에 있어야 하는 최대 샘플 수, 작으면 노이즈를 알아채기 쉬움 default=1
* max\_leaf\_nodes: 잎노드의 최대 수, 이것보다 커지면 분할을 멈춤 default=None
* n\_jobs: 병렬처리될 수 있는 최댓값 default=none

+[https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html) 얘도 함께 참조

### 3\) AdaBoost[¶]() <a id="3)-AdaBoost"></a>

사이트 설명보다 ppt설명이 자세한 듯 해서 ppt 설명 위주로 요약했습니다.

Adaptive boosting, 가장 단순한 Boosting 알고리즘 간단한 약분류기를 통해 상호보완하도록 순차적으로 학습한 후, 이들을 조합하여 강분류기의 성능을 증폭시킨다.

1. 각 Instance의 Weight를 동일하게 설정,한 후 weight를 기준으로 샘플링
2. 오차가 날 때마다 해당 instance에 가중치를 부여함 =&gt; 뽑힐 확률이 높아짐
3. 이러한 오차를 바탕으로 error를 계산 \(error= 틀린 것의 weight 총합/ 모든 가중치 \)
4. 분류기에 log\(1-error\)/error만큼 분류기에 가중치를 더해줌
5. 이러한 모델의 가중치를 exp =&gt; 각각 모델의 가중치를 곱하여 전체 앙상블에 얼마만큼 반영할지 결정

단점은 weight가 낮은 데이터 주위에 높은 weight를 가진 데이터가 있으면 성능이 크게 떨어진다 그래서 GBM으로 오류값을 최소화 하는 학습 시행

In \[ \]:

```text
#분류기
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(random_state=1)

#회귀 분석
from sklearn.ensemble import AdaBoostRegressor
model = AdaBoostRegressor()
```

★파라미터 설명★

* base\_estimators : 기본 추정기 유형 default= DecisionTreeClassifier\(max\_depth=1\)
* n\_estimators : 기본 추정기 개수 default=10, 성능을 위해 높이는 것 권장
* learning\_rate : n\_estimators와 상충관계, 최종 조합에서 추정량의 기여도를 조정 default=1.0
* max\_depth : 개별 추정기 최대 깊이, 성능을 위해 조정하는 걸 권장
* n\_jobs : 사용 가능한 프로세서 수, -1은 모든 프로세서 사용
* random\_state : 랜덤 데이터 분할의 시드, random\_state를 하나로 정할 경우 동일한 결과가 나옴

+[https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) 얘도 함께 참조

In \[ \]:

```text
from sklearn.ensemble import GradientBoostingClassifier
model= GradientBoostingClassifier(learning_rate=0.01,random_state=1)

from sklearn.ensemble import GradientBoostingRegressor
model= GradientBoostingRegressor()
```

### 5\) XGB[¶]() <a id="5)-XGB"></a>

XGBoost, extreme Gradient Boosting, GBM의 발전된 버전으로 GBM보다 예측력이 높고 10배 가량 빠르며 과적합이 적게 발생한다. 또 다양한 정규화를 포함하기에 정규화된 부스팅 기술이라고도 한다.

장점

1. 정규화-오버피팅 감소
2. 병렬 처리- 빠른 속도, Hadoop에서의 구현 지원
3. 유연성- 모델에 새로운 차원을 추가하여, 최적화 목표, 평가 기준을 사용자가 정의할 수 있다
4. 결측값 처리- 결측값 처리를 위한 루틴 내장
5. 가지치기- 지정된 max\_depth까지 분할한 뒤 다음 트리를 잘라내어 이득이 없는 가지를 제거한다
6. 교차검증 - 교차검증이 내장되어 있어서 한번의 실행으로 최적의 부스팅 횟수를 얻을 수 있다

In \[ \]:

```text
import xgboost as xgb
model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
model=xgb.XGBRegressor()
```

### 6\) Light GBM[¶]() <a id="6)-Light-GBM"></a>

데이터셋이 매우 큰 경우에 유리, 대규모 데이터셋을 실행할 때 시간을 단축시켜준다 기본적으로 트리 기반 알고리즘의 GBM 프레임 워크와 Leaf-wise 접근 방식을 따른다 데이터 수가 적으면 오버피팅이 발생하기 쉬우므로, 10000개 이하일 때는 다른 방식이 권장된다

* Leafwise \(Light GBM에서 사용\): 한쪽으로만 성장 =&gt; 학습에 걸리는 시간, 메모리 사용량 단축
* Levelwise \(XGB에서 사용\): 층계별로 균형적인 성장을 보임, GridSearch CV로 튜닝을 시행할 시 시간이 너무 오래 걸림

In \[ \]:

```text
import lightgbm as lgb
train_data=lgb.Dataset(x_train,label=y_train)

#define parameters
params = {'learning_rate':0.001}
model= lgb.train(params, train_data, 100) 
y_pred=model.predict(x_test)

for i in range(0,185):
    if y_pred[i]>=0.5: 
    y_pred[i]=1
else: 
    y_pred[i]=0
```

In \[ \]:

```text
from catboost import CatBoostClassifier
model=CatBoostClassifier()

from catboost import CatBoostRegressor
model=CatBoostRegressor()
```

