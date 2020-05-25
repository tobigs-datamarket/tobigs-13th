---
description: 인공신경망 기초 (1)
---

# Neural Net Basic \(1\)

### 과제 내용 설명

1. 인공신경망의 오차 역전파 과정을 직접 필기하여 계산해주세요
2. 인공 신경망을 구현하는 실습파일을 완성해주세요

### 우수과제 선정 이유

개념을 하나 하나 정리하신 점이 인상 깊었습니다. 또한, 코드에 대한 해설과 함께 결과 해석까지 모두 해주셨습니다.

### Assignment 1. 오차 역전파 계산

![week6\_NN\_assignment2](https://camo.githubusercontent.com/edc2d82e7d2d287d66c4b12f62bafef8b9a8ba75/68747470733a2f2f757365722d696d616765732e67697468756275736572636f6e74656e742e636f6d2f34333734393537312f37353231363830342d39353731336330302d353764382d313165612d383066642d3533666535663863343631392e6a7067)

### Assignment 2\) 인공 신경망 구현

In \[1\]:

```python
from random import seed
from random import random
import numpy as np
 
# 네트워크 초기 설정
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)] # 개수는 같게 설정 
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)] # 개수는 같게 설정 
    network.append(output_layer)
    return network
 
seed(1)
network = initialize_network(2, 1, 2) 
# (2+1(bias))개의 input node, 1개의 hidden layer, 2개의 ouput node 
# w 값은 임의로 설정한다


for layer in network:
    print(layer)

# initialize_network function
# 첫번째(input node)가 들어가서 두번째(hidden layer)가 나오고 
# 두번째(hidden layer)가 들어가서 세번째(output node)가 나온다 

# 결과값은 차례대로 
# 1) hidden layer: input node의 [w0, w1, w2], (2 input weights + bias)
# 2) output node: hidden layer의 [w00, w01], [w10, w11] (1 weight + bias)
```

```text
[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}]
[{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]
```

In \[2\]:

```python
def activate(weights, inputs):
    activation = weights[-1] # weights 벡터의 마지막 값이 bias: 그래서 [-1] 넣는 것. bias는 곱하지 않고 넣어야 하기 때문에 그냥 넣는다 
    for i in range(len(weights)-1):
        activation += inputs[i] * weights[i]  # 순전파 진행 : 곱해서 activation list에 추가한다는 뜻 
    return activation # bias + activation 한 것 = list 

def sigmoid(activation):
    return 1 / (1 + np.exp(-activation)) # 시그모이드 구현

def forward_propagate(network, row):
    inputs = row
    for layer in network: # network = U1, U2 전체를 말하는 것, 그 중 layer는 U1, U2 각각을 말하는 것  
        new_inputs = []
        for neuron in layer: # layer의 하나의 열이 neuron 
            activation = activate(neuron['weights'], inputs) # neuron['weight'] = 하나의 열에서 가중치 각각 말하는 것  
            neuron['output'] = sigmoid(activation) # 활성함수에 activation 값을 넣어서 출력시킨다 
            new_inputs.append(neuron['output']) # new_input은 다음 히든층에 들어갈 값
        inputs = new_inputs
    return inputs
```

**여기까지는 순전파 학습과정이었습니다. 이 과정이 끝나면 가중치가 바뀌나요?  
답변을 답변의 근거 코딩 결과와 함께 보여주세요.**In \[3\]:

```text
network # 똑같다: forward propagation은 weight에 전혀 영향을 주지 않는다
```

Out\[3\]:

```text
[[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
 [{'weights': [0.2550690257394217, 0.49543508709194095]},
  {'weights': [0.4494910647887381, 0.651592972722763]}]]
```

In \[4\]:

```python
row = [1, 0, None] # None = bias 의미 
output = forward_propagate(network, row)
print(output)
```

```text
[0.6629970129852887, 0.7253160725279748]
```

In \[5\]:

```python
def sigmoid_derivative(output):
    return output * (1 - output) # 시그모이드 미분

def backward_propagate_error(network, expected): # expected = 예측값: NN 통과해서 나온 값, ouput = 실제값: target 변수의 값 
    for i in reversed(range(len(network))): # network 층 거꾸로 간다 
        layer = network[i] # 마지막 층부터 고려한다 
        errors = []
        if i != len(network)-1: # 출력단인 경우
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]: 
                    error += (neuron['weights'][j] * neuron['delta']) 
                errors.append(error) 
        else: # 출력단이 아닌 경우   
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])  
        
            # hidden layer의 error = (가중치 * 에러) * 활성함수 미분(예측값)
            # backpropagation에서 hidden -> input 으로 갈 때, (U1으로 미분 과정에서) 
            # 오류 역전파의 유도 과정에서 결론적으로 다 미분하면 가중치만 남게 되므로 
            # 가중치를 곱해 주면서 에러를 업데이트 한다 
            
        
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoid_derivative(neuron['output'])  
            
            # output node의 error = (실제값 - 예측값) * 활성함수 미분(예측값)
            # output -> hidden으로 갈 때, (U2로 미분 과정에서)
            # 결론적으로 Zj만 남게 되므로
            # 업데이트 된 에러와 활성함수 미분값을 곱해 delta에 저장한다
```

In \[6\]:

```python
expected = [0, 1] # label

backward_propagate_error(network, expected)
for layer in network:
    print(layer)
```

```text
[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614], 'output': 0.7105668883115941, 'delta': -0.002711797799238243}]
[{'weights': [0.2550690257394217, 0.49543508709194095], 'output': 0.6629970129852887, 'delta': -0.14813473120687762}, {'weights': [0.4494910647887381, 0.651592972722763], 'output': 0.7253160725279748, 'delta': 0.05472601157879688}]
```

In \[7\]:

```python
def weights_update(network, row, l_rate): # 한 단계 가중치 업데이트
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j] # 가중치 갱신 : w
            neuron['weights'][-1] += l_rate * neuron['delta'] # 퍼셉트론 학습 규칙: bias - Yk의 유무의 따라서 달라지기 때문 
            

def train_network(network, train, l_rate, n_epoch, n_outputs): # 전체 학습 과정 - 무수히 가중치 업데이트 과정을 반복
    for epoch in range(n_epoch): 
        sum_error = 0
        for row in train: # train data set의 하나하나의 row 
            outputs = forward_propagate(network, row) # 순전파: 예상값  
            expected = [0 for i in range(n_outputs)] # label 
            expected[row[-1]] = 1  # 0 / 1 classification - 값이 있으면 1, 없으면 0으로 채워준다  
            sum_error += sum([(expected[i] - outputs[i])**2 for i in range(len(expected))]) # 예측값의 오차 합: euclidean distance 
            backward_propagate_error(network, expected)
            weights_update(network, row, l_rate)
        if epoch%100==0:
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
```

In \[8\]:

```python
seed(1)
dataset = [[2.7810836,2.550537003,0],
    [1.465489372,2.362125076,0],
    [3.396561688,4.400293529,0],
    [1.38807019,1.850220317,0],
    [3.06407232,3.005305973,0],
    [7.627531214,2.759262235,1],
    [5.332441248,2.088626775,1],
    [6.922596716,1.77106367,1],
    [8.675418651,-0.242068655,1],
    [7.673756466,3.508563011,1]]
```

In \[9\]:

```python
n_inputs = len(dataset[0])-1 # 입력 노드의 개수는 training data의 독립변수 개수 (X1, X2)
n_outputs = len(set([dataset[i][-1] for i in range(len(dataset))])) # output의 target 개수 = 여기에서는 2 (0, 1) 
network = initialize_network(n_inputs, 2, n_outputs)

train_network(network, dataset, 0.02, 5001, n_outputs) # 자유롭게 설정하고 최적을 찾아보세요.
```

```text
>epoch=0, lrate=0.020, error=6.697
>epoch=100, lrate=0.020, error=4.299
>epoch=200, lrate=0.020, error=3.007
>epoch=300, lrate=0.020, error=2.037
>epoch=400, lrate=0.020, error=1.415
>epoch=500, lrate=0.020, error=1.020
>epoch=600, lrate=0.020, error=0.754
>epoch=700, lrate=0.020, error=0.570
>epoch=800, lrate=0.020, error=0.444
>epoch=900, lrate=0.020, error=0.356
>epoch=1000, lrate=0.020, error=0.293
>epoch=1100, lrate=0.020, error=0.247
>epoch=1200, lrate=0.020, error=0.212
>epoch=1300, lrate=0.020, error=0.185
>epoch=1400, lrate=0.020, error=0.163
>epoch=1500, lrate=0.020, error=0.146
>epoch=1600, lrate=0.020, error=0.131
>epoch=1700, lrate=0.020, error=0.119
>epoch=1800, lrate=0.020, error=0.109
>epoch=1900, lrate=0.020, error=0.101
>epoch=2000, lrate=0.020, error=0.093
>epoch=2100, lrate=0.020, error=0.087
>epoch=2200, lrate=0.020, error=0.081
>epoch=2300, lrate=0.020, error=0.076
>epoch=2400, lrate=0.020, error=0.072
>epoch=2500, lrate=0.020, error=0.068
>epoch=2600, lrate=0.020, error=0.064
>epoch=2700, lrate=0.020, error=0.061
>epoch=2800, lrate=0.020, error=0.058
>epoch=2900, lrate=0.020, error=0.055
>epoch=3000, lrate=0.020, error=0.053
>epoch=3100, lrate=0.020, error=0.050
>epoch=3200, lrate=0.020, error=0.048
>epoch=3300, lrate=0.020, error=0.046
>epoch=3400, lrate=0.020, error=0.045
>epoch=3500, lrate=0.020, error=0.043
>epoch=3600, lrate=0.020, error=0.041
>epoch=3700, lrate=0.020, error=0.040
>epoch=3800, lrate=0.020, error=0.039
>epoch=3900, lrate=0.020, error=0.037
>epoch=4000, lrate=0.020, error=0.036
>epoch=4100, lrate=0.020, error=0.035
>epoch=4200, lrate=0.020, error=0.034
>epoch=4300, lrate=0.020, error=0.033
>epoch=4400, lrate=0.020, error=0.032
>epoch=4500, lrate=0.020, error=0.031
>epoch=4600, lrate=0.020, error=0.030
>epoch=4700, lrate=0.020, error=0.030
>epoch=4800, lrate=0.020, error=0.029
>epoch=4900, lrate=0.020, error=0.028
>epoch=5000, lrate=0.020, error=0.027
```

In \[10\]:

```python
# 업데이트 된 가중치 
for layer in network:
    print(layer)
```

```text
[{'weights': [-1.9544732139957548, 2.704321541499767, 1.427507893789529], 'output': 0.016593738310259763, 'delta': -0.000252180328244517}, {'weights': [1.134364357068215, -1.6866805418327002, -0.3324719515881516], 'output': 0.9208338348540293, 'delta': 0.0005597952940571184}]
[{'weights': [4.903406947389829, -2.108155861429033, -1.2721372756183076], 'output': 0.04180758754504879, 'delta': -0.0016747999653304656}, {'weights': [-4.566502112347264, 2.615993561526247, 0.8280139001392679], 'output': 0.9593429092162126, 'delta': 0.0015857928993322072}]
```

In \[11\]:

```python
# 학습한 네트워크로 예측값을 뽑아보자.

def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

# 데이터셋 별로 예측하는 것 = 0 / 1 라벨에 속할 '확률'값 
# 이 중에 더 큰 것이 최종적으로 예측한 값 
# output의 max의 index를 구하면 된다
```

In \[12\]:

```python
# 네트워크가 잘 학습되었는지 확인해보자. 

for row in dataset:
    prediction = predict(network, row) # 앞서 최적(학습)시킨 네트워크로 잘 학습되었는지 평가 
    print('실제값=%d, 예측값=%d' % (row[-1], prediction))
```

```text
실제값=0, 예측값=0
실제값=0, 예측값=0
실제값=0, 예측값=0
실제값=0, 예측값=0
실제값=0, 예측값=0
실제값=1, 예측값=1
실제값=1, 예측값=1
실제값=1, 예측값=1
실제값=1, 예측값=1
실제값=1, 예측값=1
```

