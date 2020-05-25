---
description: 인공신경망 기초 (2)
---

# Neural Net Basic \(2\)

### 과제 내용 설명

1. 인공신경망의 오차 역전파 과정을 직접 필기하여 계산해주세요
2. 인공 신경망을 구현하는 실습파일을 완성해주세요

### 우수과제 선정 이유

개념을 하나 하나 정리하신 점이 인상 깊었습니다. 또한, 코드에 대한 해설과 함께 결과 해석까지 모두 해주셨습니다.

### Assignment 1\) 오차 역전파 계산

![img1](https://camo.githubusercontent.com/6b30733452bf883b372d83de7da7848f2ce2a7fa/68747470733a2f2f696d6775722e636f6d2f3163677a5a45612e6a7067)

![img2](https://camo.githubusercontent.com/a9107b3bcc932f3f3937c7e614ea646e74fd22f3/68747470733a2f2f696d6775722e636f6d2f61416f416369742e6a7067)

### Assignment 2\) 인공 신경망 구현

In \[1\]:

```python
from random import seed
from random import random
import numpy as np
 
# 네트워크 초기 설정
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)] 
    #hidden layer*(1(bias term)+input수)->weight수 결정! 히든레이어로 갈때 가중치는 총 3개 발생
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    #히든레이어에서 output레이어로 넘어갈때 output수 2이므로 각각 2(히든레이어수+1)개씩 총 4개의 weight가 발생
    network.append(output_layer)
    return network
 
seed(1)
network = initialize_network(2, 1, 2)
for layer in network:
    print(layer)
```

```text
[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}]
[{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]
```

In \[2\]:

```python
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i] # 순전파 진행 
        #weight * inputs 곱한거를 return 값으로 받은 후
    return activation

def sigmoid(activation): #activation function으로 sigmoid 선택(영역을 영역으로 변환)
    return 1.0 / (1.0 + np.exp(-activation)) # 시그모이드 구현

#활성화 함수의 역할은! nn에서 입력받은 데이터를 다음층으로 출력할지를 결정

#하나의 노드가 1개이상의 노드와 연결되어있고
#데이터 입력을 받게 되는데 연결강도의 가중치의 합을 구하게 되고
#활성화 함수를 통해 weights 값의 크기에 따라 출력하게 되는 것!

#먼저 순전파 propagation을 진행한다
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)  #weight와 input값을 곱한 계산값을
            neuron['output'] = sigmoid(activation) # 나온 계산 값을 그대로 쓰나요? #아니요! 시그모이드 함수에 넣어줍니다
            new_inputs.append(neuron['output']) # new_input은 다음 히든층에 들어갈 값이죠? #넵
        inputs = new_inputs
    return inputs #한번의 순전파를 거친 output이 다음 hiddenlayer의 input값이 된다
```

**여기까지는 순전파 학습과정이었습니다. 이 과정이 끝나면 가중치가 바뀌나요?  
답변을 답변의 근거 코딩 결과와 함께 보여주세요.**In \[3\]:

```python
row = [1, 0, None]
forward_propagate(network,row)
for layer in network:
    print(layer)
    
#순전파 학습과정을 거친 후 weight는 바뀌지 않는다!
#가중치가 바뀌는 과정은 오류역전파 과정에서 일어남
```

```text
[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614], 'output': 0.7105668883115941}]
[{'weights': [0.2550690257394217, 0.49543508709194095], 'output': 0.6629970129852887}, {'weights': [0.4494910647887381, 0.651592972722763], 'output': 0.7253160725279748}]
```

In \[4\]:

```python
row = [1, 0, None]
output = forward_propagate(network, row)
print(output)
```

```text
[0.6629970129852887, 0.7253160725279748]
```

In \[9\]:

```python
for i in reversed(range(len(network))):
    print(i)
```

```text
1
0
```

In \[5\]:

```python
def sigmoid_derivative(output):
    return output * (1.0 - output) # 시그모이드 미분

#오류역전파 진행합니다
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = []
        #i = 0 일때 2번째로
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]: 
                    error += (neuron['weights'][j] * neuron['delta']) #앞에서 구한 delta값을 기반으로 error 구한다
                errors.append(error) 
        # i =1 일때 1번으로!
        else:
            for j in range(len(layer)): 
                neuron = layer[j]
                errors.append(expected[j] - neuron['output']) 
                # 역전파시 오차는 어떻게 설정했나요?
                #함수인자로 받은 예상값 expected와 앞서 순전파로 구한 output값의 차
        for j in range(len(layer)):
            neuron = layer[j] 
            neuron['delta'] =  errors[j] * sigmoid_derivative(neuron['output'])
            #델타값은 앞서 구한 오류 값 * 순전파 과정으로 구한 output을 시그모이드 미분한 함수에 넣은 값 으로 구한다
            # 시그모이드 함수를 사용한 역전파
```

In \[6\]:

```python
expected = [0, 1]

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
#역전파 과정을 토대로 가중치를 업데이트 시킨다
def weights_update(network, row, l_rate): 
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]] #앞서 구한 output을 input으로 
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j] #앞서 구한 델타값과 learning rate를 곱해서 가중치에 더하는 방식으로 업데이트시킨다
            neuron['weights'][-1] +=  l_rate * neuron['delta']  # 퍼셉트론 학습 규칙
            
#앞서 진행한 과정을 반복해서 error을 줄여나간다
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network,row) # 먼저 순전파진행 
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]- outputs[i])**2 for i in range(len(expected))])
            # 예측값의 오차 합
            #sum error확인-> 학습을 진행하면서 error가 줄어드는지 확인해야하니까
            backward_propagate_error(network, expected) #그다음 역전파 진행
            weights_update(network, row, l_rate) #역전파 기반 가중치를 업데이트 시킨다
        #이과정을 지정한 epoch수만큼 반복
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
```

In \[8\]:

```text
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

In \[12\]:

```python
n_inputs = len(dataset[0]) - 1# 뉴럴렛의 입력노드로 뭐가 들어가죠? 그럼 입력 노드의 개수는?
#데이터셋에 들어있는 개수는 끝에 output label값도 포함하고 있으므로 그건 빼고!
n_outputs = len(set([row[-1] for row in dataset])) # 뉴럴렛의 출력노드의 개수는 뭐라고 했죠? 
#데이터셋의 맨마지막 부분 label값이 몇가지인지 그걸 set집합으로 중복된거 제거해서 그 개수를 세어준다
#여기서는 0 1 두가지 이므로 2개
network = initialize_network(n_inputs, 2, n_outputs) #먼저 네트워크 초기 설정

for layer in network:
    print(layer) # 초기 네트워크 
#학습된 네트워크랑 초기 네트워크를 비교하기 위해 먼저 초기 네트워크 출력했습니다


train_network(network, dataset, 1.0, 1000, n_outputs) # 자유롭게 설정하고 최적을 찾아보세요.



# 학습된(최적화)된 네트워크가 초기 네트워크와 달리 어떻게 변하였는지 출력하시별로,hint : for문))
for layer in network:
    print(layer) # 학습된(최적화된) 네트워크
```

```text
[{'weights': [0.762280082457942, 0.0021060533511106927, 0.4453871940548014]}, {'weights': [0.7215400323407826, 0.22876222127045265, 0.9452706955539223]}]
[{'weights': [0.9014274576114836, 0.030589983033553536, 0.0254458609934608]}, {'weights': [0.5414124727934966, 0.9391491627785106, 0.38120423768821243]}]
>epoch=0, lrate=1.000, error=6.153
>epoch=1, lrate=1.000, error=5.621
>epoch=2, lrate=1.000, error=5.606
>epoch=3, lrate=1.000, error=5.569
>epoch=4, lrate=1.000, error=5.467
>epoch=5, lrate=1.000, error=5.215
>epoch=6, lrate=1.000, error=4.842
>epoch=7, lrate=1.000, error=4.397
>epoch=8, lrate=1.000, error=3.908
>epoch=9, lrate=1.000, error=3.415
>epoch=10, lrate=1.000, error=2.937
>epoch=11, lrate=1.000, error=2.438
>epoch=12, lrate=1.000, error=2.051
>epoch=13, lrate=1.000, error=1.670
>epoch=14, lrate=1.000, error=1.401
>epoch=15, lrate=1.000, error=1.204
>epoch=16, lrate=1.000, error=1.046
>epoch=17, lrate=1.000, error=0.919
>epoch=18, lrate=1.000, error=0.815
>epoch=19, lrate=1.000, error=0.730
>epoch=20, lrate=1.000, error=0.659
>epoch=21, lrate=1.000, error=0.599
>epoch=22, lrate=1.000, error=0.549
>epoch=23, lrate=1.000, error=0.505
>epoch=24, lrate=1.000, error=0.468
>epoch=25, lrate=1.000, error=0.435
>epoch=26, lrate=1.000, error=0.406
>epoch=27, lrate=1.000, error=0.381
>epoch=28, lrate=1.000, error=0.358
>epoch=29, lrate=1.000, error=0.338
>epoch=30, lrate=1.000, error=0.319
>epoch=31, lrate=1.000, error=0.303
>epoch=32, lrate=1.000, error=0.288
>epoch=33, lrate=1.000, error=0.274
>epoch=34, lrate=1.000, error=0.262
>epoch=35, lrate=1.000, error=0.250
>epoch=36, lrate=1.000, error=0.240
>epoch=37, lrate=1.000, error=0.230
>epoch=38, lrate=1.000, error=0.221
>epoch=39, lrate=1.000, error=0.212
>epoch=40, lrate=1.000, error=0.205
>epoch=41, lrate=1.000, error=0.197
>epoch=42, lrate=1.000, error=0.191
>epoch=43, lrate=1.000, error=0.184
>epoch=44, lrate=1.000, error=0.178
>epoch=45, lrate=1.000, error=0.173
>epoch=46, lrate=1.000, error=0.167
>epoch=47, lrate=1.000, error=0.162
>epoch=48, lrate=1.000, error=0.158
>epoch=49, lrate=1.000, error=0.153
>epoch=50, lrate=1.000, error=0.149
>epoch=51, lrate=1.000, error=0.145
>epoch=52, lrate=1.000, error=0.141
>epoch=53, lrate=1.000, error=0.138
>epoch=54, lrate=1.000, error=0.134
>epoch=55, lrate=1.000, error=0.131
>epoch=56, lrate=1.000, error=0.128
>epoch=57, lrate=1.000, error=0.125
>epoch=58, lrate=1.000, error=0.122
>epoch=59, lrate=1.000, error=0.119
>epoch=60, lrate=1.000, error=0.116
>epoch=61, lrate=1.000, error=0.114
>epoch=62, lrate=1.000, error=0.112
>epoch=63, lrate=1.000, error=0.109
>epoch=64, lrate=1.000, error=0.107
>epoch=65, lrate=1.000, error=0.105
>epoch=66, lrate=1.000, error=0.103
>epoch=67, lrate=1.000, error=0.101
>epoch=68, lrate=1.000, error=0.099
>epoch=69, lrate=1.000, error=0.097
>epoch=70, lrate=1.000, error=0.095
>epoch=71, lrate=1.000, error=0.094
>epoch=72, lrate=1.000, error=0.092
>epoch=73, lrate=1.000, error=0.090
>epoch=74, lrate=1.000, error=0.089
>epoch=75, lrate=1.000, error=0.087
>epoch=76, lrate=1.000, error=0.086
>epoch=77, lrate=1.000, error=0.084
>epoch=78, lrate=1.000, error=0.083
>epoch=79, lrate=1.000, error=0.082
>epoch=80, lrate=1.000, error=0.080
>epoch=81, lrate=1.000, error=0.079
>epoch=82, lrate=1.000, error=0.078
>epoch=83, lrate=1.000, error=0.077
>epoch=84, lrate=1.000, error=0.076
>epoch=85, lrate=1.000, error=0.075
>epoch=86, lrate=1.000, error=0.074
>epoch=87, lrate=1.000, error=0.073
>epoch=88, lrate=1.000, error=0.071
>epoch=89, lrate=1.000, error=0.071
>epoch=90, lrate=1.000, error=0.070
>epoch=91, lrate=1.000, error=0.069
>epoch=92, lrate=1.000, error=0.068
>epoch=93, lrate=1.000, error=0.067
>epoch=94, lrate=1.000, error=0.066
>epoch=95, lrate=1.000, error=0.065
>epoch=96, lrate=1.000, error=0.064
>epoch=97, lrate=1.000, error=0.063
>epoch=98, lrate=1.000, error=0.063
>epoch=99, lrate=1.000, error=0.062
>epoch=100, lrate=1.000, error=0.061
>epoch=101, lrate=1.000, error=0.060
>epoch=102, lrate=1.000, error=0.060
>epoch=103, lrate=1.000, error=0.059
>epoch=104, lrate=1.000, error=0.058
>epoch=105, lrate=1.000, error=0.058
>epoch=106, lrate=1.000, error=0.057
>epoch=107, lrate=1.000, error=0.056
>epoch=108, lrate=1.000, error=0.056
>epoch=109, lrate=1.000, error=0.055
>epoch=110, lrate=1.000, error=0.055
>epoch=111, lrate=1.000, error=0.054
>epoch=112, lrate=1.000, error=0.053
>epoch=113, lrate=1.000, error=0.053
>epoch=114, lrate=1.000, error=0.052
>epoch=115, lrate=1.000, error=0.052
>epoch=116, lrate=1.000, error=0.051
>epoch=117, lrate=1.000, error=0.051
>epoch=118, lrate=1.000, error=0.050
>epoch=119, lrate=1.000, error=0.050
>epoch=120, lrate=1.000, error=0.049
>epoch=121, lrate=1.000, error=0.049
>epoch=122, lrate=1.000, error=0.048
>epoch=123, lrate=1.000, error=0.048
>epoch=124, lrate=1.000, error=0.047
>epoch=125, lrate=1.000, error=0.047
>epoch=126, lrate=1.000, error=0.046
>epoch=127, lrate=1.000, error=0.046
>epoch=128, lrate=1.000, error=0.046
>epoch=129, lrate=1.000, error=0.045
>epoch=130, lrate=1.000, error=0.045
>epoch=131, lrate=1.000, error=0.044
>epoch=132, lrate=1.000, error=0.044
>epoch=133, lrate=1.000, error=0.044
>epoch=134, lrate=1.000, error=0.043
>epoch=135, lrate=1.000, error=0.043
>epoch=136, lrate=1.000, error=0.042
>epoch=137, lrate=1.000, error=0.042
>epoch=138, lrate=1.000, error=0.042
>epoch=139, lrate=1.000, error=0.041
>epoch=140, lrate=1.000, error=0.041
>epoch=141, lrate=1.000, error=0.041
>epoch=142, lrate=1.000, error=0.040
>epoch=143, lrate=1.000, error=0.040
>epoch=144, lrate=1.000, error=0.040
>epoch=145, lrate=1.000, error=0.039
>epoch=146, lrate=1.000, error=0.039
>epoch=147, lrate=1.000, error=0.039
>epoch=148, lrate=1.000, error=0.039
>epoch=149, lrate=1.000, error=0.038
>epoch=150, lrate=1.000, error=0.038
>epoch=151, lrate=1.000, error=0.038
>epoch=152, lrate=1.000, error=0.037
>epoch=153, lrate=1.000, error=0.037
>epoch=154, lrate=1.000, error=0.037
>epoch=155, lrate=1.000, error=0.037
>epoch=156, lrate=1.000, error=0.036
>epoch=157, lrate=1.000, error=0.036
>epoch=158, lrate=1.000, error=0.036
>epoch=159, lrate=1.000, error=0.035
>epoch=160, lrate=1.000, error=0.035
>epoch=161, lrate=1.000, error=0.035
>epoch=162, lrate=1.000, error=0.035
>epoch=163, lrate=1.000, error=0.034
>epoch=164, lrate=1.000, error=0.034
>epoch=165, lrate=1.000, error=0.034
>epoch=166, lrate=1.000, error=0.034
>epoch=167, lrate=1.000, error=0.034
>epoch=168, lrate=1.000, error=0.033
>epoch=169, lrate=1.000, error=0.033
>epoch=170, lrate=1.000, error=0.033
>epoch=171, lrate=1.000, error=0.033
>epoch=172, lrate=1.000, error=0.032
>epoch=173, lrate=1.000, error=0.032
>epoch=174, lrate=1.000, error=0.032
>epoch=175, lrate=1.000, error=0.032
>epoch=176, lrate=1.000, error=0.032
>epoch=177, lrate=1.000, error=0.031
>epoch=178, lrate=1.000, error=0.031
>epoch=179, lrate=1.000, error=0.031
>epoch=180, lrate=1.000, error=0.031
>epoch=181, lrate=1.000, error=0.031
>epoch=182, lrate=1.000, error=0.030
>epoch=183, lrate=1.000, error=0.030
>epoch=184, lrate=1.000, error=0.030
>epoch=185, lrate=1.000, error=0.030
>epoch=186, lrate=1.000, error=0.030
>epoch=187, lrate=1.000, error=0.030
>epoch=188, lrate=1.000, error=0.029
>epoch=189, lrate=1.000, error=0.029
>epoch=190, lrate=1.000, error=0.029
>epoch=191, lrate=1.000, error=0.029
>epoch=192, lrate=1.000, error=0.029
>epoch=193, lrate=1.000, error=0.028
>epoch=194, lrate=1.000, error=0.028
>epoch=195, lrate=1.000, error=0.028
>epoch=196, lrate=1.000, error=0.028
>epoch=197, lrate=1.000, error=0.028
>epoch=198, lrate=1.000, error=0.028
>epoch=199, lrate=1.000, error=0.028
>epoch=200, lrate=1.000, error=0.027
>epoch=201, lrate=1.000, error=0.027
>epoch=202, lrate=1.000, error=0.027
>epoch=203, lrate=1.000, error=0.027
>epoch=204, lrate=1.000, error=0.027
>epoch=205, lrate=1.000, error=0.027
>epoch=206, lrate=1.000, error=0.026
>epoch=207, lrate=1.000, error=0.026
>epoch=208, lrate=1.000, error=0.026
>epoch=209, lrate=1.000, error=0.026
>epoch=210, lrate=1.000, error=0.026
>epoch=211, lrate=1.000, error=0.026
>epoch=212, lrate=1.000, error=0.026
>epoch=213, lrate=1.000, error=0.025
>epoch=214, lrate=1.000, error=0.025
>epoch=215, lrate=1.000, error=0.025
>epoch=216, lrate=1.000, error=0.025
>epoch=217, lrate=1.000, error=0.025
>epoch=218, lrate=1.000, error=0.025
>epoch=219, lrate=1.000, error=0.025
>epoch=220, lrate=1.000, error=0.025
>epoch=221, lrate=1.000, error=0.024
>epoch=222, lrate=1.000, error=0.024
>epoch=223, lrate=1.000, error=0.024
>epoch=224, lrate=1.000, error=0.024
>epoch=225, lrate=1.000, error=0.024
>epoch=226, lrate=1.000, error=0.024
>epoch=227, lrate=1.000, error=0.024
>epoch=228, lrate=1.000, error=0.024
>epoch=229, lrate=1.000, error=0.024
>epoch=230, lrate=1.000, error=0.023
>epoch=231, lrate=1.000, error=0.023
>epoch=232, lrate=1.000, error=0.023
>epoch=233, lrate=1.000, error=0.023
>epoch=234, lrate=1.000, error=0.023
>epoch=235, lrate=1.000, error=0.023
>epoch=236, lrate=1.000, error=0.023
>epoch=237, lrate=1.000, error=0.023
>epoch=238, lrate=1.000, error=0.023
>epoch=239, lrate=1.000, error=0.022
>epoch=240, lrate=1.000, error=0.022
>epoch=241, lrate=1.000, error=0.022
>epoch=242, lrate=1.000, error=0.022
>epoch=243, lrate=1.000, error=0.022
>epoch=244, lrate=1.000, error=0.022
>epoch=245, lrate=1.000, error=0.022
>epoch=246, lrate=1.000, error=0.022
>epoch=247, lrate=1.000, error=0.022
>epoch=248, lrate=1.000, error=0.022
>epoch=249, lrate=1.000, error=0.021
>epoch=250, lrate=1.000, error=0.021
>epoch=251, lrate=1.000, error=0.021
>epoch=252, lrate=1.000, error=0.021
>epoch=253, lrate=1.000, error=0.021
>epoch=254, lrate=1.000, error=0.021
>epoch=255, lrate=1.000, error=0.021
>epoch=256, lrate=1.000, error=0.021
>epoch=257, lrate=1.000, error=0.021
>epoch=258, lrate=1.000, error=0.021
>epoch=259, lrate=1.000, error=0.021
>epoch=260, lrate=1.000, error=0.020
>epoch=261, lrate=1.000, error=0.020
>epoch=262, lrate=1.000, error=0.020
>epoch=263, lrate=1.000, error=0.020
>epoch=264, lrate=1.000, error=0.020
>epoch=265, lrate=1.000, error=0.020
>epoch=266, lrate=1.000, error=0.020
>epoch=267, lrate=1.000, error=0.020
>epoch=268, lrate=1.000, error=0.020
>epoch=269, lrate=1.000, error=0.020
>epoch=270, lrate=1.000, error=0.020
>epoch=271, lrate=1.000, error=0.020
>epoch=272, lrate=1.000, error=0.019
>epoch=273, lrate=1.000, error=0.019
>epoch=274, lrate=1.000, error=0.019
>epoch=275, lrate=1.000, error=0.019
>epoch=276, lrate=1.000, error=0.019
>epoch=277, lrate=1.000, error=0.019
>epoch=278, lrate=1.000, error=0.019
>epoch=279, lrate=1.000, error=0.019
>epoch=280, lrate=1.000, error=0.019
>epoch=281, lrate=1.000, error=0.019
>epoch=282, lrate=1.000, error=0.019
>epoch=283, lrate=1.000, error=0.019
>epoch=284, lrate=1.000, error=0.019
>epoch=285, lrate=1.000, error=0.018
>epoch=286, lrate=1.000, error=0.018
>epoch=287, lrate=1.000, error=0.018
>epoch=288, lrate=1.000, error=0.018
>epoch=289, lrate=1.000, error=0.018
>epoch=290, lrate=1.000, error=0.018
>epoch=291, lrate=1.000, error=0.018
>epoch=292, lrate=1.000, error=0.018
>epoch=293, lrate=1.000, error=0.018
>epoch=294, lrate=1.000, error=0.018
>epoch=295, lrate=1.000, error=0.018
>epoch=296, lrate=1.000, error=0.018
>epoch=297, lrate=1.000, error=0.018
>epoch=298, lrate=1.000, error=0.018
>epoch=299, lrate=1.000, error=0.018
>epoch=300, lrate=1.000, error=0.017
>epoch=301, lrate=1.000, error=0.017
>epoch=302, lrate=1.000, error=0.017
>epoch=303, lrate=1.000, error=0.017
>epoch=304, lrate=1.000, error=0.017
>epoch=305, lrate=1.000, error=0.017
>epoch=306, lrate=1.000, error=0.017
>epoch=307, lrate=1.000, error=0.017
>epoch=308, lrate=1.000, error=0.017
>epoch=309, lrate=1.000, error=0.017
>epoch=310, lrate=1.000, error=0.017
>epoch=311, lrate=1.000, error=0.017
>epoch=312, lrate=1.000, error=0.017
>epoch=313, lrate=1.000, error=0.017
>epoch=314, lrate=1.000, error=0.017
>epoch=315, lrate=1.000, error=0.017
>epoch=316, lrate=1.000, error=0.017
>epoch=317, lrate=1.000, error=0.016
>epoch=318, lrate=1.000, error=0.016
>epoch=319, lrate=1.000, error=0.016
>epoch=320, lrate=1.000, error=0.016
>epoch=321, lrate=1.000, error=0.016
>epoch=322, lrate=1.000, error=0.016
>epoch=323, lrate=1.000, error=0.016
>epoch=324, lrate=1.000, error=0.016
>epoch=325, lrate=1.000, error=0.016
>epoch=326, lrate=1.000, error=0.016
>epoch=327, lrate=1.000, error=0.016
>epoch=328, lrate=1.000, error=0.016
>epoch=329, lrate=1.000, error=0.016
>epoch=330, lrate=1.000, error=0.016
>epoch=331, lrate=1.000, error=0.016
>epoch=332, lrate=1.000, error=0.016
>epoch=333, lrate=1.000, error=0.016
>epoch=334, lrate=1.000, error=0.016
>epoch=335, lrate=1.000, error=0.015
>epoch=336, lrate=1.000, error=0.015
>epoch=337, lrate=1.000, error=0.015
>epoch=338, lrate=1.000, error=0.015
>epoch=339, lrate=1.000, error=0.015
>epoch=340, lrate=1.000, error=0.015
>epoch=341, lrate=1.000, error=0.015
>epoch=342, lrate=1.000, error=0.015
>epoch=343, lrate=1.000, error=0.015
>epoch=344, lrate=1.000, error=0.015
>epoch=345, lrate=1.000, error=0.015
>epoch=346, lrate=1.000, error=0.015
>epoch=347, lrate=1.000, error=0.015
>epoch=348, lrate=1.000, error=0.015
>epoch=349, lrate=1.000, error=0.015
>epoch=350, lrate=1.000, error=0.015
>epoch=351, lrate=1.000, error=0.015
>epoch=352, lrate=1.000, error=0.015
>epoch=353, lrate=1.000, error=0.015
>epoch=354, lrate=1.000, error=0.015
>epoch=355, lrate=1.000, error=0.015
>epoch=356, lrate=1.000, error=0.014
>epoch=357, lrate=1.000, error=0.014
>epoch=358, lrate=1.000, error=0.014
>epoch=359, lrate=1.000, error=0.014
>epoch=360, lrate=1.000, error=0.014
>epoch=361, lrate=1.000, error=0.014
>epoch=362, lrate=1.000, error=0.014
>epoch=363, lrate=1.000, error=0.014
>epoch=364, lrate=1.000, error=0.014
>epoch=365, lrate=1.000, error=0.014
>epoch=366, lrate=1.000, error=0.014
>epoch=367, lrate=1.000, error=0.014
>epoch=368, lrate=1.000, error=0.014
>epoch=369, lrate=1.000, error=0.014
>epoch=370, lrate=1.000, error=0.014
>epoch=371, lrate=1.000, error=0.014
>epoch=372, lrate=1.000, error=0.014
>epoch=373, lrate=1.000, error=0.014
>epoch=374, lrate=1.000, error=0.014
>epoch=375, lrate=1.000, error=0.014
>epoch=376, lrate=1.000, error=0.014
>epoch=377, lrate=1.000, error=0.014
>epoch=378, lrate=1.000, error=0.014
>epoch=379, lrate=1.000, error=0.014
>epoch=380, lrate=1.000, error=0.014
>epoch=381, lrate=1.000, error=0.013
>epoch=382, lrate=1.000, error=0.013
>epoch=383, lrate=1.000, error=0.013
>epoch=384, lrate=1.000, error=0.013
>epoch=385, lrate=1.000, error=0.013
>epoch=386, lrate=1.000, error=0.013
>epoch=387, lrate=1.000, error=0.013
>epoch=388, lrate=1.000, error=0.013
>epoch=389, lrate=1.000, error=0.013
>epoch=390, lrate=1.000, error=0.013
>epoch=391, lrate=1.000, error=0.013
>epoch=392, lrate=1.000, error=0.013
>epoch=393, lrate=1.000, error=0.013
>epoch=394, lrate=1.000, error=0.013
>epoch=395, lrate=1.000, error=0.013
>epoch=396, lrate=1.000, error=0.013
>epoch=397, lrate=1.000, error=0.013
>epoch=398, lrate=1.000, error=0.013
>epoch=399, lrate=1.000, error=0.013
>epoch=400, lrate=1.000, error=0.013
>epoch=401, lrate=1.000, error=0.013
>epoch=402, lrate=1.000, error=0.013
>epoch=403, lrate=1.000, error=0.013
>epoch=404, lrate=1.000, error=0.013
>epoch=405, lrate=1.000, error=0.013
>epoch=406, lrate=1.000, error=0.013
>epoch=407, lrate=1.000, error=0.013
>epoch=408, lrate=1.000, error=0.013
>epoch=409, lrate=1.000, error=0.012
>epoch=410, lrate=1.000, error=0.012
>epoch=411, lrate=1.000, error=0.012
>epoch=412, lrate=1.000, error=0.012
>epoch=413, lrate=1.000, error=0.012
>epoch=414, lrate=1.000, error=0.012
>epoch=415, lrate=1.000, error=0.012
>epoch=416, lrate=1.000, error=0.012
>epoch=417, lrate=1.000, error=0.012
>epoch=418, lrate=1.000, error=0.012
>epoch=419, lrate=1.000, error=0.012
>epoch=420, lrate=1.000, error=0.012
>epoch=421, lrate=1.000, error=0.012
>epoch=422, lrate=1.000, error=0.012
>epoch=423, lrate=1.000, error=0.012
>epoch=424, lrate=1.000, error=0.012
>epoch=425, lrate=1.000, error=0.012
>epoch=426, lrate=1.000, error=0.012
>epoch=427, lrate=1.000, error=0.012
>epoch=428, lrate=1.000, error=0.012
>epoch=429, lrate=1.000, error=0.012
>epoch=430, lrate=1.000, error=0.012
>epoch=431, lrate=1.000, error=0.012
>epoch=432, lrate=1.000, error=0.012
>epoch=433, lrate=1.000, error=0.012
>epoch=434, lrate=1.000, error=0.012
>epoch=435, lrate=1.000, error=0.012
>epoch=436, lrate=1.000, error=0.012
>epoch=437, lrate=1.000, error=0.012
>epoch=438, lrate=1.000, error=0.012
>epoch=439, lrate=1.000, error=0.012
>epoch=440, lrate=1.000, error=0.012
>epoch=441, lrate=1.000, error=0.012
>epoch=442, lrate=1.000, error=0.011
>epoch=443, lrate=1.000, error=0.011
>epoch=444, lrate=1.000, error=0.011
>epoch=445, lrate=1.000, error=0.011
>epoch=446, lrate=1.000, error=0.011
>epoch=447, lrate=1.000, error=0.011
>epoch=448, lrate=1.000, error=0.011
>epoch=449, lrate=1.000, error=0.011
>epoch=450, lrate=1.000, error=0.011
>epoch=451, lrate=1.000, error=0.011
>epoch=452, lrate=1.000, error=0.011
>epoch=453, lrate=1.000, error=0.011
>epoch=454, lrate=1.000, error=0.011
>epoch=455, lrate=1.000, error=0.011
>epoch=456, lrate=1.000, error=0.011
>epoch=457, lrate=1.000, error=0.011
>epoch=458, lrate=1.000, error=0.011
>epoch=459, lrate=1.000, error=0.011
>epoch=460, lrate=1.000, error=0.011
>epoch=461, lrate=1.000, error=0.011
>epoch=462, lrate=1.000, error=0.011
>epoch=463, lrate=1.000, error=0.011
>epoch=464, lrate=1.000, error=0.011
>epoch=465, lrate=1.000, error=0.011
>epoch=466, lrate=1.000, error=0.011
>epoch=467, lrate=1.000, error=0.011
>epoch=468, lrate=1.000, error=0.011
>epoch=469, lrate=1.000, error=0.011
>epoch=470, lrate=1.000, error=0.011
>epoch=471, lrate=1.000, error=0.011
>epoch=472, lrate=1.000, error=0.011
>epoch=473, lrate=1.000, error=0.011
>epoch=474, lrate=1.000, error=0.011
>epoch=475, lrate=1.000, error=0.011
>epoch=476, lrate=1.000, error=0.011
>epoch=477, lrate=1.000, error=0.011
>epoch=478, lrate=1.000, error=0.011
>epoch=479, lrate=1.000, error=0.011
>epoch=480, lrate=1.000, error=0.011
>epoch=481, lrate=1.000, error=0.010
>epoch=482, lrate=1.000, error=0.010
>epoch=483, lrate=1.000, error=0.010
>epoch=484, lrate=1.000, error=0.010
>epoch=485, lrate=1.000, error=0.010
>epoch=486, lrate=1.000, error=0.010
>epoch=487, lrate=1.000, error=0.010
>epoch=488, lrate=1.000, error=0.010
>epoch=489, lrate=1.000, error=0.010
>epoch=490, lrate=1.000, error=0.010
>epoch=491, lrate=1.000, error=0.010
>epoch=492, lrate=1.000, error=0.010
>epoch=493, lrate=1.000, error=0.010
>epoch=494, lrate=1.000, error=0.010
>epoch=495, lrate=1.000, error=0.010
>epoch=496, lrate=1.000, error=0.010
>epoch=497, lrate=1.000, error=0.010
>epoch=498, lrate=1.000, error=0.010
>epoch=499, lrate=1.000, error=0.010
>epoch=500, lrate=1.000, error=0.010
>epoch=501, lrate=1.000, error=0.010
>epoch=502, lrate=1.000, error=0.010
>epoch=503, lrate=1.000, error=0.010
>epoch=504, lrate=1.000, error=0.010
>epoch=505, lrate=1.000, error=0.010
>epoch=506, lrate=1.000, error=0.010
>epoch=507, lrate=1.000, error=0.010
>epoch=508, lrate=1.000, error=0.010
>epoch=509, lrate=1.000, error=0.010
>epoch=510, lrate=1.000, error=0.010
>epoch=511, lrate=1.000, error=0.010
>epoch=512, lrate=1.000, error=0.010
>epoch=513, lrate=1.000, error=0.010
>epoch=514, lrate=1.000, error=0.010
>epoch=515, lrate=1.000, error=0.010
>epoch=516, lrate=1.000, error=0.010
>epoch=517, lrate=1.000, error=0.010
>epoch=518, lrate=1.000, error=0.010
>epoch=519, lrate=1.000, error=0.010
>epoch=520, lrate=1.000, error=0.010
>epoch=521, lrate=1.000, error=0.010
>epoch=522, lrate=1.000, error=0.010
>epoch=523, lrate=1.000, error=0.010
>epoch=524, lrate=1.000, error=0.010
>epoch=525, lrate=1.000, error=0.010
>epoch=526, lrate=1.000, error=0.010
>epoch=527, lrate=1.000, error=0.010
>epoch=528, lrate=1.000, error=0.010
>epoch=529, lrate=1.000, error=0.009
>epoch=530, lrate=1.000, error=0.009
>epoch=531, lrate=1.000, error=0.009
>epoch=532, lrate=1.000, error=0.009
>epoch=533, lrate=1.000, error=0.009
>epoch=534, lrate=1.000, error=0.009
>epoch=535, lrate=1.000, error=0.009
>epoch=536, lrate=1.000, error=0.009
>epoch=537, lrate=1.000, error=0.009
>epoch=538, lrate=1.000, error=0.009
>epoch=539, lrate=1.000, error=0.009
>epoch=540, lrate=1.000, error=0.009
>epoch=541, lrate=1.000, error=0.009
>epoch=542, lrate=1.000, error=0.009
>epoch=543, lrate=1.000, error=0.009
>epoch=544, lrate=1.000, error=0.009
>epoch=545, lrate=1.000, error=0.009
>epoch=546, lrate=1.000, error=0.009
>epoch=547, lrate=1.000, error=0.009
>epoch=548, lrate=1.000, error=0.009
>epoch=549, lrate=1.000, error=0.009
>epoch=550, lrate=1.000, error=0.009
>epoch=551, lrate=1.000, error=0.009
>epoch=552, lrate=1.000, error=0.009
>epoch=553, lrate=1.000, error=0.009
>epoch=554, lrate=1.000, error=0.009
>epoch=555, lrate=1.000, error=0.009
>epoch=556, lrate=1.000, error=0.009
>epoch=557, lrate=1.000, error=0.009
>epoch=558, lrate=1.000, error=0.009
>epoch=559, lrate=1.000, error=0.009
>epoch=560, lrate=1.000, error=0.009
>epoch=561, lrate=1.000, error=0.009
>epoch=562, lrate=1.000, error=0.009
>epoch=563, lrate=1.000, error=0.009
>epoch=564, lrate=1.000, error=0.009
>epoch=565, lrate=1.000, error=0.009
>epoch=566, lrate=1.000, error=0.009
>epoch=567, lrate=1.000, error=0.009
>epoch=568, lrate=1.000, error=0.009
>epoch=569, lrate=1.000, error=0.009
>epoch=570, lrate=1.000, error=0.009
>epoch=571, lrate=1.000, error=0.009
>epoch=572, lrate=1.000, error=0.009
>epoch=573, lrate=1.000, error=0.009
>epoch=574, lrate=1.000, error=0.009
>epoch=575, lrate=1.000, error=0.009
>epoch=576, lrate=1.000, error=0.009
>epoch=577, lrate=1.000, error=0.009
>epoch=578, lrate=1.000, error=0.009
>epoch=579, lrate=1.000, error=0.009
>epoch=580, lrate=1.000, error=0.009
>epoch=581, lrate=1.000, error=0.009
>epoch=582, lrate=1.000, error=0.009
>epoch=583, lrate=1.000, error=0.009
>epoch=584, lrate=1.000, error=0.009
>epoch=585, lrate=1.000, error=0.009
>epoch=586, lrate=1.000, error=0.009
>epoch=587, lrate=1.000, error=0.008
>epoch=588, lrate=1.000, error=0.008
>epoch=589, lrate=1.000, error=0.008
>epoch=590, lrate=1.000, error=0.008
>epoch=591, lrate=1.000, error=0.008
>epoch=592, lrate=1.000, error=0.008
>epoch=593, lrate=1.000, error=0.008
>epoch=594, lrate=1.000, error=0.008
>epoch=595, lrate=1.000, error=0.008
>epoch=596, lrate=1.000, error=0.008
>epoch=597, lrate=1.000, error=0.008
>epoch=598, lrate=1.000, error=0.008
>epoch=599, lrate=1.000, error=0.008
>epoch=600, lrate=1.000, error=0.008
>epoch=601, lrate=1.000, error=0.008
>epoch=602, lrate=1.000, error=0.008
>epoch=603, lrate=1.000, error=0.008
>epoch=604, lrate=1.000, error=0.008
>epoch=605, lrate=1.000, error=0.008
>epoch=606, lrate=1.000, error=0.008
>epoch=607, lrate=1.000, error=0.008
>epoch=608, lrate=1.000, error=0.008
>epoch=609, lrate=1.000, error=0.008
>epoch=610, lrate=1.000, error=0.008
>epoch=611, lrate=1.000, error=0.008
>epoch=612, lrate=1.000, error=0.008
>epoch=613, lrate=1.000, error=0.008
>epoch=614, lrate=1.000, error=0.008
>epoch=615, lrate=1.000, error=0.008
>epoch=616, lrate=1.000, error=0.008
>epoch=617, lrate=1.000, error=0.008
>epoch=618, lrate=1.000, error=0.008
>epoch=619, lrate=1.000, error=0.008
>epoch=620, lrate=1.000, error=0.008
>epoch=621, lrate=1.000, error=0.008
>epoch=622, lrate=1.000, error=0.008
>epoch=623, lrate=1.000, error=0.008
>epoch=624, lrate=1.000, error=0.008
>epoch=625, lrate=1.000, error=0.008
>epoch=626, lrate=1.000, error=0.008
>epoch=627, lrate=1.000, error=0.008
>epoch=628, lrate=1.000, error=0.008
>epoch=629, lrate=1.000, error=0.008
>epoch=630, lrate=1.000, error=0.008
>epoch=631, lrate=1.000, error=0.008
>epoch=632, lrate=1.000, error=0.008
>epoch=633, lrate=1.000, error=0.008
>epoch=634, lrate=1.000, error=0.008
>epoch=635, lrate=1.000, error=0.008
>epoch=636, lrate=1.000, error=0.008
>epoch=637, lrate=1.000, error=0.008
>epoch=638, lrate=1.000, error=0.008
>epoch=639, lrate=1.000, error=0.008
>epoch=640, lrate=1.000, error=0.008
>epoch=641, lrate=1.000, error=0.008
>epoch=642, lrate=1.000, error=0.008
>epoch=643, lrate=1.000, error=0.008
>epoch=644, lrate=1.000, error=0.008
>epoch=645, lrate=1.000, error=0.008
>epoch=646, lrate=1.000, error=0.008
>epoch=647, lrate=1.000, error=0.008
>epoch=648, lrate=1.000, error=0.008
>epoch=649, lrate=1.000, error=0.008
>epoch=650, lrate=1.000, error=0.008
>epoch=651, lrate=1.000, error=0.008
>epoch=652, lrate=1.000, error=0.008
>epoch=653, lrate=1.000, error=0.008
>epoch=654, lrate=1.000, error=0.008
>epoch=655, lrate=1.000, error=0.008
>epoch=656, lrate=1.000, error=0.008
>epoch=657, lrate=1.000, error=0.008
>epoch=658, lrate=1.000, error=0.008
>epoch=659, lrate=1.000, error=0.008
>epoch=660, lrate=1.000, error=0.008
>epoch=661, lrate=1.000, error=0.007
>epoch=662, lrate=1.000, error=0.007
>epoch=663, lrate=1.000, error=0.007
>epoch=664, lrate=1.000, error=0.007
>epoch=665, lrate=1.000, error=0.007
>epoch=666, lrate=1.000, error=0.007
>epoch=667, lrate=1.000, error=0.007
>epoch=668, lrate=1.000, error=0.007
>epoch=669, lrate=1.000, error=0.007
>epoch=670, lrate=1.000, error=0.007
>epoch=671, lrate=1.000, error=0.007
>epoch=672, lrate=1.000, error=0.007
>epoch=673, lrate=1.000, error=0.007
>epoch=674, lrate=1.000, error=0.007
>epoch=675, lrate=1.000, error=0.007
>epoch=676, lrate=1.000, error=0.007
>epoch=677, lrate=1.000, error=0.007
>epoch=678, lrate=1.000, error=0.007
>epoch=679, lrate=1.000, error=0.007
>epoch=680, lrate=1.000, error=0.007
>epoch=681, lrate=1.000, error=0.007
>epoch=682, lrate=1.000, error=0.007
>epoch=683, lrate=1.000, error=0.007
>epoch=684, lrate=1.000, error=0.007
>epoch=685, lrate=1.000, error=0.007
>epoch=686, lrate=1.000, error=0.007
>epoch=687, lrate=1.000, error=0.007
>epoch=688, lrate=1.000, error=0.007
>epoch=689, lrate=1.000, error=0.007
>epoch=690, lrate=1.000, error=0.007
>epoch=691, lrate=1.000, error=0.007
>epoch=692, lrate=1.000, error=0.007
>epoch=693, lrate=1.000, error=0.007
>epoch=694, lrate=1.000, error=0.007
>epoch=695, lrate=1.000, error=0.007
>epoch=696, lrate=1.000, error=0.007
>epoch=697, lrate=1.000, error=0.007
>epoch=698, lrate=1.000, error=0.007
>epoch=699, lrate=1.000, error=0.007
>epoch=700, lrate=1.000, error=0.007
>epoch=701, lrate=1.000, error=0.007
>epoch=702, lrate=1.000, error=0.007
>epoch=703, lrate=1.000, error=0.007
>epoch=704, lrate=1.000, error=0.007
>epoch=705, lrate=1.000, error=0.007
>epoch=706, lrate=1.000, error=0.007
>epoch=707, lrate=1.000, error=0.007
>epoch=708, lrate=1.000, error=0.007
>epoch=709, lrate=1.000, error=0.007
>epoch=710, lrate=1.000, error=0.007
>epoch=711, lrate=1.000, error=0.007
>epoch=712, lrate=1.000, error=0.007
>epoch=713, lrate=1.000, error=0.007
>epoch=714, lrate=1.000, error=0.007
>epoch=715, lrate=1.000, error=0.007
>epoch=716, lrate=1.000, error=0.007
>epoch=717, lrate=1.000, error=0.007
>epoch=718, lrate=1.000, error=0.007
>epoch=719, lrate=1.000, error=0.007
>epoch=720, lrate=1.000, error=0.007
>epoch=721, lrate=1.000, error=0.007
>epoch=722, lrate=1.000, error=0.007
>epoch=723, lrate=1.000, error=0.007
>epoch=724, lrate=1.000, error=0.007
>epoch=725, lrate=1.000, error=0.007
>epoch=726, lrate=1.000, error=0.007
>epoch=727, lrate=1.000, error=0.007
>epoch=728, lrate=1.000, error=0.007
>epoch=729, lrate=1.000, error=0.007
>epoch=730, lrate=1.000, error=0.007
>epoch=731, lrate=1.000, error=0.007
>epoch=732, lrate=1.000, error=0.007
>epoch=733, lrate=1.000, error=0.007
>epoch=734, lrate=1.000, error=0.007
>epoch=735, lrate=1.000, error=0.007
>epoch=736, lrate=1.000, error=0.007
>epoch=737, lrate=1.000, error=0.007
>epoch=738, lrate=1.000, error=0.007
>epoch=739, lrate=1.000, error=0.007
>epoch=740, lrate=1.000, error=0.007
>epoch=741, lrate=1.000, error=0.007
>epoch=742, lrate=1.000, error=0.007
>epoch=743, lrate=1.000, error=0.007
>epoch=744, lrate=1.000, error=0.007
>epoch=745, lrate=1.000, error=0.007
>epoch=746, lrate=1.000, error=0.007
>epoch=747, lrate=1.000, error=0.007
>epoch=748, lrate=1.000, error=0.007
>epoch=749, lrate=1.000, error=0.007
>epoch=750, lrate=1.000, error=0.007
>epoch=751, lrate=1.000, error=0.007
>epoch=752, lrate=1.000, error=0.007
>epoch=753, lrate=1.000, error=0.007
>epoch=754, lrate=1.000, error=0.007
>epoch=755, lrate=1.000, error=0.007
>epoch=756, lrate=1.000, error=0.007
>epoch=757, lrate=1.000, error=0.006
>epoch=758, lrate=1.000, error=0.006
>epoch=759, lrate=1.000, error=0.006
>epoch=760, lrate=1.000, error=0.006
>epoch=761, lrate=1.000, error=0.006
>epoch=762, lrate=1.000, error=0.006
>epoch=763, lrate=1.000, error=0.006
>epoch=764, lrate=1.000, error=0.006
>epoch=765, lrate=1.000, error=0.006
>epoch=766, lrate=1.000, error=0.006
>epoch=767, lrate=1.000, error=0.006
>epoch=768, lrate=1.000, error=0.006
>epoch=769, lrate=1.000, error=0.006
>epoch=770, lrate=1.000, error=0.006
>epoch=771, lrate=1.000, error=0.006
>epoch=772, lrate=1.000, error=0.006
>epoch=773, lrate=1.000, error=0.006
>epoch=774, lrate=1.000, error=0.006
>epoch=775, lrate=1.000, error=0.006
>epoch=776, lrate=1.000, error=0.006
>epoch=777, lrate=1.000, error=0.006
>epoch=778, lrate=1.000, error=0.006
>epoch=779, lrate=1.000, error=0.006
>epoch=780, lrate=1.000, error=0.006
>epoch=781, lrate=1.000, error=0.006
>epoch=782, lrate=1.000, error=0.006
>epoch=783, lrate=1.000, error=0.006
>epoch=784, lrate=1.000, error=0.006
>epoch=785, lrate=1.000, error=0.006
>epoch=786, lrate=1.000, error=0.006
>epoch=787, lrate=1.000, error=0.006
>epoch=788, lrate=1.000, error=0.006
>epoch=789, lrate=1.000, error=0.006
>epoch=790, lrate=1.000, error=0.006
>epoch=791, lrate=1.000, error=0.006
>epoch=792, lrate=1.000, error=0.006
>epoch=793, lrate=1.000, error=0.006
>epoch=794, lrate=1.000, error=0.006
>epoch=795, lrate=1.000, error=0.006
>epoch=796, lrate=1.000, error=0.006
>epoch=797, lrate=1.000, error=0.006
>epoch=798, lrate=1.000, error=0.006
>epoch=799, lrate=1.000, error=0.006
>epoch=800, lrate=1.000, error=0.006
>epoch=801, lrate=1.000, error=0.006
>epoch=802, lrate=1.000, error=0.006
>epoch=803, lrate=1.000, error=0.006
>epoch=804, lrate=1.000, error=0.006
>epoch=805, lrate=1.000, error=0.006
>epoch=806, lrate=1.000, error=0.006
>epoch=807, lrate=1.000, error=0.006
>epoch=808, lrate=1.000, error=0.006
>epoch=809, lrate=1.000, error=0.006
>epoch=810, lrate=1.000, error=0.006
>epoch=811, lrate=1.000, error=0.006
>epoch=812, lrate=1.000, error=0.006
>epoch=813, lrate=1.000, error=0.006
>epoch=814, lrate=1.000, error=0.006
>epoch=815, lrate=1.000, error=0.006
>epoch=816, lrate=1.000, error=0.006
>epoch=817, lrate=1.000, error=0.006
>epoch=818, lrate=1.000, error=0.006
>epoch=819, lrate=1.000, error=0.006
>epoch=820, lrate=1.000, error=0.006
>epoch=821, lrate=1.000, error=0.006
>epoch=822, lrate=1.000, error=0.006
>epoch=823, lrate=1.000, error=0.006
>epoch=824, lrate=1.000, error=0.006
>epoch=825, lrate=1.000, error=0.006
>epoch=826, lrate=1.000, error=0.006
>epoch=827, lrate=1.000, error=0.006
>epoch=828, lrate=1.000, error=0.006
>epoch=829, lrate=1.000, error=0.006
>epoch=830, lrate=1.000, error=0.006
>epoch=831, lrate=1.000, error=0.006
>epoch=832, lrate=1.000, error=0.006
>epoch=833, lrate=1.000, error=0.006
>epoch=834, lrate=1.000, error=0.006
>epoch=835, lrate=1.000, error=0.006
>epoch=836, lrate=1.000, error=0.006
>epoch=837, lrate=1.000, error=0.006
>epoch=838, lrate=1.000, error=0.006
>epoch=839, lrate=1.000, error=0.006
>epoch=840, lrate=1.000, error=0.006
>epoch=841, lrate=1.000, error=0.006
>epoch=842, lrate=1.000, error=0.006
>epoch=843, lrate=1.000, error=0.006
>epoch=844, lrate=1.000, error=0.006
>epoch=845, lrate=1.000, error=0.006
>epoch=846, lrate=1.000, error=0.006
>epoch=847, lrate=1.000, error=0.006
>epoch=848, lrate=1.000, error=0.006
>epoch=849, lrate=1.000, error=0.006
>epoch=850, lrate=1.000, error=0.006
>epoch=851, lrate=1.000, error=0.006
>epoch=852, lrate=1.000, error=0.006
>epoch=853, lrate=1.000, error=0.006
>epoch=854, lrate=1.000, error=0.006
>epoch=855, lrate=1.000, error=0.006
>epoch=856, lrate=1.000, error=0.006
>epoch=857, lrate=1.000, error=0.006
>epoch=858, lrate=1.000, error=0.006
>epoch=859, lrate=1.000, error=0.006
>epoch=860, lrate=1.000, error=0.006
>epoch=861, lrate=1.000, error=0.006
>epoch=862, lrate=1.000, error=0.006
>epoch=863, lrate=1.000, error=0.006
>epoch=864, lrate=1.000, error=0.006
>epoch=865, lrate=1.000, error=0.006
>epoch=866, lrate=1.000, error=0.006
>epoch=867, lrate=1.000, error=0.006
>epoch=868, lrate=1.000, error=0.006
>epoch=869, lrate=1.000, error=0.006
>epoch=870, lrate=1.000, error=0.006
>epoch=871, lrate=1.000, error=0.006
>epoch=872, lrate=1.000, error=0.006
>epoch=873, lrate=1.000, error=0.006
>epoch=874, lrate=1.000, error=0.006
>epoch=875, lrate=1.000, error=0.006
>epoch=876, lrate=1.000, error=0.006
>epoch=877, lrate=1.000, error=0.006
>epoch=878, lrate=1.000, error=0.006
>epoch=879, lrate=1.000, error=0.006
>epoch=880, lrate=1.000, error=0.006
>epoch=881, lrate=1.000, error=0.006
>epoch=882, lrate=1.000, error=0.006
>epoch=883, lrate=1.000, error=0.006
>epoch=884, lrate=1.000, error=0.006
>epoch=885, lrate=1.000, error=0.006
>epoch=886, lrate=1.000, error=0.006
>epoch=887, lrate=1.000, error=0.006
>epoch=888, lrate=1.000, error=0.005
>epoch=889, lrate=1.000, error=0.005
>epoch=890, lrate=1.000, error=0.005
>epoch=891, lrate=1.000, error=0.005
>epoch=892, lrate=1.000, error=0.005
>epoch=893, lrate=1.000, error=0.005
>epoch=894, lrate=1.000, error=0.005
>epoch=895, lrate=1.000, error=0.005
>epoch=896, lrate=1.000, error=0.005
>epoch=897, lrate=1.000, error=0.005
>epoch=898, lrate=1.000, error=0.005
>epoch=899, lrate=1.000, error=0.005
>epoch=900, lrate=1.000, error=0.005
>epoch=901, lrate=1.000, error=0.005
>epoch=902, lrate=1.000, error=0.005
>epoch=903, lrate=1.000, error=0.005
>epoch=904, lrate=1.000, error=0.005
>epoch=905, lrate=1.000, error=0.005
>epoch=906, lrate=1.000, error=0.005
>epoch=907, lrate=1.000, error=0.005
>epoch=908, lrate=1.000, error=0.005
>epoch=909, lrate=1.000, error=0.005
>epoch=910, lrate=1.000, error=0.005
>epoch=911, lrate=1.000, error=0.005
>epoch=912, lrate=1.000, error=0.005
>epoch=913, lrate=1.000, error=0.005
>epoch=914, lrate=1.000, error=0.005
>epoch=915, lrate=1.000, error=0.005
>epoch=916, lrate=1.000, error=0.005
>epoch=917, lrate=1.000, error=0.005
>epoch=918, lrate=1.000, error=0.005
>epoch=919, lrate=1.000, error=0.005
>epoch=920, lrate=1.000, error=0.005
>epoch=921, lrate=1.000, error=0.005
>epoch=922, lrate=1.000, error=0.005
>epoch=923, lrate=1.000, error=0.005
>epoch=924, lrate=1.000, error=0.005
>epoch=925, lrate=1.000, error=0.005
>epoch=926, lrate=1.000, error=0.005
>epoch=927, lrate=1.000, error=0.005
>epoch=928, lrate=1.000, error=0.005
>epoch=929, lrate=1.000, error=0.005
>epoch=930, lrate=1.000, error=0.005
>epoch=931, lrate=1.000, error=0.005
>epoch=932, lrate=1.000, error=0.005
>epoch=933, lrate=1.000, error=0.005
>epoch=934, lrate=1.000, error=0.005
>epoch=935, lrate=1.000, error=0.005
>epoch=936, lrate=1.000, error=0.005
>epoch=937, lrate=1.000, error=0.005
>epoch=938, lrate=1.000, error=0.005
>epoch=939, lrate=1.000, error=0.005
>epoch=940, lrate=1.000, error=0.005
>epoch=941, lrate=1.000, error=0.005
>epoch=942, lrate=1.000, error=0.005
>epoch=943, lrate=1.000, error=0.005
>epoch=944, lrate=1.000, error=0.005
>epoch=945, lrate=1.000, error=0.005
>epoch=946, lrate=1.000, error=0.005
>epoch=947, lrate=1.000, error=0.005
>epoch=948, lrate=1.000, error=0.005
>epoch=949, lrate=1.000, error=0.005
>epoch=950, lrate=1.000, error=0.005
>epoch=951, lrate=1.000, error=0.005
>epoch=952, lrate=1.000, error=0.005
>epoch=953, lrate=1.000, error=0.005
>epoch=954, lrate=1.000, error=0.005
>epoch=955, lrate=1.000, error=0.005
>epoch=956, lrate=1.000, error=0.005
>epoch=957, lrate=1.000, error=0.005
>epoch=958, lrate=1.000, error=0.005
>epoch=959, lrate=1.000, error=0.005
>epoch=960, lrate=1.000, error=0.005
>epoch=961, lrate=1.000, error=0.005
>epoch=962, lrate=1.000, error=0.005
>epoch=963, lrate=1.000, error=0.005
>epoch=964, lrate=1.000, error=0.005
>epoch=965, lrate=1.000, error=0.005
>epoch=966, lrate=1.000, error=0.005
>epoch=967, lrate=1.000, error=0.005
>epoch=968, lrate=1.000, error=0.005
>epoch=969, lrate=1.000, error=0.005
>epoch=970, lrate=1.000, error=0.005
>epoch=971, lrate=1.000, error=0.005
>epoch=972, lrate=1.000, error=0.005
>epoch=973, lrate=1.000, error=0.005
>epoch=974, lrate=1.000, error=0.005
>epoch=975, lrate=1.000, error=0.005
>epoch=976, lrate=1.000, error=0.005
>epoch=977, lrate=1.000, error=0.005
>epoch=978, lrate=1.000, error=0.005
>epoch=979, lrate=1.000, error=0.005
>epoch=980, lrate=1.000, error=0.005
>epoch=981, lrate=1.000, error=0.005
>epoch=982, lrate=1.000, error=0.005
>epoch=983, lrate=1.000, error=0.005
>epoch=984, lrate=1.000, error=0.005
>epoch=985, lrate=1.000, error=0.005
>epoch=986, lrate=1.000, error=0.005
>epoch=987, lrate=1.000, error=0.005
>epoch=988, lrate=1.000, error=0.005
>epoch=989, lrate=1.000, error=0.005
>epoch=990, lrate=1.000, error=0.005
>epoch=991, lrate=1.000, error=0.005
>epoch=992, lrate=1.000, error=0.005
>epoch=993, lrate=1.000, error=0.005
>epoch=994, lrate=1.000, error=0.005
>epoch=995, lrate=1.000, error=0.005
>epoch=996, lrate=1.000, error=0.005
>epoch=997, lrate=1.000, error=0.005
>epoch=998, lrate=1.000, error=0.005
>epoch=999, lrate=1.000, error=0.005
[{'weights': [1.3334610117002106, 0.80917951597068, 0.8102576447449061], 'output': 0.9999990643000904, 'delta': -8.486019842246916e-10}, {'weights': [2.5651674647078955, -4.035903584500035, -0.8272658223717634], 'output': 0.9909373603667286, 'delta': 3.45122823718331e-05}]
[{'weights': [2.257630748020625, -8.410378564442574, 1.906725832293663], 'output': 0.015230273584542674, 'delta': -0.0002284284004134143}, {'weights': [-1.7106372012268205, 8.40633030321677, -2.450162335010934], 'output': 0.984762890956722, 'delta': 0.00022863190013066792}]
```

In \[71\]:

```python
# 학습한 네트워크로 예측값을 뽑아보자.

def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs)) # 순전파 결과에서 어떤것이 최종 아웃풋이 되나요?
    #output값으로 나온 두 값중 더 큰 값의 index값을 최종 아웃풋으로 된다!
```

In \[69\]:

```python
predict(network,row)
```

```text
[0.19890425627850458, 0.7877146498611793]
```

Out\[69\]:

```text
1
```

In \[72\]:

```python
# 네트워크가 잘 학습되었는지 확인해보자. 

for row in dataset:
    prediction = predict(network,row) # 앞서 최적(학습)시킨 네트워크로 잘 학습되었는지 평가 
    print('실제값=%d, 예측값=%d' % (row[-1], prediction)) #아주 잘 학습되었음!
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

