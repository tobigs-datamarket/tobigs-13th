---
description: 12기 이세윤
---

# CNN Alexnet

## AlexNet

* Local Response Normarlization
* Dropout \(비율 0.5\)
* Stochastic Gradient Descent Optimizer

In \[1\]:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```text
Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3aietf%3awg%3aoauth%3a2.0%3aoob&response_type=code&scope=email%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdocs.test%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive.photos.readonly%20https%3a%2f%2fwww.googleapis.com%2fauth%2fpeopleapi.readonly

Enter your authorization code:
··········
Mounted at /content/drive
```

### Pytorch

```python
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
```

```python
class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            ## [Layer 1] Convolution 
            # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
            
            ## [Layer 2] Max Pooling 
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            ## [Layer 3] Convolution 
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
            
            ## [Layer 4] Max Pooling
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            ## [Layer 5] Convolution
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            ## [Layer 6] Convolution
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            ## [Layer 7] Convolution
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            ## [Layer 8] Max Pooling
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.classifier = nn.Sequential(
            ## [Layer 9] Fully Connected Layer
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            ## [Layer 10] Fully Connected Layer
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            ## [Layer 11] Fully Connected Layer
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
```

In \[4\]:

```python
from torchsummary import summary

model = AlexNet()
model.cuda()
## Model Summary
summary(model, input_size=(3, 227, 227))
```

```text
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 96, 55, 55]          34,944
              ReLU-2           [-1, 96, 55, 55]               0
 LocalResponseNorm-3           [-1, 96, 55, 55]               0
         MaxPool2d-4           [-1, 96, 27, 27]               0
            Conv2d-5          [-1, 256, 27, 27]         614,656
              ReLU-6          [-1, 256, 27, 27]               0
 LocalResponseNorm-7          [-1, 256, 27, 27]               0
         MaxPool2d-8          [-1, 256, 13, 13]               0
            Conv2d-9          [-1, 384, 13, 13]         885,120
             ReLU-10          [-1, 384, 13, 13]               0
           Conv2d-11          [-1, 384, 13, 13]       1,327,488
             ReLU-12          [-1, 384, 13, 13]               0
           Conv2d-13          [-1, 256, 13, 13]         884,992
             ReLU-14          [-1, 256, 13, 13]               0
        MaxPool2d-15            [-1, 256, 6, 6]               0
          Dropout-16                 [-1, 9216]               0
           Linear-17                 [-1, 4096]      37,752,832
             ReLU-18                 [-1, 4096]               0
          Dropout-19                 [-1, 4096]               0
           Linear-20                 [-1, 4096]      16,781,312
             ReLU-21                 [-1, 4096]               0
           Linear-22                 [-1, 1000]       4,097,000
================================================================
Total params: 62,378,344
Trainable params: 62,378,344
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.59
Forward/backward pass size (MB): 14.73
Params size (MB): 237.95
Estimated Total Size (MB): 253.27
----------------------------------------------------------------
```

### Keras

In \[5\]:

```python
import tensorflow as tf
```

```python
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class LocalResponseNormalization(Layer):
  
    def __init__(self, n=5, alpha=1e-4, beta=0.75, k=2, **kwargs):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.k = k
        super(LocalResponseNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape
        super(LocalResponseNormalization, self).build(input_shape)

    def call(self, x):
        _, r, c, f = self.shape 
        squared = K.square(x)
        pooled = K.pool2d(squared, (self.n, self.n), strides=(1,1), padding="same", pool_mode='avg')
        summed = K.sum(pooled, axis=3, keepdims=True)
        averaged = self.alpha * K.repeat_elements(summed, f, axis=3)
        denom = K.pow(self.k + averaged, self.beta)
        return x / denom 
    
    def compute_output_shape(self, input_shape):
        return input_shape
```

In \[7\]:

```python
input_shape = (227, 227, 3)
num_classes = 1000

model = tf.keras.models.Sequential()
## [Layer 1] Convolution 
model.add(tf.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=4, 
                           padding='same', input_shape=input_shape))
    
## [Layer 2] Max Pooling 
model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2))
    
## [Layer 3] Convolution 
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=1,
                           activation="relu", padding='same'))
model.add(LocalResponseNormalization(input_shape=model.output_shape[1:]))
    
## [Layer 4] Max Pooling 
model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2))
  
## [Layer 5] Convolution
model.add(tf.keras.layers.Conv2D(filters=384, kernel_size = (3,3), strides=1,
                           activation="relu", padding="same"))
model.add(LocalResponseNormalization(input_shape=model.output_shape[1:]))
    
## [Layer 6] Convolution
model.add(tf.keras.layers.Conv2D(filters=384, kernel_size = (3,3), strides=1,
                           activation="relu", padding="same"))
    
## [Layer 7] Convolution
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size = (3,3), strides=1,
                           activation="relu", padding="same"))
    
## [Layer 8] Max Pooling
model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2))
    
## [Layer 9] Fully Connected Layer
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(4096, activation="relu"))

## [Layer 10] Fully Connected Layer
model.add(tf.keras.layers.Dense(4096, activation="relu"))

## [Layer 11] Fully Connected Layer
model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
```

```text
WARNING:tensorflow:From /tensorflow-1.15.0/python3.6/tensorflow_core/python/ops/resource_variable_ops.py:1630: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
```

In \[ \]:

```python
optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=5e-5, momentum=0.9)
model.compile(loss="categorical_crossentropy", 
              optimizer=optimizer, 
              metrics=["accuracy"])
```

In \[9\]:

```python
model.summary()
```

```text
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 57, 57, 96)        34944     
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 28, 28, 96)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 28, 28, 256)       614656    
_________________________________________________________________
local_response_normalization (None, 28, 28, 256)       0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 13, 13, 256)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 13, 13, 384)       885120    
_________________________________________________________________
local_response_normalization (None, 13, 13, 384)       0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 13, 13, 384)       1327488   
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 13, 13, 256)       884992    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 6, 6, 256)         0         
_________________________________________________________________
flatten (Flatten)            (None, 9216)              0         
_________________________________________________________________
dense (Dense)                (None, 4096)              37752832  
_________________________________________________________________
dense_1 (Dense)              (None, 4096)              16781312  
_________________________________________________________________
dense_2 (Dense)              (None, 1000)              4097000   
=================================================================
Total params: 62,378,344
Trainable params: 62,378,344
Non-trainable params: 0
_________________________________________________________________
```

### **MNIST Data에 적용해보기**

In \[ \]:

```python
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils import np_utils
import matplotlib.pyplot as plt
```

In \[11\]:

```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train[:, :, :, np.newaxis].astype('float32') / 255.0
X_test = X_test[:, :, :, np.newaxis].astype('float32') / 255.0
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
```

```text
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11493376/11490434 [==============================] - 0s 0us/step
```

In \[12\]:

```python
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

```text
(60000, 28, 28, 1) (10000, 28, 28, 1) (60000,) (10000,)
```

In \[ \]:

```python
input_shape = (28, 28, 1)
num_classes = 10

## [Layer 1] Convolution 
m = tf.keras.models.Sequential()
m.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=1, 
                           padding='same', input_shape=input_shape))

## [Layer 2] Max Pooling 
m.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

## [Layer 3] Convolution 
m.add(tf.keras.layers.Conv2D(filters=192, kernel_size=(3,3), strides=1,
                           activation="relu", padding='same'))
m.add(LocalResponseNormalization(input_shape=model.output_shape[1:]))
    
## [Layer 4] Max Pooling 
m.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
  
## [Layer 5] Convolution
m.add(tf.keras.layers.Conv2D(filters=384, kernel_size = (3,3), strides=1,
                           activation="relu", padding="same"))
m.add(LocalResponseNormalization(input_shape=model.output_shape[1:]))
    
## [Layer 6] Convolution
m.add(tf.keras.layers.Conv2D(filters=256, kernel_size = (3,3), strides=1,
                           activation="relu", padding="same"))
    
## [Layer 7] Convolution
m.add(tf.keras.layers.Conv2D(filters=256, kernel_size = (3,3), strides=1,
                           activation="relu", padding="same"))
    
## [Layer 8] Max Pooling
m.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    
## [Layer 9] Fully Connected Layer
m.add(tf.keras.layers.Flatten())
m.add(tf.keras.layers.Dense(4096, activation="relu"))

## [Layer 10] Fully Connected Layer
m.add(tf.keras.layers.Dense(2048, activation="relu"))

## [Layer 11] Fully Connected Layer
m.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
```

In \[ \]:

```python
optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=5e-5, momentum=0.9)
m.compile(loss="categorical_crossentropy", 
              optimizer=optimizer, 
              metrics=["accuracy"])
```

In \[15\]:

```text
m.summary()
```

```text
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_5 (Conv2D)            (None, 28, 28, 64)        640       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 14, 14, 64)        0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 14, 14, 192)       110784    
_________________________________________________________________
local_response_normalization (None, 14, 14, 192)       0         
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 7, 7, 192)         0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 7, 7, 384)         663936    
_________________________________________________________________
local_response_normalization (None, 7, 7, 384)         0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 7, 7, 256)         884992    
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 7, 7, 256)         590080    
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 3, 3, 256)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2304)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 4096)              9441280   
_________________________________________________________________
dense_4 (Dense)              (None, 2048)              8390656   
_________________________________________________________________
dense_5 (Dense)              (None, 10)                20490     
=================================================================
Total params: 20,102,858
Trainable params: 20,102,858
Non-trainable params: 0
_________________________________________________________________
```

In \[16\]:

```python
%%time
hist = m.fit(X_train, Y_train, epochs=10, batch_size=600,
                   validation_data=(X_test, Y_test), verbose=2)
```

```text
WARNING:tensorflow:From /tensorflow-1.15.0/python3.6/tensorflow_core/python/ops/math_grad.py:1375: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 60000 samples, validate on 10000 samples
Epoch 1/10
60000/60000 - 12s - loss: 2.2761 - acc: 0.2003 - val_loss: 2.1770 - val_acc: 0.4934
Epoch 2/10
60000/60000 - 10s - loss: 1.1234 - acc: 0.7205 - val_loss: 0.3219 - val_acc: 0.9031
Epoch 3/10
60000/60000 - 10s - loss: 0.2141 - acc: 0.9329 - val_loss: 0.1567 - val_acc: 0.9479
Epoch 4/10
60000/60000 - 10s - loss: 0.1355 - acc: 0.9564 - val_loss: 0.1091 - val_acc: 0.9630
Epoch 5/10
60000/60000 - 10s - loss: 0.0945 - acc: 0.9698 - val_loss: 0.0775 - val_acc: 0.9738
Epoch 6/10
60000/60000 - 10s - loss: 0.0749 - acc: 0.9770 - val_loss: 0.0653 - val_acc: 0.9791
Epoch 7/10
60000/60000 - 10s - loss: 0.0659 - acc: 0.9793 - val_loss: 0.0587 - val_acc: 0.9816
Epoch 8/10
60000/60000 - 10s - loss: 0.0594 - acc: 0.9810 - val_loss: 0.0704 - val_acc: 0.9769
Epoch 9/10
60000/60000 - 10s - loss: 0.0501 - acc: 0.9844 - val_loss: 0.0514 - val_acc: 0.9842
Epoch 10/10
60000/60000 - 10s - loss: 0.0489 - acc: 0.9850 - val_loss: 0.0445 - val_acc: 0.9855
CPU times: user 1min 4s, sys: 24.9 s, total: 1min 28s
Wall time: 1min 53s
```

In \[17\]:

```python
m.evaluate(X_test, Y_test, verbose=2)
```

```text
10000/10000 - 1s - loss: 0.0445 - acc: 0.9855
```

Out\[17\]:

```text
[0.04452171303179348, 0.9855]
```

In \[18\]:

```python
plt.plot(hist.history['acc'], 'b-', label="training")
plt.plot(hist.history['val_acc'], 'r:', label="test")
plt.legend()
plt.show()
```

![](data:image/png;base64,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%0A)

Test Accuracy 0.9855로 꽤 좋은 성능을 보이고 있네요!!

