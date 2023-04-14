import sys, os
sys.path.append(os.pardir) # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
import pickle

# 이미지 표시
def img_show(img):
    pil_img=Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) =\
    load_mnist(flatten=True, normalize=False) # load_mnist(flatten(1차원 배열로 만들기), normalize(이미지 픽섹을 0~1사이로 정규화), one_hot_encoding(정답만 1로))

img=x_train[0]
label=t_train[0]
print(label)

print(img.shape) # flatten 으로 1차원 배열
img=img.reshape(28, 28) # 원래 이미지 모양으로 변형 이렇게 변형 후 이미지를 표시할 수 있음
print(img.shape)

img_show(img)

# 신경망 추론 처리
def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c) # 오버플로 문제를 막기위해서 c를 뺴줌
    sum_exp_a=sum(exp_a)
    y=exp_a/sum_exp_a
    
    return y

def get_data():
    (x_train, t_train), (x_test, t_test) =\
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1=np.dot(x, W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1, W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2, W3)+b3
    y=softmax(a3)

    return y

# 정확도 평가

x, t = get_data()
network=init_network()

batch_size=100
accuracy_cnt=0

for i in range(0, len(x), batch_size):
    x_batch=x[i:i+batch_size]
    y_batch=predict(network, x_batch)
    p=np.argmax(y_batch, axis=1)
    accuracy_cnt+=np.sum(p==t[i:i+batch_size])

print('Accuracy:'+str(float(accuracy_cnt) / len(x)))