import numpy as np 
import matplotlib.pyplot as plt

# 계단 함수 구현 => 배열을 인수로 받을순 없음
def step_function(x):
    if x>0:
        return 1
    else:
        return 0

# 배열을 인수로 받을수 있는 계단 함수 그리기
def step_function(x):
    return np.array(x>0)

x=np.arange(-5.0, 5.0, 0.1)
y=step_function(x)
plt.plot(x,y)
plt.ylim(-0.1, 1.1) # y축 범위 지정
plt.show()

# sigmoid 함수 구현 => 브로드캐스트 했기 때문에 연산가능
def sigmoid(x):
    return 1/(1+np.exp(-x))

x=np.array([-1.0, 1.0, 2.0])
print(sigmoid(x))

# sigmoid 함수 그림
x=np.arange(-5.0, 5.0, 0.1)
y=sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1, 1,1)
plt.show()

# ReLU 함수 구현 => 0보다 작으면 0 그렇지 않으면 x 출력
def relu(x):
    return np.maximum(0, x) # x와 0중 큰값 출력

# 다차원 배열
A=np.array([[1,2],[3,4],[5,6]])
print(A)
print(np.ndim(A)) # 배열의 차원 수
print(A.shape) # 배열의 형상

# 행렬의 곱 => 앞쪽의 열과 뒤쪽의 행이 같은 숫자여야 가능함 결과 형상은 (앞쪽의 행x뒤쪽의 열) 형상을 가짐
A=np.array([[1,2],[3,4]])
B=np.array([[5,6],[7,8]])
print(np.dot(A,B)) # A@B 로도 표현가능

# 신경망에서의 행렬 곱
X=np.array([1,2])
W=np.array([[1,3,5],[2,4,6]])
Y=np.dot(X,W)
print(Y)

# 항등 함수 정의
def identity_function(x):
    return x

# 각 층의 신호 전달 구현
def init_network():
    network={}
    network['W1']=np.array([[0.1, 0.3, 0.5],[0.2, 0.4, 0.6]])
    network['b1']=np.array([0.1, 0.2, 0.3])
    network['W2']=np.array([[0.1, 0.4],[0.2, 0.5],[0.3, 0.6]])
    network['b2']=np.array([0.1, 0.2])
    network['W3']=np.array([[0.1, 0.3],[0.2, 0.4]])
    network['b3']=np.array([0.1, 0.2])

    return network

def forward(x):
    network=init_network()
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1=np.dot(x, W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1, W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2, W3)+b3
    y=identity_function(a3)

    return y

x=np.array([1.0, 0.5])
y=forward(x)
print(y)

# softmax 함수 구현 => softmax 함수의 출력의 합은 1이 되기 때문에 확률로 해석할 수 있음
def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c) # 오버플로 문제를 막기위해서 c를 뺴줌
    sum_exp_a=sum(exp_a)
    y=exp_a/sum_exp_a
    
    return y

a=np.array([0.3, 2.9, 4.0])
print(softmax(a))