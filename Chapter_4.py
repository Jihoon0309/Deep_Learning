import numpy as np
import matplotlib.pyplot as plt
''' 손실함수
    신경망의 성능이 얼마나 나쁜지 나타내주는 함수
    즉 손실함수가 클수록 훈련 데이터를 잘 처리하지 '못'함
'''

# # 평균 제곱 오차 => y=소프트맥수 함수의 출력(모든 수의 합이 1)이므로 확률로 해석 가능
# def mean_squared_error(y, t):
#     return 0.5*np.sum((y-t)**2)

# t=[0,0,1,0,0,0,0,0,0,0] # 정답 2
# y=[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] # 2일 확률이 가장높음
# print(mean_squared_error(np.array(y),np.array(t)))

# y=[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0] # 7일 확률이 가장높음
# print(mean_squared_error(np.array(y),np.array(t)))

# # 교차 엔프로피 오차
def cross_entropy_error(y,t):
    delta = 1e-7 # log(0)이 나오지 않게끔 엄청 작은 값을 더해줌
    return -np.sum(t*np.log(y+delta))

# t=[0,0,1,0,0,0,0,0,0,0] # 정답 2
# y=[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] # 2일 확률이 가장높음
# print(cross_entropy_error(np.array(y),np.array(t)))

# y=[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0] # 7일 확률이 가장높음
# print(cross_entropy_error(np.array(y),np.array(t)))

# 미니배치 => 랜덤으로 몇개 뽑아서 그 뽑은 것들만 사용하여 학습하는것

# import sys, os
# sys.path.append(os.pardir)
# from dataset.mnist import load_mnist

# (x_train, t_train), (x_test, t_test) =\
#     load_mnist(normalize=True, one_hot_label=True) # 정답 위치만 1(one_hot_label)

# train_size=x_train.shape[0] # 데이터의 개수
# batch_size=10 # 무작위로 뽑는 개수
# batch_mask=np.random.choice(train_size, batch_size) # 데이터의 개수중에 무작위로 뽑는 개수
# x_batch=x_train[batch_mask]
# t_batch=t_train[batch_mask]

# # (배치용) 교체 엔트로피 오차 구현 => 원-핫 인코딩일때 t가 0인 원소의 계산은 무시해도 좋다
# def corss_entropy_error(y, t):
#     if y.ndim==1:
#         t=t.reshape(1, t.size)
#         y=y.reshape(1, y.size)
    
#     batch_size=y.shape[0]
#     return -np.sum(t*np.log(y+1e-7)) / batch_size

# '''
#     손실함수 사용 이유
#     손실함수를 최소화하는 매개변수 값을 찾음
#     이때 미분을 하면서 매개변수의 값을 서서히 갱신
#     미분값이 -,+에 따라 반대방향으로 매개변수 변화 가능
#     미분값이 0이되면 매개변수 갱신을 멈춤

#     손실함수가 아닌 정확도를 사용하게되면
#     미세한 변화에는 거의 반응을 하지않음
#     계단함수를 사용하지 않는것과 같음
#     매개변수가 작게 변화하면 정확도는 바뀌지 않거나 크게 바뀌기 때문

#     따라서 어느 장소더라도 기울기가 0이 되지 않는게 신경망에서 중요한 성질임
# '''

# 수치미분
def numerical_diff(f, x):
    h=1e-4 # 0.0001 너무작은 값을 사용하면 반올림 오차가 생김
    return (f(x+h)-f(x-h))/(2*h) # 중심차분,중앙차분 (f(x+h)-f(x))->이건 전방차분

# def function_1(x):
#     return (0.01*x**2)+(0.1*x)

# x=np.arange(0.0, 20.0, 0.1)
# y=function_1(x)
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.plot(x,y)
# plt.show() # 0.01x^2+0.1x 함수의 그래프를 그림

# print(numerical_diff(function_1,5))
# print(numerical_diff(function_1,10))

# # 편미분 => 변수가 여럿인 함수에대한 미분
def function_2(x):
    return x[0]**2+x[1]**2

# def function_tmp1(x0):
#     return x0*x0+4.0**2.0 # x[0]=3, x[1]=4일 때, x[0]에 대한 편미분

# def function_tmp2(x1):
#     return 3.0**2.0+x1*x1 # x[0]=3, x[1]=4일 때, x[1]에 대한 편미분

# print(numerical_diff(function_tmp1, 3.0))
# print(numerical_diff(function_tmp2, 4.0))

# 기울기 => 편미분을 동시에 계산함 모둔 변수의 편미분을 벡터로 정리한 것
def numerical_gradient(f, x):
    h=1e-4
    grad=np.zeros_like(x)

    for idx in range(x.size):
        tmp_val=x[idx]

        x[idx]=tmp_val+h
        fxh1=x[0]**2+x[1]**2 # f(x+h)계산

        x[idx]=tmp_val-h
        fxh2=x[0]**2+x[1]**2 # f(x-h)계산

        grad[idx]=(fxh1-fxh2)/(2*h)

    return grad

# print(numerical_gradient(function_2, np.array([3.0, 4.0])))
# print(numerical_gradient(function_2, np.array([0.0, 2.0])))
# print(numerical_gradient(function_2, np.array([3.0, 0.0])))

# 경사법(경사 하강법) => 기울기를 이용해 함수의 최솟값(또는 가능한 한 작은 값)을 찾는 것
def gradient_descent(f, init_x, lr=0.01, step_num=100): # (함수, 초깃값, 학습률, 반복횟수)
    x=init_x

    for i in range(step_num):
        grad=numerical_gradient(f, x) # 기울기 함수
        x-=lr*grad
    return x

# def function_2(x):
#     return x[0]**2+x[1]**2

# init_x=np.array([-3.0, 4.0])
# result=gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
# print(result)

# result_max=gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100) # 학습률이 너무 클때
# print(result_max)

# result_min=gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100) # 학습률이 너무 작을때
# print(result_min)

def numerical_gradient_2(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 값 복원
        it.iternext()   
        
    return grad

from Chapter_3 import softmax

class SimpleNet:
    def __init__(self):
        self.W=np.random.randn(2,3) # 정규 분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z=self.predict(x)
        y=softmax(z)
        loss=cross_entropy_error(y, t)

        return loss

net=SimpleNet()
print(net.W) # 가중치 매개변수

x=np.array([0.6, 0.9]) # 입력 데이터
p=net.predict(x) # 예측 x@net.W
print(p)
print(np.argmax(p)) # 최댓값의 인덱스
if np.argmax(p)==0:
    t=np.array([1,0,0])
elif np.argmax(p)==1:
    t=np.array([0,1,0])
else:
    t=np.array([0,0,1])
print(net.loss(x,t))

def f(W):
    return net.loss(x, t)

dW=numerical_gradient_2(f, net.W)
print(dW)

