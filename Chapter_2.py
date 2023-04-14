# AND 게이트
def AND(x1, x2):
    w1, w2, there = 0.5, 0.5, 0.7
    tmp=(x1*w1)+(x2*w2)
    if tmp<=there:
        return 0
    else:
        return 1
print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))

# 편향 도입
import numpy as np
x=np.array([0,1]) # 입력
w=np.array([0.5, 0.5]) # 가중치
b=-0.7 # 편향
print(np.sum(x*w)+b)

# 가중치, 편향 구현
def AND(x1, x2):
    x=np.array([x1, x2])
    w=np.array([0.5, 0.5])
    b=-0.7
    tmp=np.sum(x*w)+b
    if tmp<=0:
        return 0
    else:
        return 1

# NAND 게이트
def NAND(x1, x2):
    x=np.array([x1, x2])
    w=np.array([-0.5, -0.5])
    b=0.7
    tmp=np.sum(x*w)+b
    if tmp<=0:
        return 0
    else:
        return 1

# OR 게이트
def OR(x1, x2):
    x=np.array([x1, x2])
    w=np.array([0.5, 0.5])
    b=-0.2
    tmp=np.sum(x*w)+b
    if tmp<=0:
        return 0
    else:
        return 1

# XOR 게이트 (다층 퍼셉트론)
def XOR(x1, x2):
    s1=NAND(x1, x2)
    s2=OR(x1, x2)
    y=AND(s1, s2)
    return y
print(XOR(0,0))
print(XOR(0,1))
print(XOR(1,0))
print(XOR(1,1))