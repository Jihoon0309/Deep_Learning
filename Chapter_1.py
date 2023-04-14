import numpy as np
import matplotlib.pyplot as plt

#브로드캐스트
A=np.array([[1,2],[3,4]])
B=np.array([10,20])
print(A*B) # 형상이 다른 행렬을 형상을 같도록 늘려서 연산해주는 기능

#그래프 그리기
x=np.arange(0, 6, 0.01) # 0~6까지 0.1 간격으로 생성
y1=np.sin(x)
y2=np.cos(x)
#기능 추가
plt.plot(x, y1, label='sin')
plt.plot(x, y2, linestyle='--', label='sin') # y2는 점선으로 그리기
plt.xlabel('x') # x축 이름
plt.xlabel('y') # y축 이름
plt.title('sin & cos') # 제목
plt.legend() # 범례 표시하기 (여기선 label을 표시해줌)
plt.show()

#이미지 표시하기
from matplotlib.image import imread
test_img=imread('lena.png')
plt.imshow(test_img)
plt.show()
