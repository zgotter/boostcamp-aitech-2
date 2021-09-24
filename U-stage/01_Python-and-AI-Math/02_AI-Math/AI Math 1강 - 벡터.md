# 벡터

## 벡터의 노름

### $L_1$ 노름

$$
||\mathbb{x}||_1 = \sum_{i=1}^d | x_i |
$$

### $L_2$ 노름

$$
||\mathbb{x}||_2 = \sqrt{\sum_{i=1}^d {| x_i |}^2}
$$



## 두 벡터 사이의 거리

벡터 $\mathbb{x}$와 벡터 $\mathbb{y}$ 사이의 거리
$$
||\mathbb{y} - \mathbb{x}|| = ||\mathbb{x} - \mathbb{y}||
$$


## 두 벡터 사이의 각도

두 벡터 사이의 각도를 구하는 것은 $L_2$ 노름에서만 가능하다.

**제 2 코사인 법칙**에 의해 두 벡터 사이의 각도를 계산할 수 있다.



벡터 $\mathbb{x}$와 벡터 $\mathbb{y}$ 사이의 각도
$$
cos \theta = \frac{||\mathbb{x}||_2^2 + ||\mathbb{y}||_2^2 - ||\mathbb{x} - \mathbb{y}||_2^2}{2 ||\mathbb{x}||_2 ||\mathbb{y}||_2}
$$


위 공식의 분자를 쉽계 계산하는 방법이 **내적**이다.
$$
cos \theta = 
\frac{2 <\mathbb{x}, \mathbb{y}>}{2 ||\mathbb{x}||_2 ||\mathbb{y}||_2} =
\frac{<\mathbb{x}, \mathbb{y}>}{||\mathbb{x}||_2 ||\mathbb{y}||_2}
$$


내적(inner product) 연산
$$
<\mathbb{x}, \mathbb{y}> = 
\sum_{i=1}^d x_i y_i
$$


## 내적을 어떻게 해석할까?

내적은 **정사영(orthogonal projection)된 벡터의 길이**와 관련 있다.

<img src="./img/001_01.jpg" style="width:200px" />

- $Proj(\mathbb{x})$ : 벡터 $\mathbb{y}$로 정사영된 벡터 $\mathbb{x}$의 **그림자**를 의미



$Proj(\mathbb{x})$ 의 길이는 **코사인법칙**에 의해 $||\mathbb{x}|| cos \theta$ 가 된다.

<img src="./img/001_02.jpg" style="width:200px" />



내적은 정사영의 길이를 **벡터 $\mathbb{y}$ 의 길이 $||\mathbb{y}||$ 만큼 조정**한 값이다.

<img src="./img/001_03.jpg" style="width:200px" />



내적은 두 벡터의 **유사도(similarity)**를 측정하는 데 사용 가능하다.



