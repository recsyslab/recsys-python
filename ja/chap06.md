<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# 第6章 次元削減

```python
import numpy as np
import numpy.linalg as LA

np.set_printoptions(precision=3)

# 縮約後の次元数
DIM = 2

# 評価履歴
Du = np.array([
            [5, 3, 3, +1],
            [6, 2, 5, +1],
            [4, 1, 5, +1],
            [8, 5, 9, -1],
            [2, 4, 2, -1],
            [3, 6, 5, -1],
            [7, 6, 8, -1],
            [4, 2, 3, np.nan],
            [5, 1, 8, np.nan],
            [8, 6, 6, np.nan],
            [3, 4, 2, np.nan],
            [4, 7, 5, np.nan],
            [4, 4, 4, np.nan],
])
print('Du = \n{}'.format(Du))
print()

# アイテム集合
I = np.arange(Du.shape[0])
print('I = {}'.format(I))
print()

# アイテムの特徴ベクトル
x = Du[:,:-1]
print('x = \n{}'.format(x))
print()

# 評価値
ru = Du[:,-1]
print('ru = {}'.format(ru))
print()
```

## 分散共分散行列


### 01 特徴量kの平均値
特徴量$$k$$の平均値$$\overline{x}_{k}$$を求めるコードを書きなさい。得られた値を`xk_mean`とすること。

コード
```python
# 特徴量kの平均値
k = 0
【問題01】
print('x{}_mean = {:.3f}'.format(k, xk_mean))
```

結果
```
x0_mean = 4.846
```

★
1. `numpy.mean()`を使う。

### 02 すべての特徴量の平均値
すべての特徴量の平均値を`ndarray`としてまとめて求めるコードを書きなさい。得られた`ndarray`を`x_mean`とすること。

コード
```python
# 各特徴量の平均値
【問題02】
print('x_mean = {}'.format(x_mean))
```

結果
```bash
x_mean =  [4.846 3.923 5.   ]
```

★
1. `numpy.mean()`を使う。

### 03 特徴量kの分散
特徴量$$k$$の分散$$s_{k}^{2}$$は次式で求められる。

$$
s_{k}^{2} = \frac{1}{\mid I \mid} \sum_{i \in I} (x_{i,k} - \overline{x}_{k})^{2}
$$

特徴量$$k$$の分散$$s_{k}^{2}$$を求めるコードを書きなさい。得られた値を`sk2`とすること。

コード
```python
# 特徴量kの分散
k = 0
【問題03】
print('s{}^2 = {:.3f}'.format(k, sk2))
```

結果
```bash
s0^2 = 3.361
```

★
1. `numpy.var()`を使う。

★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。

### 04 すべての特徴量の分散
すべての特徴量の分散を`ndarray`としてまとめて求めるコードを書きなさい。得られた`ndarray`を`s2`とすること。

コード
```python
# すべての特徴量の分散
【問題04】
print('s^2 = {}'.format(s2))
```

結果
```bash
s^2 = [3.361 3.763 4.769]
```

★
1. `numpy.var()`を使う。

★★★
1. 二重のリスト内包表記を使う。
2. `numpy.sum()`を使う。
3. `numpy.array()`を使う。

★★
1. リスト内包表記を使う。
2. 二重のリスト内包表記を使わない。
3. `numpy.sum()`を使う。
4. `numpy.array()`を使う。

★★
1. リスト内包表記を使わない。
2. `numpy.sum()`を使う。

### 05 特徴量kの標準化
特徴量$$x_{i,k}$$を標準化した値$$x_{i,k}^{'}$$は次式で求められる。

$$
x_{i,k}^{'} = \frac{x_{i,k} - \overline{x}_{k}}{s_{k}}
$$

特徴量$$k$$を標準化するコードを書きなさい。得られた値を`xik2`とすること。

コード
```python
i = 0
k = 0
【問題05】
print('x{}{}\' = {:.3f}'.format(i, k, xik2))
```

結果
```bash
x00' = 0.084
```

★
1. `numpy.sqrt()`を使う。

### 06 すべての特徴量の標準化
すべての特徴量の`ndarray`としてまとめて標準化するコードを書きなさい。得られた`ndarray`を`x2`とすること。

コード
```python
# すべての特徴量の標準化
# 06
【問題06】
print('x\' = \n{}'.format(x2))
```

結果
```bash
x'' = 
 [[ 0.699  0.264]
 [-0.139  1.065]
 [ 0.752  1.355]
 [-2.564  0.202]
 [ 1.969 -0.636]
 [ 0.373 -1.211]
 [-2.035 -0.496]
 [ 1.219  0.656]
 [-0.545  1.766]
 [-1.788 -0.601]
 [ 1.598 -0.535]
 [-0.148 -1.603]
 [ 0.611 -0.227]]
```

★★★
1. 二重のリスト内包表記を使う。
2. `numpy.sqrt()`を使う。
3. `numpy.array()`を使う。

★★
1. リスト内包表記を使う。
2. 二重のリスト内包表記を使わない。
3. `numpy.sqrt()`を使う。
4. `numpy.array()`を使う。

★★
1. リスト内包表記を使わない。
3. `numpy.sqrt()`を使う。

### 07 標準化された特徴量kと特徴量lの共分散
標準化された特徴量$$k$$と特徴量$$l$$の共分散$$s_{k,l}$$は次式で求められる。

$$
s_{k,l} = \frac{1}{\mid I \mid} \sum_{i \in I} x_{i,k}^{'} x_{i,l}^{'}
$$

標準化された特徴量$$k$$と特徴量$$l$$の共分散$$s_{k,l}$$を求めるコードを書きなさい。得られた値を`skl`とすること。

コード
```python
k = 0
l = 1
【問題07】
print('s{}{} = {:.3f}'.format(k, l, skl))
print()
```

結果
```bash
s01 = 0.191
```

★
1. `numpy.cov()`を使う。
2. `numpy.cov()`において`bias=True`を指定する。

★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。

★★
1. リスト内包表記を使わない。
2. `numpy.sum()`を使う。

### 08 分散共分散行列
各特徴量について分散共分散行列$$\boldsymbol{S}$$を`ndarray`として求めるコードを書きなさい。得られた`ndarray`を`S`とすること。

コード
```python
# 分散共分散行列
【問題08】
print('S = \n{}'.format(S))
```

結果
```bash
S = 
 [[1.    0.191 0.749]
 [0.191 1.    0.163]
 [0.749 0.163 1.   ]]
```

★
1. `numpy.cov()`を使う。
2. `numpy.cov()`において`bias=True`を指定する。

★★
1. `numpy.zeros()`を使う。
2. 二重ループを使う。

★★★
1. 三重のリスト内包表記を使う。
2. `numpy.sum()`を使う。
3. `numpy.array()`を使う。

## 固有値・固有ベクトル

### 09 固有値・固有ベクトル

分散共分散行列$$\boldsymbol{S}$$の固有値$$\lambda$$、固有ベクトル$$\boldsymbol{v}$$を求めるコードを書きなさい。`ndarray`として得られた固有値、固有ベクトルを、それぞれ`lmd`、`v`とすること。

```python
# 分散共分散行列Sの固有値lmd、固有ベクトルv
【問題09】
print('λ = {}'.format(lmd))
print('v = \n{}'.format(v))
```

```bash
λ = [1.826 0.25  0.924]
v = 
[[-0.679 -0.71   0.186]
 [-0.291  0.028 -0.956]
 [-0.674  0.704  0.225]]
```

★
1. `numpy.linalg.eig()`を使う。



