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
特徴量$$k$$の平均値$$\overline{x}_{k}$$を求めるコードを書きなさい。

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
2. 得られた値を`xk_mean`とする。

### 02 すべての特徴量の平均値
すべての特徴量の平均値を`ndarray`としてまとめて求めるコードを書きなさい。

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
2. 得られた`ndarray`を`x_mean`とする。

### 03 特徴量kの分散
特徴量$$k$$の分散$$s_{k}^{2}$$は次式で求められる。

$$
s_{k}^{2} = \frac{1}{\mid I \mid} \sum_{i \in I} (x_{i,k} - \overline{x}_{k})^{2}
$$

特徴量$$k$$の分散$$s_{k}^{2}$$を求めるコードを書きなさい。

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

★★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。
3. 得られた値を`sk2`とする。

### 04 すべての特徴量の分散
すべての特徴量の分散を`ndarray`としてまとめて求めるコードを書きなさい。

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
4. 得られた`ndarray`を`s2`とする。

★★★
1. リスト内包表記を使う。
2. 二重のリスト内包表記を使わない。
3. `numpy.sum()`を使う。
4. `numpy.array()`を使う。
5. 得られた`ndarray`を`s2`とする。

★★★
1. リスト内包表記を使わない。
3. `numpy.sum()`を使う。
4. `numpy.array()`を使う。
5. 得られた`ndarray`を`s2`とする。

### 05 特徴量kの標準化
特徴量$$k$$を標準化するコードを書きなさい。




### 02 各特徴量の分散
各特徴量$$k$$の分散$$s_{k}^{2}$$を求めるコードを書きなさい。

★★★
1. 二重のリスト内包表記を使う。
2. `numpy.sum()`を使う。
3. `numpy.array()`を使う。

コード
```python
# 各特徴量の分散
d = x.shape[1]
【問題02】
print('s^2 = {}'.format(s2))
```

結果
```
s^2 = [3.361 3.763 4.769]
```





# 各特徴量の標準化
i = 0
k = 0
xik2 = np.array((x[i,k] - x_mean[k]) / np.sqrt(s2[k]))
print('x{}{}\' = {:.3f}'.format(i, k, xik2))
x2 = np.array([[(x[i,k] - x_mean[k]) / np.sqrt(s2[k]) for k in range(0, d)] for i in I])
#x2 = np.array([(x[i] - x_mean) / np.sqrt(s2) for i in I])
#x2 = np.array((x - x_mean) / np.sqrt(s2))
print('x\' = \n', x2)
print()

# 各特徴量間の共分散
k = 0
l = 1
skl = (1 / I.size) * np.sum([x2[i, k] * x2[i, l] for i in I])
#skl = (1 / I.size) * np.sum(x2[:, k] * x2[:, l])
print('s{}{} = {:.3f}'.format(k, l, skl))
print()

# 分散共分散行列
#S = np.zeros((d, d))
#for k in range(0, d):
#    for l in range(0, d):
#        S[k,l] = (1 / I.size) * np.sum([x2[i, k] * x2[i, l] for i in I])
S = np.array([[(1 / I.size) * np.sum([x2[i, k] * x2[i, l] for i in I]) for k in range(0, d)] for l in range(0, d)])
#S = np.cov(x2, rowvar=False, bias=True)
print('S = \n', S)
print()
```
