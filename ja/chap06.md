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
            [5, 3, +1],
            [6, 2, +1],
            [4, 1, +1],
            [8, 5, -1],
            [2, 4, -1],
            [3, 6, -1],
            [7, 6, -1],
            [4, 2, np.nan],
            [5, 1, np.nan],
            [8, 6, np.nan],
            [3, 4, np.nan],
            [4, 7, np.nan],
            [4, 4, np.nan],
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


### 01 各特徴量の平均値


```python
# 各特徴量の平均値
【問題01】
print('x_mean = {}'.format(x_mean))
```

```
x_mean = [4.846 3.923]
```

```python
# 各特徴量の分散
k = 0
sk2 = np.array((1 / I.size) * np.sum([(x[i,k] - x_mean[k])**2 for i in I]))
print('s{}^2 = {:.3f}'.format(k, sk2))
d = x.shape[1]
s2 = np.array([(1 / I.size) * np.sum([(x[i,k] - x_mean[k])**2 for i in I]) for k in range(0, d)])
#s2 = np.array([(1 / I.size) * np.sum((x[:,k] - x_mean[k])**2) for k in range(0, d)])
#s2 = np.array((1 / I.size) * np.sum((x - x_mean)**2, axis=0))
#s2 = np.var(x, axis=0)
print('s^2 = ', s2)
print()

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
