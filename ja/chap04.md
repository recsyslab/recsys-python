<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# 第4章 ユーザベース協調フィルタリング

## 準備
次のコードを入力しなさい。

```python
import numpy as np

np.set_printoptions(precision=3)

# 近傍ユーザ数
K_USERS = 3
# 閾値
THETA = 0

# 評価値行列
R = np.array([
              [np.nan, 4,      3,      1,      2,      np.nan],
              [5,      5,      4,      np.nan, 3,      3     ],
              [4,      np.nan, 5,      3,      2,      np.nan],
              [np.nan, 3,      np.nan, 2,      1,      1     ],
              [2,      1,      2,      4,      np.nan, 3     ],
])
print('R = \n{}'.format(R))
# ユーザ集合
U = np.arange(R.shape[0])
print('U = {}'.format(U))
# アイテム集合
I = np.arange(R.shape[1])
print('I = {}'.format(I))
print()
```

## ユーザ類似度
ユーザ$$u$$とユーザ$$v$$のピアソンの相関係数によるユーザ類似度$$\mathrm{sim}(u, v)$$は次式で定義される。

$$
\mathrm{sim}(u, v) = \mathrm{pearson}(u, v)
$$

$$
\mathrm{sim}(u, v) = \mathrm{pearson}(u, v) \\
                   = \frac{\sum_{i \in I_{u,v}} (r_{u,i} - \overline{r}_{u})(r_{v,i} - \overline{r}_{v})}{\sqrt{\sum_{i \in I_{u,v}} (r_{u,i} - \overline{r}_{u})^{2}} \sqrt{\sum_{i \in I_{u,v}} (r_{v,i} - \overline{r}_{v})^{2}}}
$$

ここで、$$I_{u,v}$$はユーザ$$u$$とユーザ$$v$$の共通の評価済みアイテム集合である。また、$$\overline{r}_{u}$$はユーザ$$u$$の平均評価値を表し、次式で算出される。

$$
\overline{r}_{u} = \frac{\sum_{i \in I_{u}} r_{u,i}}{\mid I_{u} \mid}
$$

### 01 ユーザuの評価済みアイテム集合の取得 | ベクトルのブールインデックス参照
ユーザ0の評価済みアイテム集合$$I_{0}$$を`ndarray`のベクトル`Iu`として取得しなさい。

★
1. `numpy.isnan()`を使う。
2. `~`演算子を使う。
3. 行列のスライスを使う。

### 02 ユーザvの評価済みアイテム集合の取得 | ベクトルのブールインデックス参照
ユーザ1の評価済みアイテム集合$$I_{1}$$を`ndarray`のベクトル`Iv`として取得しなさい。

★
1. `numpy.isnan()`を使う。
2. `~`演算子を使う。
3. 行列のスライスを使う。

### 03 ユーザuとユーザvの共通の評価済みアイテム集合の取得 | 積集合
ユーザ0とユーザ1の共通の評価済みアイテム集合$$I_{0,1}$$を`ndarray`のベクトル`Iuv`として取得しなさい。

★
1. `numpy.intersect1d()`を使う。

### 04

