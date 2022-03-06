<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# 第5章 アイテムベース協調フィルタリング

## 準備
次のコードを入力しなさい。

```python
import numpy as np

np.set_printoptions(precision=3)

# 近傍アイテム数
K_ITEMS = 3
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

## コサイン類似度
アイテム$$i$$とアイテム$$j$$のコサイン類似度$$\mathrm{cos}(i, j)$$は次式で定義される。

$$
\mathrm{cos}(i, j) = \frac{\sum_{u \in U_{i,j}} r_{u,i} r_{u,j}}{\sqrt{\sum_{u \in U_{i,j}} r_{u,i}^{2}} \sqrt{\sum_{u \in U_{i,j}} r_{u,j}^{2}}}
$$

```
def cos(i, j):
    """
    評価値行列Rにおいてアイテムiとアイテムjのコサイン類似度を算出する。

    Parameters
    ----------
    i : int
        アイテムi
    j : int
        アイテムj

    Returns
    -------
    cosine : float
        コサイン類似度
    """
    # アイテムiを評価済みのユーザ集合
    【課題01】
    print('U{} = {}'.format(i, Ui))
    # アイテムjを評価済みのユーザ集合
    【課題02】
    print('U{} = {}'.format(j, Uj))
    # アイテムi、アイテムjの両方を評価済みのユーザ集合
    【課題03】
    print('U{}{} = {}'.format(i, j, Uij))
    
    # アイテムiとアイテムjのコサイン類似度
    【課題04】

    return cosine
```

```python
i = 0
j = 4
cosine = cos(i, j)
print('cos({}, {}) = {:.3f}'.format(i, j, cosine))
```

```python
U0 = [1 2 4]
U4 = [0 1 2 3]
U04 = [1 2]
cos(0, 4) = 0.996
```

### 01 アイテムiを評価済みのユーザ集合
アイテム$$i$$を評価済みのユーザ集合$$U_{i}$$を取得するコードを書きなさい。

★★
1. `numpy.isnan()`を使う。
2. `~`演算子を使う。
3. 行列のスライスを使う。
4. ブールインデックス参照を使う。
5. 得られたユーザ集合を`ndarray`として`Ui`に代入する。

### 02 アイテムjを評価済みのユーザ集合
アイテム$$j$$を評価済みのユーザ集合$$U_{j}$$を取得するコードを書きなさい。

★★
1. `numpy.isnan()`を使う。
2. `~`演算子を使う。
3. 行列のスライスを使う。
4. ブールインデックス参照を使う。
5. 得られたユーザ集合を`ndarray`として`Uj`に代入する。

### 03 アイテムi、アイテムjの両方を評価済みのユーザ集合
アイテム$$i$$、アイテム$$j$$の両方を評価済みのユーザ集合$$U_{i,j}$$を取得するコードを書きなさい。

★
1. `numpy.intersect1d`を使う。
2. 得られたユーザ集合を`ndarray`として`Uij`に代入する。

### 04 アイテムiとアイテムjのコサイン類似度
アイテム$$i$$とアイテム$$j$$のコサイン類似度を求めるコードを書きなさい。

★★★
1. リスト内包表記を使う
2. `numpy.sum()`を使う。
3. `numpy.sqrt()`を使う。
4. 得られたコサイン類似度を`cosine`に代入する。

## 調整コサイン類似度
評価値行列$$\boldsymbol{R}$$の平均中心化評価値行列$$\boldsymbol{R}^{'}$$は次式のとおりとなる。

$$
\boldsymbol{R}^{'} = \left[
            \begin{array}{rrrrrr}
                     &  1.5  &  0.5 & -1.5  & -0.5  &       \\
                 1   &  1    &  0   &       & -1    & -1    \\
                 0.5 &       &  1.5 & -0.5  & -1.5  &       \\
                     &  1.25 &      &  0.25 & -0.75 & -0.75 \\
                -0.4 & -1.4  & -0.4 &  1.6  &       &  0.6
            \end{array}
        \right]
$$

このとき、アイテム$$i$$とアイテム$$j$$の調整コサイン類似度$$\mathrm{cos}(i, j)^{'}$$は次式で定義される。

$$
\mathrm{cos}(i, j)^{'} = \frac{\sum_{u \in U_{i,j}} r_{u,i}^{'} r_{u,j}^{'}}{\sqrt{\sum_{u \in U_{i,j}} r_{u,i}^{'2}} \sqrt{\sum_{u \in U_{i,j}} r_{u,j}^{'2}}}
$$

ここで、$$r_{u,i}^{'}$$はユーザ$$u$$のアイテム$$i$$に対する平均中心化評価値を表す。



```
def adjusted_cos(i, j):
    """
    評価値行列R2においてアイテムiとアイテムjの調整コサイン類似度を算出する。

    Parameters
    ----------
    i : int
        アイテムi
    j : int
        アイテムj

    Returns
    -------
    cosine : float
        調整コサイン類似度
    """
    【課題06】

    return cosine
```

```python
# 平均中心化評価値行列
【課題05】
print('R\' = \n{}'.format(R2))

# 調整コサイン類似度
i = 0
j = 4
cosine = adjusted_cos(i, j)
print('sim({}, {}) = {:.3f}'.format(i, j, cosine))
```

```python
ru_mean =  [2.5  4.   3.5  1.75 2.4 ]
R' = 
[[  nan  1.5   0.5  -1.5  -0.5    nan]
 [ 1.    1.    0.     nan -1.   -1.  ]
 [ 0.5    nan  1.5  -0.5  -1.5    nan]
 [  nan  1.25   nan  0.25 -0.75 -0.75]
 [-0.4  -1.4  -0.4   1.6    nan  0.6 ]]
sim(0, 4) = -0.868
```

### 05 平均中心化評価値行列
評価値行列$$\boldsymbol{R}$$を平均中心化評価値行列$$\boldsymbol{R}^{'}$$に変換するコードを書きなさい。

★★
1. `numpy.nanmean()`を使う。
2. `axis`を指定する。
3. `ndarray.reshape()`を使う。
4. 得られた行列を`ndarray`として`R2`に代入する。

### 06 アイテムiとアイテムjのコサイン類似度
アイテム$$i$$とアイテム$$j$$の調整コサイン類似度を求めるコードを書きなさい。

★★★
1. `numpy.isnan()`を使う。
2. `~`演算子を使う。
3. ベクトルのブールインデックス参照を使う。
4. `numpy.intersect1d()`を使う。
5. リスト内包表記を使う。
6. `numpy.sum()`を使う。
7. `numpy.sqrt()`を使う。
8. 得られたコサイン類似度を`cosine`に代入する。

## アイテム-アイテム類似度行列

アイテム$$i$$とアイテム$$j$$のアイテム類似度$$\mathrm{sim}(i, j)^{'}$$は次式で定義される。

$$
\mathrm{sim}(i, j)^{'} = \mathrm{cos}(i, j)^{'}
$$

アイテム-アイテム類似度行列$$\boldsymbol{\mathcal{S}}$$は次式のとおりとなる。

$$
\boldsymbol{\mathcal{S}} = \left[
    \begin{array}{rrrrrr}
         1.000 &  0.842 &  0.494 & -0.829 & -0.868 & -0.987 \\
         0.842 &  1.000 &  0.896 & -0.788 & -0.910 & -0.942 \\
         0.494 &  0.896 &  1.000 & -0.583 & -0.845 & -0.514 \\
        -0.829 & -0.788 & -0.583 &  1.000 &  0.469 &  0.497 \\
        -0.868 & -0.910 & -0.845 &  0.469 &  1.000 &  1.000 \\
        -0.987 & -0.942 & -0.514 &  0.497 &  1.000 &  1.000 
    \end{array}
\right]
$$

```python
def sim(i, j):
    """
    アイテムiとアイテムjのアイテム類似度を算出する。

    Parameters
    ----------
    i : int
        アイテムi
    j : int
        アイテムj

    Returns
    -------
    cosine : float
        コサイン類似度
    """
    return adjusted_cos(i, j)
```

```python
# アイテム-アイテム類似度行列
【課題07】
print('S = \n{}'.format(S))
```

```python
S = 
 [[ 1.     0.842  0.494 -0.829 -0.868 -0.987]
 [ 0.842  1.     0.896 -0.788 -0.91  -0.942]
 [ 0.494  0.896  1.    -0.583 -0.845 -0.514]
 [-0.829 -0.788 -0.583  1.     0.469  0.497]
 [-0.868 -0.91  -0.845  0.469  1.     1.   ]
 [-0.987 -0.942 -0.514  0.497  1.     1.   ]]
```

### 07 アイテム-アイテム類似度行列
アイテム集合$$I$$内について、アイテム-アイテム類似度行列$$\boldsymbol{\mathcal{S}}$$を求めるコードを書きなさい。

★★★
1. 二重のリスト内包表記を使う。
2. `numpy.array()`を使う。
3. 得られた行列を`ndarray`として`S`に代入する。

## 類似アイテムの選定

アイテム$$i$$の類似アイテム集合を$$I^{i}$$とする。ここでは、アイテム類似度が上位$$k$$件のアイテムを類似アイテム集合として選定する。ただし、類似度がしきい値$$\theta$$未満のアイテムは除外する。

次のコードはアイテム`i`の類似アイテム集合`Ii`を選定するためのコードである。ここで、`K_ITEMS`は上位$$k$$件を表す定数であり、`THETA`はしきい値$$\theta$$を表す定数である。
```python
# アイテムiの類似アイテム集合
i = 0
# j: S[i,j]を要素とした辞書型のアイテム集合
Ii = {j: S[i,j] for j in I if i!=j}
# 類似度上位K_ITEMS件のアイテム集合
【課題08】
print('I{} = {}'.format(i, Ii))
# 類似度が閾値THETA以上のアイテム集合
【課題09】
print('I{} = {}'.format(i, Ii))
# 辞書型からndarray型に変換
Ii = np.array(list(Ii.keys()))
print('I{} = {}'.format(i, Ii))
```

### 08
`j: S[i,j]`を要素とした辞書型のアイテム集合`Ii`から類似度上位`K_ITEMS`件のアイテム集合を取得するコードを書きなさい。

★★★
1. `sorted()`を使う。
2. `lambda`式を使う。
3. リストのスライスを使う。
4. `dict()`を使う。
5. 得られた集合を辞書として`Ii`に代入する。

### 09
`j: S[i,j]`を要素とした辞書型のアイテム集合`Ii`から類似度がしきい値`THETA`以上のアイテム集合を取得するコードを書きなさい。

★★★
1. 辞書内包表記を使う。
2. `dict.items()`を使う。
3. 辞書内包表記内で`if`節を使う。
4. 得られた集合を辞書として`Ii`に代入する。

## 嗜好予測

対象ユーザ$$u$$の対象アイテム$$i$$に対する予測評価値$$\hat{r}_{u,i}$$は次式で求められる。

$$
\displaystyle
\hat{r}_{u,i} = 
 \begin{cases}
  \frac{\sum_{j \in I_{u}^{i}} \mathrm{sim}(i, j) \cdot r_{u,j}}{\sum_{j \in I_{u}^{i}} \mid \mathrm{sim}(i,j) \mid} & (I_{u}^{i} \neq \emptyset) \\
  \overline{r}_{u} & (I_{u}^{i} = \emptyset)
 \end{cases}
$$

ここで、$$I_{u}^{i}$$は、
