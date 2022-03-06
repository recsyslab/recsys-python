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

### 04 アイテムi、アイテムjのコサイン類似度
アイテム$$i$$、アイテム$$j$$のコサイン類似度を求めるコードを書きなさい。

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
    【課題07】

    return cosine
```

```python
# 各ユーザの平均評価値
【課題05】
print('ru_mean = ', ru_mean)

# 平均中心化評価値行列
【課題06】
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

