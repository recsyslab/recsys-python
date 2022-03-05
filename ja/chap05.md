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
    
    # アイテムi、アイテムjのコサイン類似度
    【課題04】
    print('cos({}, {}) = {:.3f} / ({:.3f} * {:.3f}) = {:.3f}'.format(i, j, num, den_i, den_j, cosine))

    return cosine
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



