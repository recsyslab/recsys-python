{% include header.html %}

# 第6章 アイテムベース協調フィルタリング

## 準備
次のコードを書きなさい。

```python
import pprint
import numpy as np
np.set_printoptions(precision=3)

# 近傍アイテム数
K_ITEMS = 3
# 閾値
THETA = 0

R = np.array([
              [np.nan, 4,      3,      1,      2,      np.nan],
              [5,      5,      4,      np.nan, 3,      3     ],
              [4,      np.nan, 5,      3,      2,      np.nan],
              [np.nan, 3,      np.nan, 2,      1,      1     ],
              [2,      1,      2,      4,      np.nan, 3     ],
])
U = np.arange(R.shape[0])
I = np.arange(R.shape[1])
Ui = [U[~np.isnan(R)[:,i]] for i in I]
Iu = [I[~np.isnan(R)[u,:]] for u in U]
ru_mean = np.nanmean(R, axis=1)
R2 = R - ru_mean.reshape((ru_mean.size, 1))
```

## コサイン類似度
アイテム$$i$$とアイテム$$j$$のコサイン類似度$$\mathrm{cos}(i, j)$$は次式で定義される。

$$
\mathrm{cos}(i, j) = \frac{\sum_{u \in U_{i,j}} r_{u,i} r_{u,j}}{\sqrt{\sum_{u \in U_{i,j}} r_{u,i}^{2}} \sqrt{\sum_{u \in U_{i,j}} r_{u,j}^{2}}}
$$

ここで、$$U_{i,j}$$はアイテム$$i$$とアイテム$$j$$の両方を評価済みのユーザ集合である。このコサイン類似度関数を次のコードのとおり定義する。

関数
```python
def cos(i, j):
    """
    評価値行列Rにおけるアイテムiとアイテムjのコサイン類似度を返す。

    Parameters
    ----------
    i : int
        アイテムiのID
    j : int
        アイテムjのID

    Returns
    -------
    float
        コサイン類似度
    """
    Uij = np.intersect1d(Ui, Uj)
    
    【    問01    】
    return cosine
```

コード
```python
i = 0
j = 4
cosine = cos(i, j)
print('cos({}, {}) = {:.3f}'.format(i, j, cosine))
```

結果
```python
cos(0, 4) = 0.996
```

このとき、関数の仕様を満たすように、次の問いに答えなさい。

### 01 アイテムiとアイテムjのコサイン類似度
アイテム$$i$$とアイテム$$j$$のコサイン類似度$$\mathrm{cos}(i, j)$$を求めるコードを書きなさい。得られた値を`cosine`とすること。

★★★
1. リスト内包表記を使う
2. `numpy.sum()`を使う。
3. `numpy.sqrt()`を使う。

## 調整コサイン類似度
平均中心化評価値行列$$\bolssymbol{R}^{'}$$を用いると、アイテム$$i$$とアイテム$$j$$の調整コサイン類似度$$\mathrm{cos}(i, j)^{'}$$は次式で定義される。

$$
\mathrm{cos}(i, j)^{'} = \frac{\sum_{u \in U_{i,j}} r_{u,i}^{'} r_{u,j}^{'}}{\sqrt{\sum_{u \in U_{i,j}} r_{u,i}^{'2}} \sqrt{\sum_{u \in U_{i,j}} r_{u,j}^{'2}}}
$$

この調整コサイン類似度関数を次のコードのとおり定義する。

関数
```python
def adjusted_cos(i, j):
    """
    評価値行列R2におけるアイテムiとアイテムjの調整コサイン類似度を返す。

    Parameters
    ----------
    i : int
        アイテムiのID
    j : int
        アイテムjのID

    Returns
    -------
    cosine : float
        調整コサイン類似度
    """
    Uij = np.intersect1d(Ui[i], Ui[j])
    
    【    問02    】
    return cosine
```

コード
```python
i = 0
j = 4
cosine = adjusted_cos(i, j)
print('cos({}, {})\' = {:.3f}'.format(i, j, cosine))
```

結果
```python
cos(0, 4)' = -0.868
```

このとき、関数の仕様を満たすように、次の問いに答えなさい。

### 02 アイテムiとアイテムjの調整コサイン類似度
アイテム$$i$$とアイテム$$j$$のコサイン類似度$$\mathrm{cos}(i, j)^{'}$$を求めるコードを書きなさい。得られた値を`cosine`とすること。

★★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。
3. `numpy.sqrt()`を使う。

## アイテム-アイテム類似度行列
アイテム$$i$$とアイテム$$j$$のアイテム類似度$$\mathrm{sim}(i, j)^{'}$$は次式で定義される。

$$
\mathrm{sim}(i, j)^{'} = \mathrm{cos}(i, j)^{'}
$$

このアイテム類似度関数を次のコードのとおり定義する。

関数
```python
def sim(i, j):
    """
    アイテム類似度関数：アイテムiとアイテムjのアイテム類似度を返す。

    Parameters
    ----------
    i : int
        アイテムiのID
    j : int
        アイテムjのID

    Returns
    -------
    float
        アイテム類似度
    """
    return adjusted_cos(i, j)
```

各アイテム$$i$$と各アイテム$$j$$のアイテム類似度$$\mathrm{sim}(i, j)$$を要素とした行列をアイテム-アイテム類似度行列$$\boldsymbol{\mathcal{S}}$$とする。アイテム-アイテム類似度行列$$\boldsymbol{\mathcal{S}}$$は次式のとおりとなる。

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

このとき、次の問いに答えなさい。

### 03 アイテム-アイテム類似度行列
アイテム-アイテム類似度行列$$\boldsymbol{\mathcal{S}}$$を`ndarray`として生成するコードを書きなさい。生成した`ndarray`を`S`とすること。

コード
```python
【    問03    】
print('S = \n{}'.format(S))
```

結果
```
S = 
[[ 1.     0.842  0.494 -0.829 -0.868 -0.987]
 [ 0.842  1.     0.896 -0.788 -0.91  -0.942]
 [ 0.494  0.896  1.    -0.583 -0.845 -0.514]
 [-0.829 -0.788 -0.583  1.     0.469  0.497]
 [-0.868 -0.91  -0.845  0.469  1.     1.   ]
 [-0.987 -0.942 -0.514  0.497  1.     1.   ]]
```

★★★
1. `numpy.zeros()`を使う。
2. 二重の`for`ループを使う。

★★★
1. 二重のリスト内包表記を使う。
2. `numpy.array()`を使う。

## 類似アイテムの選定
アイテム$$i$$の類似アイテム集合を$$I^{i} \subseteq I$$とする。ここでは、アイテム類似度が上位$$k$$件のアイテムを類似アイテム集合として選定する。ただし、類似度がしきい値$$\theta$$未満のアイテムは除外する。

次のコードはアイテム`i`の類似アイテム集合`Ii`を選定するためのコードである。ここで、`K_ITEMS`は上位$$k$$件を表す定数であり、`THETA`はしきい値$$\theta$$を表す定数である。

コード
```python
# アイテム-アイテム類似度行列から対象アイテムを除外した辞書
Ii = {i: {j: S[i,j] for j in I if i != j} for i in I}
print('Ii = ')
pprint.pprint(Ii)
【    問04    】
print('Ii = ')
pprint.pprint(Ii)
【    問05    】
print('Ii = ')
pprint.pprint(Ii)
# 各アイテムの類似アイテム集合をまとめた辞書
Ii = {i: np.array(list(Ii[i].keys())) for i in I}
print('Ii = ')
```

結果
```bash
Ii = 
{0: {1: 0.8418791389638738,
     2: 0.49365474375598073,
     3: -0.8291725540450335,
     4: -0.8682431421244593,
     5: -0.987241120712647},
 1: {0: 0.8418791389638738,
     2: 0.896314672184623,
     3: -0.7876958617794716,
     4: -0.9099637547345425,
     5: -0.9419581446623225},
 2: {0: 0.49365474375598073,
     1: 0.896314672184623,
     3: -0.5833076828172804,
     4: -0.8451542547285166,
     5: -0.5144957554275266},
 3: {0: -0.8291725540450335,
     1: -0.7876958617794716,
     2: -0.5833076828172804,
     4: 0.4685212856658182,
     5: 0.49665813370370504},
 4: {0: -0.8682431421244593,
     1: -0.9099637547345425,
     2: -0.8451542547285166,
     3: 0.4685212856658182,
     5: 1.0},
 5: {0: -0.987241120712647,
     1: -0.9419581446623225,
     2: -0.5144957554275266,
     3: 0.49665813370370504,
     4: 1.0}}
Ii = 
{0: {1: 0.8418791389638738, 2: 0.49365474375598073, 3: -0.8291725540450335},
 1: {0: 0.8418791389638738, 2: 0.896314672184623, 3: -0.7876958617794716},
 2: {0: 0.49365474375598073, 1: 0.896314672184623, 5: -0.5144957554275266},
 3: {2: -0.5833076828172804, 4: 0.4685212856658182, 5: 0.49665813370370504},
 4: {2: -0.8451542547285166, 3: 0.4685212856658182, 5: 1.0},
 5: {2: -0.5144957554275266, 3: 0.49665813370370504, 4: 1.0}}
Ii = 
{0: {1: 0.8418791389638738, 2: 0.49365474375598073},
 1: {0: 0.8418791389638738, 2: 0.896314672184623},
 2: {0: 0.49365474375598073, 1: 0.896314672184623},
 3: {4: 0.4685212856658182, 5: 0.49665813370370504},
 4: {3: 0.4685212856658182, 5: 1.0},
 5: {3: 0.49665813370370504, 4: 1.0}}
Ii = 
{0: array([1, 2]),
 1: array([2, 0]),
 2: array([1, 0]),
 3: array([5, 4]),
 4: array([5, 3]),
 5: array([4, 3])}
 ```

### 04 類似度上位k件のアイテム集合
`Ii`から、各アイテム$$i \in I$$について類似度上位`K_ITEMS`件のみを残した辞書を生成するコードを書きなさい。生成した辞書を`Ii`とすること。

★★★
1. 辞書内包表記を使う。
2. `sorted()`を使う。
3. `dict.items()`を使う。
4. `lambda`式を使う。
5. スライシングを使う。
6. `dict()`を使う。

### 05 類似度がしきい値以上のアイテム集合
`Ii`から、各アイテム$$i \in I$$について類似度がしきい値`THETA`以上のみを残した辞書を生成するコードを書きなさい。生成した辞書を`Ii`とすること。

★★★
1. 二重の辞書内包表記を使う。
2. `dict.items()`を使う。
3. 条件式を使う。

## 嗜好予測
ユーザ$$u$$のアイテム$$i$$に対する予測評価値$$\hat{r}_{u,i}$$は次式で求められる。

$$
\hat{r}_{u,i} = 
 \begin{cases}
  \frac{\sum_{j \in I_{u}^{i}} \mathrm{sim}(i, j) \cdot r_{u,j}}{\sum_{j \in I_{u}^{i}} \mid \mathrm{sim}(i,j) \mid} & (I_{u}^{i} \neq \emptyset) \\
  \overline{r}_{u} & (I_{u}^{i} = \emptyset)
 \end{cases}
$$

ここで、$$I_{u}^{i}$$は類似アイテム集合$$I^{i}$$の中でユーザ$$u$$が評価値を与えているアイテム集合を表す。$$\emptyset$$は空集合を表す。この予測関数を次のコードのとおり定義する。

関数
```python
def predict(u, i):
    """
    予測関数：ユーザuのアイテムiに対する予測評価値を返す。

    Parameters
    ----------
    u : int
        ユーザuのID
    i : int
        アイテムiのID
    
    Returns
    -------
    float
        ユーザuのアイテムiに対する予測評価値
    """
    【    問06    】
    print('I{}{} = {}'.format(i, u, Iiu))

    if Iiu.size <= 0: return ru_mean[u]
    【    問07    】
    
    return rui_pred
```

コード
```python
u = 0
i = 0
print('r{}{} = {:.3f}'.format(u, i, predict(u, i)))
u = 0
i = 5
print('r{}{} = {:.3f}'.format(u, i, predict(u, i)))
```

結果
```
I00 = [1 2]
r00 = 3.630
I50 = [3 4]
r05 = 1.668
```

このとき、関数の仕様を満たすように、次の問いに答えなさい。

### 06 類似アイテム集合の中でユーザuが評価値を与えているアイテム集合
類似アイテム集合$$I^{i}$$の中でユーザ$$u$$が評価値を与えているアイテム集合$$I_{u}^{i}$$を`ndarray`として生成するコードを書きなさい。生成された`ndarray`を`Iiu`とすること。

★
1. `numpy.intersect1d()`を使う。

### 07 予測評価値
$$I_{u}^{i} \neq \emptyset$$のとき、ユーザ$$u$$のアイテム$$i$$に対する予測評価値$$\hat{r}_{u,i}$$を求めるコードを書きなさい。得られた値を`rui_pred`とすること。

★★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。
3. `numpy.abs()`を使う。