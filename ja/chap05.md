<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# 第5章 ユーザベース協調フィルタリング

### 準備
次のコードを書きなさい。

```python
import numpy as np
np.set_printoptions(precision=3)

# 近傍ユーザ数
K_USERS = 3
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

## ピアソンの相関係数
ユーザ$$u$$とユーザ$$v$$のピアソンの相関係数$$\mathrm{pearson}(u, v)$$は次式で定義される。

$$
\mathrm{pearson}(u, v) = \frac{\sum_{i \in I_{u,v}} (r_{u,i} - \overline{r}_{u})(r_{v,i} - \overline{r}_{v})}{\sqrt{\sum_{i \in I_{u,v}} (r_{u,i} - \overline{r}_{u})^{2}} \sqrt{\sum_{i \in I_{u,v}} (r_{v,i} - \overline{r}_{v})^{2}}}
$$

ここで、$$I_{u,v}$$はユーザ$$u$$とユーザ$$v$$の共通の評価済みアイテム集合である。また、$$\overline{r}_{u}$$はユーザ$$u$$の平均評価値を表す。このピアソンの相関係数の関数を次のコードのとおり定義する。

関数
```
def pearson1(u, v):
    """
    評価値行列Rにおけるユーザuとユーザvのピアソンの相関係数を返す。

    Parameters
    ----------
    u : int
        ユーザuのID
    v : int
        ユーザvのID

    Returns
    -------
    float
        ピアソンの相関係数
    """
    Iuv = np.intersect1d(Iu[u], Iu[v])

    【    問01    】
    print('num = {}'.format(num))
    【    問02    】
    print('den_u = {:.3f}'.format(den_u))
    【    問03    】
    print('den_v = {:.3f}'.format(den_v))
    
    prsn = num / (den_u * den_v)
    return prsn
```

コード
```python
u = 0
v = 1
prsn = pearson1(u, v)
print('pearson1({}, {}) = {:.3f}'.format(u, v, prsn))
```

結果
```python
num = 2.0
den_u = 1.658
den_v = 1.414
pearson1(0, 1) = 0.853
```

このとき、関数の仕様を満たすように、次の問いに答えなさい。

### 01 ピアソンの相関係数（分子）
ピアソンの相関係数$$\mathrm{pearson}(u, v)$$の分子である次式を求めるコードを書きなさい。得られた値を`num`とすること。

$$
\sum_{i \in I_{u,v}} (r_{u,i} - \overline{r}_{u})(r_{v,i} - \overline{r}_{v})
$$

★★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。

### 02 ピアソンの相関係数の算出（分母左部）
ピアソンの相関係数$$\mathrm{pearson}(u, v)$$の分母の左部である次式を求めるコードを書きなさい。得られた値を`den_u`とすること。

$$
\sqrt{\sum_{i \in I_{u,v}} (r_{u,i} - \overline{r}_{u})^{2}}
$$

★★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。
3. `numpy.sqrt()`を使う。

### 03 ピアソンの相関係数の算出（分母右部）
ピアソンの相関係数$$\mathrm{pearson}(u, v)$$の分母の右部である次式を求めるコードを書きなさい。得られた値を`den_vとすること。

$$
\sqrt{\sum_{i \in I_{u,v}} (r_{v,i} - \overline{r}_{v})^{2}}
$$

★★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。
3. `numpy.sqrt()`を使う。


## 平均中心化評価値行列に基づくピアソンの相関係数
平均中心化評価値行列$$\boldsymbol{R}^{'}$$を用いると、ユーザ$$u$$とユーザ$$v$$のピアソンの相関係数$$\mathrm{pearson}(u, v)$$は次式で定義される。

$$
\mathrm{pearson}(u, v)  = \frac{\sum_{i \in I_{u,v}} r_{u,i}^{'} r_{v,i}^{'}}{\sqrt{\sum_{i \in I_{u,v}} r_{u,i}^{'2}}  \sqrt{\sum_{i \in I_{u,v}} r_{v,i}^{'2}}}
$$

ここで、$$r_{u,i}^{'}$$は$$r_{u,i}$$の平均中心化評価値を表す。このピアソンの相関係数の関数を次のコードのとおり定義する。

関数
```
def pearson2(u, v):
    """
    平均中心化評価値行列R2におけるユーザuとユーザvのピアソンの相関係数を返す。

    Parameters
    ----------
    u : int
        ユーザuのID
    v : int
        ユーザvのID

    Returns
    -------
    float
        ピアソンの相関係数
    """
    Iuv = np.intersect1d(Iu[u], Iu[v])
    
    【    問04    】
    print('num = {}'.format(num))
    【    問05    】
    print('den_u = {:.3f}'.format(den_u))
    【    問06    】
    print('den_v = {:.3f}'.format(den_v))

    prsn = num / (den_u * den_v)
    return prsn
```

コード
```python
u = 0
v = 1
similarity = pearson2(u, v)
print('pearson2({}, {}) = {:.3f}'.format(u, v, prsn))
```

結果
```python
num = 2.0
den_u = 1.658
den_v = 1.414
pearson2(0, 1) = 0.853
```

このとき、関数の仕様を満たすように、次の問いに答えなさい。

### 04 ピアソンの相関係数（分子）
ピアソンの相関係数$$\mathrm{pearson}(u, v)$$の分子である次式を求めるコードを書きなさい。得られた値を`num`とすること。

$$
\sum_{i \in I_{u,v}} r_{u,i}^{'} r_{v,i}^{'}
$$

★★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。

### 05 ピアソンの相関係数の算出（分母左部）
ピアソンの相関係数$$\mathrm{pearson}(u, v)$$の分母の左部である次式を求めるコードを書きなさい。得られた値を`den_u`とすること。

$$
\sqrt{\sum_{i \in I_{u,v}} r_{u,i}^{'2}}
$$

★★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。
3. `numpy.sqrt()`を使う。

### 06 ピアソンの相関係数の算出（分母右部）
ピアソンの相関係数$$\mathrm{pearson}(u, v)$$の分母の右部である次式を求めるコードを書きなさい。得られた値を`den_vとすること。

$$
\sqrt{\sum_{i \in I_{u,v}} r_{v,i}^{'2}}
$$

★★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。
3. `numpy.sqrt()`を使う。

## ユーザ-ユーザ類似度行列
ユーザ$$u$$とユーザ$$v$$のユーザ類似度$$\mathrm{sim}(u,v)$$は次式で定義される。

$$
\mathrm{sim}(u,v) = \mathrm{pearson}(u,v)
$$

このユーザ類似度関数を次のコードのとおり定義する。

関数
```python
def sim(u, v):
    """
    ユーザ類似度関数：ユーザuとユーザvのユーザ類似度を返す。

    Parameters
    ----------
    u : int
        ユーザuのID
    v : int
        ユーザvのID

    Returns
    -------
    float
        ユーザ類似度
    """
    return pearson2(u, v)
```

各ユーザ$$u$$と各ユーザ$$v$$のユーザ類似度$$\mathrm{sim}(u, v)$$を要素とした行列をユーザ-ユーザ類似度行列$$\boldsymbol{\mathcal{S}}$$とする。ユーザ-ユーザ類似度行列$$\boldsymbol{\mathcal{S}}$$は次式のとおりとなる。

$$
\boldsymbol{\mathcal{S}} = \left[
    \begin{array}{rrrrrr}
         1.000 &  0.853 &  0.623 &  0.582 & -0.997 \\
         0.853 &  1.000 &  0.649 &  0.968 & -0.853 \\
         0.623 &  0.649 &  1.000 &  0.800 & -0.569 \\
         0.582 &  0.968 &  0.800 &  1.000 & -0.551 \\
        -0.997 & -0.853 & -0.569 & -0.551 &  1.000
    \end{array}
\right]
$$

このとき、次の問いに答えなさい。

### 07 ユーザ-ユーザ類似度行列
ユーザ-ユーザ類似度行列$$\boldsymbol{\mathcal{S}}$$を`ndarray`として生成するコードを書きなさい。生成した`ndarray`を`S`とすること。

コード
```python
【    問07    】
print('S = \n{}'.format(S))
```

結果
```bash
S = 
[[ 1.     0.853  0.623  0.582 -0.997]
 [ 0.853  1.     0.649  0.968 -0.853]
 [ 0.623  0.649  1.     0.8   -0.569]
 [ 0.582  0.968  0.8    1.    -0.551]
 [-0.997 -0.853 -0.569 -0.551  1.   ]]
```

★★★
1. `numpy.zeros()`を使う。
2. 二重のforループを使う。

★★★
1. 二重のリスト内包表記を使う。
2. `numpy.array()`を使う。

## 類似ユーザの選定
ユーザ$$u$$の類似ユーザ集合を$$U^{u} \subseteq U$$とする。ここでは、ユーザ類似度が上位$$k$$人のユーザを類似ユーザ集合として選定する。ただし、しきい値$$\theta$$未満のユーザは除外する。

次のコードはユーザ`u`の類似ユーザ集合`Uu`を選定するためのコードである。ここで、`K_USERS`は上位$$k$$人を表す定数であり、`THETA`はしきい値$$\theta$$を表す定数である。

コード
```python
# ユーザ-ユーザ類似度行列から対象ユーザを除外した辞書
Uu = {u: {v: S[u,v] for v in U if u!=v} for u in U}
print('Uu = ')
pprint.pprint(Uu)
【    問08    】
print('Uu = ')
pprint.pprint(Uu)
【    問09    】
print('Uu = ')
pprint.pprint(Uu)
# 各ユーザの類似ユーザ集合をまとめた辞書
Uu = {u: np.array(list(Uu[u].keys())) for u in U}
print('Uu = ')
pprint.pprint(Uu)
```

結果
```bash
Uu = 
{0: {1: 0.8528028654224417,
     2: 0.6225430174794672,
     3: 0.5816750507471109,
     4: -0.9968461286620518},
 1: {0: 0.8528028654224417,
     2: 0.6488856845230501,
     3: 0.9684959969581863,
     4: -0.8528028654224418},
 2: {0: 0.6225430174794672,
     1: 0.6488856845230501,
     3: 0.7999999999999998,
     4: -0.5685352436149611},
 3: {0: 0.5816750507471109,
     1: 0.9684959969581863,
     2: 0.7999999999999998,
     4: -0.550920031004556},
 4: {0: -0.9968461286620518,
     1: -0.8528028654224418,
     2: -0.5685352436149611,
     3: -0.550920031004556}}
Uu = 
{0: {1: 0.8528028654224417, 2: 0.6225430174794672, 3: 0.5816750507471109},
 1: {0: 0.8528028654224417, 2: 0.6488856845230501, 3: 0.9684959969581863},
 2: {0: 0.6225430174794672, 1: 0.6488856845230501, 3: 0.7999999999999998},
 3: {0: 0.5816750507471109, 1: 0.9684959969581863, 2: 0.7999999999999998},
 4: {1: -0.8528028654224418, 2: -0.5685352436149611, 3: -0.550920031004556}}
Uu = 
{0: {1: 0.8528028654224417, 2: 0.6225430174794672, 3: 0.5816750507471109},
 1: {0: 0.8528028654224417, 2: 0.6488856845230501, 3: 0.9684959969581863},
 2: {0: 0.6225430174794672, 1: 0.6488856845230501, 3: 0.7999999999999998},
 3: {0: 0.5816750507471109, 1: 0.9684959969581863, 2: 0.7999999999999998},
 4: {}}
Uu = 
{0: array([1, 2, 3]),
 1: array([3, 0, 2]),
 2: array([3, 1, 0]),
 3: array([1, 2, 0]),
 4: array([], dtype=float64)}
 ```

このとき、次の問いに答えなさい。

### 08 類似度上位k人のユーザ集合
`Uu`から、各ユーザ$$u \in U$$について類似度上位`K_USERS`人のみを残した辞書を生成するコードを書きなさい。生成した辞書を`Uu`とすること。

★★★
1. 辞書内包表記を使う。
2. `sorted()`を使う。
3. `dict.items()`を使う。
4. `lambda`式を使う。
5. スライシングを使う。
6. `dict()`を使う。

### 09 類似度がしきい値以上のユーザ集合
`Uu`から、各ユーザ$$u \in U$$について類似度がしきい値`THETA`以上のみを残した辞書を生成するコードを書きなさい。生成した辞書を`Uu`とすること。

★★★
1. 二重の辞書内包表記を使う。
2. `dict.items()`を使う。
3. 辞書内包表記において`if`節を使う。

## 嗜好予測
ユーザ$$u$$のアイテム$$i$$に対する予測評価値$$\hat{r}_{u,i}$$は次式で求められる。

$$
\hat{r}_{u,i} = \begin{cases}
 \overline{r}_{u} + \frac{\sum_{v \in U_{i}^{u}} \mathrm{sim}(u, v) \cdot r_{v,i}^{'}}{\sum_{v \in U_{i}^{u}} \mid \mathrm{sim} (u, v) \mid} & (U_{i}^{u} \neq \emptyset)\\
 \overline{r}_{u} & (U_{i}^{u} = \emptyset)
\end{cases}
$$

ここで、$$U_{i}^{u}$$は類似ユーザ集合$$U^{u}$$の中でアイテム$$i$$を評価済みのユーザ集合を表す。$$\emptyset$$は空集合を表す。この予測関数を次のコードのとおり定義する。

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
    【    問10    】
    print('U{}{} = {}'.format(u, i, Uui))

    if Uui.size <= 0: return ru_mean[u]
    【    問11    】
    
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
U00 = [1 2]
r00 = 3.289
U05 = [1 3]
r05 = 1.601
```

このとき、関数の仕様を満たすように、次の問いに答えなさい。

### 10 類似ユーザ集合の中でアイテムiを評価済みのユーザ集合
ユーザ$$u$$の類似ユーザ集合$$U^{u}$$の中でアイテム$$i$$を評価済みのユーザ集合$$U_{i}^{u}$$を`ndarray`として生成するコードを書きなさい。生成された`ndarray`を`Uui`とすること。

★
1. `numpy.intersect1d()`を使う。

### 11 予測評価値
$$U_{i}^{u} \neq \emptyset$$のとき、ユーザ$$u$$のアイテム$$i$$に対する予測評価値$$\hat{r}_{u,i}$$を求めるコードを書きなさい。得られた値を`rui_pred`とすること。

★★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。
3. `numpy.abs()`を使う。

## 評価値行列の補完
評価値行列$$\boldsymbol{R}$$の欠損値を予測評価値で補完した行列$$\boldsymbol{R}^{''}$$とする。行列$$\boldsymbol{R}^{''}$$は次式のとおりとなる。

$$
\boldsymbol{R}^{''} = \left[
    \begin{array}{rrrrrr}
        3.289 & 4     & 3     & 1     & 2     & 1.601 \\
        5     & 5     & 4     & 3.449 & 3     & 3     \\
        4     & 4.747 & 5     & 3     & 2     & 2.638 \\
        2.524 & 3     & 2.384 & 2     & 1     & 1     \\
        2     & 1     & 2     & 4     & 2.400 & 3    
    \end{array}
\right]
$$

このとき、次の問いに答えなさい。

### 12 評価値行列の補完
評価値行列$$\boldsymbol{R}$$の欠損値を予測評価値で補完した行列$$\boldsymbol{R}^{''}$$を`ndarray`として生成するコードを書きなさい。生成した`ndarray`を`R3`とすること。

コード
```python
【    問12    】
print('R\'\' = \n{}'.format(R3))
```

結果
```bash
R'' = 
[[3.289 4.    3.    1.    2.    1.601]
 [5.    5.    4.    3.449 3.    3.   ]
 [4.    4.747 5.    3.    2.    2.638]
 [2.524 3.    2.384 2.    1.    1.   ]
 [2.    1.    2.    4.    2.4   3.   ]]
```

★★
1. `ndarray.copy()`を使う。
2. 二重の`for`ループを使う。
4. `continue`文を使う。

★★★
1. 二重のリスト内包表記を使う。
2. リスト内包表記内で`if`節を使う。
