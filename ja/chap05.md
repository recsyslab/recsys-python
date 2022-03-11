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

### 03ピアソンの相関係数の算出（分母右部）
ピアソンの相関係数$$\mathrm{pearson}(u, v)$$の分母の右部である次式を求めるコードを書きなさい。得られた値を`den_vとすること。

$$
\sqrt{\sum_{i \in I_{u,v}} (r_{v,i} - \overline{r}_{v})^{2}}
$$

★★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。
3. `numpy.sqrt()`を使う。





## 平均中心化評価値行列

次のコードは、評価値行列$$\boldsymbol{R}$を平均中心化評価値行列$$\boldsymbol{R}^{'}$$に変換するためのコードである。各課題に答えなさい。

```python
ru_mean = 【課題10】
print('ru_mean = {}'.format(ru_mean))

# 平均中心化評価値行列
R2 = 【課題11】
print('R\' = \n{}'.format(R2))
print()
```

実行結果
```python
ru_mean = [2.5  4.   3.5  1.75 2.4 ]
R' = 
 [[  nan  1.5   0.5  -1.5  -0.5    nan]
 [ 1.    1.    0.     nan -1.   -1.  ]
 [ 0.5    nan  1.5  -0.5  -1.5    nan]
 [  nan  1.25   nan  0.25 -0.75 -0.75]
 [-0.4  -1.4  -0.4   1.6    nan  0.6 ]]
```

### 10 各ユーザの平均評価値
各ユーザの平均評価値を`ndarray`のベクトルで返すコードを記述しなさい。

★
1. `numpy.nanmean`を使う。
2. `axis`を指定する。

### 11 平均中心化評価値行列
評価値行列$$\boldsymbol{R}$$を平均中心化評価値行列$\boldsymbol{R}^{'}$に変換するコードを記述しなさい。

★★
1. `ndarray.reshape`を使う。

## 平均中心化評価値行列に基づくピアソンの相関係数

平均中心化評価値行列$$\boldsymbol{R}^{'}$$を用いると、ユーザ$$u$$とユーザ$$v$$のピアソンの相関係数$$\mathrm{pearson}(u, v)$$は次式で定義される。

$$
\mathrm{pearson}(u, v)  = \frac{\sum_{i \in I_{u,v}} r_{u,i}^{'} r_{v,i}^{'}}{\sqrt{\sum_{i \in I_{u,v}} r_{u,i}^{'2}}  \sqrt{\sum_{i \in I_{u,v}} r_{v,i}^{'2}}}
$$

次のコードは平均中心化評価値行列$$\boldsymbol{R}^{'}$$を基にした$$\mathrm{pearson}(u, v)$$を算出するための関数である。

```python
def pearson2(u, v):
    """
    平均中心化評価値行列R2においてユーザuとユーザvのピアソンの相関係数を算出する。

    Parameters
    ----------
    u : int
        ユーザu
    v : int
        ユーザv

    Returns
    -------
    prsn : float
        ピアソンの相関係数
    """
    Iu = I[~np.isnan(R2)[u,:]]
    Iv = I[~np.isnan(R2)[v,:]]
    Iuv = np.intersect1d(Iu, Iv)
    
    num = 【課題12】
    den_u = 【課題13】
    den_v = 【課題14】
    prsn = num / (den_u * den_v)
    print('pearson({}, {}) = {:.3f} / ({:.3f} * {:.3f}) = {:.3f}'.format(u, v, num, den_u, den_v, prsn))

    return prsn
```

確認コード
```python
u = 0
v = 1
similarity = pearson2(u, v)
print('sim({}, {}) = {:.3f}'.format(u, v, similarity))
```

実行結果
```python
pearson(0, 1) = 2.000 / (1.658 * 1.414) = 0.853
sim(0, 1) = 0.853
```

### 12 ピアソンの相関係数の算出
ピアソンの相関係数$$\mathrm{pearson}(u, v)$$の分子である次式を求めるコードを記述しなさい。

$$
\sum_{i \in I_{u,v}} r_{u,i}^{'} r_{v,i}^{'}
$$

★★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。

### 13 ピアソンの相関係数の算出
ピアソンの相関係数$$\mathrm{pearson}(u, v)$$の分母の左部である次式を求めるコードを記述しなさい。

$$
\sqrt{\sum_{i \in I_{u,v}} r_{u,i}^{'2}}
$$

★★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。
3. `numpy.sqrt()`を使う。

### 14 ピアソンの相関係数の算出
ピアソンの相関係数$$\mathrm{pearson}(u, v)$$の分母の右部である次式を求めるコードを記述しなさい。

$$
\sqrt{\sum_{i \in I_{u,v}} r_{v,i}^{'2}}
$$

★★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。
3. `numpy.sqrt()`を使う。

## 類似ユーザの選定

$$
\begin{cases}
 \overline{r}_{u} + \frac{\sum_{v \in U_{i}^{u}} \mathrm{sim}(u, v) \cdot r_{v,i}^{'}}{\sum_{v \in U_{i}^{u}} \mid \mathrm{sim} (u, v) \mid} & (U_{i}^{u} \neq \emptyset)\\
 \overline{r}_{u} & (U_{i}^{u} = \emptyset)
\end{cases}
$$

```python
def sim(u, v):
    """
    ユーザuとユーザvのユーザ類似度を算出する。

    Parameters
    ----------
    u : int
        ユーザu
    v : int
        ユーザv

    Returns
    -------
    sim : float
        ユーザ類似度
    """
    return pearson2(u, v)
```

```python
def predict(u, i):
    """
    ユーザuのアイテムiに対する予測評価値を算出する。

    Parameters
    ----------
    u : int
        ユーザu
    i : int
        アイテムi

    Returns
    -------
    rui_pred : float
        ユーザuのアイテムiに対する予測評価値
    """
    # ユーザuの類似ユーザ集合
    Uu = {v: sim(u, v) for v in U if u!=v}
    print('U{} = {}'.format(u, Uu))
    # 類似度上位K_USERS人のユーザ集合
    Uu = 【課題15】
    print('U{} = {}'.format(u, Uu))
    # 類似度が閾値THETA以上のユーザ集合
    Uu = 【課題16】
    print('U{} = {}'.format(u, Uu))
    # 辞書型からndarray型に変換
    Uu = np.array(list(Uu.keys()))
    print('U{} = {}'.format(u, Uu))

    # アイテムiを評価済みのユーザ集合
    Ui = 【課題17】
#    print('U{} = {}'.format(i, Ui))
    # ユーザuの類似ユーザ集合U^{u}の中でアイテムiを評価済みのユーザ集合
    Uui = 【課題18】
#    print('U{}{} = {}'.format(u, i, Uui))

    # ユーザuのアイテムiに対する予測評価値
    ru_mean = np.nanmean(R[u])
    if Uui.size <= 0: return ru_mean
    rui_pred = 【課題19】
#    print('r{}{}_pred = {:.3f} + {:.3f} / {:.3f} = {:.3f}'.format(u, i, ru_mean, num, den, rui_pred))
```

### 15 類似度上位k人のユーザ集合
`Uu`は`v: sim(u, v)`を要素とした辞書である。`Uu`から上位`K_USERS`人のユーザ集合を辞書として取得するコードを記述しなさい。

★★★
1. `sorted()`を使う。
2. リストのスライスを使う。
3. `dict()`を使う。

### 16 類似度がしきい値以上のユーザ集合
`Uu`から類似度がしきい値`THETA`以上のユーザのみを残した辞書を作成するコードを記述しなさい。

★★★
1. 辞書内包表記を使う。
2. `dict.items()`を使う。
3. 辞書内包表記内で`if`節を使う。

### 17
アイテム$$i$$を評価済みのユーザ集合を取得するコードを記述しなさい。

★★
1. `numpy.isnan()`を使う。
2. `~`演算子を使う。
3. ベクトルのブールインデックス参照を使う。

### 18
ユーザ$$u$$の類似ユーザ集合$$U^{u}$$の中でアイテム$$i$$を評価済みのユーザ集合$$U_{i}^{u}$$を取得するコードを記述しなさい。

★
1. `numpy.intersect1d()`を使う。

### 19 予測評価値
$$U_{i}^{u} \neq \emptyset$$のとき、ユーザ$$u$$のアイテム$$i$$に対する予測評価値$$\hat{r}_{u,i}$$を算出するコードを記述しなさい。

★★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。
3. `numpy.abs()`を使う。

## 評価値行列の補完
評価値行列$$\boldsymbol{R}$$の欠損値を予測評価値で補完した行列$$\boldsymbol{R}^{''}$$を作成するコードを記述しなさい。

★★
1. `ndarray.copy()`を使う。
2. `for`文を使う。
3. 二重ループを使う。
4. `continue`文を使う。

★★★
1. 二重のリスト内包表記を使う。
2. リスト内包表記内で`if`節を使う。
