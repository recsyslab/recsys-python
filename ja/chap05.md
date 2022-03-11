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

ここで、$$I_{u,v}$$はユーザ$$u$$とユーザ$$v$$の共通の評価済みアイテム集合である。また、$$\overline{r}_{u}$$はユーザ$$u$$の平均評価値を表し、次式で算出される。

$$
\overline{r}_{u} = \frac{\sum_{i \in I_{u}} r_{u,i}}{\mid I_{u} \mid}
$$

次のコードは$$\mathrm{pearson}(u, v)$$を算出するための関数である。以下の各課題に答えながらコード中の【課題NN】を埋め、関数を完成させなさい。関数を完成後、確認コードを実行したとき、実行結果のとおりの結果が出力されること。

```
def pearson1(u, v):
    """
    評価値行列Rにおいてユーザuとユーザvのピアソンの相関係数を算出する。

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
    Iu = 【課題01】
    print('I{} = {}'.format(u, Iu))
    Iv = 【課題02】
    print('I{} = {}'.format(v, Iv))
    Iuv = 【課題03】
    print('I{}{} = {}'.format(u, v, Iuv))
    
    ru_mean = 【課題04】
    print('r{}_mean = {:.3f}'.format(u, ru_mean))
    rv_mean = 【課題05】
    print('r{}_mean = {:.3f}'.format(v, rv_mean))

    num = 【課題06】
    den_u = 【課題07】
    den_v = 【課題08】
    prsn = 【課題09】
    print('pearson({}, {}) = {:.3f} / ({:.3f} * {:.3f}) = {:.3f}'.format(u, v, num, den_u, den_v, prsn))

    return prsn
```

確認コード
```python
u = 0
v = 1
prsn = pearson1(u, v)
print('pearson({}, {}) = {:.3f}'.format(u, v, prsn))
print()
```

実行結果
```python
I0 = [1 2 3 4]
I1 = [0 1 2 4 5]
I01 = [1 2 4]
r0_mean = 2.500
r1_mean = 4.000
pearson(0, 1) = 2.000 / (1.658 * 1.414) = 0.853
pearson(0, 1) = 0.853
```

### 01 ユーザuの評価済みアイテム集合の取得 | ベクトルのブールインデックス参照
ユーザ$$u$$の評価済みアイテム集合$$I_{u}$$を取得するコードを記述しなさい。

★
1. `numpy.isnan()`を使う。
2. `~`演算子を使う。
3. 行列のスライスを使う。

### 02 ユーザvの評価済みアイテム集合の取得 | ベクトルのブールインデックス参照
ユーザ$$v$$の評価済みアイテム集合$$I_{v}$$を取得するコードを記述しなさい。

★
1. `numpy.isnan()`を使う。
2. `~`演算子を使う。
3. 行列のスライスを使う。

### 03 ユーザuとユーザvの共通の評価済みアイテム集合の取得 | 積集合
ユーザ$$u$$とユーザ$$v$$の共通の評価済みアイテム集合$$I_{u,v}$$を取得するコードを記述しなさい。

★
1. `numpy.intersect1d()`を使う。

### 04 ユーザuの平均評価値の算出 | 平均
ユーザ$$u$$の平均評価値$$\overline{r}_{u}$$を求めるコードを記述しなさい。

★
1. `numpy.nanmean()`を使う。

★★★
1. `numpy.nanmean()`を使わない。
2. リスト内包表記を使う。
3. `numpy.sum()`を使う。
4. `ndarray.size`を使う。

### 05 ユーザvの平均評価値の算出 | 平均
ユーザ$$v$$の平均評価値$$\overline{r}_{v}$$を求めるコードを記述しなさい。

★
1. `numpy.nanmean()`を使う。

★★★
1. `numpy.nanmean()`を使わない。
2. リスト内包表記を使う。
3. `numpy.sum()`を使う。
4. `ndarray.size`を使う。

### 06 ピアソンの相関係数の算出
ピアソンの相関係数$$\mathrm{pearson}(u, v)$$の分子である次式を求めるコードを記述しなさい。

$$
\sum_{i \in I_{u,v}} (r_{u,i} - \overline{r}_{u})(r_{v,i} - \overline{r}_{v})
$$

★★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。

### 07 ピアソンの相関係数の算出
ピアソンの相関係数$$\mathrm{pearson}(u, v)$$の分母の左部である次式を求めるコードを記述しなさい。

$$
\sqrt{\sum_{i \in I_{u,v}} (r_{u,i} - \overline{r}_{u})^{2}}
$$

★★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。
3. `numpy.sqrt()`を使う。

### 08 ピアソンの相関係数の算出
ピアソンの相関係数$$\mathrm{pearson}(u, v)$$の分母の右部である次式を求めるコードを記述しなさい。

$$
\sqrt{\sum_{i \in I_{u,v}} (r_{v,i} - \overline{r}_{v})^{2}}
$$

★★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。
3. `numpy.sqrt()`を使う。

### 09 ピアソンの相関係数の算出
ピアソンの相関係数$$\mathrm{pearson}(u, v)$$を求めるコードを記述しなさい。

★
1. 課題06の結果を使う。
2. 課題07の結果を使う。
3. 課題08の結果を使う。

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
