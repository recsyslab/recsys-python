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
3. `numpy.sqrt()`を使う。

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


