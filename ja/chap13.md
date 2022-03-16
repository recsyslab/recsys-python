---
title: 第13章 推薦システムの評価 | 推薦順位に基づく正確性 | recsys-python
layout: default
---

{% include header.html %}

# 第13章 推薦システムの評価 | 推薦順位に基づく正確性

## テストデータと予測評価値
次の評価値行列$$\boldsymbol{R}^{\mathit{test}}$$はテストデータである。$$\boldsymbol{R}$$の$$(u, i)$$成分はユーザ$$u$$がアイテム$$i$$に与えた評価値$$r_{u,i}$$を表す。ただし、$$-$$で示した要素はテストデータの対象ではないことを表す。また、$$\boldsymbol{R}^{\mathit{test}}$$に含まれる成分の集合を$$R^{\mathit{test}}$$と表す。

$$
\boldsymbol{R}^{\mathit{test}} = \left[
 \begin{array}{rrrrrrrrrr}
  5 & 4 & 3 & - & 5 & 4 & 2 & 2 & - & - \\
  3 & 3 & 3 & 3 & 2 & - & 4 & - & 5 & - \\
  4 & - & 3 & 5 & 4 & 3 & - & 3 & - & - \\
 \end{array}
\right]
$$

次の行列$$\hat{\boldsymbol{R}}^{A}$$は、推薦システムAによる推薦リストである。$$\hat{\boldsymbol{R}}^{A}$$の$$(u, i)$$成分は、それぞれユーザ$$u$$向けの推薦システムA、推薦システムBによる推薦リストにおける順位を表す。

$$
\hat{\boldsymbol{R}}^{A} = \left[
 \begin{array}{rrrrrrrrrr}
  1 & - & 3 & - & 4 & 2 & 5 & - & - & - \\
  4 & 1 & - & 3 & - & - & 5 & - & 2 & - \\
  - & - & 5 & 3 & 4 & 2 & - & 1 & - & - \\
 \end{array}
\right]
$$

## 準備
次のコードを書きなさい。

```python
import math
import numpy as np
np.set_printoptions(precision=3)

# 上位K件
TOP_K = 5
# 対数の底
ALPHA = 2

# テストデータ
R = np.array([
              [5, 4,      3, np.nan, 5, 4,      2,      2,      np.nan, np.nan],
              [3, 3,      3, 3,      2, np.nan, 4,      np.nan, 5,      np.nan],
              [4, np.nan, 3, 5,      4, 3,      np.nan, 3,      np.nan, np.nan],
])
U = np.arange(R.shape[0])
I = np.arange(R.shape[1])
Iu = [I[~np.isnan(R)[u,:]] for u in U]

# 推薦システムAによる推薦リスト
RA = np.array([
               [1,      np.nan, 3,      np.nan, 4,      2,      5,      np.nan, np.nan, np.nan],
               [4,      1,      np.nan, 3,      np.nan, np.nan, 5,      np.nan, 2,      np.nan],
               [np.nan, np.nan, 5,      3,      4,      2,      np.nan, 1,      np.nan, np.nan],
])

def confusion_matrix(u, RS, K):
    """
    ユーザu向け推薦リストRSの上位K件における混同行列の各値を返す。

    Parameters
    ----------
    u : int
        ユーザuのID
    RS : ndarray
        推薦リストRS
    K : int
        上位K件

    Returns
    -------
    int
        TP
    int
        FN
    int
        FP
    int
        TN
    """
    like = R[u,Iu[u]]>=4
    recommended = RS[u,Iu[u]]<=K
    TP = np.count_nonzero(np.logical_and(like, recommended))
    FN = np.count_nonzero(np.logical_and(like, ~recommended))
    FP = np.count_nonzero(np.logical_and(~like, recommended))
    TN = np.count_nonzero(np.logical_and(~like, ~recommended))
    return TP, FN, FP, TN
```

## 平均逆順位
平均逆順位$$\mathit{MRR}$$は次式で定義される。

$$
\mathit{MRR} = \frac{1}{\mid U \mid} \sum_{u \in U} \frac{1}{k_{u}}
$$

ここで、$$k_{u}$$はユーザ$$u$$向けの推薦リストにおいて最初にユーザ$$u$$が好きなアイテムが見つかったときの順位を表す。ここでは、評価値が4以上のアイテムを好きなアイテムとみなす。

###### ###### コード


```python
u = 0
【    問01    】
print('like = \n{}'.format(like))
【    問02    】
print('ku = {}'.format(ku))
【    問03    】
print('MRR = {:.3f}'.format(MRR))
```

###### ###### 結果


```bash
like = 
[[ True  True False False  True  True False False False False]
 [False False False False False False  True False  True False]
 [ True False False  True  True False False False False False]]
ku = [1. 2. 3.]
MRR = 0.611
```

このとき、次の問いに答えなさい。

### 01 好きなアイテムか否かの判定
`R`において、評価値が4以上の要素には`True`を、4未満の要素には`False`を入れたブール値配列を生成するコードを書きなさい。得られたブール値配列を`like`とすること。

★
1. 比較演算子を使う。

### 02 最初に好きなアイテムが見つかったときの順位
各ユーザ向けの推薦リストにおいて最初にそのユーザが好きなアイテムが見つかったときの順位$$k_{u}$$を`ndarray`としてまとめて求めるコードを書きなさい。得られた`ndarray`を`ku`とすること。

★★★
1. リスト内包表記を使う。
2. ブール値インデキシングを使う。
3. `numpy.nanmin()`を使う。
4. `numpy.array()`を使う。

### 03 MRR
$$\mathit{MRR}$$を求めるコードを書きなさい。得られた値を`MRR`とすること。

★★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。
3. `ndarray.size`を使う。

## 平均適合率
第$$K$$位までのユーザ$$u$$向けの推薦リストの平均逆順位$$\mathit{AP}_{u}$$は次式で定義される。

$$
\mathit{AP}_{u} = \frac{1}{\sum_{k=1}^{K} \mathit{rel}_k} \sum_{k=1}^{K} \mathit{rel}_k \cdot \mathit{precision}@k
$$

ここで、$$\mathit{precision}@k$$は順位$$k$$における適合率を表す。$$\mathit{rel}_k$$は次式で定義される。

$$
\mathit{rel}_k =
    \begin{cases}
        1 & (\text{第$k$位が好きなアイテムであるとき}) \\
        0 & (\text{otherwise})
    \end{cases}
$$

また、すべてのユーザの平均適合率を平均した$$\mathit{MAP}$$は次式で定義される。

$$
\mathit{MAP} = \frac{1}{\mid U \mid} \sum_{u \in U} \mathit{AP}_{u}
$$

###### コード

```python
# 各順位における適合率
precisions = []
for u in U:
    precisions_u = []
    for k in range(1, Iu[u].size+1):
        TP, FN, FP, TN = confusion_matrix(u, RA, k)
        precision_uk = TP / (TP + FP)
        precisions_u.append(precision_uk)
    precisions.append(precisions_u)
print('precisions = \n{}'.format(precisions))

【    問04    】
print('ranked_R = \n{}'.format(ranked_R))
【    問05    】
print('ranked_like = \n{}'.format(ranked_like))
【    問06    】
print('rel = \n{}'.format(rel))
【    問07    】
print('APu = {}'.format(APu))
【    問08    】
print('MAP = {:.3f}'.format(MAP))
```

###### 結果

```bash
precisions = 
[[1.0, 1.0, 0.6666666666666666, 0.75, 0.6, 0.6, 0.6], [0.0, 0.5, 0.3333333333333333, 0.25, 0.4, 0.4, 0.4], [0.0, 0.0, 0.3333333333333333, 0.5, 0.4, 0.4]]
ranked_R = 
[[ 5.  4.  3.  5.  2.  4. nan  2. nan nan]
 [ 3.  5.  3.  3.  4.  3.  2. nan nan nan]
 [ 3.  3.  5.  4.  3.  4. nan nan nan nan]]
ranked_like = 
[[ True  True False  True False  True False False False False]
 [False  True False False  True False False False False False]
 [False False  True  True False  True False False False False]]
rel = 
[[1 1 0 1 0 1 0 0 0 0]
 [0 1 0 0 1 0 0 0 0 0]
 [0 0 1 1 0 1 0 0 0 0]]
APu = [0.917 0.45  0.417]
MAP = 0.594
```

このとき、次の問いに答えなさい。

### 04 評価値行列の並べ替え
`RA`に示された順位にしたがって、`R`の各行をユーザごとの推薦順位の昇順に並べ替えた`ndarray`を生成するコードを書きなさい。生成した`ndarray`を`ranked_R`とすること。

★★★
1. リスト内包表記を使う。
2. `numpy.argsort()`を使う。
3. `numpy.array()`を使う。

### 05 好きなアイテムか否かの判定
`ranked_R`において、評価値が4以上の要素には`True`を、4未満の要素には`False`を入れたブール値配列を生成するコードを書きなさい。得られたブール値配列を`ranked_like`とすること。

★
1. 比較演算子を使う。

### 06 好きなアイテムか否かの判定
`ranked_like`において、`True`の要素には`1`を、`False`の要素には`0`を入れた`ndarray`を生成するコードを書きなさい。生成した`ndarray`を`rel`とすること。

★★★
1. リスト内包表記を使う。
2. `map()`を使う。
3. `list()`を使う。
4. `numpy.array()`を使う。

### 07 AP
上位`TOP_K`件の推薦リストについて各ユーザの$$\mathit{AP}_{u}$$を`ndarray`としてまとめて求めるコードを書きなさい。得られた`ndarray`を`APu`とすること。

★★★
1. 二重のリスト内包表記を使う。
2. `numpy.sum()`を使う。
3. `numpy.array()`を使う。

### 08 MAP
上位`TOP_K`件の推薦リストについて$$\mathit{MAP}$$を求めるコードを書きなさい。得られた値を`MAP`とすること。

★★
1. `numpy.sum()`を使う。
2. `ndarray.size`を使う。

## DCG
ユーザ$$u$$向けの推薦リストの$$\mathit{DCG}_{u}$$は次式で定義される。

$$
\mathit{DCG}_{u} = \sum_{i \in I_{u}^{\mathit{rec}}, k_{i} \leq K} \frac{r_{u,i}}{\max (1, \log_{\alpha} k_{i})}
$$

ここで、$$I_{u}^{\mathit{rec}}$$はユーザ$$u$$向けの推薦リストに含まれるアイテム集合である。$$k_{i}$$は推薦リストにおけるアイテム$$i$$の順位を表す。$$\alpha$$は対数の底$であり、ここでは、$$\alpha = 2$$とする。

ユーザ$$u$$向けの推薦リストの$\mathit{nDCG}_{u}$は次式で定義される。

$$
\mathit{nDCG}_{u} = \frac{\mathit{DCG}_{u}}{\mathit{IDCG}_{u}}
$$

ここで、$$\mathit{IDCG}_{u}$$は、ユーザ$$u$$のテストデータを理想的な順位（評価値が高い順）に並べ替えた推薦リストのDCGを表す。すべてのユーザのnDCGの平均値を$$\mathit{nDCG}$$とすると、次式で定義される。

$$
\mathit{nDCG} = \frac{1}{\mid U \mid} \sum_{u \in U} \mathit{nDCG}_{u}
$$

###### コード

```python
Iu_rec = [I[~np.isnan(RA[u])] for u in U]
【    問09    】
print('DCGu = {}'.format(DCGu))

【    問10    】
print('RI = \n{}'.format(RI))
【    問11    】
print('Iu_recI = \n{}'.format(Iu_recI))
【    問12    】
print('IDCGu = {}'.format(IDCGu))
【    問13    】
print('nDCGu = {}'.format(nDCGu))
【    問14    】
print('nDCG = {:.3f}'.format(nDCG))
```

###### 結果

```bash
DCGu = [14.254 13.115 12.447]
RI = 
[[ 1  3  5  8  2  4  6  7  9 10]
 [ 3  4  5  6  7  8  2  9  1 10]
 [ 2  7  4  1  3  5  8  6  9 10]]
Iu_recI = 
[[0 1 2 4 5]
 [0 1 2 6 8]
 [0 2 3 4 5]]
IDCGu = [15.816 13.685 14.316]
nDCGu = [0.901 0.958 0.869]
nDCG = 0.910
```

このとき、次の問いに答えなさい。

### 09 DCGu
各ユーザの$$\mathit{DCG}_{u}$$を`ndarray`としてまとめて求めるコードを書きなさい。ただし、$$\alpha$$は`ALPHA`とする。得られた`ndarray`を`DCGu`とすること。

★★★
1. 二重のリスト内包表記を使う。
2. `math.log()`を使う。
3. `numpy.max()`を使う。
4. `numpy.sum()`を使う。
5. `numpy.array()`を使う。

### 10 理想的な推薦順位
`R`において、各ユーザにとっての理想的な推薦順位を`ndarray`として生成するコードを書きなさい。生成した`ndarray`を`RI`とすること。

★★★
1. `numpy.argsort()`を2回使う。

### 11 理想的な推薦リスト
`RI`から上位`TOP_K`以内のアイテム集合を各ユーザにとっての理想的な推薦リストとする。このとき各ユーザにとっての理想的な推薦リストを`ndarray`として生成するコードを書きなさい。生成した`ndarray`を`Iu_recI`とすること。

★★★
1. リスト内包表記を使う。
2. ブール値インデキシングを使う。
3. `numpy.array()`を使う。

### 10 IDCGu
各ユーザの$$\mathit{IDCG}_{u}$$を`ndarray`としてまとめて求めるコードを書きなさい。ただし、$$\alpha$$は`ALPHA`とする。得られた`ndarray`を`IDCGu`とすること。

★★★
1. 二重のリスト内包表記を使う。
2. `math.log()`を使う。
3. `numpy.max()`を使う。
4. `numpy.sum()`を使う。
5. `numpy.array()`を使う。

### 11 nDCGu
各ユーザの$$\mathit{nDCG}_{u}$$を`ndarray`としてまとめて求めるコードを書きなさい。得られた`ndarray`を`nDCGu`とすること。

★
1. `DCGu`を参照する。
2. `IDCGu`を参照する。

### 12 nDCG
$$\mathit{nDCG}$$を求めるコードを書きなさい。得られた値を`nDCG`とすること。

★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。
3. `ndarray.size`を使う。

