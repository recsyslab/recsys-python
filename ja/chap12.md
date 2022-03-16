---
title: 第12章 推薦システムの評価 | 好き嫌い分類に基づく評価指標 | recsys-python
layout: default
---

{% include header.html %}

# 第12章 推薦システムの評価 | 好き嫌い分類に基づく評価指標

## テストデータと予測評価値
次の評価値行列$$\boldsymbol{R}^{\mathit{test}}$$はテストデータである。$$\boldsymbol{R}$$の$$(u, i)$$成分はユーザ$$u$$がアイテム$$i$$に与えた評価値$$r_{u,i}$$を表す。ただし、$$-$$で示した要素はテストデータの対象ではないことを表す。また、$$\boldsymbol{R}^{\mathit{test}}$$に含まれる成分の集合を$$R^{\mathit{test}}$$と表す。

$$
\boldsymbol{R}^{\mathit{test}} = \left[
 \begin{array}{rrrrrrrrrr}
  5 & 4 & 3 & - & 5 & 4 & 2 & 2 & - & - \\
 \end{array}
\right]
$$

次の行列$$\hat{\boldsymbol{R}}^{A}$$、$$\hat{\boldsymbol{R}}^{B}$$は、それぞれ推薦システムA、推薦システムBによる推薦リストである。$$\hat{\boldsymbol{R}}^{A}$$、$$\hat{\boldsymbol{R}}^{B}$$の$$(u, i)$$成分は、それぞれユーザ$$u$$向けの推薦システムA、推薦システムBによる推薦リストにおける順位を表す。

$$
\hat{\boldsymbol{R}}^{A} = \left[
 \begin{array}{rrrrrrrrrr}
  1 & 6 & 3 & - & 4 & 2 & 5 & 7 & - & - \\
 \end{array}
\right]
$$

$$
\hat{\boldsymbol{R}}^{B} = \left[
 \begin{array}{rrrrrrrrrr}
  4 & 3 & 1 & - & 6 & 7 & 2 & 5 & - & - \\
 \end{array}
\right]
$$

## 準備
次のコードを書きなさい。

```python
import numpy as np

# テストデータ
R = np.array([
              [5, 4,      3, np.nan, 5, 4,      2,      2,      np.nan, np.nan],
])
U = np.arange(R.shape[0])
I = np.arange(R.shape[1])
Iu = [I[~np.isnan(R)[u,:]] for u in U]

# 推薦システムAによる推薦リスト
RA = np.array([
               [1, 6, 3, np.nan, 4, 2, 5, 7, np.nan, np.nan],
])

# 推薦システムBによる推薦リスト
RB = np.array([
               [4, 3, 1, np.nan, 6, 7, 2, 5, np.nan, np.nan],
])
```

## 混同行列
次の表は混同行列である。

|      | 推薦された | 推薦されなかった |
| ---- | --------- | ----------------|
| 好き | 好きなアイテムが推薦された数（TP） | 好きなアイテムが推薦されなかった数（FN） |
| 嫌い | 嫌いなアイテムが推薦された数（FP） | 嫌いなアイテムが推薦されなかった数（TN） |

ここでは、評価値が4以上のアイテムを好きなアイテムとみなす。次の関数は、ユーザ`u`向け推薦リスト`RS`の上位`K`件における混同行列の各値を返す関数である。

###### 関数

```python
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
    【    問01    】
    print('like = {}'.format(like))
    【    問02    】
    print('recommended@{} = {}'.format(K, recommended))
    【    問03    】
    print('TP@{} = {}'.format(K, TP))
    【    問04    】
    print('FN@{} = {}'.format(K, FN))
    【    問05    】
    print('FP@{} = {}'.format(K, FP))
    【    問06    】
    print('TN@{} = {}'.format(K, TN))
    return TP, FN, FP, TN
```

###### コード

```python
u = 0
K = 3
TP, FN, FP, TN = confusion_matrix(u, RA, K)
print('混同行列 = \n{}'.format(np.array([[TP, FN], [FP, TN]])))
```

###### 結果

```bash
like = [ True  True False  True  True False False]
recommended@3 = [ True False  True False  True False False]
TP@3 = 2
FN@3 = 2
FP@3 = 1
TN@3 = 2
混同行列 = 
[[2 2]
 [1 2]]
```

このとき、次の問いに答えなさい。

### 01 好きなアイテムか否かの判定
`R`において、ユーザ`u`が評価済みのアイテム集合を対象に評価値が4以上の要素には`True`を、4未満の要素には`False`を入れたブール値配列を生成するコードを書きなさい。得られたブール値配列を`like`とすること。

★★
1. 整数配列インデキシングを使う。

### 02 推薦されたアイテムか否かの判定
推薦リスト`RS`において順位が`K`以内の要素には`True`を、それ以外の要素には`False`を入れたブール値配列を生成するコードを書きなさい。得られたブール値配列を`recommended`とすること。

★★
1. 整数配列インデキシングを使う。

### 03 好きなアイテムが推薦された数（TP）
上位$$K$$件の推薦リストにおいて、好きなアイテムが推薦された数$$\mathit{TP}@K$$を取得するコードを書きなさい。得られた値を`TP`とすること。

★★
1. `numpy.logical_and()`を使う。
2. `numpy.count_nonzero()`を使う。

### 04 好きなアイテムが推薦されなかった数（FN）
上位$$K$$件の推薦リストにおいて、好きなアイテムが推薦されなかった数$$\mathit{FN}@K$$を取得するコードを書きなさい。得られた値を`FN`とすること。

★★
1. `numpy.logical_and()`を使う。
2. `numpy.count_nonzero()`を使う。

### 05 嫌いなアイテムが推薦された数（FP）
上位$$K$$件の推薦リストにおいて、嫌いなアイテムが推薦された数$$\mathit{FP}@K$$を取得するコードを書きなさい。得られた値を`FP`とすること。

★★
1. `numpy.logical_and()`を使う。
2. `numpy.count_nonzero()`を使う。

### 06 嫌いなアイテムが推薦されなかった数（TN）
上位$$K$$件の推薦リストにおいて、嫌いなアイテムが推薦されなかった数$$\mathit{TN}@K$$を取得するコードを書きなさい。得られた値を`TN`とすること。

★★
1. `numpy.logical_and()`を使う。
2. `numpy.count_nonzero()`を使う。

## 真陽性率と偽陽性率
上位$$K$$件の推薦リストにおける真陽性率$$\mathit{TPR}@K$$、偽陽性率$$\mathit{FPR}@K$$は、それぞれ次式で定義される。

$$
\mathit{TPR}@K = \frac{\mathit{TP}@K}{\mathit{TP}@K + \mathit{FN}@K}
$$

$$
\mathit{FPR}@K = \frac{\mathit{FP}@K}{\mathit{FP}@K + \mathit{TN}@K}
$$

###### コード

```python
【    問07    】
print('TPR@{} = {:.3f}'.format(K, TPR))
【    問08    】
print('FPR@{} = {:.3f}'.format(K, FPR))
```

###### 結果

```bash
TPR@3 = 0.500
FPR@3 = 0.333
```

このとき、次の問いに答えなさい。

### 07 真陽性率（TPR）
上位$$K$$件の推薦リストにおける真陽性率$$\mathit{TPR}@K$$を求めるコードを書きなさい。得られた値を`TPR`とすること。

★
1. `TP`を参照する。
2. `FN`を参照する。

### 08 偽陽性率（FPR）
上位$$K$$件の推薦リストにおける偽陽性率$$\mathit{FPR}@K$$を求めるコードを書きなさい。得られた値を`TPR`とすること。

★
1. `FP`を参照する。
2. `TN`を参照する。

## 適合率と再現率
上位$$K$$件の推薦リストにおける適合率$$\mathit{precision}@K$$、再現率$$\mathit{recall}@K$$は、それぞれ次式で定義される。

$$
\mathit{precision}@K = \frac{\mathit{TP}@K}{\mathit{TP}@K + \mathit{FP}@K}
$$

$$
\mathit{recall}@K = \frac{\mathit{TP}@K}{\mathit{TP}@K + \mathit{FN}@K}
$$

また、上位$$K$$件の推薦リストにおけるF値$$F_{1}@K$$は、次式で定義される。

$$
F_{1}@K = \frac{2 \cdot \mathit{precision}@K \cdot \mathit{recall}@K}{\mathit{precision}@K + \mathit{recall}@K}
$$

###### コード

```python
【    問09    】
print('precision@{} = {:.3f}'.format(K, precision))
【    問10    】
print('recall@{} = {:.3f}'.format(K, recall))
【    問11    】
print('F1@{} = {:.3f}'.format(K, F1))
```

###### 結果

```bash
precision@3 = 0.667
recall@3 = 0.500
F1@3 = 0.571
```

このとき、次の問いに答えなさい。

### 09 適合率
上位$$K$$件の推薦リストにおける適合率$$\mathit{precision}@K$$を求めるコードを書きなさい。得られた値を`precision`とすること。

★
1. `TP`を参照する。
2. `FP`を参照する。

### 10 再現率
上位$$K$$件の推薦リストにおける再現率$$\mathit{recall}@K$$を求めるコードを書きなさい。得られた値を`recall`とすること。

★
1. `TP`を参照する。
2. `FN`を参照する。

### 11 F値
上位$$K$$件の推薦リストにおけるF値$$F_{1}@K$$を求めるコードを書きなさい。得られた値を`F1`とすること。

★
1. `precision`を参照する。
2. `recall`を参照する。
