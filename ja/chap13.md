<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# 第13章 推薦順位に基づく正確性

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
from scipy.stats import rankdata

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

# 推薦システムAによる推薦リスト
RA = np.array([
               [1,      np.nan, 3,      np.nan, 4,      2,      5,      np.nan, np.nan, np.nan],
               [4,      1,      np.nan, 3,      np.nan, np.nan, 5,      np.nan, 2,      np.nan],
               [np.nan, np.nan, 5,      3,      4,      2,      np.nan, 1,      np.nan, np.nan],
])
Iu = [I[~np.isnan(RA[u])] for u in U]

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

コード
```python
u = 0
【    問01    】
print('like = \n{}'.format(like))
【    問02    】
print('ku = {}'.format(ku))
【    問03    】
print('MRR = {:.3f}'.format(MRR))
```

結果
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
ユーザ$$u$$向けの平均逆順位$$\mathit{AP}_{u}$$は次式で定義される。

$$
\mathit{AP}_{u} = \frac{1}{\sum_{k=1}^{K} \mathit{rel}_k} \sum_{k=1}^{K} \mathit{rel}_k \cdot \mathit{precision}@k
$$

$$
\mathit{rel}_k =
    \begin{cases}
        1 & (\text{第$k$位が好きなアイテムであるとき}) \\
        0 & (\text{otherwise})
    \end{cases}
$$
ここで、$$k_{u}$$はユーザ$$u$$向けの推薦リストにおいて最初にユーザ$$u$$が好きなアイテムが見つかったときの順位を表す。ここでは、評価値が4以上のアイテムを好きなアイテムとみなす。

コード
```python
u = 0
【    問01    】
print('like = \n{}'.format(like))
【    問02    】
print('ku = {}'.format(ku))
【    問03    】
print('MRR = {:.3f}'.format(MRR))
```

結果
```bash
like = 
[[ True  True False False  True  True False False False False]
 [False False False False False False  True False  True False]
 [ True False False  True  True False False False False False]]
ku = [1. 2. 3.]
MRR = 0.611
```

このとき、次の問いに答えなさい。
