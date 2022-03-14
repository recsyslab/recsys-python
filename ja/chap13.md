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
```

