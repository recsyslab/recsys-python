<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# 第12章 好き嫌い分類に基づく評価指標

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

次の評価値行列$$\hat{\boldsymbol{R}}^{A}$$、$$\hat{\boldsymbol{R}}^{B}$$は、それぞれ推薦システムA、推薦システムBによる推薦リストである。$$\hat{\boldsymbol{R}}^{A}$$、$$\hat{\boldsymbol{R}}^{B}$$の$$(u, i)$$成分は、それぞれ推薦システムA、推薦システムBによる推薦リストにおける順位を表す。

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
             [5, 4, 3, np.nan, 5, 4, 2, 2, np.nan, np.nan],
             [3, 3, 3, 3, 2, np.nan, 4, np.nan, 5, np.nan],
             [4, np.nan, 3, 5, 4, 3, np.nan, 3, np.nan, np.nan],
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


