<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# 第11章 嗜好予測の正確性

## 準備
次のコードを書きなさい。

```python
import numpy as np

# テストデータ
R = np.array([
              [np.nan, 4, np.nan, np.nan, np.nan, np.nan, 2, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, 2, np.nan, np.nan, np.nan, 5, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3, np.nan, np.nan],
])
print('R = {}'.format(R))
U = np.arange(R.shape[0])
I = np.arange(R.shape[1])

# 推薦システムAによる予測評価値
RA = np.array([
              [np.nan, 2, np.nan, np.nan, np.nan, np.nan, 2, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, 2, np.nan, np.nan, np.nan, 3, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3, np.nan, np.nan],
])
print('RA = {}'.format(RA))

# 推薦システムBによる予測評価値
RB = np.array([
              [np.nan, 3, np.nan, np.nan, np.nan, np.nan, 1, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, 3, np.nan, np.nan, np.nan, 4, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 4, np.nan, np.nan],
])
print('RB = {}'.format(RB))
print()
```



次の評価値行列$$\boldsymbol{R}$$はテストデータである。$$\boldsymbol{R}$$の$$(u, i)$$成分はユーザ$$u$$がアイテム$$i$$に与えた評価値$$r_{u,i}$$を表す。ただし、$$-$$で示した要素はテストデータの対象ではないことを表す。

$$
\boldsymbol{R} = \left[
 \begin{array}{rrrrrrrrrr}
 -  & 4 & - & - & - & - & 2 & - & - & - \\
 -  & - & - & - & 2 & - & - & - & 5 & - \\
 -  & - & - & - & - & - & - & 3 & - & - \\
 \end{array}
\right]
$$

次の評価値行列$$\hat{\boldsymbol{R}}^{A}$$、$$\hat{\boldsymbol{R}}^{B}$$は、それぞれ推薦システムA、推薦システムBによる予測評価値である。$$\hat{\boldsymbol{R}}^{A}$$、$$\hat{\boldsymbol{R}}^{B}$$の$$(u, i)$$成分は、それぞれ推薦システムA、推薦システムBによる予測評価値$$\hat{r}_{u,i}$$を表す。

$$
\hat{\boldsymbol{R}}^{A} = \left[
 \begin{array}{rrrrrrrrrr}
 -  & 2 & - & - & - & - & 2 & - & - & - \\
 -  & - & - & - & 2 & - & - & - & 3 & - \\
 -  & - & - & - & - & - & - & 3 & - & - \\
 \end{array}
\right]
$$

$$
\hat{\boldsymbol{R}}^{B} = \left[
 \begin{array}{rrrrrrrrrr}
 -  & 3 & - & - & - & - & 1 & - & - & - \\
 -  & - & - & - & 3 & - & - & - & 4 & - \\
 -  & - & - & - & - & - & - & 4 & - & - \\
 \end{array}
\right]
$$



次のコードを書きなさい。

```python
import pprint
import numpy as np
np.set_printoptions(precision=3)
```
