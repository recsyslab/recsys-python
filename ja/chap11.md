<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# 第11章 嗜好予測の正確性

## テストデータと予測評価値
次の評価値行列$$\boldsymbol{R}^{\mathit{test}}$$はテストデータである。$$\boldsymbol{R}$$の$$(u, i)$$成分はユーザ$$u$$がアイテム$$i$$に与えた評価値$$r_{u,i}$$を表す。ただし、$$-$$で示した要素はテストデータの対象ではないことを表す。また、$$\boldsymbol{R}^{\mathit{test}}$$に含まれる成分の集合を$$R^{\mathit{test}}$$と表す。

$$
\boldsymbol{R}^{\mathit{test}} = \left[
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
```

## 平均絶対誤差
平均絶対誤差$$\mathit{MAE}$$は次式で定義される。

$$
\mathit{MAE} = \frac{\sum_{(u,i) \in R^{\mathit{test}}} \mid \hat{r}_{u,i} - r_{u,i} \mid}{\mid R^{\mathit{test}} \mid}
$$

コード
```python
【    問01    】
print('MAE_{} = {:.3f}'.format('A', MAE_A))
【    問02    】
print('MAE_{} = {:.3f}'.format('B', MAE_B))
```

結果
```bash
MAE_A = 0.800
MAE_B = 1.000
```

このとき、次の問いに答えなさい。

### 01 推薦システムAのMAE
推薦システムAの$$\mathit{MAE}^{A}$$を求めるコードを書きなさい。得られた値を`MAE_A`とすること。

★★★
1. 二重のリスト内包表記を使う。
2. `numpy.abs()`を使う。
3. `numpy.count_nonzero()`を使う。
4. `numpy.isnan()`を使う。
5. `~`演算子を使う。
6. `numpy.nansum()`を使う。


### 02 推薦システムBのMAE
推薦システムBの$$\mathit{MAE}^{B}$$を求めるコードを書きなさい。得られた値を`MAE_A`とすること。

★★★
1. 二重のリスト内包表記を使う。
2. `numpy.abs()`を使う。
3. `numpy.count_nonzero()`を使う。
4. `numpy.isnan()`を使う。
5. `~`演算子を使う。
6. `numpy.nansum()`を使う。

## 平均二乗誤差
平均二乗誤差$$\mathit{MSE}$$は次式で定義される。

$$
\mathit{MSE} = \frac{\sum_{(u,i) \in R^{\mathit{test}}} (\hat{r}_{u,i} - r_{u,i})^{2}}{\mid R^{\mathit{test}} \mid}
$$

コード
```python
【    問03    】
print('MSE_{} = {:.3f}'.format('A', MSE_A))
【    問04    】
print('MSE_{} = {:.3f}'.format('B', MSE_B))
```

結果
```bash
MSE_A = 1.600
MSE_B = 1.000
```

このとき、次の問いに答えなさい。

### 03 推薦システムAのMSE
推薦システムAの$$\mathit{MSE}^{A}$$を求めるコードを書きなさい。得られた値を`MSE_A`とすること。

★★★
1. 二重のリスト内包表記を使う。
3. `numpy.count_nonzero()`を使う。
4. `numpy.isnan()`を使う。
5. `~`演算子を使う。
6. `numpy.nansum()`を使う。

### 04 推薦システムBのMSE
推薦システムAの$$\mathit{MSE}^{B}$$を求めるコードを書きなさい。得られた値を`MSE_B`とすること。

★★★
1. 二重のリスト内包表記を使う。
3. `numpy.count_nonzero()`を使う。
4. `numpy.isnan()`を使う。
5. `~`演算子を使う。
6. `numpy.nansum()`を使う。

