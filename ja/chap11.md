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
              [np.nan, 4,      np.nan, np.nan, np.nan, np.nan, 2,      np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, 2,      np.nan, np.nan, np.nan, 5,      np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3,      np.nan, np.nan],
])
U = np.arange(R.shape[0])
I = np.arange(R.shape[1])

# 推薦システムAによる予測評価値
RA = np.array([
               [np.nan, 2,      np.nan, np.nan, np.nan, np.nan, 2,      np.nan, np.nan, np.nan],
               [np.nan, np.nan, np.nan, np.nan, 2,      np.nan, np.nan, np.nan, 3,      np.nan],
               [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3,      np.nan, np.nan],
])

# 推薦システムBによる予測評価値
RB = np.array([
              [np.nan, 3,      np.nan, np.nan, np.nan, np.nan, 1,      np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, 3,      np.nan, np.nan, np.nan, 4,      np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 4,      np.nan, np.nan],
])
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
2. `numpy.count_nonzero()`を使う。
3. `numpy.isnan()`を使う。
4. `~`演算子を使う。
5. `numpy.nansum()`を使う。

### 04 推薦システムBのMSE
推薦システムAの$$\mathit{MSE}^{B}$$を求めるコードを書きなさい。得られた値を`MSE_B`とすること。

★★★
1. 二重のリスト内包表記を使う。
2. `numpy.count_nonzero()`を使う。
3. `numpy.isnan()`を使う。
4. `~`演算子を使う。
5. `numpy.nansum()`を使う。

## 二乗平均平方根誤差
二乗平均平方根誤差$$\mathit{RMSE}$$は次式で定義される。

$$
\mathit{RMSE} = \sqrt{\frac{\sum_{(u,i) \in R^{\mathit{test}}} (\hat{r}_{u,i} - r_{u,i})^{2}}{\mid R^{\mathit{test}} \mid}}
$$

コード
```python
print('MSE_{} = {:.3f}'.format('B', MSE_B))
【    問05    】
print('RMSE_{} = {:.3f}'.format('A', RMSE_A))
【    問06    】
print('RMSE_{} = {:.3f}'.format('B', RMSE_B))
```

結果
```bash
RMSE_A = 1.265
RMSE_B = 1.000
```

このとき、次の問いに答えなさい。

### 05 推薦システムAのRMSE
推薦システムAの$$\mathit{RMSE}^{A}$$を求めるコードを書きなさい。得られた値を`RMSE_A`とすること。

★
1. `MSE_A`を参照する。
2. `numpy.sart()`を使う。

★★★
1. 二重のリスト内包表記を使う。
2. `numpy.count_nonzero()`を使う。
3. `numpy.isnan()`を使う。
4. `~`演算子を使う。
5. `numpy.nansum()`を使う。
6. `numpy.sqrt()`を使う。

### 06 推薦システムBのRMSE
推薦システムBの$$\mathit{RMSE}^{B}$$を求めるコードを書きなさい。得られた値を`RMSE_B`とすること。

★
1. `MSE_B`を参照する。
2. `numpy.sart()`を使う。

★★★
1. 二重のリスト内包表記を使う。
2. `numpy.count_nonzero()`を使う。
3. `numpy.isnan()`を使う。
4. `~`演算子を使う。
5. `numpy.nansum()`を使う。
6. `numpy.sqrt()`を使う。

## 正規化MAEと正規化RMSE
正規化MAE$$\mathit{NMAE}$$と正規化RMSE$$\mathit{NRMSE}$$は、それぞれ次式で定義される。

$$
\mathit{NMAE} = \frac{\mathit{MAE}}{r_{\mathit{max}} - r_{\mathit{min}}}
$$

$$
\mathit{NRMSE} = \frac{\mathit{RMSE}}{r_{\mathit{max}} - r_{\mathit{min}}}
$$

ここで、$$r_{\mathit{max}}$$、$$r_{\mathit{min}}$$は、それぞれ、とりうる評価値の最大値、最小値を表す。

コード
```python
# NMAE
【    問07    】
print('NMAE_{} = {:.3f}'.format('A', NMAE_A))
【    問08    】
print('NMAE_{} = {:.3f}'.format('B', NMAE_B))

# NRMSE
【    問09    】
print('NRMSE_{} = {:.3f}'.format('A', NRMSE_A))
【    問10    】
print('NRMSE_{} = {:.3f}'.format('B', NRMSE_B))
```

結果
```bash
NMAE_A = 0.200
NMAE_B = 0.250
NRMSE_A = 0.316
NRMSE_B = 0.250
```

このとき、次の問いに答えなさい。

### 07 推薦システムAのNMAE
推薦システムAの$$\mathit{NMAE}^{A}$$を求めるコードを書きなさい。ただし、$$r_{\mathit{max}}$$、$$r_{\mathit{min}}$$は、それぞれ`R_MAX`、`R_MIN`とする。得られた値を`NMAE_A`とすること。

★
1. `MAE_A`を参照する。

### 08 推薦システムBのNMAE
推薦システムAの$$\mathit{NMAE}^{B}$$を求めるコードを書きなさい。ただし、$$r_{\mathit{max}}$$、$$r_{\mathit{min}}$$は、それぞれ`R_MAX`、`R_MIN`とする。得られた値を`NMAE_B`とすること。

★
1. `MAE_B`を参照する。

### 09 推薦システムAのNRMSE
推薦システムAの$$\mathit{NRMSE}^{A}$$を求めるコードを書きなさい。ただし、$$r_{\mathit{max}}$$、$$r_{\mathit{min}}$$は、それぞれ`R_MAX`、`R_MIN`とする。得られた値を`NRMSE_A`とすること。

★
1. `RMSE_A`を参照する。

### 10 推薦システムBのNRMSE
推薦システムBの$$\mathit{NRMSE}^{B}$$を求めるコードを書きなさい。ただし、$$r_{\mathit{max}}$$、$$r_{\mathit{min}}$$は、それぞれ`R_MAX`、`R_MIN`とする。得られた値を`NRMSE_B`とすること。

★
1. `RMSE_B`を参照する。
