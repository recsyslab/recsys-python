<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# 第7章 単純ベイズ分類器

```python
import numpy as np
from fractions import Fraction

# 上位K件
TOP_K = 3
# スムージングパラメタ
ALPHA = 0
# クラス数
N = 2
# 特徴量kがとりうる値のユニーク数
M = [2, 2, 2, 2, 2, 2]

# 評価履歴
Du = np.array([
              [1, 0, 0, 0, 1, 0, +1],
              [0, 1, 0, 0, 1, 0, +1],
              [1, 1, 0, 0, 1, 0, +1],
              [1, 0, 0, 1, 1, 0, +1],
              [1, 0, 0, 0, 0, 1, +1],
              [0, 1, 0, 1, 0, 1, +1],
              [0, 0, 1, 0, 1, 0, -1],
              [0, 0, 1, 1, 1, 0, -1],
              [0, 1, 0, 0, 1, 1, -1],
              [0, 0, 1, 0, 0, 1, -1],
              [1, 1, 0, 1, 1, 0, np.nan],
              [0, 0, 1, 0, 1, 1, np.nan],
              [0, 1, 1, 1, 1, 0, np.nan],
])
print('Du = \n{}'.format(Du))
print()

# アイテム集合
I = np.arange(Du.shape[0])
print('I = {}'.format(I))
print()

# アイテムの特徴ベクトル
x = Du[:,:-1]
print('x = \n{}'.format(x))
print()

# 評価値
ru = Du[:,-1]
print('ru = {}'.format(ru))
print()

# 訓練データ
IL = I[~np.isnan(ru)]
DuL = Du[IL]
xL = x[IL]
ruL = ru[IL]
print('IL = {}'.format(IL))
print('DuL = {}'.format(DuL))
print('|DuL| = {}'.format(DuL.shape[0]))
print('xL = \n{}'.format(xL))
print('ruL = {}'.format(ruL))
print()

# 予測対象データ
IU = I[np.isnan(ru)]
DuU = Du[IU]
xU = x[IU]
print('IU = {}'.format(IU))
print('DuU = {}'.format(DuU))
print('|DuU| = {}'.format(DuU.shape[0]))
print('xL = \n{}'.format(xU))
print()

# 訓練データのうち評価値がrui=+1, rui=-1となる事例
print('DuL+ = \n{}'.format(DuL[ruL==+1]))
print('DuL- = \n{}'.format(DuL[ruL==-1]))
print('|DuL+| = {}'.format(len(DuL[ruL==+1])))
print('|DuL-| = {}'.format(len(DuL[ruL==-1])))
print()
```

## 問題設定
アイテム$$i$$ついて好む確率、嫌う確率はそれぞれ次式のように表される。

$$
P(R = +1 \mid x_{i,1}, \ldots, x_{i,d})
$$

$$
P(R = -1 \mid x_{i,1}, \ldots, x_{i,d})
$$

ここで、$$d$$は特徴ベクトル$$\boldsymbol{x}_{i}$$の次元数である。ベイズの定理および単純ベイズ仮定を用いると、それぞれ次式のように表される。

$$
P(R = +1) \prod_{k=1}^{d} P(X_{k} = x_{i,k} \mid R = +1)
$$

$$
P(R = -1) \prod_{k=1}^{d} P(X_{k} = x_{i,k} \mid R = -1)
$$

## 事前確率

上式の$$P(R = +1)$$、$$P(R = -1)$$は事前確率であり、ユーザ$$u$$が好む/嫌う確率を表す。それぞれ次式で表される。

$$
P(R = +1) = \frac{\mid D^{L+}_{u} \mid}{\mid D^{L}_{u} \mid}
$$

$$
P(R = -1) = \frac{\mid D^{L-}_{u} \mid}{\mid D^{L}_{u} \mid}
$$

次の関数は、評価値が`r`となる事前確率を返す関数`P_prior(r)`である。

関数
```python
def P_prior(r):
    """
    評価値がrとなる事前確率を返す。

    Parameters
    ----------
    r : int
        評価値

    Returns
    -------
    float
        事前確率
    """
    【問題01】
    【問題02】
    prob = Fraction(num, den, _normalize=False)
    return prob
```

コード
```python
# 事前確率
r = +1
print('P(R={:+}) = {}'.format(r, P_prior(r)))
r = -1
print('P(R={:+}) = {}'.format(r, P_prior(r)))
```

結果
```bash
P(R=+1) = 6/10
P(R=-1) = 4/10
```

## 01 評価値がrとなる事前確率（分子）
事前確率の式の分子を求めるコードを書きなさい。得られた値を`num`とすること。

★
1. `ndarray.shape`を使う。
2. 行列のブールインデックス参照を使う。

## 02 評価値がrとなる事前確率（分母）
事前確率の式の分母を求めるコードを書きなさい。得られた値を`den`とすること。

★
1. `ndarray.shape`を使う。

## 特徴量kに関する条件付き確率
上式の$$P(X_{k} = x_{i,k} \mid R = +1)$$、$$P(X_{k} = x_{i,k} \mid R = -1)$$は、特徴量$$k$$に関する条件付き確率であり、それぞれ次式のように表される。

$$
P(X_{k} = x_{i,k} \mid R = +1) = \frac{\mid D^{L+}_{u}(x_{i,k}) \mid}{\mid D^{L+}_{u} \mid}
$$

$$
P(X_{k} = x_{i,k} \mid R = -1) = \frac{\mid D^{L-}_{u}(x_{i,k}) \mid}{\mid D^{L-}_{u} \mid}
$$

関数
```python
def P_cond(i, k, r):
    """
    評価値がrとなる条件下でアイテムiの特徴量kの条件付き確率を返す。

    Parameters
    ----------
    i : int
        アイテムiのインデックス
    k : int
        特徴量kのインデックス
    r : int
        評価値

    Returns
    -------
    float
        条件付き確率
    """
    num = DuL[ruL==r][xL[:,k][ruL==r]==x[i,k]].shape[0] + ALPHA
    den = DuL[ruL==r].shape[0] + ALPHA * M[k]
    prob = Fraction(num, den, _normalize=False)
    return prob
```

コード
```python
# 特徴量ごとの条件付き確率
i = 10
k = 0
r = +1
print('P(X{}=x{},{}|R={:+}) = {}'.format(k, i, k, r, P_cond(i, k, r)))
r = -1
print('P(X{}=x{},{}|R={:+}) = {}'.format(k, i, k, r, P_cond(i, k, r)))
```


