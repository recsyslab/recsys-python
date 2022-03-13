<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# 第9章 単純ベイズ分類器

### 準備
次のコードを書きなさい。

```python
import numpy as np
from fractions import Fraction

# 上位K件
TOP_K = 3
# スムージングパラメタ
ALPHA = 1
# クラス数
N = 2
# 各特徴量がとりうる値のユニーク数
M = [2, 2, 2, 2, 2, 2]

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
I = np.arange(Du.shape[0])
x = Du[:,:-1]
ru = Du[:,-1]

Iu = I[~np.isnan(ru)]
Iu_not = np.setdiff1d(I, Iu)
DuL = Du[Iu]
xL = x[Iu]
ruL = ru[Iu]
DuU = Du[Iu_not]
xU = x[Iu_not]
```

## 問題設定
アイテム$$i$$ついて好む確率、嫌う確率はそれぞれ次式のように表される。

$$
P(R = +1 \mid x_{i,1}, \ldots, x_{i,d})
$$

$$
P(R = -1 \mid x_{i,1}, \ldots, x_{i,d})
$$

ここで、$$d$$は特徴ベクトル$$\boldsymbol{x}_{i}$$の次元数である。ベイズの定理および単純ベイズ仮定を用いると、上式はそれぞれ次式のように表される。

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

ここで、$$\mid D_{u}^{L}$$は訓練事例数である。$$\mid D_{u}^{L+}$$、$$\mid D_{u}^{L-}$$はそれぞれい訓練事例に含まれる正事例数、負事例数である。これらの事前確率を返す関数を次のコードのとおり定義する。

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
    Fraction
        事前確率
    """
    【    問01    】【    問06    】
    【    問02    】【    問07    】
    prob = Fraction(num, den, _normalize=False)
    return prob
```

コード
```python
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

このとき、関数の仕様を満たすように、次の問いに答えなさい。

### 01 評価値がrとなる事前確率（分子）
事前確率の式の分子を求めるコードを書きなさい。得られた値を`num`とすること。

★
1. `ndarray.shape`を使う。
2. ブール値インデキシングを使う。

### 02 評価値がrとなる事前確率（分母）
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

ここで、$$\mid D^{L+}_{u}(x_{i,k}) \mid$$は評価値が$$+1$$である訓練事例のうち、属性$$k$$の特徴量が対象アイテム$$i$$の特徴量$$x_{i,k}$$に一致する事例数を表す。同様に、$$\mid D^{L-}_{u}(x_{i,k}) \mid$$は評価値が$$-1$$である訓練事例のうち、属性$$k$$の特徴量が対象アイテム$$i$$の特徴量$$x_{i,k}$$に一致する事例数を表す。これらの条件付き確率を返す関数を次のコードのとおり定義する。

関数
```python
def P_cond(i, k, r):
    """
    評価値がrとなる条件下でアイテムiの特徴量kに関する条件付き確率を返す。

    Parameters
    ----------
    i : int
        アイテムiのID
    k : int
        特徴量kのインデックス
    r : int
        評価値

    Returns
    -------
    Fraction
        条件付き確率
    """
    【問題03】【問題08】
    【問題04】【問題09】
    prob = Fraction(num, den, _normalize=False)
    return prob
```

コード
```python
i = 10
k = 0
r = +1
print('P(X{}=x{},{}|R={:+}) = {}'.format(k, i, k, r, P_cond(i, k, r)))
r = -1
print('P(X{}=x{},{}|R={:+}) = {}'.format(k, i, k, r, P_cond(i, k, r)))
```

結果
```bash
P(X0=x10,0|R=+1) = 4/6
P(X0=x10,0|R=-1) = 0/4
```

このとき、関数の仕様を満たすように、次の問いに答えなさい。

### 03 特徴量kに関する条件付き確率（分子）
特徴量$$k$$に関する条件付き確率の式の分子を求めるコードを書きなさい。得られた値を`num`とすること。

★★★
1. `ndarray.shape`を使う。
2. ブール値インデキシングを使う。

### 04 特徴量kに関する条件付き確率（分母）
特徴量$$k$$に関する条件付き確率の式の分母を求めるコードを書きなさい。得られた値を`den`とすること。

★
1. `ndarray.shape`を使う。

## 嗜好予測
次男関数は、アイテム`i`の評価値が`r`となる確率を返す関数である。

関数
```python
def P(i, r):
    """
    アイテムiの評価値がrとなる確率を返す。

    Parameters
    ----------
    i : int
        アイテムiのインデックス
    r : int
        評価値

    Returns
    -------
    Fraction
        事前確率
    list of Fraction
        各特徴量に関する条件付き確率
    float
        好き嫌いの確率
    """
    pp = P_prior(r)
    pk = [P_cond(i, k, r) for k in range(0, x.shape[1])]
    【問題05】
    return pp, pk, prob
```

コード
```python
# 好き嫌いの確率
i = 10
r = +1
pp, pk, prob = P(i, r)
left = 'P(R={:+}|'.format(r) + ','.join(map(str, map(int, x[i]))) + ')'
right = str(pp) + '×' + '×'.join(map(str, pk))
print('{} = {} = {:.3f}'.format(left, right, prob))

r = -1
pp, pk, prob = P(i, r)
left = 'P(R={:+}|'.format(r) + ','.join(map(str, map(int, x[i]))) + ')'
right = str(pp) + '×' + '×'.join(map(str, pk))
print('{} = {} = {:.3f}'.format(left, right, prob))
```

結果
```bash
P(R=+1|1,1,0,1,1,0) = 6/10×4/6×3/6×6/6×2/6×4/6×4/6 = 0.030
P(R=-1|1,1,0,1,1,0) = 4/10×0/4×1/4×1/4×1/4×3/4×2/4 = 0.000
```

### 05 好き嫌いの確率
`pk`と`pk`を使って好き嫌いの確率を求めるコードを書きなさい。得られた値を`prob`とすること。

1. `numpy.prod()`を使う。
2. `float()`を使う。

## ラプラススムージング
ラプラススムージングを適用すると、事前確率はそれぞれ次式のように表される。

$$
P(R = +1) = \frac{\mid D^{L+}_{u} \mid + \alpha}{\mid D^{L}_{u} \mid + \alpha n}
$$

$$
P(R = -1) = \frac{\mid D^{L-}_{u} \mid + \alpha}{\mid D^{L}_{u} \mid + \alpha n}
$$

ここで、$$\alpha$$はスムージングパラメタであり、$$n$$はクラス数である。同様に、特徴量$$k$$に関する条件つ確率はそれぞれ次式のように表される。

$$
P(X_{k} = x_{i,k} \mid R = +1) = \frac{\mid D^{L+}_{u}(x_{i,k}) \mid + \alpha}{\mid D^{L+}_{u} \mid + \alpha m_{k}}
$$

$$
P(X_{k} = x_{i,k} \mid R = -1) = \frac{\mid D^{L-}_{u}(x_{i,k}) \mid + \alpha}{\mid D^{L-}_{u} \mid + \alpha m_{k}}
$$

ここで、$$m_{k}$$は特徴量$$k$$がとりうる値のユニーク数である。

### 05 評価値がrとなる事前確率（分子）（ラプラススムージングあり）
ラプラススムージングを適用したとき、事前確率の式の分子を求めるコードを書きなさい。得られた値を`num`とすること。

★
1. `ndarray.shape`を使う。
2. 行列のブールインデックス参照を使う。

### 06 評価値がrとなる事前確率（分母）（ラプラススムージングあり）
ラプラススムージングを適用したとき、事前確率の式の分母を求めるコードを書きなさい。得られた値を`den`とすること。

★
1. `ndarray.shape`を使う。

### 07 特徴量kに関する条件付き確率（分子）
ラプラススムージングを適用したとき、特徴量kに関する条件付き確率の式の分子を求めるコードを書きなさい。得られた値を`num`とすること。

★★★
1. `ndarray.shape`を使う。
2. ブールインデックス参照を使う。

### 08 特徴量kに関する条件付き確率（分母）
ラプラススムージングを適用したとき、特徴量kに関する条件付き確率の式の分母を求めるコードを書きなさい。得られた値を`den`とすること。

★
1. `ndarray.shape`を使う。

## 推薦
スコア関数は次式のように定義される。

$$
\mathrm{score}(u, i) = \frac{P(R = +1) \prod_{k=1}^{d} P(X_{k} = x_{i,k} \mid R = +1)}{P(R = +1) \prod_{k=1}^{d} P(X_{k} = x_{i,k} \mid R = +1) + P(R = -1) \prod_{k=1}^{d} P(X_{k} = x_{i,k} \mid R = -1)}
$$

次の関数は、ユーザ`u`のアイテム`i`に対するスコアを返す関数である。

関数
```python
def score(u, i):
    """
    スコア関数：ユーザuのアイテムiに対するスコアを返す。

    Parameters
    ----------
    u : int
        ユーザuのインデックス（ダミー）
    i : int
        アイテムiのインデックス

    Returns
    -------
    float
        スコア
    """
    【問題09】
    return scr
```

コード
```python
# 各対象アイテムのスコア
u = 0
scores = {i: score(u, i) for i in IU}
print('scores = ')
for i, scr in scores.items():
    print('{}: {:.3f}'.format(i, scr))
```

結果
```bash
scores = 
10: 0.965
11: 0.055
12: 0.189
```

### 09
ユーザ$$u$$のアイテム$$i$$に対するスコアを求めるコードを書きなさい。得られた値を`scr`とすること。

★★
1. `P()`関数を呼ぶ。


