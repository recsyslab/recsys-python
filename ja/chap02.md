<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# 第2章 評価値行列

## 準備
次のコードを書きなさい。

```python
import numpy as np
```

## 評価値行列

次の行列$$\boldsymbol{R}$$は評価値行列である。

$$
\boldsymbol{R} = \left[
 \begin{array}{rrrrrr}
  ? & 4 & 3 & 1 & 2 & ? \\
  5 & 5 & 4 & ? & 3 & 3 \\
  4 & ? & 5 & 3 & 2 & ? \\
  ? & 3 & ? & 2 & 1 & 1 \\
  2 & 1 & 2 & 4 & ? & 3 \\
 \end{array}
\right]
$$

$$\boldsymbol{R}$$の$$u$$行目はユーザ$$u \in U$$を表し、$$i$$列目はアイテム$$i \in I$$を表す。ここで、$$\boldsymbol{R}$$におけるユーザ数は$$\mid U \mid = 5$、アイテム数は$$\mid I \mid = 6$$となる。$$\boldsymbol{R}$$の$$(u, i)$$成分はユーザ$$u$$がアイテム$$i$$に与えた評価値$$r_{u,i}$$を表す。ただし、$$?$$は欠損値であることを表す。このとき、次の問いに答えなさい。

### 01 評価値行列の生成
$$\boldsymbol{R}$$を`ndarray`として生成するコードを書きなさい。得られた`ndarray`を`R`とすること。

コード
```python
# 評価値行列
【    問01    】
print('R = \n{}'.format(R))
```

結果
```bash
R = 
[[nan  4.  3.  1.  2. nan]
 [ 5.  5.  4. nan  3.  3.]
 [ 4. nan  5.  3.  2. nan]
 [nan  3. nan  2.  1.  1.]
 [ 2.  1.  2.  4. nan  3.]]
```

★
1. `numpy.nan`を使う。
2. `numpy.array()`を使う。

### 02 ユーザ集合
`R`の各行のインデックスは各ユーザ$$u$$のユーザIDに対応する。ユーザ集合$$U$$（ユーザIDを要素としたベクトル）を`ndarray`として生成するコードを書きなさい。得られた`ndarray`を`U`とすること。

コード
```python
# ユーザ集合
【    問02    】
print('U = {}'.format(U))
```

結果
```bash
U = [0 1 2 3 4]
```

★
1. `numpy.arange()`を使う。
2. `ndarray.shape`を使う。

### 03 アイテム集合
`R`の各列のインデックスは各アイテム$$i$$のアイテムIDに対応する。アイテム集合$$I$$（アイテムIDを要素としたベクトル）を`ndarray`として生成するコードを書きなさい。得られた`ndarray`を`I`とすること。

コード
```python
# ユーザ集合
【    問02    】
print('I = {}'.format(I))
```

結果
```bash
I = [0 1 2 3 4 5]
```

★
1. `numpy.arange()`を使う。
2. `ndarray.shape`を使う。

### 04 ユーザ数
ユーザ数$$\mid U \mid$$を取得するコードを書きなさい。

結果
```bash
|U| = 5
```

★
1. `ndarray.size`を使う。

### 05 アイテム数
アイテム数$$\mid I \mid$$を取得するコードを書きなさい。

結果
```bash
|I| = 6
```

★
1. `ndarray.size`を使う。

### 06 評価値
`R`からユーザ$$u$$のアイテム$$i$$に対する評価値$$r_{u,i}$$を取得するコードを書きなさい。得られた値を$$rui$$とすること。

コード
```python
# ユーザuのアイテムiに対する評価値
u = 0
i = 1
【    問06    】
print('r{}{} = {}'.format(u, i, rui))
```

結果
```bash
r01 = 4.0
```
★
1. インデキシングを使う。
