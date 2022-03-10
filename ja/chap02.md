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
評価値行列$$\boldsymbol{R}$$を`ndarray`として生成するコードを書きなさい。得られた`ndarray`を`R`とすること。

コード
```python
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
`R`の各行のインデックス`u`は各ユーザ$$u$$のユーザIDに対応する。ユーザ集合$$U$$（ユーザIDを要素としたベクトル）を`ndarray`として生成するコードを書きなさい。得られた`ndarray`を`U`とすること。

コード
```python
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
`R`の各列のインデックス`i`は各アイテム$$i$$のアイテムIDに対応する。アイテム集合$$I$$（アイテムIDを要素としたベクトル）を`ndarray`として生成するコードを書きなさい。得られた`ndarray`を`I`とすること。

コード
```python
【    問03    】
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
`R`からユーザ数$$\mid U \mid$$を取得するコードを書きなさい。

コード
```python
print('|U| = {}'.format(【    問04    】))
```

結果
```bash
|U| = 5
```

★
1. `ndarray.size`を使う。

### 05 アイテム数
`R`からアイテム数$$\mid I \mid$$を取得するコードを書きなさい。

コード
```python
print('|I| = {}'.format(【    問05    】))
```

結果
```bash
|I| = 6
```

★
1. `ndarray.size`を使う。

### 06 評価値
`R`からユーザ$$u$$のアイテム$$i$$に対する評価値$$r_{u,i}$$を取得するコードを書きなさい。

コード
```python
u = 0
i = 1
print('r{}{} = {}'.format(u, i, 【    問06    】))
```

結果
```bash
r01 = 4.0
```

★
1. インデキシングを使う。

## 評価値行列の疎性
評価値行列$$\boldsymbol{R}$$の疎性$$\mathrm{sparsity}$$は次式で求められる。

$$
\mathrm{sparsity} = 1 - \frac{\mid R \mid}{\mid U \mid \mid I \mid}
$$

ここで、$$\mid R \mid$$は評価値が与えられた成分の数、すなわち観測値数（欠損値でない要素数）を表す。このとき、次の問いに答えなさい。

### 07 評価値行列の全要素数
`R`の全要素数を取得するコードを書きなさい。ただし、欠損値も含む。

コード
```python
print('Rの全要素数 = {}'.format(【    問07    】))
```

結果
```bash
Rの全要素数 = 30
```

★
1. `ndarray.size`を使う。

### 08 観測されているか否かの判定
`R`において、観測値の要素には`True`を、欠損値の要素には`False`を入れたブール値配列を生成するコードを書きなさい。

コード
```python
print('観測値 = \n{}'.format(【    問08    】))
```

結果
```bash
観測値 = 
[[False  True  True  True  True False]
 [ True  True  True False  True  True]
 [ True False  True  True  True False]
 [False  True False  True  True  True]
 [ True  True  True  True False  True]]
```

★
1. `numpy.isnan()`を使う。
2. `~`演算子を使う。

### 09 評価値行列の観測値数
`R`における観測値数$$\mid R \mid$$を取得するコードを書きなさい。

コード
```python
print('|R| = {}'.format(【    問09    】))
```

結果
```bash
|R| = 22
```

★★
1. `numpy.isnan()`を使う。
2. `~`演算子を使う。
3. `numpy.count_nonzero()`を使う。

★★
1. `numpy.isnan()`を使う。
2. `~`演算子を使う。
3. `numpy.size`を使う。
4. ブール値インデキシングを使う。

### 10 評価値行列の疎性
評価値行列$$\boldsymbol{R}$$の疎性$$\mathrm{sparsity}$$を求めるコードを書きなさい。得られた値を`sparsity`とすること。

コード
```python
【    問10    】
print('sparsity = {:.3f}'.format(sparsity))
```

結果
```
sparsity = 0.267
```

★★
1. `numpy.isnan()`を使う。
2. `~`演算子を使う。
3. `numpy.count_nonzero()`を使う。
4. `numpy.size`を使う。

## 評価済みアイテム集合
アイテム集合$$I$$のうちユーザ$$u$$が評価済みのアイテム集合を$$I_{u} \subseteq I$$、ユーザ$$v$$が評価済みのアイテム集合を$$I_{v} \subseteq I$$とすると、ユーザ$$u$$とユーザ$$v$$の共通の評価済みアイテム集合は$$I_{u,v} = I_{u} \cap I_{v}$$と表される。また、ユーザ集合$$U$$のうちアイテム$$i$$を評価済みのユーザ集合を$$U_{i} \subseteq U$$、アイテム$$j$$を評価済みのユーザ集合を$$U_{j} \subseteq U$$とすると、アイテム$$i$$とアイテム$$j$$の両方を評価済みのユーザ集合は$U_{i,j} = U_{i} \cap U_{j}$$と表される。このとき、次の問いに答えなさい。

### 11 ユーザuが評価済みのアイテム集合
`I`からユーザ$$u$$が評価済みのアイテム集合$$I_{u}$$を`ndarray`として生成するコードを書きなさい。

コード
```python
u = 0
print('I{} = {}'.format(u, 【    問11    】))
```

結果
```bash
I0 = [1 2 3 4]
```

★★
1. `numpy.isnan()`を使う。
2. `~`演算子を使う。
3. ブール値インデキシングを使う。

### 12 各ユーザの評価済みアイテム集合
`I`から各ユーザの評価済みのアイテム集合を`ndarray`のリストとしてまとめて生成するコードを書きなさい。得られたリストを`Iu`とすること。

コード
```python
【    問12    】
print('Iu = {}'.format(Iu))
```

結果
```bash
Iu = [array([1, 2, 3, 4]), array([0, 1, 2, 4, 5]), array([0, 2, 3, 4]), array([1, 3, 4, 5]), array([0, 1, 2, 3, 5])]
```

★★★
1. リスト内包表記を使う。
2. `numpy.isnan()`を使う。
3. `~`演算子を使う。
4. ブール値インデキシングを使う。

### 13 ユーザuとユーザvの共通の評価済みアイテム集合
`Iu`からユーザ$$u$$とユーザ$$v$$の共通の評価済みアイテム集合$$I_{u,v}$$を`ndarray`として生成するコードを書きなさい。得られた`ndarray`を`Iuv`とすること。

コード
```python
u = 0
v = 1
【    問12    】
print('I{}{} = {}'.format(u, v, Iuv))
```

結果
```bash
I01 = [1 2 4]
```

★
1. `numpy.intersect1d`を使う。

### 14 アイテムiを評価済みのユーザ集合
`U`からアイテム$$i$$を評価済みのユーザ集合$$U_{i}$$を`ndarray`として生成するコードを書きなさい。

コード
```python
i = 0
print('U{} = {}'.format(i, 【    問14    】))
```

結果
```bash
U0 = [1 2 4]
```

★★
1. `numpy.isnan()`を使う。
2. `~`演算子を使う。
3. ブール値インデキシングを使う。

### 15 各アイテムの評価済みユーザ集合
`U`から各アイテムの評価済みユーザ集合を`ndarray`のリストとしてまとめて生成するコードを書きなさい。得られたリストを`Ui`とすること。

コード
```python
【    問15    】
print('Ui = {}'.format(Ui))
```

結果
```bash
Ui = [array([1, 2, 4]), array([0, 1, 3, 4]), array([0, 1, 2, 4]), array([0, 2, 3, 4]), array([0, 1, 2, 3]), array([1, 3, 4])]
```

★★★
1. リスト内包表記を使う。
2. `numpy.isnan()`を使う。
3. `~`演算子を使う。
4. ブール値インデキシングを使う。

### 16 アイテムiとアイテムjの両方を評価済みのユーザ集合
`Iu`からアイテム$$i$$とアイテム$$j$$の両方を評価済みのユーザ集合$$U_{i,j}$$を`ndarray`として生成するコードを書きなさい。得られた`ndarray`を`Uij`とすること。

コード
```python
i = 0
j = 4
【    問16    】
print('U{}{} = {}'.format(i, j, Uij))
```

結果
```bash
U04 = [1 2]
```

★
1. `numpy.intersect1d`を使う。

## 平均中心化評価値行列

