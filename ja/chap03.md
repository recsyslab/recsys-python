<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# 第3章 内容ベース推薦システム | 類似度に基づく推薦

### 準備
次のコードを書きなさい。

```python
import numpy as np

# 上位K件
TOP_K = 3

Du = np.array([
            [5, 3, +1],
            [6, 2, +1],
            [4, 1, +1],
            [8, 5, -1],
            [2, 4, -1],
            [3, 6, -1],
            [7, 6, -1],
            [4, 2, np.nan],
            [5, 1, np.nan],
            [8, 6, np.nan],
            [3, 4, np.nan],
            [4, 7, np.nan],
            [4, 4, np.nan],
])
I = np.arange(Du.shape[0])
x = Du[:,:-1]
ru = Du[:,-1]

Iu = I[~np.isnan(ru)]
Iup = I[ru==+1]
Iun = I[ru==-1]
Iu_not = np.setdiff1d(I, Iu)
```

## ユーザプロファイル
ユーザ$$u$$のユーザプロファイル$$\boldsymbol{p}_{u}$$は次式で求められる。

$$
\boldsymbol{p}_{u} = \frac{1}{\mid I_{u}^{+} \mid} \sum_{i \in I_{u}^{+}} \boldsymbol{x}_{i}
$$

ここで、$$I_{u}^{+}$$は対象ユーザ$$u$$が「好き」と評価したアイテム集合であり、$$\boldsymbol{x}_{i}$$はアイテム$i$の特徴ベクトルである。

### 01 好きなアイテム集合に含まれるアイテムの特徴ベクトルの集合
`x`から$$I_{u}^{+}$$に含まれるアイテムの特徴ベクトルの集合を`ndarray`として生成するコードを書きなさい。

コード
```python
print('x[Iu+] = \n{}'.format(【    問01    】))
```

結果
```bash
x[Iu+] = 
[[5. 3.]
 [6. 2.]
 [4. 1.]]
```

★
1. 整数配列インデキシングを使う。

★★
1. リスト内包表記を使う。

### 02 特徴ベクトルの総和
次式により、$$I_{u}^{+}$$に含まれるアイテムの特徴ベクトルの総和を求めるコードを書きなさい。

$$
\sum_{i \in I_{u}^{+}} \boldsymbol{x}_{i}
$$

コード
```python
print('sum(x[Iu+]) = {}'.format(【    問02    】))
```

結果
```bash
sum(x[Iu+]) = [15.  6.]
```

★★
1. 整数配列インデキシングを使う。
2. `numpy.sum()`を使う。

★★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。

### 03 ユーザプロファイル
ユーザ$$u$$のユーザプロファイル$$\boldsymbol{p}_{u}$$を`ndarray`として求めるコードを書きなさい。得られた`ndarray`を`pu`とすること。

コード
```python
【    問03    】
print('pu = {}'.format(pu))
```

結果
```bash
pu = [5. 2.]
```

★★
1. 整数配列インデキシングを使う。
2. `numpy.sum()`を使う。

★★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。

## コサイン類似度

ユーザプロファイル$$\boldsymbol{p}_{u}$$とアイテム$$i$$の特徴ベクトル$$\boldsymbol{x}_{i}$$のコサイン類似度は次式で定義される。

$$
\mathrm{cos}(\boldsymbol{p}_{u}, \boldsymbol{x}_{i}) = \frac{\boldsymbol{p}_{u} \cdot \boldsymbol{x}_{i}}{\| \boldsymbol{p}_{u} \| \| \boldsymbol{x}_{i} \|}
$$

ここで、$$\boldsymbol{p}_{u} \cdot \boldsymbol{x}_{i}$$は二つのベクトル$$\boldsymbol{p}_{u}$$と$$\boldsymbol{x}_{i}$$の内積であり、次式のように表される。

$$
\boldsymbol{p}_{u} \cdot \boldsymbol{x}_{i} = \sum_{k=1}^{d} p_{u,k} x_{i,k}
$$

$$d$$はベクトルの次元数である。また、$$\| \boldsymbol{p}_{u} \|$$はベクトル$$\boldsymbol{p}_{u}$$のノルム（大きさ）であり、次式のように表される。

$$
\| \boldsymbol{p}_{u} \| = \sqrt{\boldsymbol{p}_{u} \cdot \boldsymbol{p}_{u}} = \sqrt{\sum_{k=1}^{d} p_{u,k}^{2}}
$$

このコサイン類似度関数を次のコードのとおり定義する。

関数
```python
def cos(pu, xi):
    """
    コサイン類似度関数：ユーザプロファイルpuとアイテムiの特徴ベクトルxiのコサイン類似度を返す。

    Parameters
    ----------
    pu : ndarray
        ユーザuのユーザプロファイル
    xi : ndarray
        アイテムiの特徴ベクトル

    Returns
    -------
    float
        コサイン類似度
    """
    【    問04    】
    print('num = {}'.format(num))
    【    問05    】
    print('den_u = {:.3f}'.format(den_u))
    【    問06    】
    den_i = np.linalg.norm(xi)
    
    cosine = num / (den_u * den_i)
    return cosine
```

コード
```python
u = 0
i = 7
print('cos(p{}, x{}) = {:.3f}'.format(u, i, cos(pu, x[i])))
u = 0
i = 11
print('cos(p{}, x{}) = {:.3f}'.format(u, i, cos(pu, x[i])))
```

結果
```bash
num = 24.0
den_u = 5.385
den_i = 4.472
cos(p0, x7) = 0.997
num = 34.0
den_u = 5.385
den_i = 8.062
cos(p0, x11) = 0.783
```

このとき、関数の仕様を満たすように、次の問いに答えなさい。

### 04 ベクトルの内積
内積$$\boldsymbol{p}_{u} \cdot \boldsymbol{x}_{i}$$を求めるコードを書きなさい。得られた値を`num`とすること。

★
1. `@`演算子を使う。

★
1. `numpy.dot()`を使う。

★★★
1. リスト内包表記を使う。
2. `range()`を使う。
3. `numpy.sum()`を使う。

★★★
1. `numpy.sum()`を使う。
2. `@`演算子を使わない。
3. `numpy.dot()`を使わない。
4. リスト内包表記を使わない。

### 05 ユーザプロファイルのノルム
$$\boldsymbol{p}_{u}$$のノルム$$\| \boldsymbol{p}_{u} \|$$を求めるコードを書きなさい。得られた値を`den_u`とすること。

★
1. `numpy.linalg.norm()`を使う。

★★
1. `@`演算子を使う。
2. `numpy.sqrt()`を使う。

★★★
1. リスト内包表記を使う。
2. `range()`を使う。
3. `numpy.sum()`を使う。
4. `numpy.sqrt()`を使う。

★★★
1. `numpy.sum()`を使う。
2. `numpy.sqrt()`を使う。
3. `numpy.linalg.norm()`を使わない。
4. `@`演算子を使わない。
5. `numpy.dot()`を使わない。
6. リスト内包表記を使わない。

### 06 特徴ベクトルのノルム
$$\boldsymbol{x}_{i}$$のノルム$$\| \boldsymbol{x}_{i} \|$$を求めるコードを書きなさい。得られた値を`den_i`とすること。

★
1. `numpy.linalg.norm()`を使う。

## 推薦

スコア関数$$\mathrm{score}(u, i)$$はユーザ$$u$$がアイテム$$i$$を好む程度をスコアとして返す関数であり、次式のように定義される。

$$
\mathrm{score}(u, i) = \mathrm{cos}(\boldsymbol{p}_{u}, \boldsymbol{x}_{i})
$$

このスコア関数を次のコードのとおり定義する。

関数
```
def score(u, i):
    """
    スコア関数：ユーザuのアイテムiに対するスコアを返す。

    Parameters
    ----------
    u : int
        ユーザuのID（ダミー）
    i : int
        アイテムiのID

    Returns
    -------
    float
        スコア
    """
    return cos(pu, x[i])
```

順序付け関数$$\mathrm{order}(u, I)$$は、アイテム集合$$I$$が与えられたとき、ユーザ$$u$$向けの推薦リストを返す関数である。ここでは、スコア上位$$K$$件のアイテム集合を推薦リストとして返すものとする。この順序付け関数を次のコードのとおり定義する。

```python
def order(u, I):
    """
    順序付け関数：アイテム集合Iにおいて、ユーザu向けの推薦リストを返す。

    Parameters
    ----------
    u : int
        ユーザuのID
    I : ndarray
        アイテム集合

    Returns
    -------
    dict
        (アイテムID: スコア)をペアにした辞書型の推薦リスト
    """
    【    問07    】
    print('scores = ', end='')
    for i, scr in scores.items():
        print('{}: {:.3f}'.format(i, scr), end=', ')
    print()
    【    問08    】
    return rec_list
```

コード
```python
u = 0
rec_list = order(u, Iu_not)
print('rec_list = ', end='')
for i, scr in rec_list.items():
    print('{}: {:.3f}'.format(i, scr), end=', ')
print()
```

結果
```bash
scores = 7: 0.997, 8: 0.983, 9: 0.966, 10: 0.854, 11: 0.783, 12: 0.919, 
rec_list = 7: 0.997, 8: 0.983, 9: 0.966, 
```

このとき、関数の仕様を満たすように、次の問いに答えなさい。

### 07 各アイテムに対するスコア
引数に渡されたアイテム集合$$I$$について、ユーザ$$u$$の各アイテム$$i \in I$$に対するスコア$$\mathrm{score}(u, i)$$を求め、`(i: score(u, i))`をペアとした辞書を生成するコードを書きなさい。生成した辞書を`scores`とすること。

★★
1. `for`文を使う。

★★★
1. 辞書内包表記を使う。

### 08 推薦リスト
`scores`内の`(i: score(u, i))`のペアを`score(u, i)`の降順にソートし、上位`TOP_K`件のリストを生成するコードを書きなさい。得られたリストを辞書に変換したものを`rec_list`とすること。

★★★
1. `sorted()`を使う。
2. `dict.items()`を使う。
3. `lambda`式を使う。
4. スライシングを使う。
5. `dict()`を使う。
