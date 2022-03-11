<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# 第4章 内容ベース推薦システム | k近傍法

### 準備
次のコードを書きなさい。

```python
import numpy as np

# 上位K件
TOP_K = 3
# 近傍アイテム数
K_ITEMS = 3

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

## 距離
アイテム$$i$$の特徴ベクトル$$\boldsymbol{x}_{i}$$とアイテム$$j$$の特徴ベクトル$$\boldsymbol{x}_{j}$$のユークリッド距離は次式で定義される。

$$
\mathrm{dist}(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}) = \sqrt{\sum_{k=1}^{d} (x_{j,k} - x_{i,k})^{2}}
$$

ここで、$$d$$はベクトルの次元数である。

この距離関数を次のコードのとおり定義する。

関数
```python
def dist(xi, xj):
    """
    距離関数：アイテムiの特徴ベクトルxiとアイテムjの特徴ベクトルxjのユークリッド距離を返す。

    Parameters
    ----------
    xi : ndarray
        アイテムiの特徴ベクトル
    xj : ndarray
        アイテムjの特徴ベクトル

    Returns
    -------
    float
        ユークリッド距離
    """
    【    問01    】
    return distance
```

コード
```python
i = 7
j = 2
print('dist(x{}, x{}) = {:.3f}'.format(i, j, dist(x[i], x[j])))
i = 7
j = 3
print('dist(x{}, x{}) = {:.3f}'.format(i, j, dist(x[i], x[j])))
```

結果
```bash
dist(x7, x2) = 1.000
dist(x7, x3) = 5.000
```

このとき、関数の仕様を満たすように、次の問いに答えなさい。

### 01 ユークリッド距離
$$\boldsymbol{x}_{i}$$と$$\boldsymbol{x}_{j}$$のユークリッド距離\mathrm{dist}(\boldsymbol{x}_{i}, \boldsymbol{x}_{j})を求めるコードを書きなさい。得られた値を`distance`とすること。

★★★
1. `ndarray.size`を使う。
2. リスト内包表記を使う。
3. `range()`を使う。
4. `numpy.sum()`を使う。
5. `numpy.sqrt()`を使う。

★★★
1. `numpy.sum()`を使う。
2. `numpy.sqrt()`を使う。
4. リスト内包表記を使わない。

## 近傍アイテム
ユーザ$$u$$の未評価アイテム集合$$\overline{I}_{u}$$の各対象アイテム$$i \in \overline{I}_{u}$$について、ユーザ$$u$$の評価済みアイテム集合$$I_{u}$の中から近傍アイテム集合$$I_{i}$$を見つける。

### 02 対象アイテムiと各アイテムとのユークリッド距離
対象アイテム$$i$$と各アイテム$$j \in I_{u}$$のユークリッド距離$$\mathrm{dist}(\boldsymbol{x}_{i}, \boldsymbol{x}_{j})$$を求め、`(j: dist(x[i], x[j]))`をペアとした辞書を生成するコードを書きなさい。生成した辞書を`dists`とすること。

コード
```python
i = 7
【    問02    】
print('dists = ', end='')
for j, d in dists.items():
    print('{}: {:.3f}'.format(j, d), end=', ')
print()
```

結果
```bash
dists = 0: 1.414, 1: 2.000, 2: 1.000, 3: 5.000, 4: 2.828, 5: 4.123, 6: 5.000,
```

★★
1. `for`文を使う。

★★★
1. 辞書内包表記を使う。

### 03 各対象アイテムと各アイテムとのユークリッド距離
各対象アイテム$$i \in \overline{I}_{u}$$と各アイテム$$j \in I_{u}$$のユークリッド距離$$\mathrm{dist}(\boldsymbol{x}_{i}, \boldsymbol{x}_{j})$$を求め、`(i: (j: dist(x[i], x[j]))`をネストされたペアとした辞書を生成するコードを書きなさい。生成した辞書を`dists`とすること。

コード
```python
【    問03    】
for i in Iu_not:
    print('dists[{}] = '.format(i), end='')
    for j, d in dists[i].items():
        print('{}: {:.3f}'.format(j, d), end=', ')
    print()
print()
```

結果
```bash
dists[7] = 0: 1.414, 1: 2.000, 2: 1.000, 3: 5.000, 4: 2.828, 5: 4.123, 6: 5.000, 
dists[8] = 0: 2.000, 1: 1.414, 2: 1.000, 3: 5.000, 4: 4.243, 5: 5.385, 6: 5.385, 
dists[9] = 0: 4.243, 1: 4.472, 2: 6.403, 3: 1.000, 4: 6.325, 5: 5.000, 6: 1.000, 
dists[10] = 0: 2.236, 1: 3.606, 2: 3.162, 3: 5.099, 4: 1.000, 5: 2.000, 6: 4.472, 
dists[11] = 0: 4.123, 1: 5.385, 2: 6.000, 3: 4.472, 4: 3.606, 5: 1.414, 6: 3.162, 
dists[12] = 0: 1.414, 1: 2.828, 2: 3.000, 3: 4.123, 4: 2.000, 5: 2.236, 6: 3.606, 
```

★★★
1. 二重の`for`文を使う。

★★★
1. `for`文を使う。
2. 辞書内包表記を使う。

★★★
1. 二重の辞書内包表記を使う。

### 04 各対象アイテムの近傍アイテム集合
各対象アイテム$$i \in \overline{I}_{u}$$の近傍アイテム集合$$I_{i}$$を`ndarray`のリストとしてまとめて生成するコードを書きなさい。得られたリストを`Ii`とすること。

コード
```python
【    問04    】
print('Ii = {}'.format(Ii))
```

結果
```bash
Ii = {7: array([2, 0, 1]), 8: array([2, 1, 0]), 9: array([3, 6, 0]), 10: array([4, 5, 0]), 11: array([5, 6, 4]), 12: array([0, 4, 5])}
```

★★★
1. `for`文を使う。
2. `sorted()`を使う。
3. `dict.items()`を使う。
4. `lambda`式を使う。
5. スライシングを使う。
6. `dict()`を使う。
7. `dict.keys()`を使う。
8. `list()`を使う。
9. `numpy.array()`を使う。

★★★
1. 辞書内包表記を使う。
2. `sorted()`を使う。
3. `dict.items()`を使う。
4. `lambda`式を使う。
5. スライシングを使う。
6. `dict()`を使う。
7. `dict.keys()`を使う。
8. `list()`を使う。
9. `numpy.array()`を使う。

## 嗜好予測
ユーザ$$u$$のアイテム$$i$$への予測評価値$$\hat{r}_{u,i}$$は次式により定義される。

### (a)多数決方式

$$
\hat{r}_{u,i} = 
 \begin{cases}
  +1 & (\mid I_{i}^{+} \mid > \mid I_{i}^{-} \mid) \\
  -1 & (\mid I_{i}^{+} \mid < \mid I_{i}^{-} \mid) \\
  0 & (\mid I_{i}^{+} \mid = \mid I_{i}^{-} \mid)
 \end{cases}
$$

### (b) 平均方式

$$
\hat{r}_{u,i} = \frac{1}{k} \sum_{j \in I_{i}} r_{u,j}
$$

$$\mathrm{score}(u, i)$$は対象ユーザ$$u$$が対象アイテム$$i$$を好む程度を表すスコア関数であり、次式のように定義される。

$$
\mathrm{score}(u, i) = \hat{r}_{u,i}
$$

このスコア関数`score(u, i)`を次のコードのとおり定義する。
```
def score(u, i):
    """
    スコア関数：ユーザuのアイテムiに対するスコアを算出する。

    Parameters
    ----------
    u : int
        ユーザuのインデックス
    i : int
        アイテムiのインデックス

    Returns
    -------
    scr : float
        スコア
    """
    return predict2(u, i)
```

### 05 近傍アイテム集合のうち対象ユーザが「好き」と評価したアイテム集合 | ベクトルのブールインデックス参照
対象アイテム9の近傍アイテム集合$$I_{9}$$のうち、ユーザ$$u$$が「好き」と評価したアイテム集合$$I_{i}^{+}$$を`ndarray`のベクトル`Iip`として取得しなさい。

★★
1. `numpy.isin()`を使う。
2. ベクトルのブールインデックス参照を使う。

### 06 近傍アイテム集合のうち対象ユーザが「嫌い」と評価したアイテム集合 | ベクトルのブールインデックス参照
対象アイテム9の近傍アイテム集合$$I_{9}$$のうち、ユーザ$$u$$が「嫌い」と評価したアイテム集合$$I_{i}^{-}$$を`ndarray`のベクトル`Iin`として取得しなさい。

★★
1. `numpy.isin()`を使う。
2. ベクトルのブールインデックス参照を使う。

### 07 多数決方式による予測評価値 | 条件分岐
多数決方式により、対象ユーザ$$u$$の対象アイテム$$i$$に対する予測評価値$$r_{u,i}$$を求めなさい。

★
1. `if`文を使う。
2. `ndarray.size`を使う。

### 08 予測関数（多数決方式）の定義 | 関数
関数の仕様を満たすように、次のコードの【ToDo】の箇所を埋めて予測関数（多数決方式）`predict1(u, i)`を完成させなさい。確認コードを実行したとき、実行結果のとおりの結果が出力されること。

```
def predict1(u, i):
    """
    予測関数：ユーザuのアイテムiに対する評価値を予測する（多数決方式）。

    Parameters
    ----------
    u : int
        ユーザuのインデックス（ダミー）
    i : int
        アイテムiのインデックス

    Returns
    -------
    rui : float
        評価値
    """
    
    【ToDo】
    
    return rui
```

確認コード
```python
u = 0
i = 9
print('predict1({}, {}) = {:.3f}'.format(u, i, predict1(u, i)))
```

### 09 予測関数（平均方式）の定義 | 関数
関数の仕様を満たすように、次のコードの【ToDo】の箇所を埋めて予測関数（平均方式）`predict1(u, i)`を完成させなさい。確認コードを実行したとき、実行結果のとおりの結果が出力されること。

```
def predict2(u, i):
    """
    予測関数：ユーザuのアイテムiに対する評価値を予測する（平均方式）。

    Parameters
    ----------
    u : int
        ユーザuのインデックス（ダミー）
    i : int
        アイテムiのインデックス

    Returns
    -------
    rui : float
        評価値
    """
    
    【ToDo】
    
    return rui
```

確認コード
```python
u = 0
i = 9
print('predict2({}, {}) = {:.3f}'.format(u, i, predict1(u, i)))
```

実行結果
```python
predict2(0, 9) = -0.333
```

★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。

### 10 順序付け関数の定義 | 関数
関数の仕様を満たすように、次のコードの【ToDo】の箇所を埋めて順序付け類似度関数`order(u, I)`を完成させなさい。ただし、`score(u, i) < 0`となるアイテム$$i$$は推薦リストから除外すること。確認コードを実行したとき、実行結果のとおりの結果が出力されること。

```python
def order(u, I):
    """
    順序付け関数：アイテム集合Iにおいて、ユーザu向けの推薦リストを返す。

    Parameters
    ----------
    u : int
        ユーザuのインデックス
    I : ndarray
        アイテム集合

    Returns
    -------
    rec_list: dict
        推薦リスト
    """
    
    【ToDo】
    
    return rec_list
```

確認コード
```python
u = 0
rec_list = order(u, Iu_not)
print('rec_list = ')
for i, scr in rec_list.items():
    print('{}: {:.3f}'.format(i, scr))
```

実行結果
```python
rec_list = 
7: 1.000
8: 1.000
```

★★★
1. 辞書内包表記を使う。
2. 辞書内包表記内で`if`節を使う。
3. `sorted()`を使う。
4. `dict.items()`を使う。
5. `key`を指定する。
6. `reverse`を指定する。
7. `lambda`式を使う。
8. リストのスライスを使う。
9. `dict()`を使う。
