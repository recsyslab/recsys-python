<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# 第3章 内容ベース推薦システム | k近傍法

## 準備
次のコードを入力しなさい。

```python
import numpy as np

# 上位K件
TOP_K = 3
# 近傍アイテム数
K_ITEMS = 3

# 評価履歴
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

# ユーザuが評価済みのアイテム集合
Iu = I[~np.isnan(ru)]
print('Iu = {}'.format(Iu))
# ユーザuが「好き」と評価したアイテム集合
Iup = I[ru==+1]
print('Iu+ = {}'.format(Iup))
# ユーザuが「嫌い」と評価したアイテム集合
Iun = I[ru==-1]
print('Iu- = {}'.format(Iun))
print()

# ユーザuが未評価のアイテム集合
Iu_not = np.setdiff1d(I, Iu)
print('Iu_not = {}'.format(Iu_not))
print()
```

## 距離
アイテム$$i$$の特徴ベクトル$$\boldsymbol{x}_{i}$とアイテム$$j$$の特徴ベクトル$$\boldsymbol{x}_{j}$$のユークリッド距離は次式で定義される。

$$
\mathrm{dist}(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}) = \sqrt{\sum_{k=1}^{d} (x_{j,k} - x_{i,k})^{2}}
$$

ここで、$$d$$はベクトルの次元数である。

### 01 距離関数の定義 | 関数
関数の仕様を満たすように、次のコードの【ToDo】の箇所を埋めて距離関数を完成させなさい。呼出し側のコードを実行したとき、実行結果のとおりの結果が出力されること。

```python
def dist(xi, xj):
    """
    距離関数：アイテムiの特徴ベクトルxiとアイテムjの特徴ベクトルxjのユークリッド距離を算出する。

    Parameters
    ----------
    xi : ndarray
        アイテムiの特徴ベクトル
    xj : ndarray
        アイテムjの特徴ベクトル

    Returns
    -------
    distance : float
        ユークリッド距離
    """
    【ToDo】
    return distance
```

呼出し側
```python
i = 7
j = 2
print('dist(x{}, x{}) = {:.3f}'.format(i, j, dist(x[i], x[j])))
```

実行結果
```
dist(x7, x2) = 1.000
```

★★★
1. `ndarray.size`を使う。
2. リスト内包表記を使う。
3. `range()`を使う。
4. `np.sum()`を使う。
5. `np.sqrt()`を使う。

★★★
1. `ndarray.size`を使わない。
2. リスト内包表記を使わない。
3. `range()`を使わない。
4. `np.sum()`を使う。
5. `np.sqrt()`を使う。

## 近傍アイテム

### 02 対象アイテム$$i$$と各アイテムとのユークリッド距離 | 辞書内包表記
対象アイテム7について、各アイテム$$j \in I_{u}$$とのユークリッド距離のペア`j: dist(x[7], x[j])`を要素とした辞書を作成しなさい。作成した辞書を`dists`とすること。確認コードを実行したとき、実行結果のとおりの結果が出力されること。


★★
1. 辞書内包表記を使う。

確認コード
```python
print('dists = ', end='')
for j, d in dists.items():
    print('{}: {:.3f}'.format(j, d), end=', ')
print()
```

実行結果
```python
dists = 0: 1.414, 1: 2.000, 2: 1.000, 3: 5.000, 4: 2.828, 5: 4.123, 6: 5.000,
```

### 03 各対象アイテム$$i$$と各アイテムとのユークリッド距離 | 辞書のネスト
課題02で作成した辞書を、各対象アイテム$$i \in \overline{I}_{u}$$について作成し、`i: dists`を要素とした辞書を作成しなさい。作成した辞書を`dists_dict`とすること。確認コードを実行したとき、実行結果のとおりの結果が出力されること。

★★★
1. `for`文を使う。
2. 辞書内表表記を使う。
3. 辞書のネストを使う。

確認コード
```python
for i in Iu_not:
    print('dists_dict[{}] = '.format(i), end='')
    for j, d in dists_dict[i].items():
        print('{}: {:.3f}'.format(j, d), end=', ')
    print()
print()
```

実行結果
```python
dists_dict[7] = 0: 1.414, 1: 2.000, 2: 1.000, 3: 5.000, 4: 2.828, 5: 4.123, 6: 5.000, 
dists_dict[8] = 0: 2.000, 1: 1.414, 2: 1.000, 3: 5.000, 4: 4.243, 5: 5.385, 6: 5.385, 
dists_dict[9] = 0: 4.243, 1: 4.472, 2: 6.403, 3: 1.000, 4: 6.325, 5: 5.000, 6: 1.000, 
dists_dict[10] = 0: 2.236, 1: 3.606, 2: 3.162, 3: 5.099, 4: 1.000, 5: 2.000, 6: 4.472, 
dists_dict[11] = 0: 4.123, 1: 5.385, 2: 6.000, 3: 4.472, 4: 3.606, 5: 1.414, 6: 3.162, 
dists_dict[12] = 0: 1.414, 1: 2.828, 2: 3.000, 3: 4.123, 4: 2.000, 5: 2.236, 6: 3.606, 
```

### 04 対象アイテム$$i$$の近傍アイテム集合 | 辞書のソート
対象アイテム$$i$$の近傍アイテム集合$$I_{i}$$を辞書として取得しなさい。取得した辞書を`Ii`とすること。

★★★
1. `sorted()`を使う。
2. `dict.items()`を使う。
3. `key`を指定する。
4. `lambda`式を使う。
5. リストのスライスを使う。
6. `dict()`を使う。

### 05 
課題04で取得した辞書`Ii`に含まれるすべてのキーを取り出し、それを要素としたベクトルとして取得しなさい。取得したベクトルを`Ii`とすること。

★
1. `dict.keys()`を使う。
2. `list()`を使う。

## 推薦

### 06
アイテム$$i$$の近傍アイテム集合$$I_{i}$$のうち、ユーザ$$u$$が「好き」と評価したアイテム集合を取得しなさい。取得したベクトルを`Iip`とすること。

★★
1. `numpy.isin()`を使う。
2. ベクトルのブールインデックス参照を使う。

### 07
アイテム$$i$$の近傍アイテム集合$$I_{i}$$のうち、ユーザ$$u$$が「嫌い」と評価したアイテム集合を取得しなさい。取得したベクトルを`Iin`とすること。

★★
1. `numpy.isin()`を使う。
2. ベクトルのブールインデックス参照を使う。

### 08

$$
\hat{r}_{u,i} = 
 \begin{cases}
  +1 & (\mid I_{i}^{+} \mid > \mid I_{i}^{-} \mid) \\
  -1 & (\mid I_{i}^{+} \mid < \mid I_{i}^{-} \mid) \\
  0 & (\mid I_{i}^{+} \mid = \mid I_{i}^{-} \mid)
 \end{cases}
$$

