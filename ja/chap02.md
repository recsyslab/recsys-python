<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# 第2章 類似度に基づく推薦

### 準備
次のコードを入力しなさい。

```python
import numpy as np

# 上位K件
TOP_K = 3

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
print('Iu+ {}= '.format(Iup))
# ユーザuが「嫌い」と評価したアイテム集合
Iun = I[ru==-1]
print('Iu- {}= '.format(Iun))
print()
```

## ユーザプロファイル
ユーザ$u$のユーザプロファイル$\boldsymbol{p}_{u}$は次式で求められる。

$$
\boldsymbol{p}_{u} = \frac{1}{\mid I_{u}^{+} \mid} \sum_{i \in I_{u}^{+}} \boldsymbol{x}_{i}
$$

ここで、$I_{u}^{+}$はユーザ$u$が「好き」と評価したアイテム集合であり、$\boldsymbol{x}_{i}$はアイテム$i$の特徴ベクトルである。

### 01 好きなアイテム集合に含まれる特徴ベクトルの取得 | 整数配列インデックス参照
整数配列インデックス参照を用いて$I_{u}^{+}$に含まれるアイテムの特徴ベクトルをすべて取得しなさい。

★
1. ベクトルの整数配列インデックス参照を使う。

### 02 好きなアイテム集合に含まれる特徴ベクトルの取得 | リスト内包表記
リスト内包表記を用いて$I_{u}^{+}$に含まれるアイテムの特徴ベクトルをすべて取得しなさい。

★★
1. リスト内包表記を使う。

### 03 特徴ベクトルの総和 | 総和
次式により、$I_{u}^{+}$に含まれるアイテムの特徴ベクトルの総和を求めなさい。

$$
\sum_{i \in I_{u}^{+}} \boldsymbol{x}_{i}
$$

★★
1. ベクトルの整数配列インデックス参照を使う。
2. `numpy.sum()`を使う。
3. `axis`を指定する。

★★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。
3. `axis`を指定する。

### 04 ユーザプロファイルの算出 | 数式をコードに変換
ユーザ$u$のユーザプロファイル$\boldsymbol{p}_{u}$を求めなさい。求めたユーザプロファイルを`pu`とすること。

★★
1. ベクトルの整数配列インデックス参照を使う。
2. `numpy.sum()`を使う。
3. `axis`を指定する。

★★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。
3. `axis`を指定する。

## コサイン類似度
ユーザプロファイル$\boldsymbol{p}_{u}$とアイテム$i$の特徴ベクトル$\boldsymbol{x}_{i}$のコサイン類似度は次式で定義される。



$$
\mathrm{cos}(\boldsymbol{p}_{u}, \boldsymbol{x}_{i}) = \frac{\boldsymbol{p}_{u} \cdot \boldsymbol{x}_{i}}{\| \boldsymbol{p}_{u} \| \| \boldsymbol{x}_{i} \|}
$$

### 05 ベクトルの内積の算出 | ベクトルの内積
$\boldsymbol{p}_{u} \cdot \boldsymbol{x}_{i}$は二つのベクトル$\boldsymbol{p}_{u}$と$\boldsymbol{x}_{i}$の内積であり、次式のように表せる。

$$
\boldsymbol{p}_{u} \cdot \boldsymbol{x}_{i} = \sum_{k=1}^{d} p_{u,k} x_{i,k}
$$

ここで、$d$はベクトルの次元数である。内積$\boldsymbol{p}_{u} \cdot \boldsymbol{x}_{i}$を求めなさい。

★
1. `@`演算子を使う。

★
1. `numpy.dot()`を使う。

★★★
1. リスト内包表記を使う。
2. `range()`を使う。
3. `numpy.sum()`を使う。

★★★
1. `@`演算子を使わない。
2. `numpy.dot()`を使わない。
3. リスト内包表記を使わない。
4. `numpy.sum()`を使う。

### 06 ベクトルのノルムの算出 | ベクトルのノルム
$\| \boldsymbol{p}_{u} \|$はベクトル$\boldsymbol{p}_{u}$のノルム（大きさ）であり、次式のように表せる。

$$
\| \boldsymbol{p}_{u} \| = \sqrt{\boldsymbol{p}_{u} \cdot \boldsymbol{p}_{u}} = \sqrt{\sum_{k=1}^{d} p_{u,k}^{2}}
$$

ここで、$d$はベクトルの次元数である。ノルム$\| \boldsymbol{p}_{u} \|$を求めなさい。

★
1. `numpy.linalg.norm()`を使う。

★★
1. `@`演算子を使う。
2. `numpy.sqrt()`を使う。

★★★
1. リスト内包表記を使う。
2. `range()`を使う。
3. `numpy.sqrt()`を使う。

★★★
1. `numpy.linalg.norm()`を使わない。
2. `@`演算子を使わない。
3. `numpy.dot()`を使わない。
4. リスト内包表記を使わない。
5. `numpy.sqrt()`を使う。

### 07 コサイン類似度の算出 | 数式をコードに変換
ユーザプロファイル$\boldsymbol{p}_{u}$とアイテム$i$の特徴ベクトル$\boldsymbol{x}_{i}$のコサイン類似度$\mathrm{cos}(\boldsymbol{p}_{u}, \boldsymbol{x}_{i})$を求めなさい。

★★
1. 課題05と課題06の結果を使う。

### 08 コサイン類似度関数の定義 | 関数
実行結果のとおりの結果が出力されるように、次のコードの【ToDo】の箇所を埋めてコサイン類似度関数を完成させなさい。

```python
def cos(pu, xi):
    """
    コサイン類似度関数：ユーザプロファイルpuとアイテムiの特徴ベクトルxiのコサイン類似度を算出する。

    Parameters
    ----------
    pu : ndarray
        ユーザuのユーザプロファイル
    xi : ndarray
        アイテムiの特徴ベクトル

    Returns
    -------
    cosine : float
        コサイン類似度
    """
    
    【ToDo】
    
    return cosine
```

実行結果
```python
>>> cos(pu[0], x[7])
0.9965457582448796
```



