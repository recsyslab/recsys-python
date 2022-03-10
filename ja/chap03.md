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

次の関数`cos(pu, xi)`は、ユーザプロファイル`pu`とアイテム`i`の特徴ベクトル`xi`のコサイン類似度を返す関数である。関数の仕様を満たすように、次の問いに答えなさい。

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
    【    問05    】
    【    問06    】
    cosine = num / (den_u * den_i)
    return cosine
```

コード
```python
u = 0
i = 7
print('cos(p{}, x{}) = {:.3f}'.format(u, i, cos(pu, x[i])))
i = 11
print('cos(p{}, x{}) = {:.3f}'.format(u, i, cos(pu, x[i])))
```

結果
```bash
cos(p0, x7) = 0.997
cos(p0, x11) = 0.783
```

### 05 ベクトルの内積の算出 | ベクトルの内積
$$\boldsymbol{p}_{u} \cdot \boldsymbol{x}_{i}$$は二つのベクトル$$\boldsymbol{p}_{u}$$と$$\boldsymbol{x}_{i}$$の内積であり、次式のように表せる。

$$
\boldsymbol{p}_{u} \cdot \boldsymbol{x}_{i} = \sum_{k=1}^{d} p_{u,k} x_{i,k}
$$

ここで、$$d$$はベクトルの次元数である。内積$$\boldsymbol{p}_{u} \cdot \boldsymbol{x}_{i}$$を求めなさい。

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
$$\| \boldsymbol{p}_{u} \|$$はベクトル$$\boldsymbol{p}_{u}$$のノルム（大きさ）であり、次式のように表せる。

$$
\| \boldsymbol{p}_{u} \| = \sqrt{\boldsymbol{p}_{u} \cdot \boldsymbol{p}_{u}} = \sqrt{\sum_{k=1}^{d} p_{u,k}^{2}}
$$

ここで、$$d$$はベクトルの次元数である。ノルム$$\| \boldsymbol{p}_{u} \|$$を求めなさい。

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
ユーザプロファイル$$\boldsymbol{p}_{u}$$とアイテム$$i$$の特徴ベクトル$$\boldsymbol{x}_{i}$$のコサイン類似度$$\mathrm{cos}(\boldsymbol{p}_{u}, \boldsymbol{x}_{i})$$を求めなさい。

★
1. 課題05の結果を使う。
2. 課題06の結果を使う。

### 08 コサイン類似度関数の定義 | 関数
関数の仕様を満たすように、次のコードの【ToDo】の箇所を埋めてコサイン類似度関数`cos(pu, xi)`を完成させなさい。確認コードを実行したとき、実行結果のとおりの結果が出力されること。

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

確認コード
```python
cos(pu[0], x[7])
```

実行結果
```python
0.9965457582448796
```

★★
1. 課題07の結果を使う。

### 09 結果の表示 | 書式指定
課題08の結果を小数第4位を四捨五入して小数第3位までを表示しなさい。

★
1. `str.format()`を使う。
2. 書式指定を使う。

## 推薦

$$\mathrm{score}(u, i)$$は対象ユーザ$$u$$が対象アイテム$$i$$を好む程度を表すスコア関数であり、次式のように定義される。

$$
\mathrm{score}(u, i) = \mathrm{cos}(\boldsymbol{p}_{u}, \boldsymbol{x}_{i})
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
    return cos(pu[u], x[i])
```

### 10 対象ユーザが未評価のアイテム集合の取得 | 差集合
対象ユーザ$$u$$が未評価のアイテム集合$$\overline{I}_{u}$$は次式のように定義される。

$$
\overline{I}_{u} = I \setminus I_{u}
$$

ここで、$$I \setminus I_{u}$$は$$I$$から$$I_{u}$$を引いた差集合を表す。このとき、$$\overline{I}_{u}$$を`ndarray`のベクトルとして取得しなさい。

★
1. `numpy.setdiff1d()`を使う。

### 11 対象ユーザの未評価アイテム集合内の各アイテムに対するスコアの取得 | 辞書
対象ユーザ$$u$$の未評価アイテム集合$$\overline{I}_{u}$$内の各アイテムに対するスコアを求め、`{i: score(u, i)}`をペアとした辞書`scores`として生成しなさい。

★★
1. `for`文を使う。

★★★
1. 辞書内包表記を使う。

### 12 対象アイテムのIDとスコアの表示 | ループによる辞書内のすべてのキー：値ペアの表示
`scores`内の`{i: score(u, i)}`のペアをすべて表示しなさい。

★★
1. `for`文を使う。
2. `dict.items()`を使う。

### 13 スコアに基づくアイテムIDのソート | 辞書のソート
`scores`内の`{i: score(u, i)}`のペアを`score(u, i)`の降順にソートしなさい。得られたリストを`rec_list`とすること。

★★★
1. `sorted()`を使う。
2. `dict.items()`を使う。
3. `key`を指定する。
4. `reverse`を指定する。
5. `lambda`式を使う。

### 14 推薦リストの生成 | リストのスライス
`rec_list`から上位`TOP_K`件のリストを取得しなさい。得られたリストを`rec_list`とすること。

★
1. リストのスライスを使う。

### 15 推薦リストの生成 | タプルリストの辞書への変換
`ranked_list`を辞書に変換しなさい。変換された辞書を`rec_list`とすること。

★
1. `dict()`を使う。

### 16 順序付け関数の定義 | 関数
関数の仕様を満たすように、次のコードの【ToDo】の箇所を埋めて順序付け類似度関数`order(u, I)`を完成させなさい。確認コードを実行したとき、実行結果のとおりの結果が出力されること。

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
7: 0.997
8: 0.983
9: 0.966
```

★★
1. 課題11の結果を使う。
2. 課題13の結果を使う。
3. 課題14の結果を使う。
4. 課題15の結果を使う。

