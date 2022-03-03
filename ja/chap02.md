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

## ユーザプロファイルの算出

ユーザ`u`のユーザプロファイル`pu`は次式で求められる。

$u_{i}$

$$ \\bm{p}_{u} $$


$\boldsymbol{p} _{u}$

$\boldsymbol{p}_{u}$

$\boldsymbol{p}_{u}$

$\frac{1}{\mid I_{u}^{+} \mid}$

$\sum_{i \in I_{u}^{+}} \boldsymbol{x}_{i}$

$\boldsymbol{p}_{u} = \frac{1}{\mid I_{u}^{+} \mid} \sum_{i \in I_{u}^{+}} \boldsymbol{x}_{i}$
![eq_cbr1_user_profile_avg](/img/eq/eq_cbr1_user_profile_avg.png)

### 01 好きなアイテム集合に含まれる特徴ベクトルの取得 | 整数配列インデックス参照
`Iup`に含まれるアイテムの特徴ベクトルをすべて取得しなさい。

★
1. ベクトルの整数配列インデックス参照を使う。

### 02 好きなアイテム集合に含まれる特徴ベクトルの取得 | リスト内包表記
リスト内包表記を用いて`Iup`に含まれるアイテムの特徴ベクトルをすべて取得しなさい。

★★
1. リスト内包表記を使う。

### 03 特徴ベクトルの総和 | 総和
次式を計算しなさい。

![eq_cbr1_user_profile_avg_2](/img/eq/eq_cbr1_user_profile_avg_2.png)

★★
1. ベクトルの整数配列インデックス参照を使う。
2. `numpy.sum()`を使う。
3. `axis`を指定する。

★★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。
3. `axis`を指定する。

### 04 ユーザuのユーザプロファイルの算出 | 数式をコードに変換
ユーザ`u`のユーザプロファイル`pu`を求めなさい。求めたユーザプロファイルを`pu`とすること。

★★
1. ベクトルの整数配列インデックス参照を使う。
2. `numpy.sum()`を使う。
3. `axis`を指定する。

★★★
1. リスト内包表記を使う。
2. `numpy.sum()`を使う。
3. `axis`を指定する。

## 

