<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# 第10章 決定木

```python
import numpy as np

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

## ジニ係数

訓練データ$$D_{u}^{L}$$のジニ係数$$G(D^{L}_{u})$$は次式で定義される。

$$
G(D^{L}_{u}) = 1 - \{(p^{+})^{2} + (p^{-})^{2}\}
$$

ここで、$$p^{+}$$は訓練データ$$D_{u}^{L}$$における「好き」な事例が含まれる割合、$$p^{-}$$は「嫌い」な事例が含まれる割合を表し、それぞれ次式で求められる。

$$
\begin{array}{l}
    p^{+} = \frac{\mid D^{L+}_{u} \mid}{\mid D^{L}_{u} \mid} \\
    p^{-} = \frac{\mid D^{L-}_{u} \mid}{\mid D^{L}_{u} \mid}
\end{array}
$$

次の関数は、入力された訓練データ`DL`のジニ係数を返す関数`G(DL)`である。ただし、`DL`に事例が含まれていないときは`0`を返す。

関数
```python
def G(DL):
    """
    訓練データDLのジニ係数を返す。
    
    Parameters
    ----------
    DL : ndarray
        訓練データDL

    Returns
    -------
    float
        ジニ係数
        ただし、DLに事例が含まれていないときは0
    """
    if DL.shape[0] == 0: return 0
    r = DL[:,-1]
    【問題01】
    【問題02】
    【問題03】
    return gini
```

コード
```python
# ジニ係数
print('G(DuL) = {:.3f}'.format(G(DuL)))
```

結果
```bash
G(DuL) = 0.480
```

### 01 「好き」な事例が含まれる割合
$$p^{+}$$を求めるコードを書きなさい。得られた値を`pp`とすること。

★
1. ベクトルのブールインデックス参照を使う。
2. `ndarray.shape`を使う。

### 02 「嫌い」な事例が含まれる割合
$$p^{-}$$を求めるコードを書きなさい。得られた値を`pn`とすること。

★
1. ベクトルのブールインデックス参照を使う。
2. `ndarray.shape`を使う。

### 03 ジニ係数
ジニ係数を求めるコードを書きなさい。得られた値を`gini`とすること。

★
1. べき乗を使う。

## 分割の良さ
訓練データ$$D_{u}^{L}$$を$$D_{u}^{L0}$$と$$D_{u}^{L1}$$に分割したときのジニ係数は次式で定義される。

$$
G(D^{L}_{u} \rightarrow [D^{L0}_{u}, D^{L1}_{u}]) = \frac{\mid D^{L0}_{u} \mid G(D^{L0}_{u}) + \mid D^{L1}_{u} \mid G(D^{L1}_{u})}{\mid D^{L0}_{u} \mid + \mid D^{L1}_{u} \mid}
$$

次の関数は、訓練データを`DL0`と`DL1`に分割したときののジニ係数を返す関数`G_partitioned(DL0, DL1)`である。

```python
def G_partitioned(DL0, DL1):
    """
    訓練データをDL0とDL1に分割したときのジニ係数を返す。
    
    Parameters
    ----------
    DL0 : ndarray
        訓練データDL0
    DL1 : ndarray
        訓練データDL1

    Returns
    -------
    float
        ジニ係数
    """
    【問題06】
    return gini
```

### 04 特徴量kを含まない訓練事例集合
訓練データ$$D_{u}^{L}$$から特徴量$$k$$を含まない（$$x_{i,k}=0$$となる）事例集合を`ndarray`として取得するコードを作成しなさい。取得した`ndarray`を`DuL0`とすること。

コード
```python
# 特徴量kを含まない訓練事例集合
k = 0
【問題04】
print('DuL0 = \n{}'.format(DuL0))
```

結果
```bash
DuL0 = 
 [[ 0.  1.  0.  0.  1.  0.  1.]
 [ 0.  1.  0.  1.  0.  1.  1.]
 [ 0.  0.  1.  0.  1.  0. -1.]
 [ 0.  0.  1.  1.  1.  0. -1.]
 [ 0.  1.  0.  0.  1.  1. -1.]
 [ 0.  0.  1.  0.  0.  1. -1.]]
```

★★
1. ベクトルのブールインデックス参照を使う。

### 05 特徴量kを含む訓練事例集合
訓練データ$$D_{u}^{L}$$から特徴量$$k$$を含む（$$x_{i,k}=1$$となる）事例集合を`ndarray`として取得するコードを作成しなさい。取得した`ndarray`を`DuL1`とすること。

コード
```python
# 特徴量kを含まない訓練事例集合
k = 0
【問題05】
print('DuL1 = \n{}'.format(DuL1))
```

結果
```bash
DuL1 = 
 [[1. 0. 0. 0. 1. 0. 1.]
 [1. 1. 0. 0. 1. 0. 1.]
 [1. 0. 0. 1. 1. 0. 1.]
 [1. 0. 0. 0. 0. 1. 1.]]
```

★★
1. ベクトルのブールインデックス参照を使う。

### 06 特徴量kを基準に分割したときのジニ係数
特徴量$$k$$を基準に分割したときのジニ係数を求めるコードを書きなさい。得られた値を`gini`とすること。

コード
```python
# 特徴量kを基準に分割したときのジニ係数
print('G(DuL → [DuL0, DuL1]) = {:.3f}'.format(G_partitioned(DuL0, DuL1)))
```

結果
```bash
G(DuL → [DuL0, DuL1]) = 0.267
```

★★
1. `ndarray.shape`を使う。
2. `G(DL)`関数を呼ぶ。

## 決定木の学習

次の関数は、入力された訓練データ`DL`を各特徴量で分割したときの（特徴量のインデックス: ジニ係数）を対にした辞書を返す関数`get_ginis(DL)`である。

```python
def get_ginis(DL):
    """
    訓練データDLを各特徴量で分割したときの（特徴量のインデックス: ジニ係数）を対にした辞書を返す。
    
    Parameters
    ----------
    DL : ndarray
        訓練データDL

    Returns
    -------
    dict
        （特徴量のインデックス: ジニ係数）を対にした辞書
    """
    ginis = {}
    for k in range(0, x.shape[1]):
        DL0 = DL[DL[:,k]==0]
        DL1 = DL[DL[:,k]==1]
        ginis[k] = G_partitioned(DL0, DL1)
    return ginis
```

### 07 レベル0の選択基準
`get_ginis()`関数から得られた`ginis`からジニ係数が最小となる特徴量のインデックスを取得するコードを作成しなさい。得られた値を`k0`とすること。

コード
```python
# レベル0（根ノード）の選択基準
ginis = get_ginis(DuL)
print('ginis = ', end='')
for k, gini in ginis.items():
    print('{}: {:.3f}'.format(k, gini), end=', ')
print()
【問題07】
print('k0 = {}'.format(k0))
DuL0 = DuL[DuL[:,k0] == 0]
DuL1 = DuL[DuL[:,k0] == 1]
print('DuL0 = \n', DuL0)
print('DuL1 = \n', DuL1)
```

結果
```bash
ginis = 0: 0.267, 1: 0.450, 2: 0.171, 3: 0.476, 4: 0.476, 5: 0.467,
k0 = 2
DuL0 = 
 [[ 1.  0.  0.  0.  1.  0.  1.]
 [ 0.  1.  0.  0.  1.  0.  1.]
 [ 1.  1.  0.  0.  1.  0.  1.]
 [ 1.  0.  0.  1.  1.  0.  1.]
 [ 1.  0.  0.  0.  0.  1.  1.]
 [ 0.  1.  0.  1.  0.  1.  1.]
 [ 0.  1.  0.  0.  1.  1. -1.]]
DuL1 = 
 [[ 0.  0.  1.  0.  1.  0. -1.]
 [ 0.  0.  1.  1.  1.  0. -1.]
 [ 0.  0.  1.  0.  0.  1. -1.]]
```

★★
1. `min()`を使う。

### 08 レベル1の選択基準
レベル0の分割で得られた`DuL0`からレベル1a（レベル1の左端ノード）の選択基準となる特徴量のインデックスを取得するコードを作成しなさい。得られた値を`k1a`とすること。

```python
# レベル1a（レベル1の左端ノード）の選択基準
ginis = get_ginis(DuL0)
k1a = min(ginis, key=ginis.get)
print('k1a = {}'.format(k1a))
DuL00 = DuL0[DuL0[:,k1a] == 0]
DuL01 = DuL0[DuL0[:,k1a] == 1]
print('DuL00 = \n', DuL00)
print('DuL01 = \n', DuL01)
```

```bash
k1a = 0
DuL00 = 
 [[ 0.  1.  0.  0.  1.  0.  1.]
 [ 0.  1.  0.  1.  0.  1.  1.]
 [ 0.  1.  0.  0.  1.  1. -1.]]
DuL01 = 
 [[1. 0. 0. 0. 1. 0. 1.]
 [1. 1. 0. 0. 1. 0. 1.]
 [1. 0. 0. 1. 1. 0. 1.]
 [1. 0. 0. 0. 0. 1. 1.]]
 ```
 
 ★★
 1. `get_ginis()`関数を呼ぶ。
 2. `min()`を使う。

### 09 レベル2の選択基準
レベル1の分割で得られた`DuL00`からレベル2a（レベル2の左端ノード）の選択基準となる特徴量のインデックスを取得するコードを作成しなさい。得られた値を`k2a`とすること。

```python
# レベル2a（レベル2の左端ノード）の選択基準
ginis = get_ginis(DuL00)
k2a = min(ginis, key=ginis.get)
print('k2a = {}'.format(k2a))
DuL000 = DuL00[DuL00[:,k2a] == 0]
DuL001 = DuL00[DuL00[:,k2a] == 1]
print('DuL000 = \n', DuL000)
print('DuL001 = \n', DuL001)
```

```bash
k2a = 3
DuL000 = 
 [[ 0.  1.  0.  0.  1.  0.  1.]
 [ 0.  1.  0.  0.  1.  1. -1.]]
DuL001 = 
 [[0. 1. 0. 1. 0. 1. 1.]]
 ```
 
 ★★
 1. `get_ginis()`関数を呼ぶ。
 2. `min()`を使う。

## 嗜好予測

次の関数は、訓練データ`DL`から決定木を学習する関数`train(DL, key)`および対象ユーザ`u`の対象アイテム`i`の予測評価値を返す関数`predict(u, i, key)`である。

```python
def train(DL, key=0):
    """
    訓練データDLから決定木を学習する。
    
    Parameters
    ----------
    DL : ndarray
        訓練データDL
    key : int
        キー値
    """
    if len(DL) <= 0:
        return
    elif np.count_nonzero(DL[:,-1]==-1) <= 0:
        dtree[key] = '+1'
        return
    elif np.count_nonzero(DL[:,-1]==+1) <= 0:
        dtree[key] = '-1'
        return
        
    ginis = get_ginis(DL)
    k = min(ginis, key=ginis.get)
    dtree[key] = k
    DL0 = DL[DL[:,k] == 0]
    DL1 = DL[DL[:,k] == 1]
    train(DL0, key * 2 + 1)
    train(DL1, key * 2 + 2)

def predict(u, i, key=0):
    """
    対象ユーザuの対象アイテムiの予測評価値を返す。
    
    Parameters
    ----------
    u : int
        対象ユーザuのインデックス（ダミー）
    i : int
        対象アイテムiのインデックス
    key : int
        キー値

    Returns
    -------
    int
        予測評価値
    """
    if type(dtree[key]) == str: return int(dtree[key])
    k = dtree[key]
    if x[i,k] == 0:
        return predict(u, i, key * 2 + 1)
    elif x[i,k] == 1:
        return predict(u, i, key * 2 + 2)
```

### 10 予測対象データに対する嗜好予測
予測対象データ$$D_{u}^{U}$$内の各アイテム$$i$$について予測評価値$\hat{r}_{u,i}$$を求め、`i: predict(u, i)`を対とした辞書を生成するコードを作成しなさい。生成した辞書を`ruU_pred`とすること。

コード
```python
dtree = {}
train(DuL)
print('dtree = {}'.format(dtree))

u = 0
【問題10】
print('ruU_pred = {}'.format(ruU_pred))
```

結果
```
dtree = {0: 2, 1: 0, 3: 3, 7: 5, 15: '+1', 16: '-1', 8: '+1', 4: '+1', 2: '-1'}
ruU_pred = {10: 1, 11: -1, 12: -1}
```

1. 辞書内包表記を使う。
2. `predict()`関数を呼ぶ。

