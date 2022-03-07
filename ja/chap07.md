<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# 第7章 決定木

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
    gini : float
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




