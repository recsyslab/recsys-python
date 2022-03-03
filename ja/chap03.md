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
```

## 距離
アイテム$$i$$の特徴ベクトル$$\boldsymbol{x}_{i}$とアイテム$$j$$の特徴ベクトル$$\boldsymbol{x}_{j}$のユークリッド距離は次式で定義される。

$$
\mathrm{dist}(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}) = \sqrt{\sum_{k=1}^{d} (x_{j,k} - x_{i,k})^{2}}
$$
