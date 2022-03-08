<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# 第7章 単純ベイズ分類器

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
