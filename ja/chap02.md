# 第2章 類似度に基づく推薦

## 00 準備
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

![\begin{align*}   \left( \int_0^\infty \frac{\sin x}{\sqrt{x}} dx \right)^2 =   \sum_{k=0}^\infty \frac{(2k)!}{2^{2k}(k!)^2} \frac{1}{2k+1} =   \prod_{k=1}^\infty \frac{4k^2}{4k^2 - 1} = \frac{\pi}{2} \end{align*}](https://render.githubusercontent.com/render/math?math=%5Cbegin%7Balign*%7D%20%20%20%5Cleft(%20%5Cint_0%5E%5Cinfty%20%5Cfrac%7B%5Csin%20x%7D%7B%5Csqrt%7Bx%7D%7D%20dx%20%5Cright)%5E2%20%3D%20%20%20%5Csum_%7Bk%3D0%7D%5E%5Cinfty%20%5Cfrac%7B(2k)!%7D%7B2%5E%7B2k%7D(k!)%5E2%7D%20%5Cfrac%7B1%7D%7B2k%2B1%7D%20%3D%20%20%20%5Cprod_%7Bk%3D1%7D%5E%5Cinfty%20%5Cfrac%7B4k%5E2%7D%7B4k%5E2%20-%201%7D%20%3D%20%5Cfrac%7B%5Cpi%7D%7B2%7D%20%5Cend%7Balign*%7D)
