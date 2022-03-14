<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

# 第11章 嗜好予測の正確性

## 準備

$$
\boldsymbol{R} = \left[
 \begin{array}{rrrrrrrrrr}
 -  & 4 & - & - & - & - & 2 & - & - & - \\
 -  & - & - & - & 2 & - & - & - & 5 & - \\
 -  & - & - & - & - & - & - & 3 & - & - \\
 \end{array}
\right]
$$

$$
\boldsymbol{R} = \left[
 \begin{array}{rrrrrrrrrr}
 -  & 2 & - & - & - & - & 2 & - & - & - \\
 -  & - & - & - & 2 & - & - & - & 3 & - \\
 -  & - & - & - & - & - & - & 3 & - & - \\
 \end{array}
\right]
$$

$$
\boldsymbol{R} = \left[
 \begin{array}{rrrrrrrrrr}
 -  & 3 & - & - & - & - & 1 & - & - & - \\
 -  & - & - & - & 3 & - & - & - & 4 & - \\
 -  & - & - & - & - & - & - & 4 & - & - \\
 \end{array}
\right]
$$



次のコードを書きなさい。

```python
import pprint
import numpy as np
np.set_printoptions(precision=3)
```
