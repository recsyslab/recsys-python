<script type="text/x-mathjax-config">MathJax.Hub.Config({tex2jax:{inlineMath:[['\$','\$'],['\\(','\\)']],processEscapes:true},CommonHTML: {matchFontHeight:false}});</script>
<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

## 第2章 評価値行列

## 準備

## 評価値行列

次の行列$$\boldsymbol{R}$$は評価値行列である。$$\boldsymbol{R}$$の各行は各ユーザ$$u \in U$$を表し、各列は各アイテム$$i \in I$$を表す。$$\boldsymbol{R}$$の$$(u, i)$$成分はユーザ$$u$$がアイテム$$i$$に与えた評価値$$r_{u,i}$$を表す。ただし、$$?$$は欠損値であることを表す。

$$
\boldsymbol{R} = \left[
 \begin{array}{rrrrrr}
  ? & 4 & 3 & 1 & 2 & ? \\
  5 & 5 & 4 & ? & 3 & 3 \\
  4 & ? & 5 & 3 & 2 & ? \\
  ? & 3 & ? & 2 & 1 & 1 \\
  2 & 1 & 2 & 4 & ? & 3 \\
 \end{array}
\right]
$$

### 21 評価値行列の生成 | 行列の生成
$$\boldsymbol{R}$$を`ndarray`の行列`R`として生成しなさい。

★
1. `numpy.array()`を使う。

### 22 評価値行列からユーザ集合の取得 | ベクトルの生成
`R`の各行のインデックスは各ユーザ$$u \in U$$のユーザIDに対応する。ユーザ集合$$U$$を、ユーザIDを要素とした`ndarray`のベクトル`U`として生成しなさい。

★
1. `numpy.arange()`を使う。
2. `ndarray.shape`を使う。

### 23 評価値行列からアイテム集合の取得 | ベクトルの生成
`R`の各列のインデックスは各アイテム$$i \in I$$のアイテムIDに対応する。アイテム集合$$I$$を、アイテムIDを要素とした`ndarray`のベクトル`I`として生成しなさい。

★
1. `numpy.arange()`を使う。
2. `ndarray.shape`を使う。

### 24 ユーザ数の取得 | ベクトルの要素数
ユーザ数$$\mid U \mid$$を取得しなさい。

★
1. `ndarray.size`を使う。

### 25 アイテム数の取得 | ベクトルの要素数
アイテム数$$\mid I \mid$$を取得しなさい。

★
1. `ndarray.size`を使う。

### 26 評価値の取得 | ベクトルのインデックス参照
`R`からユーザ0のアイテム1に対する評価値$$r_{0,1}$$を取得しなさい。

★
1. ベクトルのインデックス参照を使う。
