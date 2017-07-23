作者：JUDGE_MENT

邮箱：gao19920804@126.com

CSDN博客：http://blog.csdn.net/sinat_23137713

最后编辑时间：2017.3.12  V1.1

声明：

1）该资料结合官方文档及网上大牛的博客进行撰写，如有参考会在最后列出引用列表。

2）本文仅供学术交流，非商用。如果不小心侵犯了大家的利益，还望海涵，并联系博主删除。

3）转载请注明出处。

4）本文主要是用来记录本人初学Tensorflow时遇到的问题，特此记录下来，因此并不是所有的方法（如安装方法）都会全面介绍。希望后人看到可以引以为鉴，避免走弯路。同时毕竟水平有限，希望有饱含学识之士看到其中的问题之后，可以悉心指出，本人感激不尽。

---

<br />

<br />

<br />

# 一. 总览

## 1. 断言和布尔检查（Asserts and boolean checks）

断言是python中的概念，就是判断一下某些东西是否满足条件，不满足就跳出异常

## 2. 建立图（Building Graphs）

对图层面的一种操作

## 3. 常量、序列和随机值（Constants, Sequences, and Random Values）

## 4. 数学（Math）

基本的算术运算、数学函数、矩阵函数

## 5. 评估指标（tf.metrics）

metric是度量标准，总看成matrix。

## 6. 神经网络（Neural Network）

这才是我们经常能用到的神经网络的层。

## 7. 张量转换变形（Tensor Transformations）

## 8. 开发版代码（tf.contrib）

易变或实验性的代码

<br />

# 二. 数学 (Math)

**https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/**

## 1. 算数运算符

- `tf.add`: 加

- `tf.subtract`: 减

- `tf.multiply`: 乘

- `tf.scalar_mul`: 标量乘以张量

- `tf.div`: 除

  其他除法：

  - [`tf.divide(x, y, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/arithmetic_operators#divide)
  - [`tf.truediv(x, y, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/arithmetic_operators#truediv)
  - [`tf.floordiv(x, y, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/arithmetic_operators#floordiv)
  - [`tf.realdiv(x, y, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/arithmetic_operators#realdiv)
  - [`tf.truncatediv(x, y, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/arithmetic_operators#truncatediv)
  - [`tf.floor_div(x, y, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/arithmetic_operators#floor_div)

- [`tf.truncatemod(x, y, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/arithmetic_operators#truncatemod)

- [`tf.floormod(x, y, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/arithmetic_operators#floormod)

- [`tf.mod(x, y, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/arithmetic_operators#mod)

- [`tf.cross(a, b, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/arithmetic_operators#cross) ：计算成对交叉积

## 2. 基本数学函数

- `tf.add_n`: 将一份由张量组成的list,全都加起来,加成一份张量

- `tf.abs`: 计算张量的绝对值

- `tf.negative`: 求反

- `tf.sign` ：

  y = sign(x) 

  = 

  -1  if  x < 0; 

  0   if  x == 0; 

  1   if  x > 0.

- [`tf.reciprocal(x, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/basic_math_functions#reciprocal) ：计算倒数
- [`tf.square(x, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/basic_math_functions#square)： 计算每个元素的平方
- [`tf.round(x, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/basic_math_functions#round)： 每个元素四舍五入
- [`tf.sqrt(x, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/basic_math_functions#sqrt)： 每个元素求根号
- [`tf.rsqrt(x, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/basic_math_functions#rsqrt)： 每个元素求根号再求倒数
- [`tf.pow(x, y, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/basic_math_functions#pow)： 计算x的y次幂
- [`tf.exp(x, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/basic_math_functions#exp)： 计算e的x次幂
- [`tf.log(x, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/basic_math_functions#log)： 计算log的e的x

## 3. 矩阵数学函数

- [`tf.diag(diagonal, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/matrix_math_functions#diag)：输入一个向量，输出基于这个向量的对角张量。

- [`tf.diag_part(input, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/matrix_math_functions#diag_part)：输入一个张量，返回这个张量的对角向量

- [`tf.trace(x, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/matrix_math_functions#trace)：计算一个张量，其对角向量的和

- [`tf.transpose(a, perm=None, name='transpose')`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/matrix_math_functions#transpose)

- [`tf.eye(num_rows, num_columns=None, batch_shape=None, dtype=tf.float32, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/matrix_math_functions#eye)： 构建单位矩阵

- [`tf.matrix_diag(diagonal, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/matrix_math_functions#matrix_diag)

- [`tf.matrix_diag_part(input, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/matrix_math_functions#matrix_diag_part)

- [`tf.matrix_band_part(input, num_lower, num_upper, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/matrix_math_functions#matrix_band_part)

- [`tf.matrix_set_diag(input, diagonal, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/matrix_math_functions#matrix_set_diag)

- [`tf.matrix_transpose(a, name='matrix_transpose')`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/matrix_math_functions#matrix_transpose)

- [`tf.matmul(a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/matrix_math_functions#matmul) ：普通矩阵相乘

  tf.batch_matmul(x, y, adj_x=None, adj_y=None, name=None)： 考虑到三维的矩阵和矩阵的相乘，去掉batch那一维度，然后相乘

  但是batch_matmul已经被放弃了，全都融入matmul了

  ​

- [`tf.norm`](https://www.tensorflow.org/versions/master/api_docs/python/tf/norm)

- [`tf.matrix_determinant(input, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/matrix_math_functions#matrix_determinant)

- [`tf.matrix_inverse(input, adjoint=None, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/matrix_math_functions#matrix_inverse)

- [`tf.cholesky(input, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/matrix_math_functions#cholesky)

- [`tf.cholesky_solve(chol, rhs, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/matrix_math_functions#cholesky_solve)

- [`tf.matrix_solve(matrix, rhs, adjoint=None, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/matrix_math_functions#matrix_solve)

- [`tf.matrix_triangular_solve(matrix, rhs, lower=None, adjoint=None, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/matrix_math_functions#matrix_triangular_solve)

- [`tf.matrix_solve_ls(matrix, rhs, l2_regularizer=0.0, fast=True, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/matrix_math_functions#matrix_solve_ls)

- [`tf.qr`](https://www.tensorflow.org/versions/master/api_docs/python/tf/qr)

- [`tf.self_adjoint_eig(tensor, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/matrix_math_functions#self_adjoint_eig)

- [`tf.self_adjoint_eigvals(tensor, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/matrix_math_functions#self_adjoint_eigvals)

- [`tf.svd(tensor, full_matrices=False, compute_uv=True, name=None)`](https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/matrix_math_functions#svd) 计算奇异值分解

## 4. 复数函数

## 5. 傅里叶变换函数 

## 6. 减少

Tensorflow提供了多种操作来减少张量维度的数学计算

- [`tf.reduce_sum`](https://www.tensorflow.org/api_docs/python/tf/reduce_sum)：
- [`tf.reduce_prod`](https://www.tensorflow.org/api_docs/python/tf/reduce_prod)
- [`tf.reduce_min`](https://www.tensorflow.org/api_docs/python/tf/reduce_min)
- [`tf.reduce_max`](https://www.tensorflow.org/api_docs/python/tf/reduce_max)
- [`tf.reduce_mean`](https://www.tensorflow.org/api_docs/python/tf/reduce_mean)
- [`tf.reduce_all`](https://www.tensorflow.org/api_docs/python/tf/reduce_all)
- [`tf.reduce_any`](https://www.tensorflow.org/api_docs/python/tf/reduce_any)
- [`tf.reduce_logsumexp`](https://www.tensorflow.org/api_docs/python/tf/reduce_logsumexp)
- [`tf.count_nonzero`](https://www.tensorflow.org/api_docs/python/tf/count_nonzero)
- [`tf.accumulate_n`](https://www.tensorflow.org/api_docs/python/tf/accumulate_n)
- [`tf.einsum`](https://www.tensorflow.org/api_docs/python/tf/einsum)

<br />

# 二. 神经网络（Neural Network）

## 1. 激活函数

提供用于神经网络的不同类型的非线性ops，包括平滑的非线性函数（sigmoid，tanh，elu，softplus和softsign），连续但不是每个地方可区分的函数（relu，relu6，crelu和relu_x）和随机正则化（dropout）。
所有激活函数应用于输入分量，并产生与输入张量相同形状的输出张量。

`tf.nn.relu  tf.nn.relu6  tf.nn.crelu  tf.nn.elu  tf.nn.softplus  tf.nn.softsign  tf.nn.dropout  tf.nn.bias_add  tf.sigmoid  tf.tanh`

## 2. 嵌入（embeddings）

### 1）tf.nn.embedding_lookup（是tf.gather函数的泛化）

就是根据train_inputs中的id，寻找embeddings中的对应元素。比如，train_inputs=[1,3,5]，则找出embeddings中下标为1,3,5的向量组成一个矩阵返回

<br />

# 三. 张量转换（Tensor Transformation）

## 1. 转换张量中数据类型

tf.string_to_number    tf.to_double    tf.to_float    tf.to_bfloat16    tf.to_int32    tf.to_int64    tf.cast    tf.bitcast    tf.saturate_cast

## 2.  张量中属性和张量变形

### 1)  查看张量属性

tf.shape

tf.size

tf.rank

### 2)  张量变形

tf.reshape

tf.squeeze

## 3. 切片和连结

### 1)  tf.slice(input_,begin,size) 张量扣取

​	从一个张量获取一部分数据，切片分割．

​	input_：张量，假设秩为ｋ

​	begin：分割的起始位置，一个ｋ维的张量

​	size：扣取张量的大小，也是一个ｋ维的张量，第一维代表在原张量第一维的步长．

### 2)  tf.strided_slice  张量跨步扣取

​	跟上面差不多，上面是提供起始位置和步长

​	这个是提供起始和终止位置

### 3)  tf.split(value, num_or_size_splits,axis)  张量沿某维度分割  

​	value:　张量，假设秩为ｋ

​	num_or_size_splits:  => num_splits: 整数，将axis这一维度平分　　＝>size_splits: 任意维度张量，其和等于axis这一维的长度.

​	axis:    按照这一维度进行分割

### 4)  tf.tile(input, multiples)  根据小张量重复创建大张量 

### 5)  tf.pad(tensor, paddings)  

​	将tensor进行拓展，可能填充０或者重复tensor的一些内容．

### 6)  tf.concat(values, axis)  

​	根据某一维度连接张量，连接之前每个张量都是秩为ｒ，连接后还是ｒ．

### 7)  tf.stack(values,axis)

​	将一系列秩为ｒ的张量打包为ｒ＋１的张量

​	values:  一个由张量组成的列表,　这里面的每个张量秩为ｒ

### 8)  tf.transpose(a,perm)  转置

​	a：输入矩阵

​	perm：参数，对于高维矩阵

对于高维矩阵





### 7)  tf.concat

​	沿一维连接张量

<br />

# 四. 开发版代码（tf.contrib）

tf.contrib中主要包含现在实验性的代码

## 1. 构建-误差-训练过程的高层API（tf.contrib.learn）

详情请见：

https://www.tensorflow.org/get_started/tflearn

https://www.tensorflow.org/get_started/input_fn

https://www.tensorflow.org/get_started/monitors

API介绍请见：

 https://www.tensorflow.org/api_guides/python/contrib.learn