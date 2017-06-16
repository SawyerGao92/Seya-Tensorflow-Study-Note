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

## 2. 切片和连结

### 1） tf.concat(concat_dim, values, name='concat')  沿一维连接张量

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