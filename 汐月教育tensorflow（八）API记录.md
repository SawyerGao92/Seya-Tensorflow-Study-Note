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