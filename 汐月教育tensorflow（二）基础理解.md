作者：JUDGE_MENT

邮箱：gao19920804@126.com

CSDN博客：http://blog.csdn.net/sinat_23137713

最后编辑时间：2016.12.5  V1.1

声明：

1）该资料结合官方文档及网上大牛的博客进行撰写，如有参考会在最后列出引用列表。

2）本文仅供学术交流，非商用。如果不小心侵犯了大家的利益，还望海涵，并联系博主删除。

3）转载请注明出处。

4）本文主要用来记录本人初学Tensorflow时遇到的问题，特此记录下来，因此并不是所有的方法都会面面俱到。希望后人看到可以引以为鉴，避免走弯路。同时毕竟水平有限，希望有饱含学识之士看到其中的问题之后，可以悉心指出，本人感激不尽。

---

<br />

<br />

<br />

# 一. 基础

## 1. 头文件

 `import tensorflow as tf`

## 2. 整体使用过程

### 1）建立图

### 2）运行图

## 3. 常量

是一种不需要输入值，输出是它内部储存的值得节点。

### 1）构建方法：

```python
node1 = tf.constant(3.0, tf.float32)  #不写float32也行，默认float32
c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
```

### 2）查看数据

```python
print node1  #不会直接输出节点中的结果，已经被封装了，必须在会话session中运行图才能看到里面的数据。
sess = tf.Session() 
print(sess.run(node1))
```

## 4. 外部输入：占位符

图可以接受外部输入，称为占位符。 占位符是对稍后提供值的承诺。

### 1）构建方法：

```python
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b         # + provides a shortcut for tf.add(a, b)
```

### 2）给占位符以数据

```python
print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))
```

## 5. 变量

其实占位符也算是变量，但是它只能接受外部。变量是，不解释了，你懂得

### 1）构建方法：

需要一个类型和初始值

```python
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
c= tf.Variable(tf.random_normal([784, 200], stddev=0.35),name="c")     # 随机初始化，高斯分布
d= tf.Variable(c.initialized_value() * 0.2, name="d")     # 从另一变量初始化，需要加上.initialized_value()
x = tf.placeholder(tf.float32)
linear_model = W * x + b
```

### 2）初始化变量(也可以用来重置，恢复到默认变量)：

当调用tf.constant时，常量被初始化，它们的值永远不会改变；相比之下，当调用tf.Variable时，变量不会被初始化。     

如下，初始化TensorFlow中的所有变量（在run之后才初始化）：

```python
init = tf.global_variables_initializer()
sess.run(init)
```

### 3）查看数据：

教你三种方法查看数据，快谢谢我，1,2,3一起读，阿里叩头~

#### <1>

```python
sess = tf.Session() 
print(sess.run(W))
```

#### <2>

```python
sess = tf.Session()
print(v.eval(sess))
```

#### <3>

```python
with tf.Session() as sess:
print(v.eval())           #在with旗下可以省略sess
```

### 4）查看属性：

```python
W.get_shape()       # 查看张量的形状
```

### 5）下标引用：

```python
W[2]
```

### 6）变量重新赋值：

```python
fixW = tf.assign(W, [-1.])
sess.run(fixW)
```

### 7）训练过程中保存变量到本地，读取变量

创建保存节点：

```python
saver = tf.train.Saver(<可以只选择变量保存/如果为空，默认保存全部变量>)
```

选择部分变量，可以传入字典，例：

```python
saver = tf.train.Saver({"my_v2": v2})   # 把v2变量进行储存，储存为my_v2这个变量名称
```

保存变量：

```python
saver.save(<sess需要保存的会话>, <保存路径>, global_step=<第几步的迭代器>)
```

读取变量：

```python
saver.restore(<sess变量恢复到会话>, <读取的地址>)
```

注意，当你从文件中恢复变量时，不需要事先对它们做初始化（不需要global_vatiables_initializer）。

<br />

# 二. 构建网络

## 1. 低层：构建过程自己写

## 2. 高层：使用tf.contrib.learn

详情见第六章api

<br />

# 三. 损失函数

## 1. 低层：损失函数可以自己建造一个，如平方差损失函数

```python
y = tf.placeholder(tf.float32)# y是目标值
squared_deltas = tf.square(linear_model - y)# 偏差的进行平方
loss = tf.reduce_sum(squared_deltas)# 平方后进行求和
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})) # loss调用linaear_model,linear_model调用了x（在5.1）,所以要输入x和y
```

## 2.高层：使用tf内部的交叉熵损失函数：

`tf.nn.sigmoid_cross_entropy_with_logits（logits, labels）`

例如这个模型一次要判断100张图是否包含10种动物，则logits和labels输入的shape都是[100, 10]。注释中还提到这10个分类之间是独立的、不要求是互斥。

`tf.nn.sigmoid_cross_entropy( )`

`tf.nn.softmax_cross_entropy_with_logits`

`tf.nn.sparse_softmax_cross_entropy_with_logits（logits，labels）`

有sparse的label是[100，1]这个尺寸，每个label是一个数字0~任何的数字。

`tf.nn.weighted_cross_entropy_with_logits( )`

参考：http://www.tuicool.com/articles/n22m2az

<br />

# 四. 训练参数

训练参数有多重途径，包括自己求导自己写循环，包括用tersorflow里面的类减少循环等等。

## 1.低层： tensorflow求梯度（导数）（人工求,容易错）+自己写训练循环  

```python
tf.gradients

tf.AggregationMethod

tf.stop_gradient

tf.hessians
```

## 2. 中层：tensorflow的API函数帮你 求梯度 + 训练（优化的循环不用自己写）：

1）设定一些优化器参数

```python
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
```

2）运行，迭代获取最优参数

```python
for i in range(1000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
```

3）输出最后的参数结果

```python
print(sess.run([W, b]))
```

## 疑问1：为什么优化器在不知道哪个是参数的时候，可以参数优化啊？

我开始一直有这么个疑问，都没有说哪个参数需要优化，你怎么优化的？后来发现是这样的：在minimize里面有个参数-var_list，这个参数可以输入需要优化的变量，如果不输入则默认为（TRAINABLE_VARIABLES）中的所有变量。其中这个（TRAINABLE_VARIABLES）是会在变量创立时自动收集的，设定`trainable=False` 可以防止该变量被数据流图的 `GraphKeys.TRAINABLE_VARIABLES` 收集, 这样我们就不会在训练的时候尝试更新它的值。

![img](http://img.blog.csdn.net/20170223102743244?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2luYXRfMjMxMzc3MTM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

## 3. 高层：tf.contrib.learn

构建方法，损失函数...等等都不需要自己写，只需要告诉输入和输出就行

详情见第六章api

可以用这个类帮助简化：

* 训练的循环

- 评估的循环


- 管理数据集


- 管理反馈

input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4, num_epochs=1000)

2）虽然乍看之下这种方法很无脑，只需要输入和输出的文件，但是也可以自己建立模型输入函数中，但我感觉就不如自己全写了。

<br />

<br />

参考附录：
感谢各位无私的奉献
1) http://dataunion.org/26447.html