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

## 3. 张量

*　张量在numpy中定义，指多维矩阵，tensorflow中的常量变量占位符都是一种缓冲区，中间包含着张量．
*　ｏｐ之间传递靠的张量

## 4. 常量

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

## 5. 外部输入：占位符

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

## 6. 变量

其实占位符也算是变量，但是它只能接受外部。变量是**包含张量的内存缓冲区**，同样常量，占位符也都是．

### 1）构建方法：

通过将张量传递到变量构建

```python
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
c= tf.Variable(tf.random_normal([784, 200], stddev=0.35),name="c")    # 随机初始化，高斯分布
d= tf.Variable(c.initialized_value() * 0.2, name="d")     　　　　　　　# 从另一变量初始化，需要加上.initialized_value()
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

### 8) 共享变量(附)

参见：<http://www.tensorfly.cn/tfdoc/how_tos/variable_scope.html>

#### 问题：

像ＲＮＮ这种结构，需要多次调用同一个inference函数，重要的是其中的变量许要共享，应该为同一个．

```python
def inference():
    １．创建变量
    ２．推测tensorflow图
```

#### 解决：

##### 方法１：创建变量的部分放到inference函数外面

方法１缺点：破坏了封装性

##### 方法２：变量作用域

```python
with tf.variable_scope("foo", reuse=True):
    v = tf.get_variable("v", [1])
```

###### 函数：get_variable

 `v = tf.get_variable(name, shape, dtype, initializer)`

​	检查所有变量中是否存在foo+v变量，如果不存在则用**initializer[shape]**创建，如果存在则调用该变量．

​	initializer：初始化器包括

​	`tf.constant_initializer(value)` 初始化一切所提供的值,

​	`tf.random_uniform_initializer(a, b)`从a到b均匀初始化,

​	`tf.random_normal_initializer(mean, stddev)` 用所给平均值和标准差初		始化均匀分布.

###### 函数：variable_scope()

​	 为变量提供命名空间，get_variable如果在两个命名空间中就是两个变量，如果在一个变量空间中添加一个`reuse_variables()`就是共享变量

* 情况１：当`tf.get_variable_scope().reuse == False`，创建新变量

​	将会创建新变量, 如果已经存在，则抛出异常ValueError

​	另外：不能直接设置 `reuse` 为 `False` ,可以输入一个重用变量作用域,然后就释放掉,就成为非重用的变量.当打开一个变量作用域时,使用`reuse=True` 作为参数是可以的.但也要注意，同一个原因，`reuse`参数是不可继承.所以当你打开一个重用变量作用域，那么所有的子作用域也将会被重用.

```python
with tf.variable_scope("root"):
    # resue = False 将会创建新变量
    assert tf.get_variable_scope().reuse == False
    with tf.variable_scope("foo"):
        # 开启一个子空间，reuse = False 同样会创建新变量
        assert tf.get_variable_scope().reuse == False
    with tf.variable_scope("foo", reuse=True):
        # 开启一个子空间，reuse = True　明确说明调用其他变量
        assert tf.get_variable_scope().reuse == True
        with tf.variable_scope("bar"):
            # 现在子空间继承了, reuse = True 会调用其他变量
            assert tf.get_variable_scope().reuse == True
    # 跳出了子空间, 又变回reuse = False了.
    assert tf.get_variable_scope().reuse == False
```

* 情况２：当`tf.get_variable_scope().reuse == True`，共享变量

  get_variable将会调用已经存在的变量，他的全称和当前变量的作用域名+所提供的名字是否相等．如果不存在，则抛出ValueError ．

  示例：

  ```python
  with tf.variable_scope("image_filters") as scope:
      result1 = my_image_filter(image1)
      scope.reuse_variables()　　　
      result2 = my_image_filter(image2)
  ```

  或者

  `tf.get_variable_scope()`可以检索当前变量作用域

  ```python
  with tf.variable_scope("foo"):
      v = tf.get_variable("v", [1])
      tf.get_variable_scope().reuse_variables()
      v1 = tf.get_variable("v", [1])
  assert v1 == v
  ```

  或者

  ```python
  with tf.variable_scope("foo"):
      v = tf.get_variable("v", [1])
  with tf.variable_scope("foo", reuse=True):
      v1 = tf.get_variable("v", [1])
  assert v1 == v
  ```

* 其他知识

  1) 创建作用域的时候,不一定用名字,也可以用作用域对象

  ```python
  with tf.variable_scope("foo") as foo_scope:
      v = tf.get_variable("v", [1])
  with tf.variable_scope(foo_scope)
      w = tf.get_variable("w", [1])
  with tf.variable_scope(foo_scope, reuse=True)
      v1 = tf.get_variable("v", [1])
      w1 = tf.get_variable("w", [1])
  assert v1 == v
  assert w1 == w
  ```

  2) 开启新作用域,如果调用的以前存在的作用域,则会跳过当前作用域前缀

  ```python
  with tf.variable_scope("foo") as foo_scope:
      assert foo_scope.name == "foo"
  with tf.variable_scope("bar")
      with tf.variable_scope("baz") as other_scope:
          assert other_scope.name == "bar/baz"
          with tf.variable_scope(foo_scope) as foo_scope2:
              assert foo_scope2.name == "foo"  # Not changed.
  ```

  3) 创建作用域时, 可以提到initializer, 则下面的get_variable都是这个initializer.

  ```python
  with tf.variable_scope("foo", initializer=tf.constant_initializer(0.4)):
      v = tf.get_variable("v", [1])
  ```

  4) 之前是变量作用域,还可以有ops作用域,name_scope可以修改op

  ```python
  with tf.variable_scope("foo"):
      with tf.name_scope("bar"):
          v = tf.get_variable("v", [1])
          x = 1.0 + v
  assert v.name == "foo/v:0"
  assert x.op.name == "foo/bar/add"
  ```

  ​

## 7. 队列

对于进行异步运算有很大帮助

FIFOQueue:　先入先出队列

过程如下：

![](http://orkjdoapd.bkt.clouddn.com/Seya-Tensorflow-Study-Note/2.1%20IncremeterFifoQueue.gif)

* enqueue: 入栈
* enqueue_many: 入栈很多元素
* dequeue: 出栈

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