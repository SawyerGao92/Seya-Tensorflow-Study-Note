作者：JUDGE_MENT

邮箱：gao19920804@126.com

CSDN博客：http://blog.csdn.net/sinat_23137713

最后编辑时间：2016.12.5  V1.1

声明：

1）该资料结合官方文档及网上大牛的博客进行撰写，如有参考会在最后列出引用列表。  

2）本文仅供学术交流，非商用。如果不小心侵犯了大家的利益，还望海涵，并联系博主删除。

3）转载请注明出处。  

4）本文主要是用来记录本人初学Tensorflow时遇到的问题，特此记录下来，因此并不是所有的方法（如安装方法）都会全面介绍。希望后人看到可以引以为鉴，避免走弯路。同时毕竟水平有限，希望有饱含学识之士看到其中的问题之后，可以悉心指出，本人感激不尽。

---

<br />

<br />

<br />

整体步骤：导入tensorflow-->启动交互式session-->构建图-->运行图-->输出图。运行普通session很难实时调试，上文也介绍了可以使用交互式的intersession进行交互式的使用。本文主要是实战演练交互式session中，构建简单的BP神经网络+softmax多分类，具体步骤如下：

注意，交互式session中，使用 InteractiveSession 代替 Session 类（）, 使用 Tensor.eval()和 Operation.run() 方法代替 Session.run(). 这样可以避免使用一个变量来持有会话。上一话中使用的sess.run( ), 而这里变成了优化参数函数.run( )。

# 一. 导入tensorflow包

`import tensorflow as tf  `

<br />

# 二. 启动交互式session

`sess = tf.InteractiveSession()  `

<br />

# 三. 模型输入

```python
from tensorflow.examples.tutorials.mnist import input_data  
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  
```

<br />

# 四. 构建图--推理（inference）--模型预测

## 1.创建输入输出占位符

输入节点：

```python
x=tf.placeholder("float",shape=[None,784])  
y_ = tf.placeholder("float", shape=[None, 10])  
```

- 输入图片x是一个2维的浮点数张量。这里，分配给它的shape为[None, 784]，其中784是一张展平的MNIST图片的维度。None表示其值大小不定，在这里作为第一个维度值，用以指代batch的大小（批量训练），意即x的数量不定。
- 输出类别值y_也是一个2维张量，其中每一行为一个10维的one-hot向量,用于代表对应某一MNIST图片的类别。
- 虽然placeholder的shape参数（shape是每一阶有多少维）是可选的，但有了它，TensorFlow能够自动捕捉因数据维度不一致导致的错误。

## 2.第一层创建变量

### <1> 变量太多，自己建立个函数来建立权重矩阵和偏置矩阵。

```python
def weight_variable(shape):  
  initial = tf.truncated_normal(shape, stddev=0.1)  
  return tf.Variable(initial)  

def bias_variable(shape):  
  initial = tf.constant(0.1, shape=shape)  
  return tf.Variable(initial)  
```

### <2> 卷积+池化。对于卷积从5*5到像素块-->输入1个通道-->输出32个通道 

```python
W_conv1 = weight_variable([5, 5, 1, 32])  
b_conv1 = bias_variable([32])  
```

## 3.第一层写出输出函数

卷积函数在tf.nn.conv2d（）,；四个参数：输入第一个参数为输入变量，第二个参数为权重，第三，第四略。

下采样函数在tf.nn.max_pool（）；同样资格参数

### <1> 因为这两个函数某些参数一直保持默认值就可以，所以为了简洁，把这两个封装成函数。

```python
def conv2d(x, W):  
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  

def max_pool_2x2(x):  
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')  
```

### <2>  写出输出函数

输入的x需要变形称conv2d接受的形式（上面softmax把数据当成一个向量了，这里要把图片当成矩阵），变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3）。

`x_image = tf.reshape(x, [-1,28,28,1])  `

 我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行maxpooling。

```python
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  
h_pool1 = max_pool_2x2(h_conv1)  
```

## 4.第二层创建变量

```python
W_conv2 = weight_variable([5, 5, 32, 64])  
b_conv2 = bias_variable([64])  
```

## 5.第二层写出输出函数

```python
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  
h_pool2 = max_pool_2x2(h_conv2)  
```

## 6.密集连接层创建变量

```python
W_fc1 = weight_variable([7 * 7 * 64, 1024])  
b_fc1 = bias_variable([1024])  
```

## 7.密集连接层写出输出函数

```python
h_pool2_flat = tf.reshape(h_pool2, [-1, 7764])  
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  
```

加入一个有1024个神经元的全连接层

## 8.Dropout

就是hilton发明的一个防止过拟合的装置，原理其实很简单，就是一部分不输出。训练的时候开启，输出时不开启。

```python
keep_prob = tf.placeholder("float")  
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  
```

## 9.softmax输出层

![暂时无法显示](https://cl.ly/2a0k2B0r0p3Y/SOFTMAX.png)



```python
W_fc2 = weight_variable([1024, 10])  
b_fc2 = bias_variable([10])  
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  
```

<br />

# 五. 构建图--损失（Loss）+训练参数选择--模型训练

## 1.损失函数选择

损失函数使用交叉熵：

`cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))  `

## 2.优化算法和步长选择设置

`train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  `

这里步长设置为1e-4，优化算法使用AdamOptimizer，来最优化损失函数

## 3.评估模型结果指标

`correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))  `

tf.argmax:  给出某个tensor对象在某一维上的其数据最大值所在的索引值,由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签.

这里返回一个布尔数组。为了计算我们分类的准确率，我们将布尔值转换为浮点数来代表对、错，然后取平均值。例如：[True, False, True, True]变为[1,0,1,1]，计算出平均值为0.75。

`accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  `

<br />

# 六. 运行图--训练

## 1.变量全都初始化

这一步是训练前必经之路 

`sess.run(tf.initialize_all_variables())  `

## 2.循环训练

```python
for i in range(20000):                           #训练20000次  
  batch = mnist.train.next_batch(50)             #这里是训练数据每次抽取50个  
  if i%100 == 0:                                 #每训练100次输出这么一个训练准确率  
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})  
    print "step %d, training accuracy %g"%(i, train_accuracy)  
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})       #这里train_step.run就用来更新权重了  
```

<br />

# 七. 输出结果

`print "test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})  `

经过20000次的迭代修正权值，最终在测试集上准确率达到99.06%的准确率。

