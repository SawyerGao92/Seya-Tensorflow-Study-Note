作者：JUDGE_MENT

邮箱：gao19920804@126.com

CSDN博客：http://blog.csdn.net/sinat_23137713

最后编辑时间：2017.4.05  V0.1

声明：

1）该资料结合官方文档及网上大牛的博客进行撰写，如有参考会在最后列出引用列表。

2）本文仅供学术交流，非商用。如果不小心侵犯了大家的利益，还望海涵，并联系博主删除。

3）转载请注明出处。

4）本文主要是用来记录本人初学Tensorflow时遇到的问题，特此记录下来，因此并不是所有的方法（如安装方法）都会全面介绍。希望后人看到可以引以为鉴，避免走弯路。同时毕竟水平有限，希望有饱含学识之士看到其中的问题之后，可以悉心指出，本人感激不尽。

---

<br />

<br />

<br />

# 一. 什么是tfdbg?

经过之前的学习，我们发现构建一个TensorFlow 项目需要经历两个步骤：
\1. 设计图
\2. 使用 Session.run() 运行图
如果在第二阶段出现了错误和 bug，我们很难进行调试，因为我们无法在运行时断点进入图中（封闭的）。所以，一个专用的运行环境调试器（debugger）是目前 TensorFlow 用户所急需的工具。

因此，伴随着Tensorflow r1.0的发布，tfdbg这一调试器也随之发布

<br />

# 二. 查看CPU/GPU运行详情

目前需要一个方法，可以查看CPU中的运行时间之类的。

```python
import tensorflow as tf
from tensorflow.Python.client import timeline

x = tf.random_normal([1000, 1000])
y = tf.random_normal([1000, 1000])
res = tf.matmul(x, y)

# Run the graph with full trace option
with tf.Session() as sess:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    sess.run(res, options=run_options, run_metadata=run_metadata)

    # Create the Timeline object, and write it to a json
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline.json', 'w') as f:
        f.write(ctf)
```

然后打开谷歌浏览器，打开网页chrome://tracing，加载timeline.json文件。

<br />

# 三. TensorBoard

将各种summary都写入路径log_dir，然后打开tensorboard从上面可以切换查看何种属性．

TensorBoard如下所示，顶部可以切换查看什么属性：

![](http://orkjdoapd.bkt.clouddn.com/Seya-Tensorflow-Study-Note/5.1%20Tensorboard.png)

## 1. 折线图

### 1). 构建网络过程中，将tf.summary op绑定到变量

summary的API包括：

https://www.tensorflow.org/api_guides/python/summary

`tf.summary.scalar`: 记录标量

`tf.summary.image`: 记录图片?

`tf.summary.histogram`: 记录柱状图？多元素的东西？矩阵？

### 2). 初始化

#### <1> 初始化1: 融合进1个op

上面一顿summary创造了很多监控op, 这些op也是需要sess.run才能运行的. 然而这么多summary op一个一个运行太恶, 把这些op全都放入一个op,用它代替就好了.

`tf.summary.merge_all`: 上面记录的好多变量，都融合进一个op，生成唯一一个摘要文件？- >变成protobuf对象 

#### <2> 初始化2: 初始化FileWriter

在循环外, 初始化FileWriter,也就是初始化输出的文件.

`tf.summary.FileWriter`: 每K次循环将proto对象写入磁盘的对象

### 3). 写入磁盘

每次循环的时候都sess.run[merge_all之后的变量]

输出的结果,添加进FileWriter:

```python
summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
train_writer.add_summary(summary, i)
```

### 4). 启动Tensorboard

```shell
python tensorflow/tensorboard/tensorboard.py --logdir=path/to/log-directory
```

或者

```shell
tensorboard --logdir=path/to/log-directory
```

然后，浏览器中输入 `localhost:6006` 来查看 TensorBoard

### 5). 演示代码

https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py

## 2. Embedding 嵌入可视化

**Embedding Projector(EP)**: 对于高维数据像词嵌入的交互式可视化.

If you are interested in embeddings of images, check out [this article](http://colah.github.io/posts/2014-10-Visualizing-MNIST/) for interesting visualizations of MNIST images. 

On the other hand, if you are interested in word embeddings, [this article](http://colah.github.io/posts/2015-01-Visualizing-Representations/) gives a good introduction.

默认情况下, EP利用主成分分析(PCA/t-SNE)将高维数据降维到3-维.

### 1) 创建词嵌入的查询变量

```python
embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
```

### 2) 周期性保存模型变量到路径为LOG_DIR的ｍodel.ckpt检查点

```python
saver = tf.train.Saver()
saver.save(session, os.path.join(LOG_DIR, "model.ckpt"), step变量)
```

### 3) 原始数据绑定到模型变量embedding(可选) 

#### step1:对于元数据的储存形式(TSV文件<Tab分割的csv>)：

* 词的元数据

下面例子中的第一行是列名，分别是单词是什么，和出现的频率，然后这里对应于embedding是根据行号确定的．

```
Word\tFrequency
Airplane\t345
Car\t241
...
```

* 图片的元数据

略

#### step2:如果你有一些原始数据（标签/图片）需要绑定到你的嵌入，可以通过：

* 在LOG_DIR位置，手动保存一个 projector_config.pbtxt

```
embeddings {
  tensor_name: 'word_embedding'
  metadata_path: '$LOG_DIR/metadata.tsv'
}
```

* 使用python API: projector来绑定,　自动生成 projector_config.pbtxt

```python
from tensorflow.contrib.tensorboard.plugins import projector
N = 10000 　# 单词字典大小
D = 200 　　# 词向量的长度
embedding_var = tf.Variable(tf.random_normal([N,D]), name='word_embedding')
# Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
config = projector.ProjectorConfig()
# 你可以添加多个embeddings，这里我们就先添加一个
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
# Link this tensor to its metadata file (e.g. labels).
embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
# Use the same LOG_DIR where you stored your checkpoint.
summary_writer = tf.summary.FileWriter(LOG_DIR)
# 这个函数生成projector_config.pbtxt文件，在目录LOG_DIR．
projector.visualize_embeddings(summary_writer, config)
```

### 4) 启动Tensorboard-指到LOG_DIR路径 

`tensorboard --logdir=LOG_DIR`

在浏览器中打开提示地址．

网页介绍：

* 数据面板：　左上方，选择运行，嵌入，和颜色．

* 投影面板：　左下方，选择投影类型（例如PCA，t-SNE）

  * PCA（线性映射）：将高维数据降低到10维，然后EP允许你10选２或３维然后画出来．
  * t-SNE（流性－非线性降维）：一篇伟大的文章
  * 自定义

* 查询面板：　右侧，查找特定点，搜寻邻居

  ​

## 3. 网络流图可视化

反正这些东西全都保存到log_dir路径，

<br />

# 四. 保存检查点

用来保存模型, 保存网络中参数权值矩阵,偏置向量等数据. 

其他网络可以读取这个保存的参数

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