作者：JUDGE_MENT

邮箱：gao19920804@126.com

CSDN博客：http://blog.csdn.net/sinat_23137713

最后编辑时间：2016.12.5  V1.1

声明：

1）该资料结合官方文档及网上大牛的博客进行撰写，如有参考会在最后列出引用列表。

2）本文仅供学术交流，非商用。如果不小心侵犯了大家的利益，还望海涵，并联系博主删除。

3）转载请注明出处。

4）本文主要用来记录本人初学Tensorflow时遇到的问题，特此记录下来，因此并不是所有的方法都会面面俱到。希望后人看到可以引以为鉴，避免走弯路。同时毕竟水平有限，希望有饱含学识之士看到其中的问题之后，可以悉心指出，本人感激不尽。

------

<br />

<br />

<br />

# 一. 理解RNN

# 1. 参考博客

理解LSTM：

英文：http://colah.github.io/posts/2015-08-Understanding-LSTMs/

中文：http://www.jianshu.com/p/9dc9f41f0b29

tensorflow：

http://www.tensorfly.cn/tfdoc/tutorials/recurrent.html

# 2. RNN理解

​	RNN本质上是与序列和列表相关的，其关键点之一就是他们可以用来连接先前的信息到当前的任务上.

所有 RNN 都具有一种重复神经网络模块的链式的形式。在标准的 RNN 中，这个重复的模块只有一个非常简单的结构，例如一个 tanh 层。

![](http://orkjdoapd.bkt.clouddn.com/Seya-Tensorflow-Study-Note/6.1%20%E6%A0%87%E5%87%86%20RNN%20%E4%B8%AD%E7%9A%84%E9%87%8D%E5%A4%8D%E6%A8%A1%E5%9D%97%E5%8C%85%E5%90%AB%E5%8D%95%E4%B8%80%E7%9A%84%E5%B1%82.png)

# 3. LSTM理解

​	RNN成功应用的关键之处就是 LSTM 的使用，可以将几乎所有的令人振奋的关于 RNN 的结果都是通过 LSTM 达到的．

### 1) 主要结构

LSTM(long short term memory network/长短时记忆网络)与RNN同样的结构，但是重复的模块拥有一个不同的结构。不同于 单一神经网络层，这里是有四个，以一种非常特殊的方式进行交互。

![](http://orkjdoapd.bkt.clouddn.com/Seya-Tensorflow-Study-Note/6.2%20LSTM%20%E4%B8%AD%E7%9A%84%E9%87%8D%E5%A4%8D%E6%A8%A1%E5%9D%97%E5%8C%85%E5%90%AB%E5%9B%9B%E4%B8%AA%E4%BA%A4%E4%BA%92%E7%9A%84%E5%B1%82.png)

​	其中，具体的为

![](http://orkjdoapd.bkt.clouddn.com/Seya-Tensorflow-Study-Note/6.3%20LSTM%E4%B8%AD%E7%9A%84%E5%9B%BE%E6%A0%87.png)

上述粉色点，是一种点乘/点加/点tanh操作．

黄色的是神经网络层，包括了 weight/bais/转移函数（sigmoid/tanh等）

### 2) 门结构

＇门＇结构：信息选择通过的方式，判断有多少量通过，包括一个sigmoid层和点乘操作．sigmoid输出在０到１，０代表不允许任何量通过

![](http://orkjdoapd.bkt.clouddn.com/Seya-Tensorflow-Study-Note/6.4%20%E9%97%A8%E7%BB%93%E6%9E%84.png)

### 3) 结构详解

第一步，决定细胞丢弃信息，因此用了一个遗忘门结构．（如果碰到了新的主语，就遗忘掉之前的主语）.sigmoid门输出值**乘以**上一阶段细胞值C{t-1}



第二步，决定什么新信息放入细胞，因此用了这么一个结构，sigmoid层用来决定什么值我们需要更新．tanh层用来产生新的候选值向量．然后将筛选后的新信息加到舍弃信息的细胞上．



最后一步，决定什么信息输出．首先将细胞状态通过tanh进行处理（得到一个-1到1之间），并且乘上一个sigmoid层用来确定将细胞状态的哪个部分输出出去．

## 4. 高级技巧

### 1) 多层LSTM层堆叠

要想给模型更强的表达能力，可以添加多层 LSTM 来处理数据。第一层的输出作为第二层的输入，以此类推。堆叠LSTM可以拟合更大的模型复杂性

水平层堆叠的是时间轴, 垂直轴堆叠的是这个多层.

类 `MultiRNNCell` 可以无缝的将其实现：

```python
lstm = rnn_cell.BasicLSTMCell(lstm_size)
stacked_lstm = rnn_cell.MultiRNNCell([lstm] * number_of_layers)

initial_state = state = stacked_lstm.zero_state(batch_size, tf.float32)
for i in range(len(num_steps)):
    # 每次处理一批词语后更新状态值.
    output, state = stacked_lstm(words[:, i], state)

    # 其余的代码.
    # ...

final_state = state
```

### 2) GRU理解

![](http://orkjdoapd.bkt.clouddn.com/Seya-Tensorflow-Study-Note/6.5%20GRU.png)

<br />

# 二. Tensorflow源码

model中源码与简略源码

源码目录：models-master/tutorials/rnn/ptb/ptb_word_lm.py

## 1. 数据集

地址：http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

该数据集已经预先处理过并且包含了全部的 10000 个不同的词语，其中包括语句结束标记符，以及标记稀有词语的特殊符号 `(<unk>)` 。

我们在 `reader.py` 中转换所有的词语，让他们各自有唯一的整型标识符，便于神经网络处理。

## 2. 运行

```shell
$ python ptb_word_lm.py --data_path=simple-examples/data/ --model small
```

程代码中有 3 个支持的模型配置参数："small"， "medium" 和 "large"。它们指的是 LSTM 的大小，以及用于训练的超参数集。

模型越大，得到的结果应该更好。在测试集中 `small` 模型应该可以达到低于 120 的困惑度（perplexity），`large` 模型则是低于 80，但它可能花费数小时来训练。

## 3.　代码详解

### 1) 参数理解

```python
init_scale = 0.1        # 相关参数的初始值为随机均匀分布，范围是[-init_scale,+init_scale]
learning_rate = 1.0     # 学习速率,在文本循环次数超过max_epoch以后会逐渐降低
max_grad_norm = 5       # 用于控制梯度膨胀，如果梯度向量的L2模超过max_grad_norm，则等比例缩小
num_layers = 2          # 多层lstm在表达上更好,只需将第一层输出当做第二层输入便可
num_steps = 20          # lstm展开后的时步数
hidden_size = 200       # 隐藏层中单元数目, 啥意思, 这个同样也是词向量的向量长度
max_epoch = 4           # epoch<max_epoch时，lr_decay值=1,epoch>max_epoch时,lr_decay逐渐减小
max_max_epoch = 13      # 指的是整个文本循环次数。
keep_prob = 1.0         # 用于dropout.每批数据输入时神经网络中的每个单元会以1-keep_prob的概率不工作，可以防止过拟合
lr_decay = 0.5          # 学习速率衰减
batch_size = 20         # 每批数据的规模，每批有20个。
vocab_size = 10000      # 词典规模，总共10K个词
```

"multi-layer" means stacking the LSTM units;

"memory cells" means hidden units.

## 2) 数据处理



### 2) 创建变量部分

用了一个技巧，创建了一个initializer，这个创建一个(0.1,0.1)的高斯分布，然后zhi