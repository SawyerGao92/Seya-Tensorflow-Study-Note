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

# 一. 为什么要有词向量？

对于物体或语音识别这一类的任务，我们所需的全部信息已经都存储在原始数据（图片中所有单个原始像素点强度值或者音频中功率谱密度的强度值）中（显然人类本身就是依赖原始数据进行日常的物体或语音识别的）。

然而，自然语言处理系统通常将词汇作为离散的单一符号，例如 "cat" 一词或可表示为 `Id537` ，而 "dog" 一词或可表示为 `Id143`。这些符号编码毫无规律，无法提供不同词汇之间可能存在的关联信息。

可见，将词汇表达为上述的独立离散符号将进一步导致数据稀疏，使我们在训练统计模型时不得不寻求更多的数据。而词汇的向量表示将克服上述的难题。

<br />

# 二. Gensim-vord2vec

## 1. 简介

Google 2013年 将词表征为实数值向量的工具

应用于NLP 相关的工作：比如聚类、找同义词、词性分析等等。

 把词当做特征，Word2vec把特征映射到 K维向量空间，可以为文本数据寻求更加深层次的特征表示 。

## 2. 安装代码：

​        `sudo/usr/local/anaconda2/bin/pip install -U gensim`

* 安装问题1：

  安装完gensim之后，出现“Intel MKL FATAL ERROR:     Cannot load libmkl_avx2.so or libmkl_def.so”

* 解决：

  输入这两条命令：

```shell
condainstall nomkl numpy scipy scikit-learn numexpr
condaremove mkl mkl-service
```

参考 <[http://blog.csdn.net/u010335339/article/details/51501246](http://blog.csdn.net/u010335339/article/details/51501246)> 

* 安装问题2：

  ​输入的时候缺少权限，然而conda没法用sudo，你得先把那个路径取得所有权：

  ​`sudo chmod -R 777 /usr/local/anaconda2`

## 3. 参考博客

[http://radimrehurek.com/gensim/tutorial.html](http://radimrehurek.com/gensim/tutorial.html)

[https://rare-technologies.com/word2vec-tutorial/#word2vec_tutorial](https://rare-technologies.com/word2vec-tutorial/#word2vec_tutorial)

## 4. 函数介绍

1. Word2Vec(sentences=None, size=100,      alpha=0.025, window=5, min_count=5, max_vocab_size=None, sample=0.001,      seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1,      hashfxn=<built-in function hash>, iter=5, null_word=0,      trim_rule=None, sorted_vocab=1, batch_words=10000)

<br />

# 三. 语言模型

参考博客：[http://blog.csdn.net/itplus/article/details/37969519](http://blog.csdn.net/itplus/article/details/37969519)

## 1.统计语言模型

用来计算一个**句子**的概率大小的模型。

![暂时无法显示](http://orkjdoapd.bkt.clouddn.com/Seya-Tensorflow-Study-Note/4.1%20%E7%BB%9F%E8%AE%A1%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.png)

​					必须说明一下，![](https://cl.ly/0z1y1R2p3f0b/w12.png])是从word1到word2的意思

举例来说，计算“我很喜欢詹姆斯”这句话的出现概率，需要用...

* 存在问题：每句话越到后面计算量越大

## 2. N-Gram 统计语言模型

每个单词就考虑前面n个推到本单词的概率，实际应用中最多就取到3。

## 3. 构造目标函数

![](http://orkjdoapd.bkt.clouddn.com/Seya-Tensorflow-Study-Note/4.2%20%E7%9B%AE%E6%A0%87%E5%87%BD%E6%95%B0.png)

其中C为语料库，它不是字典，语料库是一段很长的话。然后针对其中每一个字，从头开始，滑动窗，然后让每一个字儿的 都从他们周围推导到这个字儿。

## 4. 神经概率模型（softmax）

训练样本：任意一个词，和前面n-1个词

在神经网络中，输入：（n-1）*m个神经元【m是每个词向量的长度】；输出：语料C的词汇量大小个神经元 
【输出(0,0,0,1,0,0)这种】 .

## 5. Hierarchical Softmax 模型

## 6. Negative Sampling 方法

### (1) NS和神经概率模型的区别

这是神经网络模型，输出的部分其实相当于计算字典中每一个单词的概率，由于字典内容很多，因此softmax层的工作量很大。

![](http://orkjdoapd.bkt.clouddn.com/Seya-Tensorflow-Study-Note/4.3%20softmax-nplm.png)

下面是负采样，简而言之，目的是将输出层的节点减少，而不是多一些训练样本，虽然确实多了一些训练样本。

![](http://orkjdoapd.bkt.clouddn.com/Seya-Tensorflow-Study-Note/4.4%20nce-nplm.png)

原本的神经网络中，从隐含层到输出层是一个U，q的权值矩阵，同时输出多个元素，然后softmax。而负采样其实是**将v输出神经网络 转换成 k个单输出的神经网络**，而且只有一层，不像原来两层。下图是我重新画的：

![](http://orkjdoapd.bkt.clouddn.com/Seya-Tensorflow-Study-Note/4.5%20skip-gram.png)

### (2) 学习过程

我们想最大化 

![](http://orkjdoapd.bkt.clouddn.com/Seya-Tensorflow-Study-Note/4.6%20nce%20target.gif)

这个目标函数最大化语料库中的每一个训练样本与其负采样的几个样本，注意，原来的时候只是p(w|Context(w)，现在w变成u，而u既包括w又包括负采样的。

然后上式中间部分可以变成：

![](http://orkjdoapd.bkt.clouddn.com/Seya-Tensorflow-Study-Note/4.7%20.png)

理解一下就是还是softmax，就跟我画的图似的最左边是跟1的误差，其他都是根0的误差，然后求他们的乘积。

### (3) 算法

![](http://orkjdoapd.bkt.clouddn.com/Seya-Tensorflow-Study-Note/4.8%20CROW%E7%AE%97%E6%B3%95.png)

解读一下，首先e是一个参数在第一行，然后第三步的循环用来更新权重，因为每一个神经网络都有一个权重，所以每次神经网络的循环都要更新一次权重参数θ，然后后面的加和向量只有一个，因此所有的神经网络之后才更新一次。

对于反过来的skip-gram算法那就是：

![](http://orkjdoapd.bkt.clouddn.com/Seya-Tensorflow-Study-Note/4.9%20skip-gram%E7%AE%97%E6%B3%95.png)

# 四.  Tensorflow-nce_loss解析

tf.nn.nce_loss: 计算和返回噪声对比估计训练损失.

