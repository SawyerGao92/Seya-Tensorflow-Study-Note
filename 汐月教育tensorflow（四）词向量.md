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

​	输入的时候缺少权限，然而conda没法用sudo，你得先把那个路径取得所有权：

​	`sudo chmod -R 777 /usr/local/anaconda2`

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

![暂时无法显示](https://cl.ly/171R1C1q3V45/%E7%BB%9F%E8%AE%A1%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.png)

​					必须说明一下，![](https://cl.ly/0z1y1R2p3f0b/w12.png])是从word1到word2的意思

举例来说，计算“我很喜欢詹姆斯”这句话的出现概率，需要用...

* 存在问题：每句话越到后面计算量越大

## 2. N-Gram 统计语言模型

每个单词就考虑前面n个推到本单词的概率，实际应用中最多就取到3。

## 3. 构造目标函数

