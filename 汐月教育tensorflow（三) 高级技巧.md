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

# 一. 数据读取

http://www.tensorfly.cn/tfdoc/how_tos/reading_data.html

- 供给数据(Feeding)： 在TensorFlow程序运行的每一步， 让Python代码来供给数据。
- 从文件读取数据： 在TensorFlow图的起始， 让一个输入管线从文件中读取数据。
- 预加载数据： 在TensorFlow图中定义常量或变量来保存所有数据(仅适用于数据量比较小的情况)。

# 二. 多线程

RandomShuffleQueue: 堆栈

- 多个线程准备训练样本，并且把这些样本推入队列。
- 一个训练线程执行一个训练操作，此操作会从队列中移除最小批次的样本（mini-batches)。

函数：

## １. Coordinator类：

Coordinator类用来帮助多个线程协同工作，多个线程同步终止。 其主要方法有：

- `should_stop()`:如果线程应该停止则返回True。
- `request_stop(<exception>)`: 请求该线程停止。
- `join(<list of threads>)`:等待被指定的线程终止。

首先创建一个`Coordinator`对象，然后建立一些使用`Coordinator`对象的线程。这些线程通常一直循环运行，一直到`should_stop()`返回True时停止。 任何线程都可以决定计算什么时候应该停止。它只需要调用`request_stop()`，同时其他线程的`should_stop()`将会返回`True`，然后都停下来。

```python
# 线程体：循环执行，直到`Coordinator`收到了停止请求。
# 如果某些条件为真，请求`Coordinator`去停止其他线程。
def MyLoop(coord):
  while not coord.should_stop():
    ...do something...
    if ...some condition...:   # 类似于条件中断
      coord.request_stop()

# 主代码: 创建一个 coordinator.
coord = Coordinator()

# 创建十个线程运行 'MyLoop()'
threads = [threading.Thread(target=MyLoop, args=(coord)) for i in xrange(10)]

# 开始线程,轮流每个线程开始,然后等待他们结束
for t in threads: 
	t.start()
coord.join(threads)
```

显然，Coordinator可以管理线程去做不同的事情。上面的代码只是一个简单的例子，在设计实现的时候不必完全照搬。Coordinator还支持捕捉和报告异常, 具体可以参考[Coordinator class](http://www.tensorfly.cn/tfdoc/api_docs/python/train.html#Coordinator)的文档。

## 2. QueueRunner类

RandomShuffleQueue: 队列, 作为模型的输入

- 多个线程准备训练样本，并且把这些样本推入队列。
- 一个训练线程执行一个训练操作，此操作会从队列中移除最小批次的样本（mini-batches)。

例子过程:

### 1)　建立一个TensorFlow图,  使用队列来输入样本。

区别: 使用队列来输入样本

* 增加了处理样本并将样本推入队列
* 增加了training操作来移除队列中的样本

```python
example = ...可以创建训练样本的ｏｐ，也就是输入数据...
# 创建一个队列，和一个可以将example进行入队列的ｏｐ操作
queue = tf.RandomShuffleQueue(...)
enqueue_op = queue.enqueue(example)
# 创建一个训练图，从队列每次出栈mini-batch当作输入数据
inputs = queue.dequeue_many(batch_size)
train_op = ...用 'inputs'  建立图（训练部分）...
```

### 2)　QueueRunner运行多个线程，Coordinator控制线程停止 

在Python的训练程序中，创建一个`QueueRunner`来运行几个线程， 这几个线程处理样本，并且将样本推入队列。创建一个`Coordinator`，让queue runner使用`Coordinator`来启动这些线程，创建一个训练的循环， 并且使用`Coordinator`来控制`QueueRunner`的线程们的终止。

```python
# 创建一个queue runner同时运行四个线程用来并行进栈
qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)

# 运行图
sess = tf.Session()

# 创建一个coordinator, launch the queue runner threads.
coord = tf.train.Coordinator()
enqueue_threads = qr.create_threads(sess, coord=coord, start=True)


# 这里相当于前一部分每一个线程（训练循环）何时使用coord停止训练的部分
for step in xrange(1000000):
    if coord.should_stop():
        break
    sess.run(train_op)
# When done, ask the threads to stop.
coord.request_stop()

# 用协调器停止线程
coord.join(threads)
```

