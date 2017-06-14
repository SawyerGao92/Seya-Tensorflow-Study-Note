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

Tensorflow中自带示例的MNIST预测源代码位于/tensorflow/examples/tutorials/mnist/fully_connected_feed.py文件中，本文的主要内容就是详解此代码中的内容。关于

此代码和/tensorflow/Models/image/mnist/convolutional.py 可以看出来都是解决MNSIT识别，但是是出自两个人之手，两个人代码的都有借鉴价值。

examples：纯为了官网上的教程写的示例代码，就是不同人以不同风格写的

models：比较靠谱的，固定到代码中的代码结构。

目前针对于MNIST这个代码，我感觉example中的那个人写的更好点呢。

本文看的代码在源代码中的“源代码tensorflow目录/examples/tutorials/mnist/fully_connected_feed”



# 一. 概览

利用了tensorflow目录下/examples里的input_data和mnist文件；

然后我记得在0.8.0时是使用gflags解析命令行的，在1.0.0中怎么变成argparse了？  gflags是google的一个开源的处理命令行参数的库，比getopt简单的多，建议不会的去看看；

然后接下来是四个函数：placeholder_inputs、placeholder_inputs、do_eval、run_training、最下面是主函数。

<br />

# 二. run_training函数详解 

run_training算是这个代码中的main函数了，那么就好好看看具体的步骤是怎样。

## 1. 导入数据

竟然调用了examples中的代码来导入数据，我说example中别的都删了怎么就mnist的不删呢，原来model里的作者懒得再写一遍了吧就...最后返回的是个字典

·data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)  ·

## 2. 模型图将会建立到默认图中

tf.Graph这个函数就是一个系统中已经默认弄好的图，我们创建图就在这个图基础上增加点东西就好。然后建立session的时候不传入图的参数，那就是默认用这个默认图；如果想做多个图，也可以自己在加图，之后会介绍如何做。

`with tf.Graph().as_default():  `

## 3. 创建输入、输出节点占位符（外部输入）

`images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)  `

## 4. 构建图

构件图的四个部分，本代码将他们放到三个函数中了，跟我上一节分类分的差不多。分别是，inference函数：构建图的输出公式；loss：选择损失和训练参数；training：优化算法和步长选择设置；evaluation：评估模型结果指标。接下来会分别详细讲这三个函数，这里暂略。

evaluation函数调用tf.nn.in_top_k函数：如果在K个最有可能的预测中可以发现真的标签，那么这个操作就会将模型输出标记为正确。在本文中，我们把K的值``设置为1，也就是只有在预测是真的标签时，才判定它是正确的。

```python
# 构建图的输出公式
logits = mnist.inference(images_placeholder,FLAGS.hidden1,FLAGS.hidden2)  

# 选择损失和训练参数
loss = mnist.loss(logits, labels_placeholder)  

# 优化算法和步长选择设置  
train_op = mnist.training(loss, FLAGS.learning_rate)     

# 评估模型结果指标  
eval_correct = mnist.evaluation(logits, labels_placeholder)  
```

## 5. 运行图：

### 1）初始化

​	上一节中采用sess.run(tf.initialize_all_variables()); 这里愣是让拆成了init =tf.initialize_all_variables() 和 sess.run(init) 还离着那么远，不知道有何用，看着就麻烦。

merge_all_summaries函数： 为了把图可视化，需要把中间数据和都放入一个记录发生了什么事件的文件中，简称事件文件（event file）。

+SummaryWriter函数：之前的相当于创建了个存事件的记事本节点，这个函数就是创建一个笔，向summary这个记事本中写入事件。

```python
summary = tf.merge_all_summaries()          #把发生的事件记录下来  
init = tf.initialize_all_variables()        #初始化变量    
saver = tf.train.Saver()                    #建立一个保存训练中间数据的存档点  
sess = tf.Session()                         #建立session会话   也可以 with tf.Session() as sess; 创建session时未传入参数，所以默认调用我们的默认图了。  

summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)  
# And then after everything is built:  
# Run the Op to initialize the variables.  
sess.run(init)  
```

### 2）包括 循环训练：

````python
for step in xrange(FLAGS.max_steps):  
  start_time = time.time()     #计时  
  # 将占位符填充上我们的训练数据  
  feed_dict = fill_feed_dict(data_sets.train,images_placeholder,labels_placeholder)  
  _, loss_value = sess.run([train_op, loss],feed_dict=feed_dict)  
  duration = time.time() - start_time   #计时  
````

## 6. 输出

### 1) 输出运行状态，每训练100次输出一下状态，然后将状态记录到事件

```python
# 都属于上面的循环，每一百行输出一个结果  
if step % 100 == 0:  
  # 输出状态  
  print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))  
  # 更新事件summary这个记事本  
  summary_str = sess.run(summary, feed_dict=feed_dict)  
  summary_writer.add_summary(summary_str, step)  
  summary_writer.flush()  
```

### 2) 将中间权重和偏置都保存到中间检查点，以后可以从这里再开始训练

save函数之后，以后可以通过restore函数重载模型的参数，继续从这里开始训练。

```python
if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
   checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')  
   saver.save(sess, checkpoint_file, global_step=step)  
```

### 3) 需要定期估计一下模型，每隔一千个训练步骤，使用`do_eval`函数三次，分别使用训练数据集、验证数据集合测试数据集看看准确率。其实应该不用对test数据集进行评估，你训练完了再评估啊！用验证集合来停止训练就够了。

do_eval函数：传入了评价标准eval_correct，具体内容可以自己看

```python
# Evaluate against the training set.  
print('Training Data Eval:')  
do_eval(sess,eval_correct,images_placeholder,labels_placeholder, data_sets.train)  
# Evaluate against the validation set.  
print('Validation Data Eval:')  
do_eval(sess,eval_correct,images_placeholder,labels_placeholder,data_sets.validation)  
# Evaluate against the test set.  
print('Test Data Eval:')  
do_eval(sess,eval_correct,images_placeholder,labels_placeholder,data_sets.test)  
```

<br />

# 三. 构件图三个函数详解

接下来这里主要介绍“二.4”中构建图的三个主要自建函数：inference( ) loss( ) 和 training( )， 建议跟着上一节课一起参照着看，其实差不多啊明明。

## 1）inference( ) : 构建图输出函数。输入：输入占位符

name_scope函数：Returns a context manager that creates hierarchical names for operations.似乎的意思就是有一个记录器，把你所有创立的节点名字都记录下来，回头也好画流程图是吧。当这些层是在`hidden1`作用域下生成时，赋予权重变量的独特名称将会是"`hidden1/weights`"。

`truncated_normal`函数：初始化权重变量,将根据所得到的均值和标准差，生成一个随机分布。输入参数：[connect from的单元数量，connect to单元数量]。

`tf.zeros`函数：初始化偏差变量（biases），确保所有偏差的起始值都是0，输入参数：connect to的单元数量。

这里只拿出第一层来举例说明：

```python
# Hidden 1  
with tf.name_scope('hidden1'):<span style="white-space:pre">              </span>             
	weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units], stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),name='weights')  
	biases = tf.Variable(tf.zeros([hidden1_units]),name='biases')  
	hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
```

## 2）loss( )：损失函数选择

`softmax_cross_entropy_with_logits`函数：用来比较inference()函数与1-hot标签所输出的logits Tensor

`reduce_mean`函数：计算batch维度下交叉熵的平均值，将将该值作为总损失。

```python
labels = tf.to_int64(labels)  
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')  
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')  

```

## 3）training( )：优化算法和步长选择设置

`scalar_summary函数+summarywriter函数`：向事件中加入汇总值，感觉就是记录的作用。

train.GradientDescentOptimizer函数+optimizer.minimize函数：选择梯度下降的步长和最小化偏差，上一集中将这两个函数整合成一个了，是一个功效。

```python

```

