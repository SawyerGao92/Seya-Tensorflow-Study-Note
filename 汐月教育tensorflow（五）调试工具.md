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