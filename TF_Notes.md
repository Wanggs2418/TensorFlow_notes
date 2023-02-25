# TF_Notes

## 1.深度学习框架简介

### 1.1 Tensorflow

- Theano 是最早的学习框架之一，整体基于 Python 开发，支持 GPU 和 CPU 运算，但开发效率低、模型编译时间长，目前已停止维护。
- Keras 是基于 Theano 和 TensorFlow 等框架提供的底层运算而实现的高层框架，提供了大量快速训练、测试网络的高层接口。开发效率高，但缺少底层的实现，运行效率不高，灵活性一般。

Keras 可以看作一套高层 API 的设计规范，Keras 本身对这套规范有官方实现，TF 也实现了这套规范，称为 tf.keras 模块，并将 tf.keras 作为 TF 2.x 版本的唯一高层接口，避免出现接口重复冗余的问题。

TensorFlow 1.x 版本由 Google 于 2015 年发布，但开发调试困难，一度被业界诟病。直到 2019 年，Google 正式推出 TensorFlow 2.x 版本，避免了 TensorFlow 1.x 多种缺陷，采用动态图优先模式运行。但注意的是，TF 2.x 并不兼容 TF 1.x，二者在编程风格、函数接口设计等方面有很大的不同。

TF 2.x 采用**动态图(优先)模式开发**，调试方便，所见即所得。开发效率高，但运行效率低于静态图模式。TF 2.x 也支持通过 tf.function 将动态图优先模式的代码转化为静态图模式，同时取得较高的开发和运行效率。

**TF 框架的三大核心功能**

1. 加速计算：利用 GPU 加速
2. 自动梯度：无需手动编写导数的表达式，TF 可由函数的表达式自动求导
3. 常用神经网络接口：内建了常用神经网络计算、常用网络层、网络训练、模型保存于加载、网络部署等一系列深度学习系统的便捷功能。

### 1.2 Pytorch

- Torch 是一个优秀的科学计算库，基于 Lua 语言开发，灵活性高，容易实现自定网络层。

- Caffe 由贾扬清于 2013 年开发，主要面向卷积神经网络 (CNN) 的应用场合。基于 C++开发，支持 GPU 和 CPU，也提供了 Python 等语言的接口。

而 Pytorch 是 Facebook 基于 Torch 框架采用 Python 语言的深度学习框架。其借鉴了 Chainer 设计风格，采用命令式编程，使得搭接和调试网络非常方便。

2017 年正式发布，但由于 Pytorch 精良紧凑的接口设计，其在学术界获得了广泛好评。Pytorch 1.0 版本后，原来的 Pytorch 与 Caffe2 进行了合并，弥补了其咋工业部署方面的不足。

## 2.TF 基础

### 2.1 数据类型和数值精度

内部数据保存在张量上 (Tensor)，所有的操作 (Operation,OP) 也是基于张量对象进行的。

TF 中的数据类型包括：数值类型、字符串类型、布尔类型。

**数值类型** 

根据维度划分：

- 标量 (Scalar)：单个实数，维度为 0，shape=[ ]
- 向量 (Vector)：中括号包裹的 n 个数的有序集合，维度 1，shape = [n, ]
- 矩阵 (Matrix)：n 行 m 列实数的有序集合，维度 2，shape = [n, m]
- 张量 (Tensor)：dim > 2 的数组，维度/轴一般代表具体的物理意义

```python
#标量
a = tf.constant(1.2)
#向量
a = tf.constant([1.2])
a = tf.constant([1,2,3])
#字符串
a = tf.constant('TensorFlow hub')
```

**字符串类型**

`tf.string.lower()`，常见的字符串类型工具函数：`length(),split()` 等。

**布尔类型**

注意的是 TF 的布尔类型与 Python 的布尔类型不等价：

```python
a = tf.constant(True)
```

**数值精度**

一般使用 `tf.int32,tf.float32` 可满足。可在定义时指定精度类型，不满足则通过 `tf.cast` 转换：

```python
a= tf.constant(np.pi, dtype=tf.float16)
tf.cast(a, tf.double)
```

### 2.2 待优化张量

专门的数据类型来支持梯度信息的记录：`tf.Variable`。其在普通张量的基础上添加了 name，trainable 等属性。由于梯度运算消耗大量的计算资源，故需要通过 `tf.Variable` 包裹以便 TF 跟踪相关的梯度信息。

创建 Variable 对象默认启用优化标志，也可设置 trainable=False 来设置不需要优化。

### 2.3 创建张量

`tf.constant()` 和 `tf.convert_to_tensor()` 都可以自动将 Numpy 数组或者 Python 列表数据转化为 Tensor 类型，使用其一即可。

```python
tf.convert_to_tensor(np.array([[1, 2.],[3, 4]]))
```

- `tf.zeros()` 和 `tf.ones()`：创建全 0 或全 1张量
-  `tf.zeros_like(a)` 和 `tf.ones_like()` 可以很方便地创建于某个张量 shape 一致，当然也可通过 `tf.zeros(a.shape)` 实现

- `tf.fill(shape, value)`：创建自定义数值地张量
- `tf.random.normal(shape, mean=,stddev=)`：创建正态分布 $N(mean,stddev^2)$
- `tf.random.uniform(shape,minval=,maxval=,dtype=tf.float32)`：创建采样在 [minval, maxval) 区间地均匀分布张量
- `tf.range(limit, delta=)`：创建在 [0, limit) 之间，步长为 delta 地整型序列

### 2.4 张量的典型应用

**标量**

简单一个数字，维度 0，shape = [ ]。典型的用途是误差值的表示、各种测量指标的表示，如准确度 (Accuracy, acc)，精度 (Precision)和召回率 (Recall)等。

**向量**

如偏置向量 $b$，类型为 Variable，因为 $W,b$ 都是待优化参数。

**矩阵**

如**全连接层的**批量输入张量 $X$ 的形状：$[b, d_{in}]$，$b$ 表示输入样本数— Batch Size，$d_{in}$ 表示输入特征长度。

**三维张量**
$$
X=[b,\quad sequencelen,\quad feature len]
$$

- b：序列信号的数量；

- sequence len：序列信号在时间维度上的采样点或步数；
- feature len：表示每个点的特征长度；

在自然语言处理 (NLP) 的情感分类网络中，一般将单词通过嵌入层编码为固定长度的向量，如 "a" 编码为某个长度为 3 个向量，则等长的句子序列可表示为 shape= [2, 5, 3] 的形式。2 表示句子个数，5 表示单词数量，3 表示单词长度。

**四维张量**

用于保存特征图 (Feature maps) 数据，

一般定义为：$[b, h, w, c]$— 样本数，高，宽，通道数

### 2.5 索引与切片

**索引-支持两种方式**

```python
#第2张图片，第10行，第3列的数据
x = tf.random.normal([4, 32, 32, 3])
x[1, 9, 1]
x[1][9][1]
```

**切片**

`start:end:step`，[start, end)

```python
#以shape=[4,32,32,3]为例
#读取第一张图片的所有行
x[0,::] #与 x[0] 等价
```

<img src="image/01.jpg" style="zoom:80%;" />

### 2.6 维度变换

**改变视图-reshape**

同一存储，在不同的角度下观察数据，可以从产生不同的视图，视图的产生是非常灵活的。在存储数据，内存都是以平铺的方式写入，维度的概念只是人为规定的为了管理方便。

<img src="image/02.jpg" style="zoom:80%;" />

可通过 `x.ndim` 或 `x.shape` 获取张量的维度和形状列表。

在保证存储维度顺序性的前提下，可通过 `tf.reshape(x, new_shape)` 进行维度变换。

```python
#参数为-1表示当前轴长度需要张量总元素不变的法则自动推导
tf.reshape(x, [2,-1])
```

**增加维度**

增添长度为 1 的维度：数据不需改变，改变的只是理解方式。

如 shape=[28,28] 的灰度图片，末尾增加 1 维度，shape=[28,28,1]

`tf.expand_dims(x,axis)` 在指定的 axis 轴前插入**一个**新的维度

```python
x = [28,28]
x = tf.expand_dims(x,axis=2)
#x.shape=[28,28,1]

#x.shape=[1,28,28]
x = tf.expand_dims(x, axis=0)
```

<img src="image/03.jpg" style="zoom:80%;" />

**删除维度**

只能删除长度为 1 的维度，不会改变张量的存储。`tf.squeeze(x, axis)`

```python
#x.shape=[1,28,28]
x = tf.squeeze(x, axis=0)

#不指定维度参数 axis，默认删除所有长度为 1 的维度
x = tf.squeeze(x)
```

**交换维度**

调整张量的存储顺序，改变了张量的视图。图片默认的格式为：`[b,h,w,c]`，但部分库图片的格式为通道先行模式：`[b,c,h,w]`，此时就需要完成维度交换 (Transpose)。

`tf.transpose(x,perm)` 函数完成维度交换操作，参数 `perm` 表示新维度的顺序 

```python
#原来的维度索引axis=[0,1,2,3]
#从[b,h,w,c]->[b,c,h,w]
x =tf.transpose(x, perm=[0,3,1,2]) #交换维度
```

需要注意的是，一定那维度交换完成后，则存储顺序完全改变，后续的操作都是基于交换后的顺序进行。

**复制数据**

通过 `tf.tile(x,multiples)` 函数完成数据在指定维度上的复制操作，mutiples 分别指定每个维度上的复制倍数，对应位置为 1 表示不复制，为 2 表示是原来长度的 2 倍。

```python
#在axis=0 维度复制1次；在 axis=1维度不复制
b= tf.constant([1,2])
tf.tile(b,multiples=[2,1])
```

`tf.tile()` 会创建一个新的张量来保存复制后的张量，复制操作涉及到大量的 IO 运算，计算代价相对较高。更推荐的是广播机制 (Broadcasting)。

**广播机制 (Broadcasting)**

一种**轻量级**的张量复制方法，Broadcasting 不会立刻复制数据，会在逻辑上改变张量的形状，使得视图变为复制后的形状，并通过深度学习框架的优化手段避免实际复制数据而完成逻辑运算。

尽管与 `tf.tile()` 的最终效果一样，但是 `Broadcasting` 机制节省了大量的计算机资源，提升了效率。

Broadcasting 只针对最常见的场景自动完成增加维度并复制数据，因为其机制的核心思想是普适性的—长度为 1 的维度。

### 2.7 数学运算

运算符号已经重载，可直接使用。

整除：`\\`；余除：`%`。

```python
a**(x)
tf.pow(a,x)
tf.square(a)#平方
tf.sqrt(a)#开方
tf.exp(a)#e^x
tf.math.log(a)#log_e(a)
```

**矩阵相乘**

通过 `@` 符号相乘；通过函数 `tf.matmul(a,b)` 实现。

注意：当张量的维度大于 2 时，TF 会选择最后两个维度的矩阵相乘，前面的维度都视作 Batch 维度。

















