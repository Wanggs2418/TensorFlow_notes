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

## 3.TF 进阶

### 3.1 合并与分割

**合并**

通过拼接 (Concatenate) 和堆叠 (Stack) 的实现。

 `tf.concat(tensor, axis)` 函数拼接张量，不会产生新维度，仅在现有的维度上进行合并。

理论上，拼接合并操作可在任意维度上进行，唯一的约束都会非合并维度的长度必须一致。

`tf.stack(tensor, axis)`，当 axis ≥ 0 时，在 axis 之前插入；axis < 0 时，在 axis 之后插入。

```python
#拼接concat
tf.concat([a,b], axis=0)
#堆叠stack
a =tf.random.normal([35,8])
b =tf.random.normal([35,8])
tf.stack([a,b], axis=0)
#结果为 [2,35,8]
```

**分割**

`tf.split(tensor, num_or_size_splits, axis)`

- `num_or_size_splits` 为单个数值时，表示等长分割的份数；为 List 表示每份的长度
- `axis` 指定分割的维度索引号

```python
#x.shape=[10, 35, 8]
#在第 0 维度上分割成 4 份
result = tf.split(x, num_or_size_splits=[4,2,2,2], axis=0)
#len(reslut)=10
result[0].shape=[4,35,8]
result[1].shape=[2,35,8]
result[2].shape=[2,35,8]
result[3].shape=[2,35,8]
```

维度按长度为 1 的方式分割：

`tf.unstack(x, axis)`，切割长度固定为 1，只需要指定 axis 即可。

### 3.2 数据统计

**向量范数**

Vector Norm，表征向量长度。TF 中用来表示张量的权值大小、梯度大小。

TF 中，可以通过 `tf.norm(x,ord)` 求解张量的 $L_1,L_2,\infty$ 等范数

- $L_1$ 范数，向量 $x$ 的所有元素绝对值之和 $\displaystyle ||x||_1=\sum_i|x_i|$
- $L_2$ 范数，向量 $x$ 的所有元素平方和开根号 $||x||_2=\displaystyle \sqrt{\sum_i|x_i|^2}$
- $\infty$ 范数，向量 $x$ 所有元素绝对值的最大值 $\displaystyle ||x||_{\infty}=\max_i(|x_i|)$

```python
tf.norm(x,ord=np.inf)
tf.norm(x,ord=1)
```

**最值、均值、和**

`tf.reduce_max,tf.reduce_mean,tf.reduce_sum`

不指定 axis 的时候，会自动求解出全局元素相关的值。

`tf.argmax(x,axis)` 和 `tf.argmin(a,axis)` 求解 axis 轴上，x 最大子、最小值所在的索引号。

**张量比较**

`tf.equal(a,b)` 或 `tf.math.equal(a,b)` 比较 2 个张量是否相等。

### 3.3 填充和复制

**填充—padding**

`tf.pad(x,paddings)`

```python
x=tf.random.normal([4,28,28,1])
#填充长宽,上下左右填充2个单元
tf.pad(x,[[0,0],[2,2],[2,2],[0,0]])

#复制,图片数复制一份，高宽复制一份，通道深度不管 
tf.tile(x,[2,3,3,1])
```

**复制**

`tf.tile(x,multiples)`

**限幅**

- `tf.maximum(x,a)`，即 $x \in [a, +\infty)$
- `tf.minimum(x,a)`，即 $x \in (-\infty,a]$

### 3.4 高级操作

1.**`tf.gather(x,index_list,axis)`** ：实现根据索引号收集数据的目的。

非常适合收集索引号不规则的集合，而切片操作的索引号规则。

假设有 4 个班级，每个班级 35 个学生，8 门科目，shape = [4,35,8]，查找每个班级第 1 、4、9 学生的成绩：

```python
tf.gather(x, [0,3,8], axis=1)
#索引号乱序排列，收集的数据也是对应的顺序
tf.gather(x, [3,1,0], axis=0)
```

2.**`tf.gather_nd(x,indices_list)`**：通过指定每次采样点的多维坐标来实现采样多个点。

如抽查第 2 个班级第 2 个同学的所有科目，第 3 个班级第 3 个同学的所有科目，第 4 个班级第 4 个同学的所有科目，则这 3 个采样点索引坐标为：
[1,1]，[2,2]，[3,3]

```python
#x.shape=[4,35,8]
tf.gather_nd(x,[[1,1],[2,2],[3,3]])
```

3.**`tf.boolean_mask(x,mask_lsit,axis)`**：给定掩码 (Mask) 方式采样。

掩码采取：mask = [True,Fasle,False,True]，采样第 1 和第 4 个班级的数据，注意掩码的维度需与对应维度长度一致。

**tf.boolean_mask 与 tf.gather 类似，一个通过掩码方式，一个通过索引号采样。**

4.**`tf.where(cond,a,b)`**：可根据 cond 条件的真假从参数 a 或 b 中读取数据。

- conda 为 True，则 a
- conda 为 False，则 b

当参数 a=b=None 时，即参数 a，b 不指定，`tf.where` 会返回 cond 张量中所有 True 的元素索引坐标。

```python
#提取张量中所有正数的数据和索引
x= tf.rand.normal([3,3])
#获取对应的掩码值
mask= x > 0
#提取正数
int_num = tf.boolean_mask(x, mask)
#提取正数的索引
int_indices = tf.where(mask)
```

**5.`scatter_nd(indices,updates,shape)`**：高效刷新张量的部分数据，而其只能在全为 0 的白板张量上执行刷新操作。

<img src="../deep_learning_notes/image/217.jpg" style="zoom:80%;" />

```python
#新数据的位置参数
indices = tf.constant([4],[3],[1],[7])
#需要写入的数据
updates = tf.constant([4.4, 3.3, 1.1, 7.7])
#在长度为 8 的全 0 张量上刷新
tf.scatter_nd(indices, updates, [8])
```

**`tf.meshgrid()`**：方便生成二维网格的采样坐标，便于可视化。

### 3.5 经典数据集加载

在 TF 中，`keras.datasets` 模块提供了常用经典数据集的自动下载、管理、加载与转换功能。并提供了 `tf.data.Dataset` 数据集对象，方便实现多线程，预处理，随机打散和批训练等常用数据集功能。

- Boston Housing，波士顿房价趋势数据集，用于回归模型训练和测试
- CIFAR 10/100，真实图片数据集，用于图片分类任务
- MNIST/Fashion_MNIST，手写图片数据集，用于图片分类任务
- IMDB，感情分类任务数据集，用于文本分类任务

`datasets.xxx.load_data()` 函数实现经典数据集自动加载，如 MNIST 数据：

```python
#记载数据集
(x,y), (x_test, y_test)=datasets.mnist.load_data()
print("x：{0},y：{1},x_test：{2},y_test：{3}".format(x.shape, y.shape, x_test.shape, y_test.shape))
#数据加载后，需要转换为 Dataset对象
train_db = tf.data.Dataset.from_tensor_slices((x, y))
```

数据加载并转换为 Dataset 对象后，执行数据集的标准处理步骤

```python
# 1.随机打散，返回一个 Dataset 新对象
# Dataset.shuffle(buffer_size),其中buffer_size参数指定缓冲池的大小
train_db = train_db.shuffle(10000) #随机打散样本，但是不打散样本与标签的映射关系

# 2.批训练，同时计算多个样本
# 一次并行 128 个样本数据，数量大小根据显存配置
train_db = train_db.batch(128)

#3.预处理自定义函数
# 预处理，使得数据集的各式满足模型的输入要求
def preprocess(x, y):
    """
    自定义预处理函数
    参数：
    x -- 待处理的数据集,维度[b,28,28]
    y -- 待处理的数据集,[b]

    返回值：
    x -- 标准化和扁平化后的 x
    y -- 转换为 one_hot 向量
    """
    x = tf.cast(x, dtype=tf.float32) / 255. #标准化0-1
    x = tf.reshape(x, [-1, 28*28]) #扁平化

    y = tf.cast(x, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)

    return x,y
train_db = train_db.map(preprocess)
```

## 4.神经网络

### 4.1 全连接层

感知机 (Perception) 模型，感知机为代表的线性模型不能解决亦或 (XOR) 等线性不可分问题。

<img src="image/04.jpg" style="zoom:80%;" />

<img src="image/05.jpg" style="zoom:80%;" />

每个**输出节点与全部的输入节点相连**接，这种网络层称为全连接层 (Fully-connected Layer)，或者稠密连接层 (Dense Layer)。

**张量方式实现**

```python
o1 = tf.matmul(x, w1)+b1
o1 = tf.nn.relu(o1)
```

**层方式的实现**

更高层、使用更方便的层实现方式：`layers.Dense(units,activation)`

```python
from tensorflow.keras import layers
x = tf.random.normal([4,28*28])
#创建全连接层，指定输出节点数和激活函数
fc =layers.Dense(512, activation=tf.nn.relu)
#获取权值矩阵
fc.kernel
#获取偏置张量
fc.bias
#返回待优化参数列表
fc.trainable_variables
#返回所有参数列表
fc.variables
```

### 4.2 神经网络

神经网络的前向传播过程是：数据张量 (Tensor) 从第一层流动 (Flow) 至输出层的过程，前向传播最后一步完成误差的计算。

误差反向传播 (Backward Propagation，BP) 算法求解梯度信息，同时用 (Gradient Descent，CD) 算法迭代更新参数。

对于回归问题，除了用 MSE 均方误差衡量模型的测试性能，还可以用平均绝对误差 MAE 衡量模型的性能。

## 5.反向传播算法

### 5.1 激活函数的导数

Backpropagation，简称 BP。反向传播算法在 1960 年早期被提出，并未引起重视。直到 1986 年，Geoffrey Hinton 等人在神经网络上应用了反向传播算法。

**sigmoid**
$$
\sigma(x)=\frac{1}{1+e^{-x}}\\
\frac{d}{dx}\sigma(x)=\sigma(1-\sigma)
$$
<img src="image/06.jpg" style="zoom:80%;" />

```python
import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def derivative(x):
    return sigmoid(x) * (1-sigmoid(x))
```

**ReLU**
$$
ReLU(x)=\max(0,x)\\
\frac{d}{dx}ReLU=    
\left\{
    \begin{aligned}
         & 1 & x \ge 0     \\
         & 0 & x < 0
    \end{aligned}
  \right.
$$
<img src="image/07.jpg" style="zoom:80%;" />

```python 
def relu(x):
    return np.maximum(0,x)
def derivative(x):
    d=np.array(x, copy=True)
    d[x < 0] = 0
    d[x >= 0] = 1
    return d
```

**LeakyReLU**

<img src="image/08.jpg" style="zoom:80%;" />

```python
def derivate(x, p):
    dx = np.ones_like(x)
    dx[x < 0] = p
    return dx
```

**Tanh**
$$
\tanh=\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$
<img src="image/09.jpg" style="zoom:80%;" />

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def tanh(x):
    return 2 * sigmoid(2 * x) -1
def derivative(x):
    return 1 - tanh(x) ** 2
```

## 6.keras 高层接口

在 TF 2 版本中，keras 被正式确定为 TF 的高层唯一接口 API，取代了 TF 1 版本中自带的 `tf.layers` 等高层接口。

### 6.1 网络容器

通过 Keras 提供的网络容器 Sequential 将多个网络层封装成 一个大型网络，只需调用网络模型的实例一次即可完成数据从第一层到最后一层的顺序传播运算。

```python
from tensorflow.keras import layers,Sequential
network = Sequential([
    layers.Dense(3, activation=None),
    layers.ReLU(),
    layers.Dense(3, activation=None),
    layers.ReLU()
])
x = tf.random.normal([4,3])
out = network(x)

# 另一种方法
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(8))
model.add(tf.keras.layers.Dense(4))
# 创建网路层参数
model.build(input_shape=(5,5))
# 通过summary()函数打印网络结构和参数量
model.summary()
```

###  6.2 模型装配与训练

在 keras 中，有 2 个比较特殊的类：keras.Model 和 keras.layers.Layer 类。

- `keras.layers.Layer` 类是网络层的母类，定义了网络层一些常见的功能，如添加权值、管理权值列表等。
- `keras.Model`  是网络的母类，除了 `Layer` 类的功能外，还有保存模型、加载模型、训练和测试模型等功能，其中 `Sequential` 就是 `Model` 的子类。

**装配**

通过 complie 函数指定使用的优化对象、损失函数类型等：

```python
from tensorflow.keras import optimizers, losses
# 采用 Adam 优化器，学习率为 0.01，交叉熵损失函数(包含softmax)，测定指标为准确率
model.compile(optimizer=optimizers.Adam(learning_rate=0.01),loss=losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
```

**训练**

```python
#指定训练集为 train_db，验证集为 val_db，训练 5 个epochs，每 2 个epoch验证一次
#返回训练的轨迹信息保存在 his_log 中
his_log = model.fit(train_db, epochs=5, validation_data=val_db, validation_freq=2)
```

训练中产生的历史数据可以通过返回值对象取得。通过 compile 和 fit 方式实现的代码非常简洁和高效，大大缩减了开发时间。但注意的是，由于接口非常高层，灵活性也相应的降低。

**预测**

`Model.predict(x)` 方法可以完成模型的预测

```python
model.predict(x)#模型预测
# 简单地测试模型的性能，可通过循环测试完 db 数据集上的所有样本
model.evaluate(db_test)
```

### 6.2 模型的保存

**张量保存方式**

在拥有网络结构源文件的条件下，直接保存网络张量参数到文件系统上是最轻量级的一种方式。但是它需要使用相同的网络结构才能正确恢复网络状态，**一般在拥有相同网络源文件的情况下使用。**

```python
# 可将当前的网络参数保存在 path 文件中
model.save_weights(path)
# 从参数文件中读取数据并写入当前网络
model.load_weights(path)
```

**网络方式**

不需要神经网络的源文件，仅仅需要模型参数文件即可恢复出网络模型。即不用提前创建模型即可从文件中恢复出网络  model 对象。

```python
#保存模型结构与模型参数到文件
model.save('model.h5')
#从文件恢复网络结构与网格参数
model =keras.models.load_model('model.h5')
```

**SaveModel 方式**

将模型部署到其他平台上时，采用 TF 提出的 SaveModel 方式更具有平台无关性。

```python
tf.saved_model.save(model, 'model_savemodel')
#恢复网络结构与网络参数
model = tf.saved_model.load('model_savemodel')
```

### 6.3 自定义网络

在创建自定义**网络中的网络层类时**，需要继承 `layers.Layer` 基类；创建自定义**网络类时**，需要继承自 `keras.Model` 基类，从而能够利用 `Layer/Model` 基类提供的参数管理等功能，同时也可与其他标准网络层类交互使用。































