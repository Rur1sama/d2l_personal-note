[toc]

# 引言

## 关键组件

1. 数据
2. 模型
3. 目标函数：我们需要定义模型的优劣程度的度量，这个度量在大多数情况是“可优化”的，这被称之为目标函数  
4. 优化算法：当我们获得了一些数据源及其表示、一个模型和一个合适的损失函数，接下来就需要一种算法，它能够搜索出最佳参数，以最小化损失函数。  

## 各种机器学习问题

### 监督学习

- 回归
  - 当标签取任意数值时，我们称之为回归问题，此时的目标是生成
    一个模型，使它的预测非常接近实际标签值  
  
- 分类
  - 标记一个数据集的不同种类，机器在样本都是在相互排斥的情况下进行学习

- 标记问题
  - 标记一个数据集的不同种类，不需要做分类工作，因为让机器去学习预测不相互排斥的类别的问题称为多标签分类

- 搜索
  - 谷歌的PageRank、百度

- 推荐系统

- 序列学习
  - 带记忆的，类似ChatGPT


### 非监督学习
比如，老板可能会给我们一大堆数据，然后要求用它做一些数据科学研究，却没有对结果有要求。
- 聚类（ clustering）问题：没有标签的情况下，我们是否能给**数据分类**呢？  
  - 比如，给定一组照片，我们能把它们分成风景照片、狗、婴儿、猫和山峰的照片吗？同样，给定一组用户的网页浏览记录，我们能否将具有相似行为的用户聚类呢？  

- 主成分分析（ principal component analysis）问题：我们能否找到少量的参数来准确地捕捉数据的**线**
  **性相关属性**？  
  - 比如，一个球的运动轨迹可以用球的速度、直径和质量来描述。
  - 比如，裁缝们已经开发出了一小部分参数，这些参数相当准确地描述了人体的形状，以适应衣服的需要。
  - 另一个例子：在欧几里得空间中是否存在一种（任意结构的）对象的表示，使其符号属性能够很好地匹配?这可以用来描述实体及其关系，例如“罗马” - “意大利” + “法国” = “巴黎”。  

- 因果关系（ causality）和概率图模型（ probabilistic graphical models）问题：我们能否描述观察到的许多**数据**的**根本原因**？  
  - 例如，如果我们有关于房价、污染、犯罪、地理位置、教育和工资的人口统计数
    据，我们能否简单地根据经验数据发现它们之间的关系？  
- 生成对抗性网络（ generative adversarial networks）：
  - 为我们提供一种合成数据的方法，甚至像图像和音频这样复杂的非结构化数据。  

### 与环境互动

我们可能会期望**人工智能**不仅能够做出预测（监督/非监督学习下的人工智能），而且能够与真实环境互动。与预测不同，“与真实环境互动”实际上会影响环境。这里的人工智能是“**智能代理**”，而不仅是“预测模型”。因此，我们必须考虑到它的行为可能会影响未来的观察结果——**强化学习**。  

监督学习与非监督学习：预先获取大量数据，然后启动模型，不再与环境交互（算法与环境断开后进行**离线学习**）

强化学习：与环境交互的

- 环境还记得我们以前做过什么吗？
- 环境是否有助于我们建模？例如，用户将文本读入语音识别器。
- 环境是否想要打败模型？例如，一个对抗性的设置，如垃圾邮件过滤或玩游戏？
- 环境是否重要？
- 环境是否变化？例如，未来的数据是否总是与过去相似，还是随着时间的推移会发生变化？是自然变化
  还是响应我们的自动化工具而发生变化？  

分布偏移 (distribution shift)：训练和测试数据不同时  

### 强化学习

在强化学习问题中，**智能体（agent）**在一系列的时间步骤上与环境交互。  在每个特定时间点，智能体从环境接收一些观察（observation），并且必须选择一个动作（action），然后通过某种机制（有时称为执行器）将其传输回环境，最后智能体从环境中获得奖励（reward）。  此后新一轮循环开始，智能体接收后续观察，并选择后续操作，依此类推。  请注意，强化学习的**目标**是产生一个**好的策略**（policy）。强化学习智能体选择的“动作”受策略控制，即一个从环境观察映射到行动的功能。  

![强化学习和环境之间的相互作用](assets/Dive into deep learning/强化学习和环境之间的相互作用.png)

1. 强化学习框架的通用性十分强大。例如可以将任何监督学习问题转化为强化学习问题。

2. 强化学习还可以解决许多监督学习无法解决的问题。

   - 例如，在监督学习中，我们总是希望输入与正确的标签相关联。

   - 但在强化学习中，我们并不假设环境告诉智能体每个观测的最优动作。一般来说，智能体只是得到一些奖励。此外，环境甚至可能不会告诉是哪些行为导致了奖励。  

3. 强化学习者必须处理**学分分配（ credit assignment）问题**：决定哪些行为是值得奖励的，哪些行为是需要惩罚的。

   - 就像一个员工升职一样，这次升职很可能反映了前一年的大量的行动。要想在未来获得更多的晋升，就需要弄清楚这一过程中哪些行为导致了晋升。  

   - 以强化学习在国际象棋的应用为例。唯一真正的奖励信号出现在游戏结束时：当智能体获胜时，智能体可以得到奖励1；当智能体失败时，智能体将得到奖励‐1。  

4. 强化学习可能还必须处理**部分可观测性问题**：当前的观察结果可能无法阐述有关当前状态的所有
   信息。  

   - 比方说，一个清洁机器人发现自己被困在一个许多相同的壁橱的房子里。推断机器人的精确位置（从而推断其状态），需要在进入壁橱之前考虑它之前的观察结果。  

   

5. 最后，在任何时间点上，强化学习智能体可能知道一个好的策略，但可能有许多**更好的**策略从未尝试过的。
   
      - 强化学习智能体必须不断地做出选择：是应该利用当前最好的策略，还是探索新的策略空间（放弃一些短期回报来换取知识）。  

智能体的动作会影响后续的观察，而奖励只与所选的动作相对应。环境可以是完整观察到的，也可以是部分观察到的，解释所有这些复杂性可能会对研究人员要求太高。此外，并不是每个实际问题都表现出所有这些复杂性。 因此一些特殊情况下的强化学习问题：

1. 当环境可被完全观察到时，强化学习问题被称为马尔可夫决策过程（Markov decision process）。

2. 当状态不依赖于之前的操作时，我们称该问题为上下文赌博机（contextual bandit problem）。

3. 当没有状态，只有一组最初未知回报的可用动作时，这个问题就是经典的多臂赌博机（multi‐armed bandit problem）。  

# 预备知识

## 数据操作

Tensor（张量）实际上就是一个多维数组，在深度学习中支持自动微分。

具有**一个轴**的张量对应数学上的向量（vector）；具有**两个轴**的张量对应数学上的矩阵（matrix）；具有两个轴以上的张量没有特殊的数学名称。  

### Torch入门

我们可以使用 `arange` 创建一个行向量 x。这个行向量包含以0开始的前12个整数，它们默认创建为整
数。也可指定创建类型为浮点数。张量中的每个值都称为张量的元素（**element**）。  

```python
import torch
x = torch.arange(12)
print(x)
```

输出：

```python
tensor([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
```


可以通过张量的`shape`属性来访问张量（沿每个轴的长度）的形状。  

```python
print( x.shape )
print( x.numel() )
```

输出：
```python
torch.Size([12])
12
```

张量中元素的总数 (即形状的所有元素乘积)，可以检查它的大小`size`。因为这里在处理的是一个向量，所以它的`shape`与它的`size` **相同**。  

**要想改变一个张量的形状而不改变元素数量和元素值，可以调用`reshape`函数。** 例如，可以把张量`x`从形状为（12）的行向量转换为形状为（3,4）的矩阵。 这个新的张量包含与转换前相同的值，但是它被看成一个3行4列的矩阵。 要重点说明一下，虽然张量的形状发生了改变，但其元素值并没有变。 注意，通过改变张量的形状，张量的大小不会改变。

```python
X = x.reshape(3, 4)
X
```

Output：（3行4列）

```python
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
```

我们不需要通过手动指定每个维度来改变形状。 也就是说，如果我们的目标形状是（高度,宽度）， 那么在知道**宽度**后，**高度**会被自动计算得出，不必我们自己做除法。 在上面的例子中，为了获得一个3行的矩阵，我们手动指定了它有3行和4列。 幸运的是，我们可以通过`-1`来调用此自动计算出维度的功能。 即我们可以用`x.reshape(-1,4)`或`x.reshape(3,-1)`来取代`x.reshape(3,4)`。



有时，我们希望使用**全0、全1、其他常量**，或者**从特定分布中随机采样**的数字来初始化矩阵。 我们可以创建一个形状为（2,3,4）的张量，其中所有元素都设置为0。代码如下：

```
torch.zeros((2, 3, 4))
torch.ones ((2, 3, 4))
```

Output:

```python
#例子（第一个有修改）
tensor
([
    [[0., 0., 0., 0.],
     [0., 0., 0., 0.],
     [0., 0., 0., 0.]
    ],

    [[0., 0., 0., 0.],
     [0., 0., 0., 0.],
     [0., 0., 0., 0.]
    ]
])

tensor([[[1., 1., 1., 1.],
		 [1., 1., 1., 1.],
		 [1., 1., 1., 1.]],
        
		[[1., 1., 1., 1.],
		 [1., 1., 1., 1.],
		 [1., 1., 1., 1.]]])

#2：最外层有 2 个“大块”（batch 维度）
#3：每个大块里有 3 行
#4：每行有 4 列
```

有时我们想通过从某个特定的概率分布中随机采样来得到张量中每个元素的值。例如，当我们构造数组来作为神经网络中的参数时，我们通常会随机初始化参数的值。

以下代码创建一个形状为（3,4）的张量。其中的每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样。 

 ```python
 torch.randn(3, 4)
 ```

output：

~~~
tensor([[ 0.9834, -1.7370,  0.0534, -1.2094],
        [ 0.4558, -1.0064,  1.4297, -0.6837],
        [-2.2369, -0.4688,  0.6639, -0.1350]])
~~~

我们还可以**通过提供包含数值的Python列表（或嵌套列表），来为所需张量中的每个元素赋予确定值**。 在这里，最外层的列表对应于轴0，内层的列表对应于轴1。

~~~python
# 轴0即行，共3行
# 轴1即列，共4列
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
~~~

~~~python
tensor([[2, 1, 4, 3],
        [1, 2, 3, 4],
        [4, 3, 2, 1]])
~~~



### 运算符

我们的兴趣不仅限于读取数据和写入数据。 我们想在这些数据上执行数学运算，其中最简单且最有用的操作是***按元素***（elementwise）运算。 它们将标准标量运算符应用于数组的每个元素。 对于将两个数组作为输入的函数，按元素运算将二元运算符应用于两个数组中的每对位置对应的元素。 我们可以基于任何从标量到标量的函数来创建按元素函数。



在数学表示法中，我们将通过符号$$f: \mathbb{R} \rightarrow \mathbb{R}$$来表示*一元*标量运算符（只接收一个输入）。这意味着该函数从任何实数（$\mathbb{R}$）映射到另一个实数。同样，我们通过符号$f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$表示*二元*标量运算符，这意味着该函数接收两个输入，并产生一个输出。给定同一形状的任意两个向量$\mathbf{u}$和$\mathbf{v}$和二元运算符$f$，我们可以得到向量$\mathbf{c} = F(\mathbf{u},\mathbf{v})$。具体计算方法是$c_i \gets f(u_i, v_i)$，其中$c_i$、$u_i$和$v_i$分别是向量$\mathbf{c}$、$\mathbf{u}$和$\mathbf{v}$中的元素。在这里，我们通过将标量函数升级为按元素向量运算来生成向量值$F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$。



对于任意具有相同形状的张量， **常见的标准算术运算符（`+`、`-`、`\*`、`/`和`\**`）都可以被升级为按元素运算**。 我们可以在同一形状的任意两个张量上调用按元素操作。 在下面的例子中，我们使用逗号来表示一个具有5个元素的元组，其中每个元素都是按元素操作的结果。

```python
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # **运算符是求幂运算
```

output:

```
(tensor([ 3.,  4.,  6., 10.]),
 tensor([-1.,  0.,  2.,  6.]),
 tensor([ 2.,  4.,  8., 16.]),
 tensor([0.5000, 1.0000, 2.0000, 4.0000]),
 tensor([ 1.,  4., 16., 64.]))
```

(**“按元素”方式可以应用更多的计算**)，包括像求幂这样的一元运算符。

```python
# e的x次方
torch.exp(x)
```

output:

```
tensor([2.7183e+00, 7.3891e+00, 5.4598e+01, 2.9810e+03])
```

### 广播机制（broadcasting）
两个张量形状不同，我们可以通过调用广播机制（ broadcasting mechanism）来执行按元素操作。
1. 通过适当复制元素来扩展一个或两个数组，以便在转换之后，两个张量具有相同的形状；
2. 对生成的数组执行按元素操作

在大多数情况下，我们将沿着数组中长度为1的轴进行广播，如下例子：
```python
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```
Output:
```python
(tensor([[0],
		 [1],
		 [2]]),
tensor([[0, 1]]))
```

由于a和b分别是3 × 1和1 × 2矩阵，如果让它们相加，它们的形状不匹配。我们将两个矩阵广播为一个更大的3 × 2矩阵，如下所示：矩阵a将复制列，矩阵b将复制行，然后再按元素相加。
```python
a + b

# output
tensor([[0, 1],
[1, 2],
[2, 3]])
```
### 索引和切片

张量中的元素可以通过索引访问 ，第一个元素的索引是0，最后一个元素索引是‐1。可以指定范围以包含第一个元素和最后一个之前的元素。  
$$
		[ 0,  1,  2,  3]\\
        [ 4,  5,  6,  7]\\
        [ 8,  9, 10, 11]
$$


如下所示，我们可以用[-1]选择最后一个元素，可以用[1:3]选择第二个和第三个元素：  

```python
X[-1], X[1:3]

#output
(tensor([ 8.,  9., 10., 11.]),
 tensor([[ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.]]))
```

### 节省内存

**运行一些操作可能会导致为新结果分配内存**
例如，如果我们用`Y = X + Y`，我们将取消引用`Y`指向的张量，而是指向新分配的内存处的张量。这会导致内存浪费，因为原先Y的内存位置没有被使用。

**如果在后续计算中没有重复使用`Y`， 我们可以使用`Y[:] = X + Y`或`Y += X`来减少操作的内存开销。**

使用`Y += X` 而不要使用 ~~Y = X + Y~~

### 转换为其他Python对象

将深度学习框架定义的张量[**转换为NumPy张量（`ndarray`）**]很容易，反之也同样容易。 torch张量和numpy数组将共享它们的底层内存，就地操作更改一个张量也会同时更改另一个张量。

```python
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)
```

output:

```python
(numpy.ndarray, torch.Tensor)
```

要(**将大小为1的张量转换为Python标量**)，我们可以调用`item`函数或Python的内置函数。

```python 
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

output:

```gfm
(tensor([3.5000]), 3.5, 3.5, 3)
```

## 数据预处理（pandas）

通常用pandas库做数据分析

### 读取数据集

**创建一个人工数据集，并存储在CSV（逗号分隔值）文件**

```python
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
```

**从创建的CSV文件中加载原始数据集**

~~~python
import pandas as pd

data = pd.read_csv(data_file)
print(data)
~~~

output：

~~~gfm
   NumRooms Alley   Price
0       NaN  Pave  127500
1       2.0   NaN  106000
2       4.0   NaN  178100
3       NaN   NaN  140000
~~~

### 处理缺失值

**为了处理缺失的数据，典型的方法包括插值法和删除法。这里我们将考虑插值法*****
通过位置索引`iloc`，我们将`data`分成`inputs`和`outputs`， 其中前者为`data`的前两列，而后者为`data`的最后一列。 对于`inputs`中缺少的数值，我们用同一列的均值替换`NaN`项。

```python
# iloc (index location) 通过行、列的索引位置来寻找数据
# .iloc[:,:]读取所有行列数据、.iloc[:,a]读取a列所有数据、.iloc[a,:]读取a行所有数据
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

output

~~~gfm
   NumRooms Alley
0       3.0  Pave
1       2.0   NaN
2       4.0   NaN
3       3.0   NaN
~~~

**对于`inputs`中的类别值或离散值，我们将“NaN”视为一个类别。**

由于“巷子类型”（`Alley`）列只接受两种类型的类别值“Pave”和“NaN”， `pandas`可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。 巷子类型为“Pave”的行会将“Alley_Pave”的值设置为1，“Alley_nan”的值设置为0。 缺少巷子类型的行会将“Alley_Pave”和“Alley_nan”分别设置为0和1。

官方文档：[pandas.get_dummies](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html) : 将分类变量转换为虚拟/指标变量。每个变量都转换为任意数量的 0/1 变量值。输出中的每个列都以一个值命名；如果输入是一个 **DataFrame**(此处指代input数据框，包含除dummy-coded 列外的其他列），则原始变量的名称会附加到值前面。



```python
# dummy_na bool，默认 Fals; 如果忽略 False NaN，则添加一列以指示 NaN。
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

Output:

```gfm
   NumRooms  Alley_Pave  Alley_nan
0       3.0           1          0
1       2.0           0          1
2       4.0           0          1
3       3.0           0          1
```

### 转换为张量格式

**现在`inputs`和`outputs`中的所有条目都是数值类型，它们可以转换为张量格式。**

```python
import torch

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y
```

Output:
```python
(tensor([[3., 1., 0.],
         [2., 0., 1.],
         [4., 0., 1.],
         [3., 0., 1.]], dtype=torch.float64),
 tensor([127500, 106000, 178100, 140000]))
```

## 线性代数

本节将介绍线性代数中的基本数学对象、算术和运算，并用数学符号和相应的代码实现来表示它们。  

### 标量

严格来说，仅包含一个数值被称为**标量**（scalar）。  

例如，北京的温度为$52^{\circ}F$。如果要将此华氏度值转换为更常用的摄氏度，则可以计算表达式$c=\frac{5}{9}(f-32)$，并将$f$赋为$52$。
在此等式中，每一项（$5$、$9$和$32$）都是标量值。 符号$c$和$f$称为*变量*（variable），它们表示未知的标量值。  

本书采用了数学表示法，其中标量变量由普通小写字母表示（例如，$x$、$y$ 和 $z$ ）。
本书用$\mathbb{R}$表示所有（连续）*实数*标量的空间，之后将严格定义*空间*（space）是什么，但现在只要记住表达式$x\in\mathbb{R}$是表示 $x$ 是一个实值标量的正式形式。
符号$\in$称为“属于”，它表示“是集合中的成员”。
例如 $x, y \in \{0,1\}$可以用来表明 $x$ 和 $y$ 是值只能为$0$或$1$的数字。

(**标量由只有一个元素的张量表示**)。
下面的代码将实例化两个标量，并执行一些熟悉的算术运算，即加法、乘法、除法和指数。

```python
import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```

output:

```pyt
(tensor(5.), tensor(6.), tensor(1.5000), tensor(9.))
```

### 向量

**向量可以被视为标量值组成的列表**。这些标量值被称为向量的**元素**（element）或**分量**（component）。
当向量表示**数据集中**的样本时，它们的值具有一定的现实意义。例如，如果我们正在训练一个模型来预测贷款违约风险，可能会将每个申请人与一个向量相关联， 其分量与其收入、工作年限、过往违约次数和其他因素相对应。 如果我们正在研究医院患者可能面临的心脏病发作风险，可能会用一个向量来表示每个患者， 其分量为最近的生命体征、胆固醇水平、每天运动时间等。在数学表示法中，向量通常记为粗体、小写的符号 （例如，、和）。

在数学表示法中，向量通常记为粗体、小写的符号
（例如，$\mathbf{x}$、$\mathbf{y}$ 和 $\mathbf{z}$ ）。

人们通过一维张量表示向量。一般来说，张量可以具有任意长度，取决于机器的内存限制。

```python
import torch
x = torch.arange(4)
x
```

output:
```python
tensor([0, 1, 2, 3])
```



我们可以使用下标来引用向量的任一元素，例如可以通过$x_i$来引用第$i$个元素。注意，元素$x_i$是一个**标量**，所以我们在引用它时不会加粗。
大量文献认为**列**向量是向量的**默认方向**，在本书中也是如此。
在数学中，向量 $\mathbf{x}$ 可以写为：
$$
\mathbf{x} =\begin{bmatrix}x_{1}  \\x_{2}  \\ \vdots  \\x_{n}\end{bmatrix}
$$
其中$x_1,\ldots,x_n$是向量的元素。在代码中，我们(**通过张量的索引来访问任一元素**)。

```
x[3]
```

output:

```python
# 3号位，第四个
tensor[3]
```

### 长度、维度和形状

向量只是一个数字数组，就像每个数组都有一个长度一样，每个向量也是如此。

在数学表示法中，如果我们想说一个向量$\mathbf{x}$由$n$个实值标量组成，
可以将其表示为$\mathbf{x}\in\mathbb{R}^n$。
向量的长度通常称为向量的*维度*（dimension）。

与普通的Python数组一样，我们可以通过调用Python的内置`len()`函数来**访问张量的长度**。

```python
len(x)
```

output:

```python
4
```

当用张量表示一个向量（只有**一个轴**）时，我们也可以通过`.shape`属性访问向量的长度。
形状（shape）是一个元素组，列出了张量沿每个轴的长度（维数）。
对于(**只有一个轴的张量，形状只有一个元素。**)

```python
x.shape
```

output:
```python
torch.Size([4])
```

请注意，**维度**（dimension）这个词在不同上下文时往往会有不同的含义，这经常会使人感到困惑。
为了清楚起见，我们在此明确一下：
**向量**或**轴**的**维度**被用来表示向量或轴的**长度**，即向量或轴的**元素数量**。
然而，**张量**（一个可在深度学习中微分的多维数组）的**维度**用来表示张量具有的**轴数**。
在这个意义上，张量的某个轴的维数就是这个轴的长度。

### 矩阵 (Matrix)

正如向量将标量从零阶推广到一阶，矩阵将向量从一阶推广到二阶。
矩阵，我们通常用粗体、大写字母来表示（例如，$\mathbf{X}$、$\mathbf{Y}$和$\mathbf{Z}$），
在代码中表示为具有两个轴的张量。

数学表示法使用$\mathbf{A} \in \mathbb{R}^{m \times n}$ 来表示矩阵$\mathbf{A}$，其由$m$行和$n$列的实值标量组成。
我们可以将任意矩阵$\mathbf{A} \in \mathbb{R}^{m \times n}$视为一个表格，其中每个元素$a_{ij}$属于第$i$行第$j$列：

<span id="eq_matrix_def">**矩阵A的定义**</span>
$$
\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}
$$


对于任意$\mathbf{A} \in \mathbb{R}^{m \times n}$，$\mathbf{A}$的形状是（$m$,$n$）或 $m \times n$。当矩阵具有相同数量的行和列时，其形状将变为正方形；因此，它被称为**方阵**（square matrix）。

当调用函数来实例化张量时，我们可以**通过指定两个分量 $m$ 和 $n$ 来创建一个形状为 $m \times n$的矩阵**。

 ```python
 A = torch.arange(20).reshape(5, 4)
 A
 ```

output: 

```python
# A 矩阵
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19]])
```

我们可以通过行索引（$i$）和列索引（$j$）来访问矩阵中的标量元素$a_{ij}$，例如$[\mathbf{A}]_{ij}$。
如果没有给出矩阵$\mathbf{A}$的标量元素，如在[矩阵定义](#矩阵 (Matrix)) 那样，
我们可以简单地使用矩阵$\mathbf{A}$的小写字母索引下标$a_{ij}$ 来引用$[\mathbf{A}]_{ij}$。
为了表示起来简单，只有在必要时才会将逗号插入到单独的索引中，例如$a_{2,3j}$和$[\mathbf{A}]_{2i-1,3}$。



当我们交换矩阵的行和列时，结果称为矩阵的*转置*（transpose）。
通常用$\mathbf{a}^\top$来表示矩阵的转置，如果$\mathbf{B}=\mathbf{A}^\top$，
则对于任意$i$和$j$，都有$b_{ij}=a_{ji}$。
因此，在$\mathbf{A}$的转置是一个形状为$n \times m$的矩阵：
$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$
作为方阵的一种特殊类型，[***对称矩阵*（symmetric matrix）$\mathbf{A}$等于其转置：$\mathbf{A} = \mathbf{A}^\top$**]。
这里定义一个对称矩阵$\mathbf{B}$：

```python
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
C = B.T
B == C
```

Output:
```python
# B
tensor([[1, 2, 3],
        [2, 0, 4],
        [3, 4, 5]])
# C
tensor([[1, 2, 3],
        [2, 0, 4],
        [3, 4, 5]])

tensor([[True, True, True],
        [True, True, True],
        [True, True, True]])
```



### 张量(tensor)

就像向量是标量的推广，矩阵是向量的推广一样，我们可以构建具有更多轴的数据结构。张量（本小节中的“张量”指代数对象）是描述具有任意数量轴的维数组的通用方法。例如向量是一阶张量，矩阵是二阶张量。张量用特殊字体的大写字母表示（例如，$\mathsf{X}$、$\mathsf{Y}$ 和 $\mathsf{Z}$ ），它们的索引机制（例如 $x_{ijk}$ 和 $[\mathsf{X}]_{1,2i-1,3}$ ）与矩阵类似。

~~~python
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
A, A + B
~~~

output:
```python
# A 
(tensor([[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.],
         [12., 13., 14., 15.],
         [16., 17., 18., 19.]]),
 # A + B
 tensor([[ 0.,  2.,  4.,  6.],
         [ 8., 10., 12., 14.],
         [16., 18., 20., 22.],
         [24., 26., 28., 30.],
         [32., 34., 36., 38.]]))
```

### 张量算法的基本性质

标量、向量、矩阵和任意数量轴的张量（本小节中的“张量”指代数对象）有一些实用的属性。 例如，从按元素操作的定义中可以注意到，任何按元素的一元运算都不会改变其操作数的形状。 同样，**给定具有相同形状的任意两个张量，任何按元素二元运算的结果都将是相同形状的张量**。 例如，将两个相同形状的矩阵相加，会在这两个矩阵上执行元素加法。

#### 矩阵的Hadamard积

若$$A=(a_{ij})$$和$$B=(b_{ij})$$是两个同阶矩阵，若$$c_{ij}=a_{ij}×b_{ij}$$,则称矩阵$C= (c_{ij}$) 为A和B的哈达玛积，或称基本积。具体而言，**两个矩阵的按元素乘法称为*Hadamard积*（Hadamard product）（数学符号$\odot$）**。

对于矩阵$\mathbf{B} \in \mathbb{R}^{m \times n}$，其中第$i$行和第$j$列的元素是$b_{ij}$。
矩阵$\mathbf{A}$和$\mathbf{B}$的Hadamard积为：
$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

~~~python
A * B
~~~

其中 $\mathbf{A}$ 和$\mathbf{B}$ 矩阵均为是

```python
(tensor([[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.],
         [12., 13., 14., 15.],
         [16., 17., 18., 19.]]),
```

output:

```python
tensor([[  0.,   1.,   4.,   9.],
        [ 16.,  25.,  36.,  49.],
        [ 64.,  81., 100., 121.],
        [144., 169., 196., 225.],
        [256., 289., 324., 361.]])
```

将张量乘以或加上一个标量不会改变张量的形状，其中张量的每个元素都将**与标量相加或相乘**。

```python
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

output:

```python
(tensor([[[ 2,  3,  4,  5],
          [ 6,  7,  8,  9],
          [10, 11, 12, 13]],
 
         [[14, 15, 16, 17],
          [18, 19, 20, 21],
          [22, 23, 24, 25]]]),
 torch.Size([2, 3, 4]))
```

#### 降维(lin-alg-reduction)

我们可以对任意张量进行的一个有用的操作是**计算其元素的和**。
数学表示法使用 $\sum$ 符号表示求和。
为了表示长度为 $d$ 的向量中元素的总和，可以记为$\sum_{i=1}^dx_i$。
在代码中可以调用计算求和的函数：

~~~python
x = torch.arange(4, dtype=torch.float32)
x, x.sum()
~~~

output:
```
(tensor([0., 1., 2., 3.]), tensor(6.))
```

我们可以**表示任意形状张量的元素和**
例如，矩阵$\mathbf{A}$中元素的和可以记为$\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$。

```python
A.shape, A.sum()
```

output:
```python
(torch.Size([5, 4]), tensor(190.))
```

默认情况下，调用求和函数会沿所有的轴降低张量的维度，使它变为一个标量。 我们还可以**指定张量沿哪一个轴来通过求和降低维度**。

 以矩阵为例，为了通过求和所有**行**的元素来降维（**轴0**），可以在调用函数时指定`axis=0`。 由于输入矩阵沿0轴降维以生成输出向量，因此输入轴0的维数在输出形状中消失。

```python
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

output:
```python
(tensor([40., 45., 50., 55.]), torch.Size([4]))
```

指定`axis=1`将通过汇总所有**列**的元素降维（**轴1**）。因此，输入轴1的维数在输出形状中消失。

~~~python
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
~~~

output:
```python
(tensor([ 6., 22., 38., 54., 70.]), torch.Size([5]))
```

沿着行和列对矩阵求和，等价于对矩阵的所有元素进行求和。
```python
A.sum(axis=[0, 1])  # 结果和A.sum()相同
```

output:
```python
tensor(190.)
```



一个与求和相关的量是 **平均值**（mean或average）。 我们通过将总和除以元素总数来计算平均值。 在代码中，我们可以调用函数来计算任意形状张量的平均值。

```python
A.mean(), A.sum() / A.numel()
```

output:
```python
(tensor(9.5000), tensor(9.5000))
```

同样，计算平均值的函数也可以沿指定轴降低张量的维度。

```python
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

output:
```python
(tensor([ 8.,  9., 10., 11.]), tensor([ 8.,  9., 10., 11.]))
```

##### 非降维求和

:label:`subseq_lin-alg-non-reduction`

但是，有时在调用函数来**计算总和或均值时保持轴数不变**会很有用。(看起来相对于不加`keepdims`，只有形状变合理了)

回顾矩阵$\mathbf{A}$: 

```python
(tensor([[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.],
         [12., 13., 14., 15.],
         [16., 17., 18., 19.]]),
```

将A矩阵的以列向量的形式加起来

~~~python
sum_A = A.sum(axis=1, keepdims=True)
sum_A
~~~

output:
```python
# 得到一个列向量样式的张量
tensor([[ 6.],
        [22.],
        [38.],
        [54.],
        [70.]])
```

例如，由于`sum_A`在对每行进行求和后仍保持两个轴，我们可以**通过广播将`A`除以`sum_A`**。
```python
A / sum_A
```

```python
tensor([[0.0000, 0.1667, 0.3333, 0.5000],
        [0.1818, 0.2273, 0.2727, 0.3182],
        [0.2105, 0.2368, 0.2632, 0.2895],
        [0.2222, 0.2407, 0.2593, 0.2778],
        [0.2286, 0.2429, 0.2571, 0.2714]])
```

如果我们想沿**某个轴计算`A`元素的累积总和**， 比如`axis=0`（按行计算），可以调用`cumsum`函数。 此函数不会沿任何轴降低输入张量的维度。
```python
A.cumsum(axis=0)
```

output:

```python
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  6.,  8., 10.],
        [12., 15., 18., 21.],
        [24., 28., 32., 36.],
        [40., 45., 50., 55.]])
```

### 点积 (Dot product)

**左转为值，右转有秩**

我们已经学习了按元素操作、求和及平均值。
另一个最基本的操作之一是点积。给定两个向量$\mathbf{x},\mathbf{y}\in\mathbb{R}^d$，它们的*点积*（dot product）$\mathbf{x}^\top\mathbf{y}$（或$\langle\mathbf{x},\mathbf{y}\rangle$）是相同位置的按元素乘积的和：$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$。

~~~pyt
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)
~~~

output：

```python
(tensor([0., 1., 2., 3.]), tensor([1., 1., 1., 1.]), tensor(6.))
```

点积在很多场合都很有用。例如，给定一组由向量$\mathbf{x} \in \mathbb{R}^d$表示的值，和一组由$\mathbf{w} \in \mathbb{R}^d$表示的权重。$\mathbf{x}$中的值根据权重$\mathbf{w}$的加权和，可以表示为点积$\mathbf{x}^\top \mathbf{w}$。当权重为非负数且和为1（即$\left(\sum_{i=1}^{d}{w_i}=1\right)$）时，点积表示*加权平均*（weighted average）。将两个向量规范化得到单位长度后，点积表示它们夹角的余弦。本节后面的内容将正式介绍*长度*（length）的概念。



### 矩阵-向量积

现在我们知道如何计算点积，可以开始理解*矩阵-向量积*（**matrix-vector product**）。
回顾分别在中定义的[矩阵](#eq_matrix_def)$\mathbf{A} \in \mathbb{R}^{m \times n}$和[向量](#向量)$\mathbf{x} \in \mathbb{R}^n$。

---



解释 $\mathbf{A}$ 矩阵：

假设矩阵  

$$
\mathbf{A} \in \mathbb{R}^{2 \times 3}, \quad 
\mathbf{A} =
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}.
$$

#### 行向量表示 $\mathbf{a}_i^\top$

- 第 1 行：
$$
\mathbf{a}_1^\top = [1 \;\; 2 \;\; 3] \in \mathbb{R}^3
$$

- 第 2 行：
$$
\mathbf{a}_2^\top = [4 \;\; 5 \;\; 6] \in \mathbb{R}^3
$$



---

#### 列向量表示 $\mathbf{b}_j$

- 第 1 列：
$$
\mathbf{b}_1 =
\begin{bmatrix}
1 \\
4
\end{bmatrix} \in \mathbb{R}^2
$$

- 第 2 列：
$$
\mathbf{b}_2 =
\begin{bmatrix}
2 \\
5
\end{bmatrix} \in \mathbb{R}^2
$$

- 第 3 列：

$$
\mathbf{b}_3 =
\begin{bmatrix}
3 \\
6
\end{bmatrix} \in \mathbb{R}^2
$$

---


可将矩阵$\mathbf{A}$用它的行向量表示：
$$
\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},
$$
其中每个 $\mathbf{a}^\top_{i} \in \mathbb{R}^n$ 都是行向量，表示矩阵的第$i$行。
**矩阵向量积$\mathbf{A}\mathbf{x}$是一个长度为$m$的列向量，其第$i$个元素是点积$\mathbf{a}^\top_i \mathbf{x}$**：
$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$


我们可以把一个矩阵$\mathbf{A} \in \mathbb{R}^{m \times n}$乘法看作一个从$\mathbb{R}^{n}$到$\mathbb{R}^{m}$向量的转换。这些转换是非常有用的，例如可以用方阵的乘法来表示旋转。后续章节将讲到，我们也可以使用矩阵-向量积来描述在给定前一层的值时，求解神经网络每一层所需的复杂计算。

在代码中使用张量表示矩阵-向量的积，我们使用`mv`函数。
当我们为矩阵`A`和向量`x`调用`torch.mv(A, x)`时，会执行矩阵-向量的积。
注意，`A`的列维数（沿轴1的长度）必须与`x`的维数（其长度）相同。

~~~python
A.shape, x.shape, torch.mv(A, x)
~~~

output:

```python
(torch.Size([5, 4]), torch.Size([4]), tensor([ 14.,  38.,  62.,  86., 110.]))
```

### 矩阵-矩阵乘法

在掌握点积和矩阵-向量积的知识后，那么**矩阵-矩阵乘法**（matrix-matrix multiplication）应该很简单。

假设有两个矩阵$\mathbf{A} \in \mathbb{R}^{n \times k}$和$\mathbf{B} \in \mathbb{R}^{k \times m}$：
$$
\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.
$$
用行向量$\mathbf{a}^\top_{i} \in \mathbb{R}^k$表示矩阵$\mathbf{A}$的第$i$行，并让列向量$\mathbf{b}_{j} \in \mathbb{R}^k$作为矩阵$\mathbf{B}$的第$j$列。要生成矩阵积$\mathbf{C} = \mathbf{A}\mathbf{B}$，最简单的方法是考虑$\mathbf{A}$的行向量和$\mathbf{B}$的列向量:
$$
\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$
当我们简单地将每个元素$c_{ij}$计算为点积$\mathbf{a}^\top_i \mathbf{b}_j$:

$$
\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$
**我们可以将矩阵-矩阵乘法$\mathbf{AB}$看作简单地执行$m$次矩阵-向量积，并将结果拼接在一起，形成一个$n \times m$矩阵**。
在下面的代码中，我们在`A`和`B`上执行矩阵乘法。这里的`A`是一个5行4列的矩阵，`B`是一个4行3列的矩阵。两者相乘后，我们得到了一个5行3列的矩阵。

```python
A = torch.arange(20,torch.dtype=float32).reshape(5,4)
B = torch.ones(4, 3) # 生成一个浮点型的矩阵（张量）
torch.mm(A, B)
```



:exclamation: 矩阵-矩阵乘法可以简单地称为**矩阵乘法**，不应与"Hadamard积"混淆。

### 范数 (norm)

<span id = "subsec_lin-algebra-norms"></span>

线性代数中最有用的一些运算符是**范数**（norm）。
非正式地说，向量的**范数**是表示一个向量有多大。这里考虑的*大小*（size）概念不涉及维度，而是分量的大小

在线性代数中，向量范数是将向量映射到标量的函数$f$。
给定任意向量$\mathbf{x}$，向量范数要满足一些属性。
第一个性质是：如果我们按常数因子$\alpha$缩放向量的所有元素，其范数也会按相同常数因子的*绝对值*缩放：
$$
f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x}).
$$
第二个性质是熟悉的三角不等式:
$$
f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y}).
$$
第三个性质简单地说范数必须是非负的:
$$
f(\mathbf{x}) \geq 0.
$$
这是有道理的。因为在大多数情况下，任何东西的最小的*大小*是0。最后一个性质要求范数最小为0，当且仅当向量全由0组成（零向量）。
$$
\forall i, [\mathbf{x}]_i = 0 \Leftrightarrow f(\mathbf{x})=0.
$$
范数听起来很像距离的度量。
欧几里得距离和毕达哥拉斯定理中的非负性概念和三角不等式可能会给出一些启发。事实上，欧几里得距离是一个***$L_2$范数***：假设$n$维向量$\mathbf{x}$中的元素是$x_1,\ldots,x_n$，其 **$L_2$*范数* 是向量元素平方和的平方根：**
$$
\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2},
$$
其中，在***$L_2$范数***中常常省略下标$2$，也就是说$\|\mathbf{x}\|$等同于$\|\mathbf{x}\|_2$。在代码中，我们可以按如下方式计算向量的$L_2$范数。

~~~python
u = torch.tensor([3.0, -4.0])
torch.norm(u)
~~~

output:
```
tensor(5.)
```

深度学习中更经常地使用$L_2$范数的平方，也会经常遇到[**$L_1$范数，它表示为向量元素的绝对值之和：**]

$$
\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.
$$


与***$L_2$范数***相比，***$L_1$范数***受异常值的影响较小。
为了计算$L_1$范数，我们将绝对值函数和按元素求和组合起来。

```python
torch.abs(u).sum()
```

output:

```
tensor(7.)
```



$L_2$范数和$L_1$范数都是更一般的$L_p$范数的特例：
$$
\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.
$$


类似于向量的$L_2$范数，**矩阵 **$\mathbf{X} \in \mathbb{R}^{m \times n}$的***Frobenius范数*（Frobenius norm）是矩阵元素平方和的平方根：**

$$
\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.
$$


Frobenius范数满足向量范数的所有性质，它就像是矩阵形向量的$L_2$范数。
调用以下函数将计算矩阵的Frobenius范数。

~~~python
torch.norm(torch.ones(4, 9))
~~~

output:

```
tensor(6.)
```

#### 范数和目标
<span id = 'subsec_norms_and_objectives'>范数和目标</span>

在深度学习中，我们经常试图解决优化问题： **最大化**分配给观测数据的概率; **最小化**预测和真实观测之间的距离。 用向量表示物品（如单词、产品或新闻文章），以便最小化相似项目之间的距离，最大化不同项目之间的距离。 目标，或许是深度学习算法最重要的组成部分（除了数据），通常被表达为范数。



[范数](subsec_norms_and_objectives)


## 微积分

### 导数和微分

### 偏导数

### 梯度
<span id = 'subsec_calculus-grad'>梯度很重要! </span>

我们可以连结一个多元函数对其所有变量的偏导数，以得到该函数的*梯度*（gradient）向量。
具体而言，设函数$f:\mathbb{R}^n\rightarrow\mathbb{R}$的输入是一个$n$维向量$\mathbf{x}=[x_1,x_2,\ldots,x_n]^\top$，并且输出是一个标量。
函数$f(\mathbf{x})$相对于$\mathbf{x}$的梯度是一个包含$n$个偏导数的向量:
$$
\nabla_{\mathbf{x}} f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_n}\bigg]^\top,
$$
其中$\nabla_{\mathbf{x}} f(\mathbf{x})$通常在没有歧义时被$\nabla f(\mathbf{x})$取代。

假设$\mathbf{x}$为$n$维向量，在微分多元函数时经常使用以下规则:

* 对于所有$\mathbf{A} \in \mathbb{R}^{m \times n}$，都有$\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$
* 对于所有$\mathbf{A} \in \mathbb{R}^{n \times m}$，都有$\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$
* 对于所有$\mathbf{A} \in \mathbb{R}^{n \times n}$，都有$\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$
* $\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$

同样，对于任何矩阵$\mathbf{X}$，都有$\nabla_{\mathbf{X}} \|\mathbf{X} \|_F^2 = 2\mathbf{X}$。

正如我们之后将看到的，梯度对于设计深度学习中的优化算法有很大用处。

### 链式法则

然而，上面方法可能很难找到梯度。
这是因为在深度学习中，多元函数通常是*复合*（composite）的，
所以难以应用上述任何规则来微分这些函数。幸运的是，链式法则可以被用来微分复合函数。

## 自动微分

### 非标量变量的反向传播
### 分离计算

## 概率论

skip

# 线性神经网络

##  nn.Linear实现方式

<span id = "线性模型">`nn.Linear(in_features, out_features)`</span> 

- 作用：做一个 **线性变换**
  $$
  y = x W^T + b
  $$
  其中：

  - `x`：输入张量，形状是 `(batch_size, in_features)`
  - `W`：权重矩阵，形状是 `(out_features, in_features)`
  - `b`：偏置，形状是 `(out_features,)`
  - 输出 `y`：形状是 `(batch_size, out_features)`

### 权重weight

为什么权重是 `(out_features, in_features)`？

因为 **矩阵乘法规则**：

- 输入 `x`：`(batch_size, in_features)`
- 权重 `W.T`：`(in_features, out_features)`
- 乘法 `x @ W.T` → 结果是 `(batch_size, out_features)`

所以 `W` 本身必须是 `(out_features, in_features)`，这样 `W.T`（即权重的转置张量） 就是 `(in_features, out_features)`，才能和输入匹配。

### 举例 

`nn.Linear(4, 8)`

- `in_features = 4`，`out_features = 8`

- 输入 `X.shape = (batch_size, 4)`

- 权重 `W.shape = (8, 4)`

- 偏置 `b.shape = (8,)`

- 计算：
  $$
  Y = X \cdot W^T + b
  $$
  → `Y.shape = (batch_size, 8)` ✅

# 深度学习计算

[【动手学深度学习】第五章笔记：层与块、参数管理、自定义层、读写文件、GPU - bringlu - 博客园](https://www.cnblogs.com/bringlu/p/17359969.html)

## 层和块

事实证明，研究讨论“比单个层大”但“比整个模型小”的组件更有价值。例如，在计算机视觉中广泛流行的ResNet-152 架构就有数百层，这些层是由**层组**（groups of layers）的重复模式组成。

为了实现这些复杂的网络，我们引入了神经网络**块**的概念。**块**（block）可以描述单个层、由多个层组成的组件或整个模型本身。使用块进行抽象的一个好处是可以将一些块组合成更大的组件。通过定义代码来按需生成任意复杂度的块，我们可以通过简洁的代码实现复杂的神经网络。

从编程的角度来看，块由**类**（class）表示。它的任何子类都必须定义一个将其输入转换为输出的前向传播函数，并且必须存储任何必需的参数。注意，有些块不需要任何参数。最后，为了计算梯度，块必须具有反向传播函数。在定义我们自己的块时，由于自动微分提供了一些后端实现，我们**只需要考虑前向传播函数和必需的参数**。

之后原书中举的例子为实例化一个包含两个线性层的多层感知机。该代码中，通过实例化 `nn.Sequential` 来构建模型，层的执行顺序是作为参数传递的。简而言之，`nn.Sequential`定义了一种特殊的 `Module`，即在 PyTorch 中表示一个块的类，它维护了一个由 `Module` 组成的有序列表。注意，两个全连接层都是 `Linear` 类的实例，`Linear` 类本身就是 `Module` 的子类。另外，到目前为止，我们一直在通过 `net(X)` 调用我们的模型来获得模型的输出。这实际上是 `net.__call__(X)` 的简写。

### 自定义块

每个块必须提供的基本功能：

1. 将输入数据作为其前向传播函数的参数。
2. 通过前向传播函数来生成输出。请注意，输出的形状可能与输入的形状不同。例如，我们上面模型中的第一个全连接的层接收一个20维的输入，但是返回一个维度为256的输出。
3. 计算其输出关于输入的梯度，可通过其反向传播函数进行访问。通常这是自动发生的。
4. 存储和访问前向传播计算所需的参数。
5. 根据需要初始化模型参数。

下面的代码片段包含一个多层感知机，其具有256个隐藏单元的隐藏层和一个10维输出层。 注意，下面的`MLP`类继承了表示块的类。 我们的实现只需要提供我们自己的构造函数（Python中的`__init__`函数）和前向传播函数。

```python
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))
```

### 顺序块

为了构建我们自己的简化的`MySequential`， 我们只需要定义两个关键函数：

1. 一种将块逐个追加到列表中的函数；
2. 一种前向传播函数，用于将输入按追加块的顺序传递给块组成的“链条”。

```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
```

### 在前向传播函数中执行代码

并不是所有的架构都是简单的顺序架构。 当需要更强的灵活性时，我们需要定义自己的块。 此外，我们可能希望执行任意的数学运算， 而不是简单地依赖预定义的神经网络层。

### 效率

读者可能会开始担心操作效率的问题。 毕竟，我们在一个高性能的深度学习库中进行了大量的字典查找、 代码执行和许多其他的Python代码。 Python的问题[全局解释器锁](https://wiki.python.org/moin/GlobalInterpreterLock) 是众所周知的。 在深度学习环境中，我们担心速度极快的GPU可能要等到CPU运行Python代码后才能运行另一个作业。

## 参数管理

训练阶段，我们的目标是找到使损失函数最小化的模型参数值。 经过训练后，我们将需要使用这些参数来做出未来的预测。 此外，有时我们希望提取参数，以便在其他环境中复用它们， 将模型保存下来，以便它可以在其他软件中执行， 或者为了获得科学的理解而进行检查。

首先看一下具有单隐藏层的多层感知机 <span id = "net">**net**</span>，本文后续都使用这个作为示例。

~~~python
import torch
from torch import nn

net = nn.Sequential(
    nn.Linear(4, 8),	# 第1层：全连接层，输入维度=4，输出维度=8
    nn.ReLU(),			# 第2层：激活函数 ReLU
    nn.Linear(8, 1))	# 第3层：全连接层，输入维度=8，输出维度=1
X = torch.rand(size=(2, 4)) # 形状为 (2,4) 的随机张量，batch_size=2，每个样本有4个特征
net(X)
~~~

把 `X` 输入网络，前向传播的过程是：

- `X` → `(2,4)`
- 经过第1个 `Linear(4,8)` → `(2,8)`
- 经过 `ReLU()` → `(2,8)`（只是做非线性变换，不改变形状）
- 经过第2个 `Linear(8,1)` → `(2,1)`

最终输出形状是 `(2,1)`。



### 参数访问

```python
# 单隐藏层的多层感知机
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)

print(net[2].state_dict())
```

output:
```python
#orderedDict 排好序的字典
# weight 1 * 8 的
OrderedDict([('weight', tensor([[-0.0427, -0.2939, -0.1894,  0.0220, -0.1709, -0.1522, -0.0334, -0.2263]])), ('bias', tensor([0.0887]))])
```

输出的结果告诉我们一些重要的事情： 首先，这个全连接层包含两个参数，分别是该层的权重和偏置。 两者都存储为单精度浮点数（float32）。 注意，参数名称允许唯一标识每个参数，即使在包含数百个层的网络中也是如此。

#### 目标参数

注意，每个参数都表示为参数类的一个实例。 要对参数执行任何操作，首先我们需要访问底层的数值。 有几种方法可以做到这一点。有些比较简单，而另一些则比较通用。 

下面的代码从第二个全连接层（即第三个神经网络层）提取偏置， 提取后返回的是一个参数类实例，并进一步访问该参数的值。

~~~python
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
~~~

output：

```python
<class 'torch.nn.parameter.Parameter'> # 可以优化的参数
Parameter containing:
tensor([0.0887], requires_grad=True) # 偏移值
tensor([0.0887]) #.data // 还有.grad 表示梯度
```

参数是复合的对象，包含值、梯度和额外信息。 这就是我们需要显式参数值的原因。 除了值之外，我们还可以访问每个参数的梯度。 在上面这个网络中，由于我们还没有调用反向传播，所以参数的梯度处于初始状态（None）。

```python
net[2].weight.grad == None
# 输出为True
```

#### 一次性访问所有参数

当我们需要对所有参数执行操作时，逐个访问它们可能会很麻烦。 当我们处理更复杂的块（例如，嵌套块）时，情况可能会变得特别复杂， 因为我们需要递归整个树来提取每个子块的参数。 
下面，我们将通过演示来比较访问第一个全连接层的参数和访问所有层。

`这个地方是返回的生成器对象，用*解包得到参数`

```python
# net[0] 指的是第一个子层（输入层） nn.Linear(4, 8)。
# 它有两个参数：weight → (8, 4) ；bias → (8,)
print(*[(name, param.shape) for name, param in net[0].named_parameters()])

# 遍历整个网络的参数（所有子层的）。
# 对 net[0] (Linear(4,8))：
	# 0.weight → (8, 4)
	# 0.bias → (8,)
# nn.ReLU() 无参数，不会在全链接层中显示
# 对 net[2] (Linear(8,1))：
	# 2.weight → (1, 8)
	# 2.bias → (1,)   
print(*[(name, param.shape) for name, param in net.named_parameters()])
```

output:
```python
('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))

('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) 
('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1])) 
```

另一种访问网络参数的方式：
~~~python
net.state_dict()['2.bias'].data
~~~



#### 从嵌套块收集参数

如果我们将多个块相互嵌套，参数命名约定是如何工作的？ 我们首先定义一个生成块的函数（可以说是“块工厂”），然后将这些块组合到更大的块中。

```python
def block1():
    # 一个线性参，两个参数
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1()) # 给你一个字符串名字
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1)) # 再套一个
rgnet(X)
```

output：

~~~python
tensor([[0.2596],
        [0.2596]], grad_fn=<AddmmBackward0>)
~~~

设计了网络后，我们看看它是如何工作的。

~~~python
print(rgnet)
~~~

output：
~~~python
Sequential(
  (0): Sequential(
    (block 0): Sequential( #block 0 是我们在嵌套那一步传入的字符串
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 1): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 2): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 3): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
  )
  (1): Linear(in_features=4, out_features=1, bias=True) # 自己传入的线性参
)

~~~

因为层是分层嵌套的，所以我们也可以像通过嵌套列表索引一样访问它们。 下面，我们访问第一个主要的块中、第二个子块的第一层的偏置项。

```python
rgnet[0][1][0].bias.data
```

output：
~~~python
#例子 符合8列即可
tensor([ 0.1999, -0.4073, -0.1200, -0.2033, -0.1573,  0.3546, -0.2141, -0.2483])
~~~

### 参数初始化

我们在 [4.8节](https://zh-v2.d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html#sec-numerical-stability)中讨论了良好初始化的必要性。深度学习框架提供默认随机初始化， 也允许我们创建自定义初始化方法， 满足我们通过其他规则实现初始化权重。

默认情况下，PyTorch会根据一个范围均匀地初始化权重和偏置矩阵， 这个范围是根据输入和输出维度计算出的。 PyTorch的`nn.init`模块提供了多种预置初始化方法。

#### 内置初始化

让我们首先调用内置的初始化器。 下面的代码将所有权重参数初始化为标准差为0.01的高斯随机变量， 且将偏置参数设置为0。[net](#net) [线性模型Linear](#线性模型)

```python
def init_normal(m): # m 指的就是moudle
    if type(m) == nn.Linear: # 如果是全连接层，
        nn.init.normal_(m.weight, mean=0, std=0.01) # 下划线为原地操作，用均值 0、标准差 0.01 的正态分布直接覆盖m.weight的权重张量（所谓初始化）
        nn.init.zeros_(m.bias) # 赋0
        
# nn.Module.apply(fn) 会对网络中的每一层调用 fn 函数
# 因此 init_normal函数 会被应用到 net 里的所有子层：        
net.apply(init_normal) 

net[0].weight.data[0], net[0].bias.data[0] 
# 记住权重weight是(out_features, in_features)，转置后与多层感知机(batch_size, in_features)做矩阵叉乘 即Net(batch_size, in) × weight.T (in, out)
# bias 是(out_features, )
```

output

~~~python
(tensor([-0.0214, -0.0015, -0.0100, -0.0058]), tensor(0.)) 
# 也许是正态分布? 
# weight[0] 是(8,4) ，此处是net[0],(4, 8)的第一个权重，即4列; 偏差已经设为0
~~~

我们还可以对某些块应用**不同的**初始化方法。

 例如，下面我们使用Xavier初始化方法初始化第一个神经网络层， 然后将第三个神经网络层初始化为常量值42。

~~~python
def init_xavier(m): # Glorot 初始化,根据输入和输出的维度自动选择一个合适的均匀分布范围
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m): # 把所有权重都初始化成常数 42。
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier) 	 # net[0] (4,8) 所以weight[0] 是 (8,4)，有8行
net[2].apply(init_42) 		 # net[2] (8,1) 所以weight[1] 是 (1,8)
print(net[0].weight.data[0]) # weight[0] 的第一行
print(net[2].weight.data)
~~~

```scss
tensor([ 0.5236,  0.0516, -0.3236,  0.3794])
tensor([[42., 42., 42., 42., 42., 42., 42., 42.]])
```

#### 自定义初始化

有时，深度学习框架没有提供我们需要的初始化方法。 在下面的例子中，我们使用以下的分布为任意权重参数定义初始化方法：
$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \text{ 可能性 } \frac{1}{4} \\
            0    & \text{ 可能性 } \frac{1}{2} \\
        U(-10, -5) & \text{ 可能性 } \frac{1}{4}
    \end{cases}
\end{aligned}
$$
同样，我们实现了一个`my_init`函数来应用到[`net`](#net)。

~~~python
def my_init(m):
    if type(m) == nn.Linear:
        print(
            "Init",
            *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10) # 在 [-10, 10] 区间均匀分布随机初始化 
        m.weight.data *= m.weight.data.abs() >= 5 #保留绝对值大于5的权重，其他赋值0

net.apply(my_init)
net[0].weight[:2] # 打印第一层 (Linear(4,8)) 权重矩阵的前两行。
~~~

解释：

- `m.weight.data.abs() >= 5` 会生成一个 **布尔掩码**，形状和权重相同，True 表示该元素绝对值 ≥ 5，False 表示 < 5。
- 乘法会把 `True` 当作 `1`，`False` 当作 `0`，所以：
  - |w| ≥ 5 → 保留原值
  - |w| < 5 → 被乘以 0，直接清零

因此，最后权重矩阵只保留“绝对值至少 5”的元素，其余变成 0。

输出（例子）：

```scss
Init weight torch.Size([8, 4])
Init weight torch.Size([1, 8])

tensor([[5.4079, 9.3334, 5.0616, 8.3095],
        [0.0000, 7.2788, -0.0000, -0.0000]], grad_fn=<SliceBackward0>)
```

注意，我们始终可以直接设置参数。

```python
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

~~~scss
tensor([42.0000, 10.3334,  6.0616,  9.3095])
~~~

### 参数绑定

有时我们希望在多个层间共享参数： 我们可以定义一个稠密层，然后使用它的参数来设置另一个层的参数。

~~~python
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
~~~

输出：

~~~scss
tensor([True, True, True, True, True, True, True, True])
tensor([True, True, True, True, True, True, True, True])
~~~

这个例子表明第三个和第五个神经网络层的参数是绑定的。 它们不仅值相等，而且由相同的张量表示。 因此，如果我们改变其中一个参数，另一个参数也会改变。 这里有一个问题：当参数绑定时，梯度会发生什么情况？ 答案是由于模型参数包含梯度，因此在反向传播期间第二个隐藏层 （即第三个神经网络层）和第三个隐藏层（即第五个神经网络层）的梯度会加在一起。



## 延后初始化（Pytorch没有这部分内容）

到目前为止，我们忽略了建立网络时需要做的以下这些事情：

- 我们定义了网络架构，但没有**指定输入维度**。
- 我们添加层时没有**指定前一层的输出维度**。
- 我们在初始化参数时，甚至没有足够的信息来**确定模型应该包含多少参数**

​	深度学习框架无法判断网络的输入维度是什么。 这里的诀窍是框架的*延后初始化*（defers initialization）， 即直到数据第一次通过模型传递时，框架才会动态地推断出每个层的大小。在以后，当使用卷积神经网络时， 由于输入维度（即图像的分辨率）将影响每个后续层的维数， 有了该技术将更加方便。 现在我们在编写代码时无须知道维度是什么就可以设置参数， 这种能力可以大大简化定义和修改模型的任务。

## 自定义层

深度学习成功背后的一个因素是神经网络的灵活性： 我们可以用创造性的方式组合不同的层，从而设计出适用于各种任务的架构。 例如，研究人员发明了专门用于处理图像、文本、序列数据和执行动态规划的层。 有时我们会遇到或要自己发明一个现在在深度学习框架中还不存在的层。 在这些情况下，必须构建自定义层。本节将展示如何构建自定义层。



# 卷积神经网络

## 全连接层到卷积

对全连接层使用平级不变性和局部性得到卷积层

1. 平移不变性：参数不共享（全连接层）→参数共享（卷积层）
2. 局部性：卷积核大小（只受卷积核大小影响输出，所谓不用远离输入位置的参数）

什么时候用卷积：检测的对象不因所处位置而改变，且一般具有局部的特征。


$$
[\mathbf{H}]_{i, j} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.
$$
:eqlabel:`eq_conv-layer`

<span id = "eq_conv-layer">卷积层公式</span>

是一个*卷积层*（convolutional layer），而卷积神经网络是包含卷积层的一类特殊的神经网络。
