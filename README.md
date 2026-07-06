# 🔥 从零开始学 AI：Micrograd 深度学习入门

> **目标**：用最少的代码搞懂 AI 训练的核心原理——正向传播、反向传播、梯度下降。
>
> **方法论**：把每个数学概念翻译成硬件直觉（增益、信号流、放大器），不做符号推演，只求物理通感。
>
> **教材**： [Andrej Karpathy 的 micrograd](https://github.com/karpathy/micrograd)（两个文件，共 150 行 Python）
>
> **预计总时长**：约 7 小时，分 7 课。

---

## 学习路径总览

```text
第一课 → Value 的前向传播（今天就能完成）
第二课 → 导数 = 增益（别怕，不需要手推公式）
第三课 → 反向传播 = 信号倒灌（核心，真正的分水岭）
第四课 → 三个本地导数规则（加法、乘法、tanh）
第五课 → 神经元 = 乘法器 + 加法器 + 非线性
第六课 → 训练循环：正向 → 反向 → 更新 → 重复
第七课 → XOR：验证 2-bit 权重到底需要什么
```

---

## 第一课：Value 的前向传播（30 分钟）

**读**： [engine.py](https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py) 前 40 行—— `__init__` 和 `__add__`、`__mul__`、`__repr__`

**核心问题**： 一个数怎么记住自己是从哪两个数算出来的？

```python
a = Value(3.0)
b = Value(2.0)
c = a + b      # c._prev = {a, b}, c._op = '+'

# 跟你硬件中的对应:
# a, b = 两个输入信号
# c = 一个二输入加法器的输出
# _prev = 反推输入引脚
```

**动手**： 打开 Python，手动建几个 Value，print 它们的 `_prev` 和 `_op`。
从 `+`、`*`、`**`、`tanh` 各试一遍。

---

## 第二课：导数只讲直觉（45 分钟）

**不看公式，只看例子：**

```
y = 2x       → x 每变 1，y 变 2    → 导数 = 2
y = x²       → x=3 时，x 微变，y 大约以 6 倍跟着变 → 导数 = 6
y = sin(x)   → x=0 时导数是 1（最敏感），x=π/2 时导数是 0（峰顶，不敏感）
```

**跟你的硬件对应**： 导数 = 这一级电路的增益。

**动手**： 画三条曲线—— y=2x（直线）、y=x²（抛物线）、y=sin(x)。
随手挑几个点，估算"x 微变，y 变多少"。不用任何公式，就用眼睛看陡不陡。

---

## 第三课：反向传播 = 各段增益相乘（1 小时）

**读**： [engine.py](https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py) 的 `backward()` 函数。

**核心问题**： 怎么从输出端把梯度沿着计算图往回倒？

```
你的硬件直觉:

  信号路径: A → [放大器 ×3] → [放大器 ×2] → 输出

  正向: 信号流过去
  反向: 梯度流回来
  规律: 经过放大器 ×3，正向是乘 3，反向也是乘 3
       经过放大器 ×2，正向是乘 2，反向也是乘 2

  全链路梯度 = 3 × 2 = 6
  意思是: 输入端微微一动，输出端以 6 倍跟着动

micrograd 的 backward():
  就是从输出端出发，沿着 _prev 反向走
  每经过一个节点，乘以那个操作的本地导数
  一直走到最开始的输入
```

**动手**： 自己手算一个简单链条的梯度：

```python
x = Value(2.0)
y = x * 3      # y = 6
z = y * 2      # z = 12
z.backward()
print(x.grad)  # 应该是 6，你手算一下验证
```

---

## 第四课：加法、乘法、tanh 的本地导数（1 小时）

**读**： [engine.py](https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py) 的 `_backward` 方法—— 只关注 `+`、`*`、`tanh` 三种。

**三个规则，记在脑子里：**

```
加法:  c = a + b
  → a 的导数 = 1.0 × c 的梯度
  → b 的导数 = 1.0 × c 的梯度
  理解: 信号直接过，不分叉

乘法:  c = a * b
  → a 的导数 = b × c 的梯度   (b 的值决定 a 的重要性)
  → b 的导数 = a × c 的梯度   (a 的值决定 b 的重要性)
  理解: b 越大，a 就越是关键参数

tanh:  c = tanh(a)
  → a 的导数 = (1 - c²) × c 的梯度
  理解: 信号在激励函数的陡坡上过得快，在平台区几乎不传
```

**动手**： 把 Karpathy 视频里手算反向传播那个例子（大概 5 个节点），自己用纸笔走一遍。
然后 Python 跑一遍，对比结果。

---

## 第五课：神经元是怎么用 Value 搭出来的（45 分钟）

**读**： [nn.py](https://github.com/karpathy/micrograd/blob/master/micrograd/nn.py)—— 总共 50 行。

**核心问题**： 一个神经元，从硬件角度看，是什么？

```
硬件视角:
  输入 [x₁, x₂, x₃] → 乘各自的权重 → 求和 → 加偏置 → 过激活函数 → 输出

  = 一组乘法器 + 一个加法器 + 一个非线性模块

micrograd 视角:
  out = tanh(x₁*w₁ + x₂*w₂ + x₃*w₃ + b)

  每个 w 和 b 都是 Value
  全部操作链自动建计算图
```

**动手**： 用 nn.py 创建一个 2 输入 1 输出的神经元，print 它这次的输出。不做训练，只看前向。

---

## 第六课：训练 = 反复跑正向 + 反向 + 更新（1 小时）

**读**： [demo.ipynb](https://github.com/karpathy/micrograd/blob/master/demo.ipynb)—— 从头到尾看一遍。

**核心循环，背下来：**

```python
for i in range(100):
    # 1. 正向传播: 输入 → 模型 → 预测
    y_pred = model(x)

    # 2. 算损失: 预测跟真实值差多少
    loss = (y_pred - y_true) ** 2

    # 3. 反向传播: 算每个参数的梯度
    model.zero_grad()
    loss.backward()

    # 4. 更新参数: 沿着梯度的反方向走一小步
    for p in model.parameters():
        p.data -= 0.01 * p.grad
```

**动手**： 自己写一个训练循环，学 y = 3x + 1。用 5 组 (x, y) 数据。
亲眼看着 w 从 0 一步步跑到 3，b 从 0 跑到 1。

---

## 第七课：XOR 问题——为什么一层不行两层就行（1 小时）

**核心问题**： XOR 是你之前提出的"1-bit 权重只有赞成票"问题的数学版。

```
一层网络: y = w₁x₁ + w₂x₂ + b
  → 只能画直线，XOR 四个点无论如何割不开
  → 你之前说的：纯 0/1 权重表达能力不够

两层网络: y = w₃×tanh(w₁x₁+w₂x₂+b₁) + w₄×tanh(...) + b₂
  → 第一层画两条线，第二层组合它们
  → 等价于：-1/0/+1 三元权重，有了"反对票"
```

**动手**： 用 micrograd 训一个两层的 MLP 去学 XOR 函数，观察 w 的变化。

---

## 学习路径总览

```
第一课 ← 今天就能完成
第二课 ← 别怕，只学到"导数 = 增益"的程度
第三课 ← 这是核心，反向传播 = 信号倒灌
第四课 ← 记三个规则，最多
第五课 ← 神经元 = 乘法器 + 加法器 + 非线性
第六课 ← 到这里你就知道"训练"到底是什么了
第七课 ← 验证你之前对 2-bit 权重的直觉
```

## 参考资料

| 资源 | 链接 |
|------|------|
| micrograd 源码 | <https://github.com/karpathy/micrograd> |
| engine.py | <https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py> |
| nn.py | <https://github.com/karpathy/micrograd/blob/master/micrograd/nn.py> |
| demo.ipynb | <https://github.com/karpathy/micrograd/blob/master/demo.ipynb> |
| Karpathy 视频教程 | <https://www.youtube.com/watch?v=VMj-3S1tku0> |
| 3Blue1Brown 神经网络系列 | <https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi> |
