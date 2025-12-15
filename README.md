# PyTorch学习笔记

一、PyTorch 的核心架构与主要模块
从宏观上看，PyTorch 的架构可以分为底层核心层和上层应用层，主要模块如下：
1. 核心基础模块：张量（Tensor）
Tensor 是 PyTorch 中最基础的数据结构，相当于 “深度学习版的 NumPy 数组”，它是存储和处理数据的基本单元。
可以在 CPU/GPU 上运行，支持各种数值运算（加减乘除、矩阵运算、卷积等）。
所有深度学习模型的输入、输出以及模型的参数，本质上都是 Tensor。
与 NumPy 的区别：Tensor 支持自动求导（Autograd）和 GPU 加速。
2. 核心功能模块：自动求导（Autograd）
Autograd 是 PyTorch 实现反向传播的核心，它会自动记录 Tensor 的运算过程，并根据链式法则计算梯度（导数），这是训练深度学习模型的关键。
当你对 Tensor 设置requires_grad=True时，Autograd 会追踪所有对该张量的操作。
调用.backward()时，会自动计算所有相关的梯度，并存储在.grad属性中。
3. 模型构建模块：nn 模块
torch.nn是 PyTorch 用于构建深度学习模型的核心模块，它提供了大量预定义的层（如卷积层、全连接层、循环层）、激活函数、损失函数等，让你可以像搭积木一样构建模型。
nn.Module：所有自定义模型的基类，你只需要继承它，并实现forward()方法（定义前向传播逻辑），反向传播会由 Autograd 自动完成。
预定义层：如nn.Conv2d（二维卷积）、nn.Linear（全连接）、nn.LSTM（长短期记忆网络）等。
损失函数：如nn.CrossEntropyLoss（交叉熵损失）、nn.MSELoss（均方误差损失）等。
4. 优化器模块：optim 模块
torch.optim提供了各种优化算法，用于更新模型的参数，最小化损失函数，比如 SGD、Adam、RMSprop 等。
你只需要将模型的参数传入优化器，然后在训练循环中调用optimizer.step()来更新参数，optimizer.zero_grad()来清空梯度（避免梯度累积）。
5. 数据处理模块：utils.data
torch.utils.data提供了处理数据集的工具，核心是Dataset和DataLoader：
Dataset：抽象类，用于定义自定义数据集，需要实现__getitem__（获取单个样本）和__len__（获取数据集长度）方法。
DataLoader：将 Dataset 包装成迭代器，支持批量加载数据、打乱数据、多线程加载等，是训练模型时读取数据的常用工具。

二、PyTorch 的典型使用流程（宏观）
1.数据准备：使用Dataset和DataLoader加载并预处理数据（如归一化、数据增强）。
2.构建模型：继承nn.Module，定义模型的层和前向传播逻辑。
3.定义损失函数和优化器：选择合适的损失函数（如交叉熵）和优化器（如 Adam）。
4.训练模型：在循环中执行 “前向传播（计算预测值）→计算损失→反向传播（计算梯度）→更新参数” 的步骤。
5.模型评估与部署：用测试集评估模型性能，之后可以将模型导出为 ONNX 格式，部署到生产环境（如 TensorRT、ONNX Runtime）。