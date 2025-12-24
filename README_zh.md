[English](README.md)

# SGL-MindSpore

本仓库为[SGLang](https://github.com/sgl-project/sglang)提供MindSpore模型支持。您需要准备Python 3.11环境，并安装SGLang和本仓库。

## 支持矩阵

| 模型 | Ascend 910B/C | Ascend 310P |
|  ----  | ----  | ----  |
| Qwen-3 Dense | &#x2705; | &#x2705; |
| Qwen-3 MoE | &#x2705; | &#x2705; |
| DeepSeek V3 | &#x2705; |  |

## 安装

我们会一步步指引，帮助您在SGLang中运行MindSpore模型。

### 1. 安装CANN

请安装[社区版8.3.RC1](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0000.html)。
该版本适合MindSpore 2.7.1每日构建版。如果您需要使用MindSpore发行版，**必须安装CANN 8.5并更新NPU驱动至25.5**。
需要安装的软件包包括**toolkit, kernels和nnal**。请根据您的NPU型号选择合适的软件包。

### 2. 基于昇腾平台，安装SGLang

```
git clone https://github.com/sgl-project/sglang.git
cd sglang
cp python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_npu]"
```

### 3. 安装sgl-kernel-npu

此步骤需要GCC版本>=9。您可以使用 `gcc -v` 检查GCC版本。如果版本低于9，请安装更新的版本。

```
git clone https://github.com/sgl-project/sgl-kernel-npu.git
cd sgl-kernel-npu
bash build.sh -a kernels
pip install output/*.whl
```

### 4. 安装MindSpore模型仓库
```
git clone https://github.com/mindspore-lab/sgl-mindspore.git
cd sgl-mindspore
pip install -e .
```
该命令会自动安装MindSpore 2.7.1每日构建包。如果您需要使用发行版，请执行`pip install mindspore==2.7.2`，并升级CANN至8.5。

## 使用示范

运行前请设置以下环境变量：
```
export ASCEND_RT_VISIBLE_DEVICES=0  # 指定你想使用的NPU设备ID
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python  # 避免protobuf版本不匹配
```

### 离线推理演示：

```
python examples/offline_infer.py --model-path /path/to/your/model
```

要启用数据或张量并行，请修改以上脚本中的 `dp_size` 和 `tp_size`。

### 服务化推理演示：

此脚本使用Python启动SGLang server，并发送一个样例请求。
```
python examples/server_infer.py --model-path /path/to/your/model
```

也可以使用bash脚本启动服务，并使用curl命令发送请求：
```
bash examples/start_server.sh
```
请求格式请参阅[SGLang官方文档](https://docs.sglang.io/basic_usage/send_request.html#Using-cURL).

### 性能测试

测试单个batch的推理性能：
```
bash examples/bench_one_batch.sh
```

测试服务化推理性能，请先启动服务，然后运行
```
bash examples/bench_serving.sh
```
`host`和`port`参数必须和启动服务时的参数匹配。

可以在脚本内修改测试参数。


## 许可证

Apache License 2.0
