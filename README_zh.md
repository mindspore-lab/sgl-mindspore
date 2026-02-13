[English](README.md)

# SGL-MindSpore

本仓库为[SGLang](https://github.com/sgl-project/sglang)提供MindSpore模型支持。您需要准备Python 3.11环境，并安装SGLang和本仓库。

## 支持矩阵

| 模型 | Ascend 910B/910C | Ascend 310P |
|  ----  | ----  | ----  |
| Qwen-3 Dense | &#x2705; | &#x2705; |
| Qwen-3 MoE | &#x2705; | &#x2705; |
| DeepSeek V3 | &#x2705; |  |

## 安装

我们会一步步指引，帮助您在SGLang中运行MindSpore模型。

### 1. 安装CANN

请安装[CANN 8.5社区版](https://www.hiascend.com/cann/download)。需要安装的软件包包括**toolkit, kernels和nnal**。如果您使用CANN 8.3且不希望更换版本，也可以选择MindSpore 2.7.1每日构建版。详情请参考第4节。

### 2. 基于昇腾平台，安装SGLang

```
git clone https://github.com/sgl-project/sglang.git
cd sglang
cp python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_npu]"
```

### 3. 安装sgl-kernel-npu

首先下载并解压安装包:
```
# 910B 或 310P
wget https://github.com/sgl-project/sgl-kernel-npu/releases/download/2026.01.21/sgl-kernel-npu_2026.01.21_8.5.0_910b.zip -O tmp.zip && unzip tmp.zip && rm -f tmp.zip

# A3 (910C)
wget https://github.com/sgl-project/sgl-kernel-npu/releases/download/2026.01.21/sgl-kernel-npu_2026.01.21_8.5.0_a3.zip -O tmp.zip && unzip tmp.zip && rm -f tmp.zip
```
然后安装:
```
pip install output/*.whl
```

如果您在运行SGLang时遇到类似`version GLIBCXX_3.4.29 not found`的报错，表示您系统的GLIBCXX版本低于编译环境的版本。这种情况下建议从源码构建：
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
该命令会安装MindSpore 2.8.0。如果您使用CANN 8.3且不希望更换版本，请安装MindSpore 2.7.1每日构建版：
```
pip install http://repo.mindspore.cn/mindspore/mindspore/version/202512/20251211/master_20251211010018_65a9c09590c14021cbe38cb8720acb5dad022901_newest/unified/aarch64/mindspore-2.7.1-cp311-cp311-linux_aarch64.whl
```

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

### 在昇腾310P上运行

昇腾310P不支持Triton。直接运行代码会导致triton编译错误。请根据您的SGLang安装目录应用补丁：
```
cd /path/to/sglang
git apply --3way /path/to/sgl-mindspore/patches/310p.patch
```
如果您需要更新SGLang的代码，请先丢弃补丁，拉取最新代码，然后重新应用补丁。

## 许可证

Apache License 2.0
