[中文版](README_zh.md)

# SGL-MindSpore

This is the MindSpore model repository for [SGLang](https://github.com/sgl-project/sglang). Please prepare a Python 3.11 environment, then install SGLang and this repository.

## Support Matrix

| Model | Ascend 910B/C | Ascend 310P |
|  ----  | ----  | ----  |
| Qwen-3 Dense | &#x2705; | &#x2705; |
| Qwen-3 MoE | &#x2705; | &#x2705; |
| DeepSeek V3 | &#x2705; |  |

## Installation

This is a step-by-step guide helping you to run MindSpore models in SGLang.

### 1. Install CANN

Please install the [CANN 8.3.RC1 community edition](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0000.html). This version is compatible with the MindSpore 2.7.1 nightly build. If you need to use a released version of MindSpore, then you **must install CANN 8.5 and update the NPU driver version to 25.5**. The packages you need to install include **toolkit, kernels and nnal**. Please choose the appropriate packages according to your NPU type.

### 2. Install SGLang for the Ascend platform

```
git clone https://github.com/sgl-project/sglang.git
cd sglang
cp python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_npu]"
```

### 3. Install sgl-kernel-npu

This step requires GCC version >= 9. You can use `gcc -v` to check you GCC version. If it is lower than 9, please install a newer version.

```
git clone https://github.com/sgl-project/sgl-kernel-npu.git
cd sgl-kernel-npu
bash build.sh -a kernels
pip install output/*.whl
```

### 4. Install MindSpore models repo
```
git clone https://github.com/mindspore-lab/sgl-mindspore.git
cd sgl-mindspore
pip install -e .
```
It will automatically install MindSpore 2.7.1 nightly build. If you need to use a released version, please `pip install mindspore==2.7.2` and upgrade CANN to 8.5.

## Usage

Please set the following environment variables before you run:
```
export ASCEND_RT_VISIBLE_DEVICES=0  # specify the NPU device id
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python  # avoid protobuf version mismatch
```

### Demo for offline inference:

```
python examples/offline_infer.py --model-path /path/to/your/model
```

To enable data or tensor parallelism, please modify `dp_size` and `tp_size` in the above script.

### Demo for server inference:

This script starts a server and sends a sample request in Python.
```
python examples/server_infer.py --model-path /path/to/your/model
```

Alternatively, start a server with the bash script and send a request with the curl command:
```
bash examples/start_server.sh
```
Please refer to the [official SGLang doc](https://docs.sglang.io/basic_usage/send_request.html#Using-cURL) for request format.

### Benchmark

To benchmark a single batch：
```
bash examples/bench_one_batch.sh
```

To benchmark in server mode, first start a server, then run：
```
bash examples/bench_serving.sh
```
The `host` and `port` arguments must match the server's setting.

You can modify the test arguments inside the scripts.


## License

Apache License 2.0
