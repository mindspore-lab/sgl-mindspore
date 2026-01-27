[中文版](README_zh.md)

# SGL-MindSpore

This is the MindSpore model repository for [SGLang](https://github.com/sgl-project/sglang). Please prepare a Python 3.11 environment, then install SGLang and this repository.

## Support Matrix

| Model | Ascend 910B/910C | Ascend 310P |
|  ----  | ----  | ----  |
| Qwen-3 Dense | &#x2705; | &#x2705; |
| Qwen-3 MoE | &#x2705; | &#x2705; |
| DeepSeek V3 | &#x2705; |  |

## Installation

This is a step-by-step guide helping you to run MindSpore models in SGLang.

### 1. Install CANN

Please install the [CANN 8.5 community edition](https://www.hiascend.com/cann/download). The packages you need to install include **toolkit, kernels and nnal**. If you are using CANN 8.3 and do not wish to upgrade, you may use a nightly build of MindSpore 2.7.1. Please refer to Section 4 for details.

### 2. Install SGLang for the Ascend platform

```
git clone https://github.com/sgl-project/sglang.git
cd sglang
cp python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_npu]"
```

### 3. Install sgl-kernel-npu

First download and unzip the packages:
```
# for 910B or 310P
wget https://github.com/sgl-project/sgl-kernel-npu/releases/download/2026.01.21/sgl-kernel-npu_2026.01.21_8.5.0_910b.zip -O tmp.zip && unzip tmp.zip && rm -f tmp.zip

# for A3 (910C)
wget https://github.com/sgl-project/sgl-kernel-npu/releases/download/2026.01.21/sgl-kernel-npu_2026.01.21_8.5.0_a3.zip -O tmp.zip && unzip tmp.zip && rm -f tmp.zip
```
Then install the downloaded packages:
```
pip install output/*.whl
```
Alternatively, you can install from source following the guides: https://github.com/sgl-project/sgl-kernel-npu/

### 4. Install MindSpore models repo
```
git clone https://github.com/mindspore-lab/sgl-mindspore.git
cd sgl-mindspore
pip install -e .
```
It will install MindSpore 2.8.0. If you are using CANN 8.3 and do not wish to change the CANN version, install MindSpore 2.7.1 nightly build instead:
```
pip install http://repo.mindspore.cn/mindspore/mindspore/version/202512/20251211/master_20251211010018_65a9c09590c14021cbe38cb8720acb5dad022901_newest/unified/aarch64/mindspore-2.7.1-cp311-cp311-linux_aarch64.whl
```


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

### Run on Ascend 310P

Triton is not supported on Ascend 310P. Directly running the code will cause triton compiler errors. Please apply the patch under your SGLang directory:
```
cd /path/to/sglang
git apply --3way /path/to/sgl_mindspore/patches/310p.patch
```
If you want to update SGLang's code, you'll need to discard the patch, pull the newest code, and apply the patch again.


## License

Apache License 2.0
