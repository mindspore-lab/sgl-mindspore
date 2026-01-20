# MindSpore Eagle Speculative Decoding

SGLang on MindSpore supports Eagle (specifically Eagle3) speculative decoding to accelerate inference.

## Run with Speculative Decoding

You can run Qwen3 with Eagle3 speculative decoding using the following command.

```bash
python3 -m sglang.launch_server \
    --device npu \
    --model-impl mindspore \
    --attention-backend ascend \
    --trust-remote-code \
    --tp-size 4 \
    --model-path /path/to/Qwen3-32B \
    --mem-fraction-static 0.8 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path /path/to/Qwen3-32B-Eagle3 \
    --speculative-num-steps 4 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 5
```

## Python API

Please refer to the example script [run_qwen_eagle3.py](../examples/run_qwen_eagle3.py) for Python API usage.
