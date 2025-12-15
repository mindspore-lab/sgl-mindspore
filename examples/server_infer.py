import argparse

import requests
from sglang.utils import launch_server_cmd, print_highlight, wait_for_server

parser = argparse.ArgumentParser("sglang-mindspore offline infer")

parser.add_argument(
    "--model-path,",
    required=False,
    default="/home/ckpt/Qwen3-8B",
    help="the model path",
    type=str,
)

args = parser.parse_args()


def main():
    server_process, port = launch_server_cmd(
        f"python3 -m sglang.launch_server \
            --model-path {args.model_path} \
            --host 0.0.0.0 \
            --device npu \
            --model-impl mindspore \
            --attention-backend ascend \
            --mem-fraction-static 0.8 \
            --tp-size 1 \
            --dp-size 1",
        port=37654,
    )

    wait_for_server(f"http://localhost:{port}")

    url = f"http://localhost:{port}/generate"
    data = {
        "text": "The capital of France is",
        "sampling_params": {"temperature": 0, "max_new_tokens": 100},
    }

    response = requests.post(url, json=data)
    print_highlight(response.json())


if __name__ == "__main__":
    main()
