#!/usr/bin/env python3
"""
scripts/generate_data.py: 调用 DataBuilder 构建各种数据集，并保存到指定路径
"""
import os
import argparse
from dotenv import load_dotenv

# 确保项目根目录已安装 databuilder、llm_apis 包
from src.dataset_builder import DataBuilder
from llm_apis.llm_factory import get_llm_api

# 加载 .env 中的 API Key 和模型路径
load_dotenv()

# 支持的构造步骤
ALL_STEPS = [
    "close_book",        # 生成 memory answers
    "counter",           # 生成 counter answers
    "direct",            # 生成 direct evidence
    "direct_patch",      # 生成 paraphrase patch
    "sentence",          # 生成 sentence-level evidence
    "update_sentence",   # 更新 sentence-level evidence
    "support",           # 生成 supporting evidence
    "update_support"     # 更新 supporting evidence
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate datasets for context-faithfulness experiments"
    )
    p.add_argument("--data_path", type=str, required=True,
                   help="原始 JSONL 数据路径，例如 data/raw/nq.jsonl 或 popQA.jsonl")
    p.add_argument("--save_path", type=str, required=True,
                   help="输出保存目录，例如 build_dataset/NQ_openai")
    p.add_argument("--dataset_name", type=str, required=True,
                   help="数据集标签，如 NQ 或 popQA，用于多处方法调用中的 dataset_name 参数")
    p.add_argument("--test_llm_type", type=str, required=True,
                   choices=["openai", "claude", "llama2", "llama3"],
                   help="用于测试的 LLM 类型")
    p.add_argument("--test_llm_model", type=str, required=False,
                   help="测试 LLM 的模型名或本地路径，如 gpt-3.5-turbo 或 /path/to/llama2-model")
    p.add_argument("--worker_llm_type", type=str, default="openai",
                   choices=["openai", "claude", "llama2", "llama3"],
                   help="用于生成数据（Worker）的 LLM 类型")
    p.add_argument("--worker_llm_model", type=str, required=False,
                   help="Worker LLM 的模型名或本地路径")
    p.add_argument("--steps", nargs="+", choices=ALL_STEPS, default=["close_book", "counter", "direct"] ,
                   help=f"运行的构造步骤，支持: {ALL_STEPS}")
    p.add_argument("--sentence_numbers", nargs="+", type=int, default=[2,3],
                   help="句子级别 evidence 的句子数列表，例如 2 3")
    p.add_argument("--evidence_type", type=str, default="Detailed",
                   help="生成Supporting Evidence时的类型，如 Detailed, Reference 等")
    return p.parse_args()


def instantiate_llm(llm_type: str, model_name: str = None, model_path: str = None):
    """
    根据 llm_type 返回对应的 LLM API 实例。
    对于 openai/claude 使用环境变量中的 KEY；
    对于 llama2/llama3 使用本地模型路径（model_path）。
    model_name 用于 openai/claude 指定模型版本。
    """
    if llm_type == "openai":
        return get_llm_api("openai", model=model_name)
    if llm_type == "claude":
        return get_llm_api("claude", model=model_name)
    if llm_type == "llama2":
        return get_llm_api("llama2", model_path=model_path)
    if llm_type == "llama3":
        return get_llm_api("llama3", model_path=model_path)
    raise ValueError(f"Unsupported LLM type: {llm_type}")


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    # 准备 LLM 实例
    test_llm = instantiate_llm(
        llm_type=args.test_llm_type,
        model_name=args.test_llm_model,
        model_path=args.test_llm_model
    )
    worker_llm = instantiate_llm(
        llm_type=args.worker_llm_type,
        model_name=args.worker_llm_model,
        model_path=args.worker_llm_model
    )

    # 初始化 DataBuilder
    db = DataBuilder(
        data_path=args.data_path,
        save_dir=args.save_path,
        test_llm=test_llm,
        worker_llm=worker_llm,
        # test_llm_name=args.test_llm_model or args.test_llm_type,
        # worker_llm_name=args.worker_llm_model or args.worker_llm_type,
        dataset_name=args.dataset_name,
    )

    # 依次执行用户指定的构造步骤
    if "close_book" in args.steps:
        print("[Step] close book answers...")
        db.close_book()
    if "counter" in args.steps:
        print("[Step] counter answers...")
        db.counter()
    if "direct" in args.steps:
        print("[Step] direct evidence...")
        db.direct()
    if "direct_patch" in args.steps:
        print("[Step] direct evidence patch...")
        db.direct_patch(patch_size=2)
    if "sentence" in args.steps:
        print("[Step] sentence-level evidence...")
        db.sentence(sentence_numbers=args.sentence_numbers)
    if "update_sentence" in args.steps:
        print("[Step] update sentence-level evidence...")
        db.update_sentence(sentence_numbers=args.sentence_numbers)

    # print("[Done] Data generation complete.")


if __name__ == "__main__":
    main()
