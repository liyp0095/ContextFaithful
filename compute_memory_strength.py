#!/usr/bin/env python3
"""
scripts/compute_memory_strength.py: 统一计算 NQ 和 popQA 数据集的 memory strength，支持闭卷回答、增量更新、结果顺序保留。
"""
import os
import argparse
import json
import math

from src.utils import load_jsonl, save_jsonl, close_book_prompt, answer_consistency_prompt
from src.nli_model import NLIModel
from llm_apis.llm_factory import get_llm_api
from tqdm import tqdm


def load_items(dataset, questions_file, paraphrase_file, answered_file):
    """
    加载待回答条目：
      - dataset: 'nq' 或 'popqa'
      - questions_file: 原始 JSONL
      - paraphrase_file: JSON
      - answered_file: 已回答 JSON（闭卷答案）
    返回条目列表，每条包含 question, paraphrases, 以及 NQ 的 uid/question_type 或 popQA 的 relation
    """
    items = []
    raw = load_jsonl(questions_file)
    with open(paraphrase_file, 'r', encoding='utf-8') as f:
        paras = json.load(f)

    # 构造 paraphrase 映射
    if dataset.lower() == 'nq':
        para_map = {p['Question']: p['Paraphrases'] for p in paras}
    else:
        para_map = {rel: info['Paraphrases'] for rel, info in paras.items()}

    # 已回答集合
    answered = {}
    if os.path.exists(answered_file):
        with open(answered_file, 'r', encoding='utf-8') as f:
            answered = {u['question']: u['answers'] for u in json.load(f)}

    # 生成待回答条目
    if dataset.lower() == 'nq':
        for unit in raw:
            q = unit.get('question')
            if not unit.get('direct_flag', True):
                continue
            if q in answered:
                continue
            items.append({
                'uid': unit['uid'],
                'question': q,
                'question_type': unit.get('question_type',''),
                'paraphrases': para_map.get(q, [])
            })
    else:
        for unit in raw:
            q = unit.get('question')
            rel = unit.get('relation')
            subj = unit.get('subject','')
            if q in answered:
                continue
            templates = para_map.get(rel, [])
            phr = [t.replace('[Subject]', subj) for t in templates]
            items.append({
                'question': q,
                'relation': rel,
                'paraphrases': phr
            })
    return items


def collect_answers(llm, items, answered_file):
    """
    对 paraphrases 依次做闭卷回答，并追加写入 answered_file。
    返回所有回答条目（包括 answers 字段）。
    """
    out = []
    for idx, unit in enumerate(tqdm(items, desc="Collecting closed-book answers")):
        answers = []
        for phr in unit['paraphrases']:
            prompt = close_book_prompt.format(phr)
            ans, _ = llm.generate(prompt)
            answers.append(ans.strip())
        unit['answers'] = answers
        out.append(unit)

    # 合并写入文件
    if os.path.exists(answered_file):
        existing = json.load(open(answered_file, 'r', encoding='utf-8'))
        existing.extend(out)
        data = existing
    else:
        data = out
    with open(answered_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(out)} answered items to {answered_file}")
    return data

def check_two_answer_consistency(llm, question, answer1, answer2):
    prompt = answer_consistency_prompt.format(question, answer1, answer2)
    text, _ = llm.generate(prompt)
    if "Same" in text:
        return True 
    else:
        return False

def compute_strength(answered_items, nli_model, llm=None):
    """
    基于 NLI 聚类 answers 并计算熵作为 memory strength。
    answered_items: 列表，每条包含 'question' 与 'answers'
    返回 strength 列表，每条 {'question','strength','clusters'}。
    """
    results = []
    for unit in tqdm(answered_items, desc="Computing memory strength"):
        answers = unit['answers']
        clusters = []
        for i, a in enumerate(answers):
            placed = False
            for c in clusters:
                j = c[0]
                if (nli_model.nli_inference(a, answers[j]) == 'entailment' and
                    nli_model.nli_inference(answers[j], a) == 'entailment'):
                    c.append(i)
                    placed = True
                    break
                elif check_two_answer_consistency(llm, unit['question'], answers[j], a):
                    c.append(i)
                    placed = True
                    break
            if not placed:
                clusters.append([i])
        n = len(answers) if answers else 1
        entropy = 0.0
        for c in clusters:
            p = len(c) / n
            entropy += p * math.log(p)
        results.append({
            'question': unit.get('question'),
            'strength': entropy,
            'clusters': clusters
        })
    return results

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
    parser = argparse.ArgumentParser(description="Compute memory strength for NQ or popQA")
    parser.add_argument('--dataset', required=True, choices=['nq','popqa'])
    parser.add_argument('--questions', required=True, help='Questions JSONL file')
    parser.add_argument('--paraphrases', required=True, help='Paraphrases JSON file')
    parser.add_argument('--answered', required=True, help='Answered JSON file')
    parser.add_argument('--output', required=True, help='Output strength JSONL')
    parser.add_argument('--test_llm', default='llama3', choices=['openai','claude','llama2','llama3'])
    parser.add_argument('--test_model', required=True, help='Model name or path')
    parser.add_argument('--worker_llm', default='openai', choices=['openai','claude','llama2','llama3'])
    parser.add_argument('--worker_model', required=True, help='Worker model name or path')
    args = parser.parse_args()

    # 实例化 LLM
    test_llm = instantiate_llm(args.test_llm, model_name=args.test_model, model_path=args.test_model)
    print(f"Using test LLM: {args.test_llm}, model: {args.test_model}")
    worker_llm = instantiate_llm(args.worker_llm, model_name=args.worker_model, model_path=args.worker_model)
    print(f"Using Worker LLM: {args.worker_llm}, model: {args.worker_model}")

    # 1. load items
    items = load_items(args.dataset, args.questions, args.paraphrases, args.answered)
    items = items
    # 2. collect closed-book answers incrementally
    answered = collect_answers(test_llm, items, args.answered)
    # 3. compute strength with incremental and ordering
    nli = NLIModel()
    # load existing strengths
    if os.path.exists(args.output):
        existing = load_jsonl(args.output)
        existing_qs = {u.get('question') for u in existing}
    else:
        existing = []
        existing_qs = set()
    to_compute = [u for u in answered if u.get('question') not in existing_qs]
    new_strengths = []
    if to_compute:
        print(f"Computing memory strength for {len(to_compute)} new items...")
        new_strengths = compute_strength(to_compute, nli, llm=worker_llm)
    else:
        print("No new items to compute strength.")
    all_strengths = existing + new_strengths
    # reorder to match answered sequence
    strength_map = {u['question']: u for u in all_strengths}
    ordered = [strength_map[u['question']] for u in answered if u['question'] in strength_map]
    # 4. save
    save_jsonl(ordered, args.output)
    print(f"Memory strength saved to {args.output} (total {len(ordered)} records)")

if __name__ == '__main__':
    main()
