import os
import json
import logging
import torch
import jsonlines
import re

from src.utils import (
    load_jsonl, save_jsonl,
    diff_string, get_overlap,
    check_question_type, key_term,
    close_book_prompt, change_answer_to_counter_prompt,
    paraphrase_prompt, patch_paraphrase_prompt,
    sentence_evidence_prompt, supporting_evidence_prompt_detailed,
    supporting_evidence_prompt_reference
)

from src.nli_model import NLIModel
from tqdm import tqdm


class DataBuilder:
    """
    构造 NQ / PopQA 数据集：闭卷回答、对抗回答、不同风格证据生成与验证
    支持：close_book, counter, direct, direct_patch,
         sentence, update_sentence, support, update_support
    """
    def __init__(
        self,
        data_path: str,
        save_dir: str,
        test_llm,
        worker_llm,
        dataset_name: str
    ):
        self.data_path = data_path
        self.save_dir = save_dir
        self.dataset_name = dataset_name  # e.g. "NQ" or "popQA"
        self.test_llm = test_llm
        self.worker_llm = worker_llm
        self.nli = NLIModel()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        os.makedirs(save_dir, exist_ok=True)
        self.raw_data = load_jsonl(data_path)
        self.error_questions = set()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _save(self, data, name: str):
        path = os.path.join(self.save_dir, f"{self.dataset_name}_{name}.jsonl")
        save_jsonl(data, path)
        self.logger.info(f"Saved {len(data)} records to {path}")

    # ---------- Step 1: Memory Answer (Close-book) ----------

    def close_book(self):
        out = []
        for unit in tqdm(self.raw_data, desc="Processing close-book answers"):
            q = unit.get('query') or unit.get('question')
            qtype = check_question_type(q)
            prompt = close_book_prompt.format(q)
            answer = self._retry_llm(self.test_llm, prompt, 
                                     lambda a: self._validate_close_book(a, q))
            unit['memory_answer'] = answer
            unit['memory_flag'] = answer not in [None, 'Not found']
            out.append(unit)
        self._save(out, 'memory_answer')

    def _validate_close_book(self, ans, question):
        if not ans or ans in ['Not found']: return False
        lines = [l for l in ans.splitlines() if l.strip()]
        if len(lines) != 1: return False
        ans = lines[0].strip()
        overlap = get_overlap(ans, question)
        return len(overlap.split()) >= 2 and 4 <= len(ans.split()) <= 20

    # ---------- Step 2: Counter-memory Answer ----------
    def counter(self):
        data = load_jsonl(os.path.join(self.save_dir, f"{self.dataset_name}_memory_answer.jsonl"))
        out = []
        for unit in tqdm(data, desc="Processing counter answers"):
            close_ans = unit['memory_answer']
            q = unit.get('query') or unit.get('question')
            kt = key_term(check_question_type(q))
            if not unit['memory_flag'] or kt == 'other':
                unit['counter_answer'] = 'Not found'
                unit['counter_flag'] = False
            else:
                prompt = change_answer_to_counter_prompt.format(close_ans, kt)
                ans = self._retry_llm(self.worker_llm, prompt,
                                      lambda a: self._validate_contradiction(a, close_ans))
                unit['counter_answer'] = ans
                unit['counter_flag'] = ans not in [None, 'Not found']
            out.append(unit)
        self._save(out, 'counter_answer')

    def _validate_contradiction(self, ans, close_ans):
        if not ans: return False
        cpart, _ = diff_string(close_ans, ans)
        if len(cpart)/len(close_ans) > 0.6: return False
        label = self.nli.nli_inference(close_ans, ans)
        return label == 'contradiction'

    # ---------- Step 3: Direct Evidence ----------
    def direct(self):
        data = load_jsonl(os.path.join(self.save_dir, f"{self.dataset_name}_counter_answer.jsonl"))
        out = []
        for unit in tqdm(data, desc="Processing direct evidence"):
            cnt = unit['counter_answer']
            if not unit['counter_flag']:
                unit['direct_evidence'] = 'Not found'
                unit['direct_flag'] = False
            else:
                prompt = paraphrase_prompt.format(cnt)
                ans = self._retry_llm(self.worker_llm, prompt,
                                        lambda a: self._validate_direct_evidence(a, cnt))
                unit['direct_evidence'] = ans
                unit['direct_flag'] = ans not in [None, 'Not found']
            out.append(unit)
        self._save(out, 'direct_evidence')

    def _validate_direct_evidence(self, ans, cnt):
        if not ans or ans in ['Not found']: return False
        lines = [l for l in ans.splitlines() if l.strip()]
        if len(lines) != 1: return False
        line = lines[0].strip()
        if self.nli.nli_inference(line, cnt) != 'entailment' or self.nli.nli_inference(line, cnt) != 'entailment':
            return False
        return True

    # ---------- Step 4: Paraphrase Patch ----------
    def direct_patch(self, patch_size: int = 2):
        data = load_jsonl(os.path.join(self.save_dir, f"{self.dataset_name}_counter_answer.jsonl"))
        out = []
        for unit in tqdm(data, desc="Processing paraphrase patches"):
            cnt = unit['counter_answer']
            if not unit['counter_flag']:
                unit['paraphrase_patch'] = []
            else:
                prompt = patch_paraphrase_prompt.format(patch_size, cnt)
                ans = self._retry_llm(self.worker_llm, prompt,
                                      lambda a: self._validate_patch(a, cnt, patch_size))
                if ans == 'Not found':
                    unit['paraphrase_patch'] = []
                    unit['paraphrase_patch_flag'] = False
                else:
                    paraphrases = [line.strip() for line in ans.splitlines() if line.strip()]
                    paraphrases = [re.sub(r'^\d+\.', '', r.strip().strip("-").strip()) for r in paraphrases]
                    unit['paraphrase_patch'] = paraphrases
                    unit['paraphrase_patch_flag'] = len(unit['paraphrase_patch']) >= patch_size
            out.append(unit)
        self._save(out, 'paraphrase_patch')

    def _validate_patch(self, ans, cnt, patch_size=2):
        ans = ans.splitlines()
        if not ans or len(ans) != patch_size: return False
        for line in ans:
            if not line.strip(): continue
            line = line.strip().strip("-").strip()
            line = re.sub(r'^\d+\.', '', line)
            if not self._validate_direct_evidence(line, cnt):
                return False
        return True

    # ---------- Step 5: Sentence-level Evidence ----------
    def sentence(self, sentence_numbers=[2,3]):
        data = load_jsonl(os.path.join(self.save_dir, f"{self.dataset_name}_paraphrase_patch.jsonl"))
        out = []
        for unit in tqdm(data, desc="Processing sentence-level evidence"):
            cnt = unit['counter_answer']
            mem = unit['memory_answer']
            if not unit['counter_flag'] or not unit['memory_flag']:
                unit.update({f'{n}_sent': 'Not found' for n in sentence_numbers})
                unit.update({f'{n}_sent_flag': False for n in sentence_numbers})
            else:
                # generate sentence-level evidence
                for n in sentence_numbers:
                    prompt = sentence_evidence_prompt.format(cnt, n)
                    ans = self._retry_llm(self.worker_llm, prompt,
                                          lambda a: self._validate_sentence_evidence(a, cnt, mem, n))
                    unit[f'{n}_sent'] = ans
                    unit[f'{n}_sent_flag'] = ans not in [None, 'Not found']
            out.append(unit)
        self._save(out, 'sentence_evidence')

    # sentence evidence has enough sentences, and nli inference is satisfied
    def _validate_sentence_evidence(self, ans, cnt, mem, sentence_number):
        if not ans or ans in ['Not found']: return False
        lines = [l for l in ans.split(". ") if l.strip()]
        if len(lines) != sentence_number: return False
        if self.nli.nli_inference(ans, cnt) != 'entailment' or self.nli.nli_inference(ans, mem) == 'entailment':
            return False
        
        return True

    # # ---------- Step 6: Supporting Evidence ----------
    # def support(self, evidence_type='Detailed'):
    #     data = load_jsonl(os.path.join(self.save_dir, f"{self.dataset_name}_paraphrase_patch.jsonl"))
    #     out = []
    #     for unit in data:
    #         cnt = unit['counter_answer']
    #         if unit.get('paraphrase_patch'):
    #             prompt = (supporting_evidence_prompt_detailed if evidence_type=='Detailed'
    #                       else supporting_evidence_prompt_reference).format(cnt)
    #             ans = self._retry_llm(self.worker_llm, prompt,
    #                                   lambda a, _: self.nli.nli_inference(a, cnt) == 'entailment')
    #             unit[f'support_{evidence_type}'] = ans
    #             unit[f'support_{evidence_type}_flag'] = ans not in [None,'Not found']
    #         out.append(unit)
    #     self._save(out, f'support_{evidence_type}')

    # ---------- Update Methods ----------
    def update_sentence(self, sentence_numbers=[2,3]):
        # if update_sentence not exists, load from sentence_evidence, else load from update_sentence_evidence
        if not os.path.exists(os.path.join(self.save_dir, f"{self.dataset_name}_update_sentence_evidence.jsonl")):
            data = load_jsonl(os.path.join(self.save_dir, f"{self.dataset_name}_sentence_evidence.jsonl"))
        else:
            data = load_jsonl(os.path.join(self.save_dir, f"{self.dataset_name}_update_sentence_evidence.jsonl"))
        out = []
        for unit in tqdm(data, desc="Updating sentence-level evidence"):
            cnt = unit['counter_answer']
            mem = unit['memory_answer']
            if not unit['counter_flag'] or not unit['memory_flag']:
                unit.update({f'{n}_sent': 'Not found' for n in sentence_numbers})
                unit.update({f'{n}_sent_flag': False for n in sentence_numbers})
                out.append(unit)
                continue
            for n in sentence_numbers:
                if not unit.get(f'{n}_sent_flag', False):
                    prompt = sentence_evidence_prompt.format(cnt, n)
                    ans = self._retry_llm(self.worker_llm, prompt,
                                          lambda a: self._validate_sentence_evidence(a, cnt, mem, n))
                    unit[f'{n}_sent'] = ans
                    unit[f'{n}_sent_flag'] = ans not in [None, 'Not found']
            out.append(unit)
        self._save(out, 'update_sentence_evidence')

    # ---------- Helper: retry wrapper ----------
    def _retry_llm(self, llm, prompt, validate_fn, retries=5, **gen_kwargs):
        for _ in range(retries):
            text, _ = llm.generate(prompt, **gen_kwargs)
            print(f"[LLM] Generated text: {text}")
            print("*" * 20)
            ans = text.strip().replace('Answer:','')
            if validate_fn(ans):
                return ans
        return 'Not found'
