# ContextFaithful

A tool for generating data and computing memory strength in context-faithful scenarios.

## Overview

This repository provides tools and scripts to:

1. **Generate context-faithfulness datasets** (NQ or popQA) with memory answers, counter answers, and various evidence formats
2. **Compute memory strength of LLMs** by measuring answer consistency across paraphrases

## Installation

```bash
# Requires Python 3.11
git clone <repository-url>
cd ContextFaithful
pip install -r requirements.txt
```

Change .env.example to .env and add your keys for OPENAI and ANTHROPIC there. 

## Data Generation

### Input Format

Input files should be in JSON format with the following structure:

```json
{
    "uid": "corpus-sub-0_6357c3655b524feb8d0e398ff61dfabf",
    "question": "how many episodes are in chicago fire season 4",
    "question_type": "how_many"
}
```

> **Note:** `question_type` is optional. See `./data/datasets/NQ_data_question.json` for an example.

### Output Format

Generated data will be saved in the following format:

```json
{
    "uid": "corpus-sub-0_6357c3655b524feb8d0e398ff61dfabf",
    "question": "how many episodes are in chicago fire season 4",
    "question_type": "how_many",
    "memory_answer": "Chicago Fire Season 4 has 23 episodes.",
    "memory_flag": true,
    "counter_answer": "Chicago Fire Season 4 has 24 episodes.",
    "counter_flag": true,
    "paraphrase_patch": [
        "There are a total of 24 episodes in Season 4 of Chicago Fire.",
        "The fourth season of Chicago Fire consists of 24 episodes."
    ],
    "paraphrase_patch_flag": true,
    "2_sent": "In Season 4 of Chicago Fire, there are 24 episodes in total, each running for approximately 42 minutes. The season finale titled \"Superhero\" aired on May 17, 2016, marking the end of the 24-episode season.",
    "2_sent_flag": true,
    "3_sent": "Chicago Fire Season 4 indeed has 24 episodes. The season premiered on October 13, 2015, and concluded on May 17, 2016, with a total of 24 episodes airing over the course of the season. Fans of the show eagerly awaited each new episode as the season unfolded with its thrilling storylines and character developments.",
    "3_sent_flag": true
}
```

### Generation Process

Data generation consists of five sequential steps:
1. `close_book` - Generate initial responses without context
2. `counter` - Create counter-factual answers
3. `direct` - Generate direct evidence
4. `direct_patch` - Create paraphrased patches
5. `sentence` - Generate sentence-level evidence
6. `update_sentence` - Update and refine sentence evidence

### Script Usage

```bash
python generate_dataset.py \
    --data_path data/datasets/NQ_small_question.json \
    --save_path data/build_datasets/NQ_chatgpt \
    --dataset_name NQ \
    --test_llm_type llama2 \
    --test_llm_model path_to_Llama-2-7b-chat-hf \
    --worker_llm_type openai \
    --worker_llm_model gpt-3.5-turbo \
    --steps close_book counter direct direct_patch sentence update_sentence
```

## Memory Strength Computation

### Input Requirements

The memory strength computation requires two main input files:

- **`NQ_direct_evidence.jsonl`**: Contains questions with their corresponding direct evidence and model responses
- **`NQ_question_paraphrases.json`**: Contains paraphrased versions of the original questions used to test answer consistency across different phrasings

### Output Format

Memory strength scores are output in the following format:

```json
{
    "question": "how many episodes are in chicago fire season 4",
    "strength": 0.0,
    "clusters": [[0, 1, 2, 3, 4, 5, 6]]
}
```

Where:
- `strength`: A numerical score representing the memory strength
- `clusters`: Groups of similar responses, indicating consistency patterns

### Script Usage

```bash
python compute_memory_strength.py \
    --dataset nq \
    --questions data/build_datasets/NQ_llama2_7b/NQ_direct_evidence.jsonl \
    --paraphrases data/datasets/NQ_question_paraphrases.json \
    --answered data/build_datasets/NQ_memory_strength/NQ_llama2_7b_answerd.json \
    --output data/build_datasets/NQ_memory_strength/NQ_llama2_7b_scores.json \
    --test_llm llama2 \
    --test_model path_to_Llama-2-7b-chat-hf \
    --worker_llm llama3 \
    --worker_model path_to_Meta-Llama-3.1-8B-Instruct
```

## File Structure

```
ContextFaithful/
├── data/
│   ├── datasets/
│   │   ├── NQ_data_question.json
│   │   └── NQ_question_paraphrases.json
│   └── build_datasets/
├── generate_dataset.py
├── compute_memory_strength.py
└── requirements.txt
```