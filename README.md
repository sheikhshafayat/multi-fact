# Multi-FAct: Assessing Multilingual LLMs’ Multi-Regional Knowledge using FActScore

## Abstract

Large Language Models (LLMs) are prone to factuality hallucination, generating text that contradicts established knowledge. While extensive research has addressed this in English, little is known about multilingual LLMs. This paper systematically evaluates multilingual LLMs' factual accuracy across languages and geographic regions. We introduce a novel pipeline for multilingual factuality evaluation, adapting FActScore \citep{min2023factscore} for diverse languages. Our analysis across nine languages reveals that English consistently outperforms others in factual accuracy and quantity of generated facts. Furthermore, multilingual models demonstrate a bias towards factual information from Western continents. These findings highlight the need for improved multilingual factuality assessment and underscore geographical biases in LLMs' fact generation.


## Introduction
This codebase contains the code for the paper "Multi-FAct: Assessing Multilingual LLMs’ Multi-Regional Knowledge using FActScore" [arxiv](https://arxiv.org/abs/2402.18045). 

This codebase is heavily built on top of the FActScore, which can be found [here](https://github.com/shmsw25/FActScore). Please cite the original paper too if you find this code useful.

The original FActScore codebase is modified to use open source mistral model for both fact generation and fact verification. This modification of FActScore results in a slightly better performance than what is reported in the original paper, which is likely due to the Mistral model. 

All experiment results reported in Multi-FAct paper are generated using Mistral 7B instruct for fact generation and retrieval + mistral + npm for fact verification.


## Getting Started



The codebase is organized as follows:

- `generated_bios/`: contains the data used in the paper
- `necessary_dicts/`: contains the necessary dictionaries for the code
- `factscore.py`: contains the modified code for the FActScore metric
- `run_multifact.py`: contains the code to run the multifact evaluation on the translated bios
- `generate_bio.py`: contains the code to generate bios
- `translate_bio_w_gpt.py`: contains the code to translate the bios to different languages
- `quickstart.ipynb`: contains a quickstart guide to play with the code


### Small Note
Current Mistral model based atomic fact generation uses a fixed five shot prompting for atomic fact breakdown, which is slightly different from the original implementation, which prompts using top k most similar prompts among given examples. We found that fixed five shot prompting works provides pretty good accuracy for our tasks, but feel free to adopt the original prompting approach if you want to.

If you find Multi-FAct useful, please cite the paper as follows:

```
@misc{shafayat2024multifact,
      title={Multi-FAct: Assessing Multilingual LLMs' Multi-Regional Knowledge using FActScore}, 
      author={Sheikh Shafayat and Eunsu Kim and Juhyun Oh and Alice Oh},
      year={2024},
      eprint={2402.18045},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

