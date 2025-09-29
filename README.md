# On the Self-awareness of Large Reasoning Models' Capability Boundaries

This is the code repository of our work: On the Self-awareness of Large Reasoning Models' Capability Boundaries

This repository contains the code to:
1. Extract last token hidden states for input questions.
2. Classify hidden states of solvable and unsolvable questions, and plot capability boundaries.
3. Optimize reasoning with self-awareness prompt suffix and output prefix.

## Abstract
Large Reasoning Models (LRMs) have shown impressive performance on complex reasoning tasks such as mathematics, yet they also display misbehaviors that expose their limitations. In particular, when faced with hard questions, LRMs often engage in unproductive reasoning until context limit, producing wrong answers while wasting substantial computation. This phenomenon reflects a fundamental issue: current answering paradigms overlook the relationship between questions and LRMs' capability boundaries. In this paper, we investigate whether LRMs possess self-awareness of capability boundaries. We begin by an observation that LRMs may know what they cannot solve through expressed reasoning confidence. For black-box models, we find that reasoning expressions reveal boundary signals, with accelerated growing confidence trajectory for solvable problems but convergent uncertainty trajectory for unsolvable ones. For white-box models, we show that hidden states of the last input token encode boundary information, with solvable and unsolvable problems linearly separable even before reasoning begins. Building on these findings, we propose two simple yet effective optimization strategies: reasoning expression monitoring and hidden states monitoring. Experiments demonstrate that these boundary-aware strategies enable LRMs to avoid unproductive reasoning without sacrificing accuracy, significantly improving reliability and efficiency by cutting token usage up to 62.7 – 93.6%.

<p align="center">
  <img src="overview.png" width="800">
</p>

## Setup

### Environment setup
1. Create a new conda environment:
```bash
conda create -n CB python=3.10 -y  # Python >= 3.10 required
conda activate CB
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 1. Extract last token hidden states

Follow `./code/extract_hidden_LRM.sh` and `./code/extract_hidden_LRM.py` to extract last token hidden states for LRMs on input questions.

Please first put the inference results in  `./data/inference`.

We provide an example of **gpt-oss-20b** in `./data/inference/aime24/gpt-oss-20b`.

## 2. Classify hidden states and plot capability boundaries

Follow `./code/capabilityBoundary.sh` and `./code/capabilityBoundary.py` to classify hidden states of solvable and unsolvable questions, and plot capability boundaries.

We provide an example of **gpt-oss-20b** in `./data/capabilityBoundaries/gpt-oss-20b`.

## 3. Optimize reasoning

Follow `./code/optimize.sh` and `./code/optimize.py` to optimize reasoning with self-awareness prompt suffix and output prefix.

We provide an example of **gpt-oss-20b** in `./data/optimize`.
