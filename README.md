# D rank：Layer-wise dynamic rank for compressing large language models
This is the code for the paper "Layer-wise dynamic rank for compressing large language models". Some config examples are added in config directory.
 If you are confused with the settings, you can also refer to [link](https://github.com/TUDa-HWAI/Basis_Sharing).

## Abstract
Large language models (LLMs) have rapidly scaled in size, bringing severe memory and computational challenges that hinder their deployment. Singular Value Decomposition (SVD)-based compression has emerged as an appealing post-training compression technique for LLMs, yet most existing methods apply a uniform compression ratio across all layers, implicitly assuming homogeneous information included in various layers. This overlooks the substantial intra-layer heterogeneity observed in LLMs, where middle layers tend to encode richer information while early and late layers are more redundant. In this work, we revisit the existing SVD-based compression method and propose D-Rank, a framework with layer-wise balanced Dynamic Rank allocation for LLMs compression. We first introduce effective rank as a principled metric to measure the information density of weight matrices, and then allocate ranks via a Lagrange multiplier-based optimization scheme to adaptively assign more capacity to groups with higher information density under a fixed compression ratio. Moreover, we rebalance the allocated ranks across attention layers to account for their varying importance and extend D-Rank to latest LLMs with grouped-query attention. Extensive experiments on various LLMs with different scales and compression ratios demonstrate that D-Rank consistently outperforms baselines, achieving more than 15 lower perplexity on the C4 dataset with LLaMA-3-8B at 20% compression ratio and up to 5% higher zero-shot reasoning accuracy with LLaMA-7B at 40% compression ratio, while also delivering higher throughput.


## Installation
Step 1: Clone this repository and navigate to D_rank folder
```
git clone https://github.com/Rainbow0427/D_rank.git
cd D_rank
```
Step 2: Create the conda environment:
```
conda create -n D_rank python=3.9
conda activate D_rank
```
Step 3: Install relevant packages:
```
pip install -r requirements.txt
```

## Run D rank
Our important arguments are all in the yaml file in the *tasks* folder.
- `--model_name`: The identifier for the LLaMa model on the Hugging Face model hub.
- `--share_part`: The weight matrices which can be grouped for SVD.
- `--private_part`: The weight matrices which are not grouped for SVD.
- `--group_size`: How much layers of weight matrices you want to share basis.
- `--untrained_model_path`: Which path you want to save your compressed model.
- `--dataset_name`: which datasets you want to use for compression and test.
- `--build_calib` : Whether to build your calibration data.
- `--calib_path`: Path to save your calibration data.


To run D rank on LLaMA-7B for generation tasks, run
```
python test.py --cf tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml
```
make sure to set *build_calib* as true for a model, when you want to compress it for the first time.
We use *tax_rate* as a hyperparameter to control the distribution of retained rank between the QK and V matrices in the attention layer.
Specifically, it determines how much of the rank preserved in the QK matrices is transferred to the V matrix, and you can adjust this value in the function of *create_model*.

After compress with WikiText, to test with other dataset run
~~~
python test.py --cf tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml --dataset_name <ptb, C4, WikiText>
~~~
For C4 you need to download them from [link](https://drive.google.com/drive/folders/123Id1MkZVsKySGy_sMO4RgiJKrtPcvUp?usp=drive_link). Don't forget to update *dataset_cache_dir* in config file.

## Run LoRA
~~~
python lora.py  --cf tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml
~~~

## Run Reasoning tasks
~~~
python test_adapter.py --cf tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml
~~~

## Run Throughput tasks
~~~
python test_throughput.py --cf tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml
~~~

## Some instruction
This codebase contains basis sharing version compression and if you want to run basis sharing, just to modify the llama.py, group.py and model_factory.py. This codebase may not be the final version and we will keeping updating it.

## Reference
@misc{mi2025layerwisedynamicrankcompressing,
      title={Layer-wise dynamic rank for compressing large language models}, 
      author={Zhendong Mi and Bian Sun and Grace Li Zhang and Shaoyi Huang},
      year={2025},
      eprint={2509.25622},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.25622}, 
}

For any correspondance, email us at shuang59@stevens.edu
