# D rank
This is the code for the paper "Layer-wise dynamic rank for compressing large language models". Some config examples are added in config directory.
This code is based on "BASIS SHARING: CROSS-LAYER PARAMETER SHARING FOR LARGE LANGUAGE MODEL COMPRESSION". If you are confused with the settings, you can also refer to [link](https://github.com/TUDa-HWAI/Basis_Sharing).

## Run D rank
To run D rank on LLaMA-7B for generation tasks, run
```
python test.py --cf tasks/configs/wikitext_ppl/llama/share2/share_llama_7b_20.yaml
```
make sure to set *build_calib* as true for a model, when you want to compress it for the first time.

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

