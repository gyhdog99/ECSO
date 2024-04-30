# ECSO
[![arXiv](https://img.shields.io/badge/arXiv-2403.09572-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2403.09572) [![arXiv](https://img.shields.io/badge/Web-ECSO-blue.svg?style=plastic)](https://gyhdog99.github.io/projects/ecso/)

This repository contains the implementation of the paper:

> ECSO: Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation <br>
> [Yunhao Gou](https://gyhdog.github.io/), [Kai Chen](https://kaichen1998.github.io/), [Zhili Liu](https://scholar.google.com/citations?user=FdR09jsAAAAJ&hl=zh-CN), [Lanqing Hong](https://scholar.google.com/citations?hl=zh-CN&user=2p7x6OUAAAAJ&view_op=list_works&sortby=pubdate), [Hang Xu](https://xuhangcn.github.io/), [Aoxue Li](https://dblp.org/pid/152/6095.html), [Zhenguo Li](https://zhenguol.github.io/), [Dit-Yan Yeung](https://sites.google.com/view/dyyeung/home), [James T. Kwok](https://www.cse.ust.hk/~jamesk/), [Yu Zhang](https://yuzhanghk.github.io/) <br>


<img src="./assets/framework.png" alt="drawing" width="800"/>


## Installation


1. Clone this repository and navigate to ECSO folder.

   ```bash
   git clone https://github.com/gyhdog99/ecso/
   cd ECSO-main
   ```
2. Install Package

   ```bash
    conda create -n ecso python=3.10 -y
    conda activate ecso
    pip install --upgrade pip  # enable PEP 660 support
    pip install -e .
   ```


## Demo

We show the 4 core steps (i.e, 1. direct answer, 2. harm detect, 3. query-aware I2T caption, 4. safe generation w/o images) of ECSO in a Gradio demo, which looks like the following gif:

<img src="./assets/demo.gif" alt="drawing" width="800"/>

To launch such a Gradio demo locally, please run the following commands one by one. 

**Launch a controller**

```shell
python -m llava.serve.controller --host 0.0.0.0 --port 10000
```

**Launch a gradio web server**

```
python -m llava.serve.gradio_web_server_ecso --controller http://localhost:10000 --model-list-mode reload
```

You just launched the Gradio web interface. Now, you can open the web interface with the URL printed on the screen. You may notice that there is no model in the model list. Do not worry, as we have not launched any model worker yet. It will be automatically updated when you launch a model worker

**Launch a model worker**

```
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path llava-v1.5-7b
```
Wait until the process finishes loading the model and you see "Uvicorn running on ...". Now, refresh your Gradio web UI, and you will see the model you just launched in the model list.

## Evaluation on Safety Benchmarks

**Data/Model Preparation**

Download [VLSafe](https://huggingface.co/datasets/YangyiYY/LVLM_NLF/tree/main/VLSafe), [MM-SafetyBench](https://github.com/isXinLiu/MM-SafetyBench) and [COCO images](http://images.cocodataset.org/zips/train2017.zip)

**VLSafe**

1. Generate direct/ECSO responses
    ```shell
    bash scripts/v1_5/eval_safe/gen_vlsafe.sh
    bash scripts/v1_5/eval_safe/gen_vlsafe_tell_ask.sh
    ```

2. Evaluation
    ```shell
    bash llava/eval/eval_vlsafe.sh
    ```

**MM-SafetyBench**

1. Generate direct/ECSO responses
    ```shell
    bash scripts/v1_5/eval_safe/gen_mmsafe.sh
    bash scripts/v1_5/eval_safe/gen_mmsafe_tell_ask.sh
    ```

2. Evaluation
    ```shell
    bash llava/eval/eval_mmsafe_loop.sh
    ```

## Evaluating Utilities on MLLM benchmarks

**Data/Model Preparation**

Follow the [guideline](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) to download the evaluation data of MME, MMBench and MM-Vet.

**MME**

Generate direct/ECSO responses
```shell
bash scripts/v1_5/eval/mme.sh
bash scripts/v1_5/eval_safe/gen_mme_unsafe_ask.sh
```

**MMBench**

Generate direct/ECSO responses
```shell
bash scripts/v1_5/eval/mmbench.sh
bash scripts/v1_5/eval_safe/gen_mmbench_unsafe_ask.sh.sh
```

**MM-Vet**

Generate direct/ECSO responses
```shell
bash scripts/v1_5/eval/mmvet.sh
bash scripts/v1_5/eval_safe/gen_mm-vet_unsafe_ask.sh.sh
```

## Acknowledgement
+ [LLaVA](https://github.com/haotian-liu/LLaVA) This repository is built upon LLaVA!

## Citation

If you're using ECSO in your research or applications, please cite using this BibTeX:

```bibtex
@article{gou2024eyes,
  title={Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation},
  author={Gou, Yunhao and Chen, Kai and Liu, Zhili and Hong, Lanqing and Xu, Hang and Li, Zhenguo and Yeung, Dit-Yan and Kwok, James T and Zhang, Yu},
  journal={arXiv preprint arXiv:2403.09572},
  year={2024}
}
```
