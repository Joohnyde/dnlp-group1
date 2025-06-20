# LoRA: Low-Rank Adaptation of Large Language Models

This repo is a modified version of https://github.com/microsoft/lora to avoid unnecessary duplication.

## Repository Overview

There are several directories in this repo:
* [loralib/](loralib) contains the source code for the package `loralib`, which needs to be installed to run the examples we provide;
* [NLU/](NLU) contains an example implementation of LoRA in RoBERTa and DeBERTa using our package, which produces competitive results on the GLUE benchmark;
* See how we use `loralib` in [RoBERTa](NLU/src/transformers/models/roberta/modeling_roberta.py), and [DeBERTa v2](NLU/src/transformers/models/deberta_v2/modeling_deberta_v2.py)

## Reproduction

Follow [NLU/README.md](NLU/README.md) to reproduce the results.

## Using Loralib

 1. Installing `loralib` is simply
 ```bash
 pip install loralib
 # Alternatively
 # pip install git+https://github.com/microsoft/LoRA
 ```

 2. You can choose to adapt some layers by replacing them with counterparts implemented in `loralib`. We only support `nn.Linear`, `nn.Embedding`, and `nn.Conv2d` for now. We also support a `MergedLinear` for cases where a single `nn.Linear` represents more than one layers, such as in some implementations of the attention `qkv` projection (see Additional Notes for more).
 ```python
 # ===== Before =====
 # layer = nn.Linear(in_features, out_features)

 # ===== After ======
 import loralib as lora
 # Add a pair of low-rank adaptation matrices with rank r=16
 layer = lora.Linear(in_features, out_features, r=16)
 ```

 3. Before the training loop begins, mark only LoRA parameters as trainable.
 ```python
 import loralib as lora
 model = BigModel()
 # This sets requires_grad to False for all parameters without the string "lora_" in their names
 lora.mark_only_lora_as_trainable(model)
 # Training loop
 for batch in dataloader:
    ...
 ```
 4. When saving a checkpoint, generate a `state_dict` that only contains LoRA parameters.
 ```python
 # ===== Before =====
 # torch.save(model.state_dict(), checkpoint_path)
 # ===== After =====
 torch.save(lora.lora_state_dict(model), checkpoint_path)
 ```
 5. When loading a checkpoint using `load_state_dict`, be sure to set `strict=False`.
 ```python
 # Load the pretrained checkpoint first
 model.load_state_dict(torch.load('ckpt_pretrained.pt'), strict=False)
 # Then load the LoRA checkpoint
 model.load_state_dict(torch.load('ckpt_lora.pt'), strict=False)
 ```

#### Now training can proceed as usual.

## Additional Notes

1. While we focus on a simple yet effect setup, namely adapting only the `q` and `v` projection in a Transformer, in our examples, LoRA can be apply to any subsets of pre-trained weights. We encourage you to explore different configurations, such as adapting the embedding layer by replacing `nn.Embedding` with `lora.Embedding` and/or adapting the MLP layers. It's very likely that the optimal configuration varies for different model architectures and tasks.

2. Some Transformer implementation uses a single `nn.Linear` for the projection matrices for query, key, and value. If one wishes to constrain the rank of the updates to the individual matrices, one has to either break it up into three separate matrices or use `lora.MergedLinear`. Make sure to modify the checkpoint accordingly if you choose to break up the layer.
```python
# ===== Before =====
# qkv_proj = nn.Linear(d_model, 3*d_model)
# ===== After =====
# Break it up (remember to modify the pretrained checkpoint accordingly)
q_proj = lora.Linear(d_model, d_model, r=8)
k_proj = nn.Linear(d_model, d_model)
v_proj = lora.Linear(d_model, d_model, r=8)
# Alternatively, use lora.MergedLinear (recommended)
qkv_proj = lora.MergedLinear(d_model, 3*d_model, r=8, enable_lora=[True, False, True])
```
3. Training bias vectors in tandem with LoRA might be a cost-efficient way to squeeze out extra task performance (if you tune the learning rate carefully). While we did not study its effect thoroughly in our paper, we make it easy to try in `lora`. You can mark some biases as trainable by passing "all" or "lora_only" to `bias=` when calling `mark_only_lora_as_trainable`. Remember to pass the corresponding `bias=` argument to `lora_state_dict` when saving a checkpoint.
```python
# ===== Before =====
# lora.mark_only_lora_as_trainable(model) # Not training any bias vectors
# ===== After =====
# Training all bias vectors associated with modules we apply LoRA to 
lora.mark_only_lora_as_trainable(model, bias='lora_only')
# Alternatively, we can train *all* bias vectors in the model, including LayerNorm biases
lora.mark_only_lora_as_trainable(model, bias='all')
# When saving a checkpoint, use the same bias= ('all' or 'lora_only')
torch.save(lora.lora_state_dict(model, bias='all'), checkpoint_path)
```
4. Calling `model.eval()` will trigger the merging of LoRA parameters with the corresponding pretrained ones, which eliminates additional latency for subsequent forward passes. Calling `model.train()` again will undo the merge. This can be disabled by passing `merge_weights=False` to LoRA layers.

## Results

We obtain result comparable or superior to full finetuning on the GLUE benchmark using [RoBERTa (Liu et al., 2019)](https://arxiv.org/abs/1907.11692) base and large and [DeBERTa (He et al., 2020)](https://arxiv.org/abs/2006.03654) XXL 1.5B, while only training and storing a fraction of the parameters. Click the numbers below to download the RoBERTa and DeBERTa LoRA checkpoints.

|   |         | RoBERTa base <br> Fine-tune  |  RoBERTa base <br> LoRA  | DeBERTa XXL <br> Fine-tune | DeBERTa XXL <br> LoRA  |
|---|-------------------------|----------------|--------------------------|-----------------|-----------------|
|   | # of Trainable Params.  | 125M | 0.8M | 1.5B | 4.7M     |
|   | MNLI (m-Acc/mm-Acc)     | <b>87.6</b> | [<b>87.5</b>±.3/86.9±.3](https://github.com/microsoft/LoRA/releases/download/RoBERTa-base/roberta_base_lora_mnli.bin) |91.7/<b>91.9</b>| [<b>91.9</b>±.1/<b>91.9</b>±.2](https://github.com/microsoft/LoRA/releases/download/DeBERTa/deberta_v2_xxlarge_lora_mnli.bin)       |
|   | SST2 (Acc)              | 94.8 | [<b>95.1</b>±.2](https://github.com/microsoft/LoRA/releases/download/RoBERTa-base/roberta_base_lora_sst2.bin) | <b>97.2</b>    | [96.9±.2](https://github.com/microsoft/LoRA/releases/download/DeBERTa/deberta_v2_xxlarge_lora_sst2.bin)                    |
|   | MRPC (Acc)              | <b>90.2</b> | [<b>89.7</b>±.7](https://github.com/microsoft/LoRA/releases/download/RoBERTa-base/roberta_base_lora_mrpc.bin) | 92.0           | [<b>92.6</b>±.6](https://github.com/microsoft/LoRA/releases/download/DeBERTa/deberta_v2_xxlarge_lora_mrpc.bin)             |
|   | CoLA (Matthew's Corr)   | <b>63.6</b> | [<b>63.4</b>±1.2](https://github.com/microsoft/LoRA/releases/download/RoBERTa-base/roberta_base_lora_cola.bin) | <b>72.0</b>    | [<b>72.4</b>±1.1](https://github.com/microsoft/LoRA/releases/download/DeBERTa/deberta_v2_xxlarge_lora_cola.bin)           |
|   | QNLI (Acc)              | 92.8 | [<b>93.3</b>±.3](https://github.com/microsoft/LoRA/releases/download/RoBERTa-base/roberta_base_lora_qnli.bin) | <b>96.0</b>    | [<b>96.0</b>±.1](https://github.com/microsoft/LoRA/releases/download/DeBERTa/deberta_v2_xxlarge_lora_qnli.bin)            |
|   | QQP (Acc)               | <b>91.9</b> | [90.8±.1](https://github.com/microsoft/LoRA/releases/download/RoBERTa-base/roberta_base_lora_qqp.bin) | 92.7           | [<b>92.9</b>±.1](https://github.com/microsoft/LoRA/releases/download/DeBERTa/deberta_v2_xxlarge_lora_qqp.bin)           |
|   | RTE (Acc)               | 78.7 | [<b>86.6</b>±.7](https://github.com/microsoft/LoRA/releases/download/RoBERTa-base/roberta_base_lora_rte.bin) | 93.9           | [<b>94.9</b>±.4](https://github.com/microsoft/LoRA/releases/download/DeBERTa/deberta_v2_xxlarge_lora_rte.bin)           |
|   | STSB (Pearson/Spearman Corr) | 91.2 | [<b>91.5</b>±.2/<b>91.3</b>±.2](https://github.com/microsoft/LoRA/releases/download/RoBERTa-base/roberta_base_lora_stsb.bin) |<b>92.9</b>/92.6| [<b>93.0</b>±.2/<b>92.9</b>±.3](https://github.com/microsoft/LoRA/releases/download/DeBERTa/deberta_v2_xxlarge_lora_stsb.bin)      |
|   | Average  | 86.40 | <b>87.24</b> | 91.06 | <b>91.32</b> |

<i>Note: You still need the original pre-trained checkpoint from [Hugging Face](https://huggingface.co/) to use the LoRA checkpoints.</i>

Fine-tuning numbers are taken from [Liu et al. (2019)](https://arxiv.org/abs/1907.11692) and [He et al. (2020)](https://arxiv.org/abs/2006.03654).  We include confidence intervals on results from our experiments. Please follow the instructions in `examples/NLU/` to reproduce our results.