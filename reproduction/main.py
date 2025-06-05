
from train_baseline import run_baseline
from train_lora import run_lora

if __name__ == "__main__":
    models = [("roberta-base", ["query", "value"]), ("microsoft/deberta-base",["in_proj"])]
    tasks = ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte"]

    for (model, target_modules) in models:
        for task in tasks:
            print(f"\nRunning baseline: {model} on {task}")
            run_baseline(model, task)

            print(f"\nRunning LoRA: {model} on {task}")
            run_lora(model, task, target_modules)
