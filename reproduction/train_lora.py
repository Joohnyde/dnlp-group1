from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,
                          Trainer, DataCollatorWithPadding)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from utils import preprocess_function, compute_metrics


def run_lora(model_name, task_name):
    sentence_keys = {
        "cola": ("sentence", None),
        "sst2": ("sentence", None),
        "mrpc": ("sentence1", "sentence2"),
        "qqp": ("question1", "question2"),
        "mnli": ("premise", "hypothesis"),
        "qnli": ("question", "sentence"),
        "rte": ("sentence1", "sentence2")
    }

    dataset = load_dataset("glue", task_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentence1_key, sentence2_key = sentence_keys[task_name]

    encoded_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer, sentence1_key, sentence2_key), batched=True)
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=dataset["train"].features["label"].num_classes)

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["query", "value"]
    )
    model = get_peft_model(model, config)

    args = TrainingArguments(
        output_dir=f"./results/{task_name}_{model_name.replace('/', '_')}_lora",
        save_strategy="no",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        logging_dir=f"./logs/{task_name}_{model_name.replace('/', '_')}_lora"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation_matched"] if task_name == "mnli" else encoded_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, task_name),
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"LoRA - {model_name} on {task_name}:")
    print(metrics)
