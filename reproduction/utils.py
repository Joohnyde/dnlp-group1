import numpy as np
import evaluate


def preprocess_function(examples, tokenizer, sentence1_key, sentence2_key=None):
    if sentence2_key:
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True,
                         padding="max_length", max_length=128)
    else:
        return tokenizer(examples[sentence1_key], truncation=True, padding="max_length", max_length=128)


def compute_metrics(eval_pred, task_name):
    metric = evaluate.load("glue", task_name)
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # Compute default GLUE metric (MCC for CoLA)
    results = metric.compute(predictions=predictions, references=labels)

    # Add accuracy for all tasks (including CoLA)
    if task_name == "cola":
        accuracy_metric = evaluate.load("accuracy")
        results["accuracy"] = accuracy_metric.compute(
            predictions=predictions, references=labels
        )["accuracy"]
    return results

