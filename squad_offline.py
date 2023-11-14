# %% [markdown]
# # Set up

# %%
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import Trainer, TrainingArguments
from peft import (
    get_peft_config,
    get_peft_model,
    PromptTuningInit,
    PromptTuningConfig,
    TaskType,
    PeftType,
)
import torch
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate
import collections
import numpy as np

device = "cuda"
model_name_or_path = "bert-base-cased"
tokenizer_name_or_path = "bert-base-cased"
peft_config = PromptTuningConfig(
    task_type=TaskType.QUESTION_ANS,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8,
    prompt_tuning_init_text="Please answer the following question based on the context:",
    tokenizer_name_or_path=model_name_or_path,
)

dataset_name = "squad"
checkpoint_name = f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_v1.pt".replace(
    "/", "_"
)
# context = "Tweet text"
# label_column = "text_label"
max_length = 384
lr = 2e-5
num_epochs = 50
batch_size = 12

# %% [markdown]
# # Dataset

# %%
squad = load_dataset("squad")  # smaller subset
# get random 5 example in train data
# squad["train"] = squad["train"].select(range(2,7))
# squad["validation"] = squad["validation"].select(range(120,127))


squad["train"].filter(lambda x: len(x["answers"]["text"]) != 1)

# %% [markdown]
# # Preprocess dataset

# %%
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# context = squad["train"][0]["context"]
# question = squad["train"][0]["question"]

# inputs = tokenizer(question, context)
# print(tokenizer.decode(inputs["input_ids"]))

# %%
max_length = 384
stride = 128


def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


train_dataset = squad["train"].map(
    preprocess_training_examples,
    batched=True,
    remove_columns=squad["train"].column_names,
)
len(squad["train"]), len(train_dataset)


# %%
def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


validation_dataset = squad["validation"].map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=squad["validation"].column_names,
)
len(squad["validation"]), len(validation_dataset)

# %%
train_dataset.set_format("torch")
validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
validation_set.set_format("torch")

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(
    validation_set, collate_fn=default_data_collator, batch_size=8
)

# %% [markdown]
# # Train

# %%
model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
print(model.config)

# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

# %%
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

# %%
n_best = 20
max_answer_length = 30
metric = evaluate.load("squad")


def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]
            # print(len(offsets))
            # print(start_logit.shape)
            # print(end_logit.shape)

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            # print(start_indexes)
            # print(end_indexes)

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip invalid predictions
                    if start_index >= len(offsets):
                        continue
                    if end_index >= len(offsets):
                        continue
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[
                            offsets[start_index][0] : offsets[end_index][1]
                        ],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [
        {"id": ex["id"], "answers": ex["answers"]} for ex in examples
    ]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)


# %%
import wandb

# Create your wandb run
run = wandb.init(
    name="peft(bert)_squad",  ## Wandb creates random run names if you skip this field
    # reinit = True, ### Allows reinitalizing runs when you re-run this cell
    # id ="dv086dml", ### Insert specific run id here if you want to resume a previous run
    # resume = "must", ### You need this to resume previous runs, but comment out reinit = True when using this
    project="idl_project",  ### Project should be created in your wandb account
    # config = config, ### Wandb Config for your run
    entity="yifansu",
)

if run.entity == "yifansu":
    wandb.save("sandbox.ipynb")

# %%
model = model.to(device)
torch.cuda.empty_cache()


from tqdm.auto import tqdm
import torch


for epoch in range(num_epochs):
    # Training
    model.train()
    progress_bar = tqdm(train_dataloader)
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    start_logits = []
    end_logits = []
    accelerator.print("Evaluation!")
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
        end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(validation_dataset)]
    end_logits = end_logits[: len(validation_dataset)]

    metrics = compute_metrics(
        start_logits, end_logits, validation_dataset, squad["validation"]
    )
    print(f"epoch {epoch}:", metrics)

    # save
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        checkpoint_name,
    )
    wandb.log(metrics)

    # Save and upload
    # accelerator.wait_for_everyone()
    # unwrapped_model = accelerator.unwrap_model(model)
    # unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    # if accelerator.is_main_process:
    #     tokenizer.save_pretrained(output_dir)
    #     repo.push_to_hub(
    #         commit_message=f"Training in progress epoch {epoch}", blocking=False
    #     )
run.finish()
