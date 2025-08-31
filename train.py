from datasets import load_dataset
import json

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

MODEL_NAME = "google/flan-t5-small"
MODEL_SAVE_PATH = "model"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 512

dataset_s = load_dataset("natural_questions", split="train", streaming=True)

with open("small_nq.json", "w") as f:
    for i, example in enumerate(dataset_s):
        f.write(json.dumps(example) + "\n")
        if i >= 2999:
            break

dataset = load_dataset("json", data_files="small_nq.json")["train"]

dataset = dataset.shuffle(seed=42)

print(dataset)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

sample = dataset[0]

print("Question:", sample.get("question"))
print("Annotations:", sample.get("annotations"))

import json

for i in range(1):  # Just print first 2 rows
    sample = dataset[i]
    output = {
        "question": sample.get("question"),
        "annotations": sample.get("annotations")
    }
    print(json.dumps(output, indent=1))

def print_structure(example, indent=0):
    prefix = " " * indent
    if isinstance(example, dict):
        for key, value in example.items():
            print(f"{prefix}- {key}: {type(value).__name__}")
            print_structure(value, indent + 2)
    elif isinstance(example, list) and len(example) > 0:
        print(f"{prefix}  [List of {type(example[0]).__name__}]")
        print_structure(example[0], indent + 4)
    else:
        # Base case for non-dict/list
        pass

print("Dataset structure:")
print_structure(dataset[0])

train_dataset = dataset.select(range(0, 2700))
valid_dataset = dataset.select(range(2700, 2850))
test_dataset = dataset.select(range(2850, 3000))

def preprocess_function(examples):
  inputs = []
  targets = []

  for i in range(len(examples["id"])):
    question = examples["question"][i]["text"].strip()

    doc_tokens = examples["document"][i]["tokens"]["token"]

    long_answer_text = "No answer provided."

    long_answer_list = examples["annotations"][i]["long_answer"]
    if long_answer_list and isinstance(long_answer_list, list):
        long_answer = long_answer_list[0]
        candidate_index = long_answer.get("candidate_index", -1)
    else:
        candidate_index = -1


    if candidate_index != -1:
      start_token = examples["long_answer_candidates"][i]["start_token"][candidate_index]
      end_token = examples["long_answer_candidates"][i]["end_token"][candidate_index]

      answer_tokens = doc_tokens[start_token:end_token]
      long_answer_text = " ".join(answer_tokens).strip()

    context_tokens = [
      tok for tok, is_html in zip(
          examples["document"][i]["tokens"]["token"],
          examples["document"][i]["tokens"]["is_html"]
      ) if not is_html
    ]

    context_text = " ".join(context_tokens).strip()

    input_text = f"answer the question: {question} context: {context_text}"
    inputs.append(input_text)
    targets.append(long_answer_text)

  model_inputs = tokenizer(
      inputs,
      max_length=MAX_INPUT_LENGTH,
      truncation=True,
      padding="max_length"
  )
  labels = tokenizer(
      targets,
      max_length=MAX_TARGET_LENGTH,
      truncation=True,
      padding="max_length"
  )
  model_inputs["labels"] = labels["input_ids"]
  return model_inputs

tokenized_train = train_dataset.map(
      preprocess_function,
      batched=True,
      remove_columns=train_dataset.column_names
)
tokenized_valid = valid_dataset.map(
      preprocess_function,
      batched=True,
      remove_columns=valid_dataset.column_names
)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model
)

training_args = Seq2SeqTrainingArguments(
    output_dir = "./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,

    save_total_limit=1,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    generation_num_beams=2,
    fp16=False,

    eval_strategy="steps",
    eval_steps=200,
    logging_steps=50,
    save_steps=400,
    report_to=[]
)

trainer = Seq2SeqTrainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_train,
    eval_dataset = tokenized_valid,
    tokenizer = tokenizer,
    data_collator = data_collator
)

trainer.train()

metrics = trainer.evaluate()
print(metrics)

MODEL_SAVE_PATH = "context-aware-qa"

model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)