import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Union

import bitsandbytes as bnb
import evaluate
import torch
from datasets import DownloadConfig, load_dataset, load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from peft.optimizers import create_loraplus_optimizer
from transformers import (
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

SIZE = 20000
# ds = load_from_disk(f"dataset_local/ds")
ds = load_dataset("OrcinusOrca/YouTube-Cantonese", streaming=True)
print(ds)
ds = ds.select(random.sample(range(len(ds)), min(SIZE, len(ds))))
ds = ds.train_test_split(test_size=0.3)
ds


# ## Load Models

# ### Processor (Feature Extractor & Tokenizer)

model_id = "openai/whisper-large-v3-turbo"
processor: WhisperProcessor = WhisperProcessor.from_pretrained(
    model_id,
    language="yue",
    task="transcribe",
)
feature_extractor: WhisperFeatureExtractor = processor.feature_extractor
tokenizer: WhisperTokenizer = processor.tokenizer


# ### Model with Quantization

quantization_config: BitsAndBytesConfig = BitsAndBytesConfig(load_in_8bit=True)
local_rank = int(os.getenv("LOCAL_RANK", "0"))
base_model: WhisperForConditionalGeneration = WhisperForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map={"": local_rank},
)


# Since the Whisper model uses Convolutional layers in the Encoder, checkpointing disables grad computation.
# To avoid this we specifically need to make the inputs trainable.
def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)


base_model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)


# ### PEFT Model

peft_config = LoraConfig(
    r=32,
    # target_modules=["q_proj", "v_proj"],
    target_modules="all-linear",
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    use_rslora=True,
)
peft_model: PeftModel = get_peft_model(base_model, peft_config)
peft_model.print_trainable_parameters()


# ### Data Collator


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Collate input features and labels for training or evaluation.

        Args:
            features (List[Dict[str, Union[List[int], torch.Tensor]]]): A list of dictionaries containing input features and labels.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the collated input features and labels.
        """
        # Prepare input features for the model (audio log-Mel spectrograms)
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Cast to half if we're on GPU / using fp16
        if batch["input_features"].dtype == torch.float32 and torch.cuda.is_available():
            batch["input_features"] = batch["input_features"].half()

        # Prepare label features (tokenized text) for the model
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Set padding tokens in labels to -100 so they're ignored in loss computation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove BOS token if present at the start (it will be added during training)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


# ## Metric

cer_metric = evaluate.load("cer")


def normalize_caption(string: str) -> str:
    """Remove all parentheses and their contents from a string and filter garbage characters."""

    # Remove parentheses and square brackets and their content
    string = re.sub(r"[\(\[].*?[\)\]]", "", string)

    # Patterns for valid characters
    ENGLISH_PATTERN = r"([a-zA-Z])"
    CHINESE_PATTERN = r"([\u4e00-\u9fff])"
    DIGIT_PATTERN = r"(\d)"

    def is_valid_char(c: str) -> bool:
        return bool(re.match(ENGLISH_PATTERN, c) or re.match(CHINESE_PATTERN, c) or re.match(DIGIT_PATTERN, c) or c == " ")

    # Filter out invalid characters
    string = "".join(c for c in string if is_valid_char(c))

    # Insert spaces between English and Chinese characters
    string = re.sub(f"{ENGLISH_PATTERN}{CHINESE_PATTERN}", r"\1 \2", string)
    string = re.sub(f"{CHINESE_PATTERN}{ENGLISH_PATTERN}", r"\1 \2", string)

    # Collapse multiple spaces and trim
    string = re.sub(r" {2,}", " ", string).strip()
    return string


def compute_metrics(pred):
    """Compute metrics for the model.

    Args:
        pred (dict): The predictions from the model.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str = [normalize_caption(p) for p in pred_str]
    label_str = [normalize_caption(l) for l in label_str]

    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


# ## Training

repo_id = "Whisper-Cantonese"

training_args = Seq2SeqTrainingArguments(
    # I/O
    output_dir=repo_id,
    overwrite_output_dir=True,
    report_to=["tensorboard"],  # only tensorboard
    logging_dir=f"{repo_id}/runs",  # explicit log folder
    push_to_hub=True,
    hub_strategy="every_save",
    hub_model_id=repo_id,
    hub_private_repo=True,
    # What to do
    eval_strategy="steps",
    eval_steps=10,  # eval every 200 steps
    save_strategy="steps",
    save_steps=10,
    save_total_limit=3,  # keep last 3 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    # Batch & optimization
    # per_device_train_batch_size=64,
    # per_device_eval_batch_size=64,
    auto_find_batch_size=True,
    learning_rate=1e-5,
    weight_decay=0,
    warmup_ratio=0.1,  # 10% warmup
    max_grad_norm=1.0,
    num_train_epochs=20,
    # Precision & memory
    fp16=True,  # 16-bit mixed precision
    gradient_checkpointing=False,
    # Logging
    logging_strategy="steps",
    logging_steps=50,
    logging_first_step=True,
    logging_nan_inf_filter=True,
    # Data loading
    eval_accumulation_steps=2,
    dataloader_num_workers=os.cpu_count(),  # maximize throughput
    dataloader_pin_memory=True,
    group_by_length=True,  # Changed to True to reduce padding (comment corrected)
    remove_unused_columns=False,  # required for PEFT wrapper
    label_names=["labels"],
    # Generation
    predict_with_generate=True,  # compute generative metrics if any
    generation_max_length=128,
    # Repro & devices
    seed=42,
)


# This callback helps to save only the adapter weights and remove the base model weights.
class SavePeftModelCallback(TrainerCallback):
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


trainer = Seq2SeqTrainer(
    model=peft_model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    processing_class=processor.feature_extractor,
    compute_metrics=compute_metrics,
    callbacks=[
        SavePeftModelCallback,
        EarlyStoppingCallback(early_stopping_patience=3),
    ],
    optimizers=(
        create_loraplus_optimizer(
            model=peft_model,
            optimizer_cls=bnb.optim.Adam8bit,
            lr=2e-5,
            loraplus_lr_ratio=16,
        ),
        None,
    ),
)

trainer.train()
