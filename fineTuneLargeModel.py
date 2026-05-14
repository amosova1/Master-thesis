import io
import soundfile as sf
import numpy as np
import torch
from functools import partial
import evaluate
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Trainer
)

def decode_audio(batch):
    audio = batch["audio"]
    audio_bytes = audio["bytes"]

    # decode WAV/FLAC/OGG bytes -> waveform
    waveform, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)

    return {
        "speech": waveform,
        "sampling_rate": sr
    }


def preprocess(batch, processor):
    inputs = processor(
        batch["speech"],
        sampling_rate=batch["sampling_rate"],
        text=batch["human_transcript"]
    )

    return {
        "input_features": inputs.input_features[0],
        "labels": inputs.labels
    }

wer_metric = evaluate.load("wer")

def data_collator(features, processor):
    input_features = [{"input_features": f["input_features"]} for f in features]
    label_features = [{"input_ids": f["labels"]} for f in features]

    batch = processor.feature_extractor.pad(
        input_features,
        return_tensors="pt"
    )

    labels_batch = processor.tokenizer.pad(
        label_features,
        return_tensors="pt"
    )

    labels = labels_batch["input_ids"].masked_fill(
        labels_batch["attention_mask"] == 0,
        -100
    )

    batch["labels"] = labels
    return batch


def compute_metrics(pred, processor, wer_metric):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    if pred_ids.ndim == 3:
        pred_ids = np.argmax(pred_ids, axis=-1)

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(
        pred_ids, skip_special_tokens=True
    )
    label_str = processor.tokenizer.batch_decode(
        label_ids, skip_special_tokens=True
    )

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def main():
    print("Loading dataset...")

    dataset = load_dataset("parquet", data_files={
        "train": "dataset/train-00000-of-00417.parquet",
        "test": "dataset/test-00000-of-00005.parquet"
    })

    # first try
    # dataset["train"] = dataset["train"].select(range(100))
    # dataset["test"] = dataset["test"].select(range(50))

    # second try
    dataset["train"] = dataset["train"].select(range(300))
    dataset["test"] = dataset["test"].select(range(10))

    dataset = dataset.cast_column("audio", Audio(decode=False))

    print("Decoding audio...")
    dataset = dataset.map(decode_audio, num_proc=1)

    print("Loading model...")

    processor = WhisperProcessor.from_pretrained("whisper-large-v3-sk")
    model = WhisperForConditionalGeneration.from_pretrained("whisper-large-v3-sk")
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    print("Preprocessing...")

    preprocess_fn = partial(preprocess, processor=processor)

    dataset = dataset.map(
        preprocess_fn,
        remove_columns=dataset["train"].column_names,
        num_proc=1
    )

    # =========================
    # TRAINING ARGS
    # =========================

    training_args = Seq2SeqTrainingArguments(
        output_dir="whisper-sk-finetuned-second",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=50,
        num_train_epochs=3,
        learning_rate=1e-5,
        fp16=torch.cuda.is_available(),
        report_to="none",
        predict_with_generate=True,
    )

    # =========================
    # TRAINER
    # =========================

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=partial(data_collator, processor=processor),
        compute_metrics=partial(
                            compute_metrics,
                            processor=processor,
                            wer_metric=wer_metric
                        ),
        processing_class=processor
    )

    # =========================
    # TRAIN
    # =========================

    print("Training...")
    trainer.train()

    # =========================
    # EVAL (WER)
    # =========================

    print("Evaluating...")
    metrics = trainer.evaluate()

    print("FINAL WER:", metrics["eval_wer"])

    # =========================
    # SAVE MODEL
    # =========================

    print("Saving model...")

    trainer.save_model("./whisper-sk-final-second")
    processor.save_pretrained("./whisper-sk-final-second")

    print("DONE")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()