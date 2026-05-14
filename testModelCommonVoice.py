import torch
import os
import librosa
import pandas as pd
import re  # Potrebné pre čistenie textu
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from jiwer import wer

def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = " ".join(text.split())
    return text

# Paths

model_path = "./whisper-sk-final"
# model_path = "./whisper-large-v3-sk"
base_data_path = "./dataset/cv-corpus-25.0-2026-03-09/sk"
test_tsv_path = os.path.join(base_data_path, "test.tsv")

print("Načítavam model...")
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")


df = pd.read_csv(test_tsv_path, sep="\t")
print(f"Načítaných {len(df)} riadkov z test.tsv")

references = []
predictions = []

count = 0
for index, row in df.iterrows():
    audio_file_path = os.path.join(base_data_path, "clips", row["path"])

    if not os.path.exists(audio_file_path):
        continue

    try:
        speech_array, _ = librosa.load(audio_file_path, sr=16000)
        inputs = processor(speech_array, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(input_features, language="slovak")

        pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        clean_ref = normalize_text(row["sentence"])
        clean_pred = normalize_text(pred_text)

        references.append(clean_ref)
        predictions.append(clean_pred)

        count += 1
        if count % 10 == 0:
            print(f"Spracované: {count} vzoriek")
            # print(f"  REF: {clean_ref}")
            # print(f"  PRED: {clean_pred}")

        if count == 500:
            break

    except Exception as e:
        print(f"Chyba pri súbore {row['path']}: {e}")

output_file = "wer_finetuned.txt"

if references:
    final_wer = wer(references, predictions)
    output_text = (
            "\n" + "=" * 30 + "\n"
            f"VÝSLEDNÁ WER (Normalizovaná): {final_wer * 100:.2f} %\n"
            + "=" * 30
    )

    print(output_text)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"\nVýsledok bol uložený do súboru: {output_file}")

else:
    msg = "Neboli spracované žiadne vzorky."
    print(msg)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(msg)