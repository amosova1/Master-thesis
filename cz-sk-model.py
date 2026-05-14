from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import torchaudio

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("fav-kky/wav2vec2-base-sk-17k")
model = Wav2Vec2Model.from_pretrained("fav-kky/wav2vec2-base-sk-17k")

speech_array, sampling_rate = torchaudio.load("pokus2.wav")
inputs = feature_extractor(
    speech_array,
    sampling_rate=16_000,
    return_tensors="pt"
)["input_values"][0]

output = model(inputs)
embeddings = output.last_hidden_state.detach().numpy()[0]

print("Spracovanie úspešne dokončené!")
print(f"Počet vygenerovaných časových rámcov: {embeddings.shape[0]}")
print(f"Dimenzia embeddingu: {embeddings.shape[1]}")
print("Ukážka dát (prvých pár čísel):")
print(embeddings[0, :10])