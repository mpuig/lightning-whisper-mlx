from lightning_whisper_mlx import LightningWhisperMLX

models = [
    "tiny",
    "large-v3-turbo",
    "large-v3-turbo-4bit",
]

if __name__ == "__main__":
    expected_text = "So, if you insist on this newscasting route, you're going to need to do some serious filtering. Strip out those verbal tics. Force it to adopt a more sophisticated vocabulary. And for the love of all that is unholy, teach it to be concise!"
    print(f"Expected text:\n{expected_text}")
    for model in models:
        whisper = LightningWhisperMLX(model=model, batch_size=12, quant=None)
        text = whisper.transcribe(audio_path="./audio.wav")['text']
        print(f"{model}:\n{text}")
