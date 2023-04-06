from flair.data import Sentence
from flair.models import SequenceTagger
import time


def predict(sentence: str):
    tokenized_sentence = Sentence(sentence)
    tagger.predict(tokenized_sentence)

    return tokenized_sentence


model_directory = "best-model.pt"
tagger = SequenceTagger.load(model_directory)

start_time = time.time()
input_file = "sample_texts/Englisch The Washington Post.TXT"
with open(input_file, encoding='utf-8') as f:
    for content in f:
        sentences = content.split(".")
        for sentence in sentences:
            if sentence != "":
                tokenized_sentence = predict(sentence)
                for entity in tokenized_sentence.get_spans('ner'):
                    print(entity.text)

end_time = time.time()

print(f"Inference took {end_time-start_time} seconds")
