import youtokentome as yttm
import json

from hw_asr.utils import ROOT_PATH

with open(ROOT_PATH / "data" / "datasets" / "librispeech" / "train-clean-100_index.json") as f:
    parsed = json.load(f)

train_data_path = ROOT_PATH / "data" / "bpe" / "train-clean-100.txt"

with open(train_data_path, "w+") as f:
    for item in parsed:
        f.write(item["text"])
        f.write('\n')

yttm.BPE.train(data=str(train_data_path), vocab_size=100, model=str(ROOT_PATH / "bpe.model"))
