import json
import logging
import os
import shutil
from pathlib import Path

import pandas as pd
import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from hw_asr.base.base_dataset import BaseDataset
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)

URL_LINK = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


class LibrispeechDataset(BaseDataset):
    def __init__(self, data_dir=None, *args, **kwargs):

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = Path(data_dir)
        index = self._get_or_load_index()

        super().__init__(index, *args, **kwargs)

    def _load_part(self):
        arch_path = self._data_dir / "LJSpeech-1.1.tar.bz2"
        download_file(URL_LINK, arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LJSpeech").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LibriSpeech"))

    def _get_or_load_index(self):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_part(part)

        df = pd.read_csv()

        flac_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".flac") for f in filenames]):
                flac_dirs.add(dirpath)
        for flac_dir in tqdm(
                list(flac_dirs), desc=f"Preparing librispeech folders: {part}"
        ):
            flac_dir = Path(flac_dir)
            trans_path = list(flac_dir.glob("*.trans.txt"))[0]
            with trans_path.open() as f:
                for line in f:
                    f_id = line.split()[0]
                    f_text = " ".join(line.split()[1:]).strip()
                    flac_path = flac_dir / f"{f_id}.flac"
                    t_info = torchaudio.info(str(flac_path))
                    length = t_info.num_frames / t_info.sample_rate
                    index.append(
                        {
                            "path": str(flac_path.absolute().resolve()),
                            "text": f_text.lower(),
                            "audio_len": length,
                        }
                    )
        return index


if __name__ == "__main__":
    text_encoder = CTCCharTextEncoder.get_simple_alphabet()
    config_parser = ConfigParser.get_default_configs()

    ds = LibrispeechDataset(
        "dev-clean", text_encoder=text_encoder, config_parser=config_parser
    )
    item = ds[0]
    print(item)
