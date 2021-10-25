from torch import nn
from torch.nn import Sequential, LSTM, Linear
import itertools
from hw_asr.base import BaseModel


class DeepSpeech(BaseModel):
    def __init__(self, n_feats, n_class, lstm_hidden, fc_hidden: list, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.lstm = Sequential(
            LSTM(fc_hidden[-1], lstm_hidden, batch_first=True, bidirectional=True)
        )
        self.lstm_activation = nn.ReLU()

        self.feature_extraction = Sequential(*itertools.chain.from_iterable([Linear(fc_hidden[i - i] if i > 0 else n_feats, fc_hidden[i]), nn.ReLU()] for i in range(len(fc_hidden))))
        self.head = Linear(lstm_hidden, n_class)
        self.lstm_hidden = lstm_hidden

    def forward(self, spectrogram, *args, **kwargs):
        lstm_output = self.lstm(self.feature_extraction(spectrogram))[0]
        lstm_output = self.lstm_activation(lstm_output[:, :, :self.lstm_hidden] + lstm_output[:, :, self.lstm_hidden:])
        return {"logits": self.head(lstm_output)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
