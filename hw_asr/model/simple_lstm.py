from torch import nn
from torch.nn import Sequential, LSTM

from hw_asr.base import BaseModel


class SimpleLSTM(BaseModel):
    def __init__(self, n_feats, n_class, lstm_hidden, fc_hidden, lstm_layers, fc_layers,  *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.lstm = LSTM(n_feats, lstm_hidden, lstm_layers, batch_first=True)
        if fc_layers == 1:
            head = [nn.ReLU(), nn.Linear(lstm_hidden, n_class)]
        else:
            head = [nn.ReLU(), nn.Linear(lstm_hidden, fc_hidden), nn.ReLU()] +\
                   [nn.Linear(fc_hidden, fc_hidden), nn.ReLU()] * (fc_layers - 2) +\
                   [nn.Linear(fc_hidden, n_class)]

        self.head = Sequential(*head)

    def forward(self, spectrogram, *args, **kwargs):
        return {"logits": self.head(self.lstm(spectrogram)[0])}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
