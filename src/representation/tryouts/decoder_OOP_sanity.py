# src/representation/tryouts/decoder_OOP_sanity.py

from representation.decoders.gpt2_decoder import GPT2Decoder
from representation.data.qev_datamodule import QEvasionDataModule

dm = QEvasionDataModule().prepare()
sample = dm.get_split("train").texts[0]

decoder = GPT2Decoder()
embedding = decoder(sample)

print("Decoder embedding shape:", embedding.shape)
