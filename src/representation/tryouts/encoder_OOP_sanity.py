# src/representation/tryouts/encoder_sanity_check.py

from representation.encoders.distilbert_encoder import DistilBERTEncoder
from representation.data.qev_datamodule import QEvasionDataModule

dm = QEvasionDataModule().prepare()
sample = dm.get_split("train").texts[0]

encoder = DistilBERTEncoder()
embedding = encoder(sample)

print("Encoder embedding shape:", embedding.shape)
