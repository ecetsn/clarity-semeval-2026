import torch

from src.representation.models.fusion_model import RepresentationFusionModel

# Simulate encoder & decoder outputs
encoder_vec = torch.randn(1, 768)
decoder_vec = torch.randn(1, 768)

model = RepresentationFusionModel(
    encoder_dim=768,
    decoder_dim=768,
    projection_dim=256,
    num_classes=3,  # e.g., Clear / Vague / Evasive
)

model.eval()

with torch.no_grad():
    logits = model(encoder_vec, decoder_vec)

print("Encoder vector shape:", encoder_vec.shape)
print("Decoder vector shape:", decoder_vec.shape)
print("Logits shape:", logits.shape)
print("Logits:", logits)
