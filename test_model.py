
import torch
from transformers import AutoProcessor, AutoModel

model_id = "google/medsiglip-448"

model = AutoModel.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)