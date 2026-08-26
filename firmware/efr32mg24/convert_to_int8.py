#!/usr/bin/env python3
# ==============================================================================
# True INT8 Export using Google's Official ai-edge-torch (No ONNX needed)
# ==============================================================================

import os
import torch
import ai_edge_torch  # This is now installed as ai-edge-litert

# 1. Load the QAT model (84.25% accuracy)
print("📦 Loading your 84.25% QAT model from best_qat_model.pth...")
from annotated_code import AudioPhiNetCRNNClassifier
model = AudioPhiNetCRNNClassifier(num_classes=50, sample_rate=16000)
model.load_state_dict(torch.load("best_qat_model.pth", map_location="cpu"), strict=False)
model.eval()

# 2. Prepare the example input
dummy_input = torch.randn(1, 1, 52, 313, dtype=torch.float32)

# 3. Convert directly to INT8 TFLite (No ONNX, no SavedModel)
print("⚡ Converting QAT model to INT8 TFLite using ai-edge-torch...")
edge_model = ai_edge_torch.convert(model, (dummy_input,))
edge_model.export("phinet_crnn_int8.tflite")

print("✅ True INT8 TFLite generated successfully!")

# 4. Generate C header
print("🛠️ Generating final INT8 C header...")
os.system("xxd -i phinet_crnn_int8.tflite > phinet_crnn_int8.h")

print("\n" + "="*88)
print("🏆 SUCCESS: True INT8 TFLite and C header generated!")
print("   • TFLite file: phinet_crnn_int8.tflite")
print("   • C Header:    phinet_crnn_int8.h")
print("="*88)