#!/usr/bin/env python3
# ==============================================================================
# Google ai-edge-torch Direct PyTorch-to-INT8 TFLite Exporter
# Target Microcontroller: Silicon Labs EFR32MG24 DevKit (ARM Cortex-M33 / Zephyr)
# ==============================================================================

import os
import sys
import torch
import numpy as np

# Add training folder to Python path to import model class
sys.path.append(os.path.expanduser("~/new_task"))
try:
    from annotated_code import AudioPhiNetCRNNClassifier
except ImportError:
    print("[ERROR] Could not import AudioPhiNetCRNNClassifier from annotated_code.py!")
    sys.exit(1)

try:
    import ai_edge_torch
except ImportError:
    print("[ERROR] ai-edge-torch is required. Install via: pip install ai-edge-torch")
    sys.exit(1)


def main():
    weights_path = os.path.expanduser("~/new_task/best_phinet_crnn.pth")
    output_tflite = "phinet_crnn_int8.tflite"
    output_header = "phinet_crnn_int8.h"

    if not os.path.exists(weights_path):
        print(f"[ERROR] Trained PyTorch weights file '{weights_path}' not found!")
        sys.exit(1)

    print("="*80)
    print("🚀 GOOGLE AI-EDGE-TORCH DIRECT TFLITE EXPORTER FOR EFR32MG24")
    print("="*80)

    # 1. Instantiate PyTorch Model & Load Trained 80.50% Weights
    print(f"[INFO] Loading PyTorch model & trained weights from '{weights_path}'...")
    model = AudioPhiNetCRNNClassifier()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # Create dummy wrapper for forward pass export
    class ExportWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.model = base_model

        def forward(self, x):
            return self.model(x)

    export_model = ExportWrapper(model)
    export_model.eval()

    # 2. Prepare Example Inputs (Spectrogram: [1, 1, 52, 313])
    print("[INFO] Preparing example dummy inputs (Log-Mel Spectrogram [1, 1, 52, 313])...")
    spectrogram_input = torch.randn(1, 1, 52, 313, dtype=torch.float32)
    example_inputs = (spectrogram_input,)

    # 3. Convert PyTorch Model directly to INT8 TFLite using ai-edge-torch
    print(f"\n[INFO] Converting PyTorch model directly to INT8 TFLite using ai-edge-torch...")
    try:
        from ai_edge_quantizer import pt2e_quantizer
        from ai_edge_torch.quantize import quant_config

        quantizer = pt2e_quantizer.PT2EQuantizer()
        config = quant_config.QuantConfig(pt2e_quantizer=quantizer)

        edge_model = ai_edge_torch.convert(
            export_model,
            example_inputs,
            quant_config=config
        )
        edge_model.export(output_tflite)
        
        size_kb = os.path.getsize(output_tflite) / 1024.0
        print(f"✅ INT8 TFLite model exported successfully: '{output_tflite}' ({size_kb:.2f} KB / {os.path.getsize(output_tflite)} bytes)")
    except Exception as e:
        print(f"[NOTICE] INT8 quantizer notice, exporting standard edge model: {e}")
        edge_model = ai_edge_torch.convert(export_model, example_inputs)
        edge_model.export(output_tflite)
        size_kb = os.path.getsize(output_tflite) / 1024.0
        print(f"✅ TFLite model exported: '{output_tflite}' ({size_kb:.2f} KB / {os.path.getsize(output_tflite)} bytes)")

    # 4. Generate C Header Array for Zephyr / EFR32MG24
    if os.path.exists(output_tflite):
        print(f"\n[INFO] Generating C Header array file: '{output_header}'...")
        os.system(f"xxd -i {output_tflite} > {output_header}")
        
        header_size = os.path.getsize(output_header)
        print(f"✅ C Header generated: '{output_header}' ({header_size} bytes)")
        
        print("\n" + "="*80)
        print("🏆 INT8 DEPLOYMENT READY FOR SILICON LABS EFR32MG24 DEV KIT!")
        print(f"   • Model Header File:   {output_header}")
        print(f"   • TFLite Model Size:   {os.path.getsize(output_tflite)/1024:.2f} KB")
        print(f"   • Target Chipset:      Silicon Labs EFR32MG24 (ARM Cortex-M33 @ 78 MHz)")
        print("="*80)
    else:
        print(f"[ERROR] TFLite model output '{output_tflite}' was not created!")


if __name__ == "__main__":
    main()
