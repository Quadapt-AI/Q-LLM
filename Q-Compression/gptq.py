"""
GPTQ (Group-wise Product Quantization) Implementation from Scratch

This module implements the GPTQ algorithm for compressing large language models.
The implementation follows the methodology described in the paper:
"GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
"""

import os
import shutil
import time
import math
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class GPTQQuantizer:
    """
    Implements GPTQ (Group-wise Product Quantization) for compressing neural network weights.
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        act_order: bool = False,
        use_hessian: bool = True,
    ):
        """
        Initialize the GPTQ quantizer.

        Args:
            bits: Number of bits to use for quantization (default: 4)
            group_size: Size of groups for group-wise quantization (default: 128)
            act_order: Whether to use activation-based ordering (default: False)
            use_hessian: Whether to use Hessian information for error minimization (default: True)
        """
        self.bits = bits
        self.group_size = group_size
        self.act_order = act_order
        self.use_hessian = use_hessian
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1

    def quantize_layer_weights(
        self,
        layer: nn.Linear,
        input_activations: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Quantize the weights of a single layer using GPTQ.

        Args:
            layer: The layer whose weights are to be quantized
            input_activations: Sample input activations for the layer (for Hessian computation)

        Returns:
            Dict containing quantized weights and quantization parameters
        """
        if not hasattr(layer, "weight"):
            raise ValueError("Layer must have a 'weight' attribute")

        weight = layer.weight.data.clone()
        hessian = None
        if self.use_hessian and input_activations is not None:
            hessian = self._compute_hessian(input_activations)

        # ordering for quantization
        ordering = None
        if self.act_order and input_activations is not None:
            ordering = self._determine_ordering(input_activations)

        quantized_results = self._quantize_weight_matrix(
            weight, ordering, hessian
        )

        return quantized_results

    def _compute_hessian(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the Hessian matrix approximation based on input activations.
        """
        if inputs.dim() > 2:
            inputs = inputs.reshape(-1, inputs.size(-1))

        H = torch.matmul(inputs.t(), inputs) / inputs.size(0)
        H = H + 1e-6 * torch.eye(H.size(0), device=H.device)

        return H

    def _determine_ordering(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Determine the order in which to process columns for quantization.
        """
        if inputs.dim() > 2:
            inputs = inputs.reshape(-1, inputs.size(-1))

        importance = torch.sum(torch.abs(inputs), dim=0)
        _, indices = torch.sort(importance, descending=True)

        return indices

    def _quantize_weight_matrix(
        self,
        weight: torch.Tensor,
        ordering: Optional[torch.Tensor] = None,
        hessian: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Quantize a weight matrix using GPTQ algorithm.
        """
        out_features, in_features = weight.shape
        num_groups = math.ceil(in_features / self.group_size)
        scales = torch.zeros((out_features, num_groups), device=weight.device)
        zeros = torch.zeros((out_features, num_groups), device=weight.device)
        quantized_weight = torch.zeros_like(
            weight, dtype=torch.int8, device=weight.device
        )
        W = weight.clone()
        if ordering is not None:
            W_perm = W[:, ordering]
            permutation = ordering
        else:
            W_perm = W
            permutation = torch.arange(in_features, device=weight.device)

        for i in range(0, in_features, self.group_size):
            group_end = min(i + self.group_size, in_features)
            group_size = group_end - i
            group_idx = i // self.group_size
            W_group = W_perm[:, i:group_end]
            scale, zero, q_group = self._quantize_group(W_group)
            scales[:, group_idx] = scale
            zeros[:, group_idx] = zero

            for j in range(i, group_end):
                col_idx = j - i
                if j < len(permutation):
                    perm_idx = permutation[j].item()
                    if perm_idx < quantized_weight.shape[1]:
                        quantized_weight[:, perm_idx] = q_group[:, col_idx]

            if hessian is not None and group_end < in_features:
                W = self._minimize_error(
                    W,
                    W_perm,
                    q_group,
                    permutation,
                    hessian,
                    i,
                    group_end,
                    group_size,
                )

        return {
            "quantized_weight": quantized_weight,
            "scales": scales,
            "zeros": zeros,
            "bits": self.bits,
            "group_size": self.group_size,
        }

    def _quantize_group(
        self, W_group: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize a group of weights.
        """
        w_min = torch.min(W_group, dim=1)[0]
        w_max = torch.max(W_group, dim=1)[0]
        scale = (w_max - w_min) / (self.qmax - self.qmin)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        zero = self.qmin - torch.round(w_min / scale)
        scale_expanded = scale.unsqueeze(1)
        zero_expanded = zero.unsqueeze(1)
        quantized = torch.round(W_group / scale_expanded + zero_expanded)
        quantized = torch.clamp(quantized, self.qmin, self.qmax).to(torch.int8)

        return scale, zero, quantized

    def _minimize_error(
        self,
        W: torch.Tensor,
        W_perm: torch.Tensor,
        q_group: torch.Tensor,
        permutation: torch.Tensor,
        hessian: torch.Tensor,
        start_idx: int,
        end_idx: int,
        group_size: int,
    ) -> torch.Tensor:
        """
        Minimize quantization error using the Hessian matrix.
        """
        out_features, in_features = W.shape
        W_group = W_perm[:, start_idx:end_idx]
        E = W_group - q_group.float()
        future_size = in_features - end_idx
        if future_size > 0:
            if start_idx < hessian.size(0) and end_idx <= hessian.size(1):
                H_current = hessian[start_idx:end_idx, start_idx:end_idx]
                H_future = hessian[start_idx:end_idx, end_idx:in_features]
                for i in range(out_features):
                    error_row = E[i]
                    future_indices = permutation[end_idx:in_features]
                    for j, idx in enumerate(future_indices):
                        if j < future_size and idx < W.size(1):
                            W[i, idx] -= torch.mean(error_row)

        return W

    def dequantize(self, quantized_data: Dict) -> torch.Tensor:
        """
        Dequantize the weights using the stored quantization parameters.
        """
        quantized_weight = quantized_data["quantized_weight"]
        scales = quantized_data["scales"]
        zeros = quantized_data["zeros"]
        group_size = quantized_data["group_size"]
        dequantized = torch.zeros_like(quantized_weight, dtype=torch.float)
        out_features, in_features = quantized_weight.shape

        for i in range(0, in_features, group_size):
            group_end = min(i + group_size, in_features)
            group_idx = i // group_size
            scale = scales[:, group_idx].unsqueeze(1)
            zero = zeros[:, group_idx].unsqueeze(1)
            # dequantize
            dequantized[:, i:group_end] = scale * (
                quantized_weight[:, i:group_end].float() - zero
            )

        return dequantized


class QuantizedLinear(nn.Module):
    """
    Linear layer with quantized weights for efficient inference.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quantized_weight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        bits: int = 4,
        group_size: int = 128,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("quantized_weight", quantized_weight)
        self.register_buffer("scales", scales)
        self.register_buffer("zeros", zeros)
        self.bits = bits
        self.group_size = group_size

        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using quantized weights.
        """
        dequantized = torch.zeros(
            self.out_features, self.in_features, device=x.device, dtype=x.dtype
        )
        # dequantize the weights
        for i in range(0, self.in_features, self.group_size):
            group_end = min(i + self.group_size, self.in_features)
            group_idx = i // self.group_size
            scale = self.scales[:, group_idx].unsqueeze(1)
            zero = self.zeros[:, group_idx].unsqueeze(1)
            dequantized[:, i:group_end] = scale * (
                self.quantized_weight[:, i:group_end].float() - zero
            )
        output = F.linear(x, dequantized, self.bias)

        return output

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        quantizer: GPTQQuantizer,
        input_activations: Optional[torch.Tensor] = None,
    ):
        """
        Convert a standard linear layer to a quantized one.
        """
        q_data = quantizer.quantize_layer_weights(linear, input_activations)

        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            quantized_weight=q_data["quantized_weight"],
            scales=q_data["scales"],
            zeros=q_data["zeros"],
            bits=quantizer.bits,
            group_size=quantizer.group_size,
            bias=linear.bias,
        )


class ModelCompressor:
    """
    Utility for compressing transformer models using GPTQ.
    """

    def __init__(
        self,
        model_id: str,
        bits: int = 4,
        group_size: int = 128,
        act_order: bool = False,
        use_hessian: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_id = model_id
        self.bits = bits
        self.group_size = group_size
        self.act_order = act_order
        self.use_hessian = use_hessian
        self.device = device
        self.base_name = self.model_id.split("/")[-1]
        self.quantizer = GPTQQuantizer(
            bits=bits,
            group_size=group_size,
            act_order=act_order,
            use_hessian=use_hessian,
        )

    def load_model(self):
        """
        Load the model and tokenizer from Hugging Face.
        """
        print(f"Loading model {self.model_id}...")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = self.model.to(self.device)
        print(f"Model loaded: {self.model_id}")
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"Parameter count: {param_count:,}")
        self._save_original_model()

        return self.model, self.tokenizer

    def _save_original_model(self):
        """
        Save the original model to disk.
        """
        save_dir = f"{self.base_name}"
        if os.path.exists(save_dir):
            print(f"Directory {save_dir} already exists. Clearing...")
            shutil.rmtree(save_dir)

        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving original model to {save_dir}...")

        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        print(f"Original model saved to {save_dir}")

    def _generate_calibration_data(self, num_samples=32):
        """
        Generate calibration data for Hessian computation.
        """
        print("Generating calibration data...")
        prompts = [
            "Once upon a time",
            "The quick brown fox",
            "In the beginning",
            "It was a dark and stormy night",
            "To be or not to be",
            "Four score and seven years ago",
            "A long time ago in a galaxy far, far away",
            "Call me Ishmael",
            "It was the best of times, it was the worst of times",
            "Happy families are all alike",
        ]

        while len(prompts) < num_samples:
            prompts.extend(prompts[: num_samples - len(prompts)])

        prompts = prompts[:num_samples]
        encoded_prompts = []
        for prompt in prompts:
            encoded = self.tokenizer(prompt, return_tensors="pt").to(
                self.device
            )
            encoded_prompts.append(encoded)

        return encoded_prompts

    def _trace_activations(self, calibration_data):
        """
        Trace activations through the model to get inputs for each layer.
        """
        print("Tracing activations through the model...")
        self.model.eval()
        layer_inputs = {}
        handles = []

        def get_hook(name):
            def hook(module, input, output):
                if name not in layer_inputs:
                    layer_inputs[name] = []
                flat_input = input[0].detach().view(input[0].size(0), -1)
                layer_inputs[name].append(flat_input)

            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(get_hook(name))
                handles.append(handle)

        with torch.no_grad():
            for batch in tqdm(
                calibration_data, desc="Processing calibration batches"
            ):
                self.model(**batch)

        for handle in handles:
            handle.remove()

        for name in layer_inputs:
            if layer_inputs[name]:
                try:
                    layer_inputs[name] = torch.cat(layer_inputs[name], dim=0)
                except RuntimeError as e:
                    print(
                        f"Warning: Could not concatenate tensors for layer {name}. Using first batch only."
                    )
                    layer_inputs[name] = layer_inputs[name][0]
            else:
                layer_inputs[name] = None

        return layer_inputs

    def compress_model(self):
        """Compress the model using GPTQ."""
        if not hasattr(self, "model"):
            raise ValueError("Model not loaded. Call load_model() first.")

        print(
            f"Compressing model {self.model_id} with {self.bits}-bit quantization..."
        )

        # calibration data and trace activations
        calibration_data = self._generate_calibration_data()
        layer_inputs = self._trace_activations(calibration_data)
        config = self.model.config
        if hasattr(type(self.model), "from_pretrained"):
            quantized_model = type(self.model).from_pretrained(
                self.model_id,
                config=config,
                torch_dtype=torch.float32,
                device_map="auto",
            )
        else:
            model_class = type(self.model)
            quantized_model = model_class(config)

        quantized_model = quantized_model.to(self.device)

        for name, module in tqdm(
            list(self.model.named_modules()), desc="Quantizing layers"
        ):
            if isinstance(module, nn.Linear):
                try:
                    activations = layer_inputs.get(name, None)
                    q_layer = QuantizedLinear.from_linear(
                        module, self.quantizer, activations
                    )
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]

                    parent_module = get_module_by_name(
                        quantized_model, parent_name
                    )
                    if parent_module is not None:
                        setattr(parent_module, child_name, q_layer)
                except Exception as e:
                    print(f"Error quantizing layer {name}: {e}")
        self._save_compressed_model(quantized_model)

        return quantized_model

    def _save_compressed_model(self, quantized_model):
        """
        Save the compressed model to disk.
        """
        save_dir = f"{self.base_name}_gptq_{self.bits}bit"
        if os.path.exists(save_dir):
            print(f"Directory {save_dir} already exists. Clearing...")
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving compressed model to {save_dir}...")
        quantized_model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        config = {
            "bits": self.bits,
            "group_size": self.group_size,
            "act_order": self.act_order,
            "use_hessian": self.use_hessian,
            "original_model": self.model_id,
            "compression_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(
            os.path.join(save_dir, "quantization_config.json"), "w"
        ) as f:
            json.dump(config, f, indent=2)

        print(f"Compressed model saved to {save_dir}")

    def compare_models(self, original_model, quantized_model):
        """
        Compare the original and quantized models.
        """
        print("\nComparing models:")
        orig_params = sum(p.numel() for p in original_model.parameters())
        quant_params = sum(p.numel() for p in quantized_model.parameters())
        print(f"Original model parameters: {orig_params:,}")
        print(f"Quantized model parameters: {quant_params:,}")
        orig_memory = sum(
            p.numel() * p.element_size() for p in original_model.parameters()
        )
        quant_params = 0
        for name, module in quantized_model.named_modules():
            if isinstance(module, QuantizedLinear):
                # Count 4-bit weights (1/8 of float32)
                quant_params += module.quantized_weight.numel() / 8
                quant_params += module.scales.numel()
                quant_params += module.zeros.numel()
            elif isinstance(module, nn.Parameter):
                quant_params += module.numel()

        print(f"Original model parameters: {orig_params:,}")
        print(f"Quantized model parameters (equivalent): {quant_params:,}")
        test_prompt = "Today I learned something interesting about"

        print("\nGenerating sample output from both models:")

        original_model.eval()
        quantized_model.eval()

        input_ids = self.tokenizer(
            test_prompt, return_tensors="pt"
        ).input_ids.to(self.device)

        with torch.no_grad():
            orig_output = original_model.generate(
                input_ids,
                max_length=50,
                num_return_sequences=1,
                temperature=0.7,
            )

            quant_output = quantized_model.generate(
                input_ids,
                max_length=50,
                num_return_sequences=1,
                temperature=0.7,
            )

        print("\nOriginal model output:")
        print(self.tokenizer.decode(orig_output[0], skip_special_tokens=True))

        print("\nQuantized model output:")
        print(self.tokenizer.decode(quant_output[0], skip_special_tokens=True))


def get_module_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    """
    Get a module from a model by its name.
    """
    if name == "":
        return model

    for n, m in model.named_modules():
        if n == name:
            return m

    return None


def main():
    # Define argument parser
    parser = argparse.ArgumentParser(
        description="Compress a HuggingFace model using GPTQ"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
        help="HuggingFace model ID to compress",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="Number of bits for quantization (default: 4)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Group size for quantization (default: 128)",
    )
    parser.add_argument(
        "--use-hessian",
        action="store_true",
        help="Use Hessian for error minimization",
    )
    parser.add_argument(
        "--act-order",
        action="store_true",
        help="Use activation-based ordering",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for compression",
    )

    args = parser.parse_args()

    # model compressor initialization
    compressor = ModelCompressor(
        model_id=args.model,
        bits=args.bits,
        group_size=args.group_size,
        act_order=args.act_order,
        use_hessian=args.use_hessian,
        device=args.device,
    )
    original_model, tokenizer = compressor.load_model()
    # compress model
    quantized_model = compressor.compress_model()
    compressor.compare_models(original_model, quantized_model)
    print("Compression completed successfully!")


if __name__ == "__main__":
    main()
