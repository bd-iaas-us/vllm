import unittest
import torch
from vllm.model_executor.models.llama import CpuOffloadLlamaDecoderLayer
import tempfile
from vllm.model_executor.parallel_utils.parallel_state import (
    destroy_model_parallel, initialize_model_parallel)

class TestCpuOffloadLlamaDecoderLayer(unittest.TestCase):
    def setUp(self):
        # Define a configuration
        class LlamaConfig:
            def __init__(self):
                self.hidden_size = 512
                self.num_attention_heads = 8
                self.num_key_value_heads = 8
                self.intermediate_size = 2048
                self.hidden_act = "silu"
                self.rms_norm_eps = 1e-6
                self.rope_theta = 10000
                self.rope_scaling = None
                self.max_position_embeddings = 8192

        config = LlamaConfig()

        # Check if distributed environment is initialized
        if not torch.distributed.is_initialized():
            # Initialize distributed environment
            temp_file = tempfile.mkstemp()[1]
            torch.distributed.init_process_group(
                backend="nccl",
                world_size=1,
                rank=0,
                init_method=f"file://{temp_file}",
            )
            torch.distributed.all_reduce(torch.zeros(1).cuda())

            initialize_model_parallel(1, 1)

        # Create an instance of the model
        self.model = CpuOffloadLlamaDecoderLayer(config)

    def test_forward(self):
        # Define input tensors
        positions = torch.randn(2, 10, 512)  # Example shape
        hidden_states = torch.randn(2, 10, 512)  # Example shape
        kv_cache = [torch.randn(2, 10, 512), torch.randn(2, 10, 512)]  # Example shape
        input_metadata = None
        residual = torch.randn(2, 10, 512)  # Example shape

        # Forward pass
        output_hidden_states, output_residual = self.model(positions, hidden_states, kv_cache, input_metadata, residual)

        # Assertions
        self.assertEqual(output_hidden_states.shape, (2, 10, 512))  # Example output shape
        self.assertEqual(output_residual.shape, (2, 10, 512))  # Example output shape

    # def test_forward_no_residual(self):
    #     # Define input tensors
    #     positions = torch.randn(2, 10, 512)  # Example shape
    #     hidden_states = torch.randn(2, 10, 512)  # Example shape
    #     kv_cache = [torch.randn(2, 10, 512), torch.randn(2, 10, 512)]  # Example shape
    #     input_metadata = None
    #     residual = None

    #     # Forward pass
    #     output_hidden_states, output_residual = self.model(positions, hidden_states, kv_cache, input_metadata, residual)

    #     # Assertions
    #     self.assertEqual(output_hidden_states.shape, (2, 10, 512))  # Example output shape
    #     self.assertEqual(output_residual.shape, (2, 10, 512))  # Example output shape

if __name__ == '__main__':
    unittest.main()
