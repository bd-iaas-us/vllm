from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

count = 0
class BitsAndBytesConfig(QuantizationConfig):
    """Config class for BitsAndBytes Quantization.

    Reference: https://arxiv.org/abs/2305.14314
    """

    def __init__(
        self,
        load_in_8bit: bool,
        load_in_4bit: bool,
        bnb_4bit_compute_dtype: str,
        bnb_4bit_quant_type: str,
        bnb_4bit_use_double_quant: bool,
        llm_int8_enable_fp32_cpu_offload: bool,
        llm_int8_has_fp16_weight: bool,
        llm_int8_skip_modules: Optional[Any],
        llm_int8_threshold: float
    ) -> None:
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        self.llm_int8_enable_fp32_cpu_offload = llm_int8_enable_fp32_cpu_offload
        self.llm_int8_has_fp16_weight = llm_int8_has_fp16_weight
        self.llm_int8_skip_modules = llm_int8_skip_modules
        self.llm_int8_threshold = llm_int8_threshold

    def __repr__(self) -> str:
        return "BitsAndBytesConfig"

    @classmethod
    def get_name(self) -> str:
        return "bitsandbytes"

    @classmethod
    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float32, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> List[str]:
        return [
            "adapter_config.json",
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BitsAndBytesConfig":
        load_in_8bit = cls.get_from_keys(config, ["load_in_8bit"])
        load_in_4bit = cls.get_from_keys(config, ["load_in_4bit"])
        bnb_4bit_compute_dtype = cls.get_from_keys(config, ["bnb_4bit_compute_dtype"])
        bnb_4bit_quant_type = cls.get_from_keys(config, ["bnb_4bit_quant_type"])
        bnb_4bit_use_double_quant = cls.get_from_keys(config, ["bnb_4bit_use_double_quant"])
        llm_int8_enable_fp32_cpu_offload = cls.get_from_keys(config, ["llm_int8_enable_fp32_cpu_offload"])
        llm_int8_has_fp16_weight = cls.get_from_keys(config, ["llm_int8_has_fp16_weight"])
        llm_int8_skip_modules = cls.get_from_keys(config, ["llm_int8_skip_modules"])
        llm_int8_threshold = cls.get_from_keys(config, ["llm_int8_threshold"])
        

        return cls(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload,
            llm_int8_has_fp16_weight=llm_int8_has_fp16_weight,
            llm_int8_skip_modules=llm_int8_skip_modules,
            llm_int8_threshold=llm_int8_threshold
        )

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["BitsAndBytesLinearMethod"]:
        if isinstance(layer, LinearBase):
            return BitsAndBytesLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return ["gelu", "gelu_fast", "gelu_new", "gelu_pytorch_tanh"]


class BitsAndBytesLinearMethod(LinearMethodBase):
    """Linear method for BitsAndBytes.

    Args:
       quant_config: The BitsAndBytes quantization config.
    """

    def __init__(self, quant_config: BitsAndBytesConfig):
        try:
            import bitsandbytes
            if bitsandbytes.__version__ < "0.42.0":
                raise ImportError("bitsandbytes version is wrong. Please "
                                  "install bitsandbytes>=0.42.0.")
        except ImportError as err:
            raise ImportError("Please install bitsandbytes>=0.42.0 via "
                              "`pip install bitsandbytes>=0.42.0` to use "
                              "bitsandbytes quantizer.") from err

        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        if self.quant_config.load_in_8bit:
            qweight = Parameter(torch.empty(sum(output_partition_sizes),
                                       input_size_per_partition,
                                       dtype=torch.int8,),
                           requires_grad=False)     
            set_weight_attrs(
                qweight,
                {
                    "input_dim": 0,
                    # In bitsandbytes_4bit, a tensor of shape [n,m] is quantized to
                    #[n*m/pack_ratio, 1],so the output_dim is 0
                    "output_dim": 0,
                    "pack_factor": 1,
                    "use_bitsandbytes_8bit": True,
                })
        else:
            quant_ratio = 0
            if params_dtype.is_floating_point:
                quant_ratio = torch.finfo(params_dtype).bits // torch.iinfo(
                    torch.uint8).bits
            else:
                quant_ratio = torch.iinfo(params_dtype).bits // torch.iinfo(
                    torch.uint8).bits

            if input_size_per_partition * sum(
                    output_partition_sizes) % quant_ratio != 0:
                raise ValueError(
                    "The input size is not aligned with the quantized "
                    "weight shape. ")
            qweight = Parameter(
                torch.empty(
                    input_size_per_partition * sum(output_partition_sizes) //
                    quant_ratio,
                    1,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )

            set_weight_attrs(
                qweight,
                {
                    "input_dim": 0,
                    # In bitsandbytes_4bit, a tensor of shape [n,m] is quantized to
                    #[n*m/pack_ratio, 1],so the output_dim is 0
                    "output_dim": 0,
                    "pack_factor": quant_ratio,
                    "use_bitsandbytes_4bit": True,
                })
        layer.register_parameter("qweight", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if self.quant_config.load_in_8bit:
            return self._apply_8bit_weight(layer, x, bias)
        else:
            return self._apply_4bit_weight(layer, x, bias)

    def _apply_8bit_weight(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        # only load the bitsandbytes module when needed
        from bitsandbytes import matmul, MatmulLtState
        
        global count


        original_type = x.dtype
        bf_x = x.to(torch.bfloat16)

        qweight = layer.qweight
        quant_states = qweight.bnb_quant_state
        offsets = qweight.bnb_shard_offsets

        out_dim_0 = x.shape[0]
        out_dim_1 = sum(
            [quant_state[1].shape[0] for quant_state in quant_states.items()])
        out = torch.empty(out_dim_0,
                          out_dim_1,
                          dtype=torch.float16,
                          device=x.device)
        
        # state = MatmulLtState()
        
        current_index = 0
        for i in range(len(quant_states)):
            count += 1
            output_size = quant_states[i].shape[0]
            weight = qweight[offsets[i]:offsets[i + 1]]

            state = MatmulLtState()
            state.CB = weight
            state.SCB = quant_states[i]
            state.threshold = self.quant_config.llm_int8_threshold
            state.has_fp16_weights = self.quant_config.llm_int8_has_fp16_weight
            state.is_training = False
            if state.threshold > 0.0 and not state.has_fp16_weights:
                state.use_pool = True

            new_x = x.unsqueeze(0)
            
            intermediate_result = matmul(new_x, weight, state=state )
            out[:, current_index:current_index + output_size] = intermediate_result[0]

            current_index += output_size
            if count == 228:
                print(f"~~~~~~~{count}")


        #print(out.shape)
 


        # current_index = 0
        # for i in range(len(quant_states)):
        #     output_size = quant_states[i].shape[0]
        #     # It is more efficient to use out kwarg like
        #     # matmul_4bit(..., out = ...).  Infeasible now due to the bug
        #     # https://github.com/TimDettmers/bitsandbytes/issues/1235.
        #     # Need to change  after the bug is fixed.
        #     out[:, current_index:current_index + output_size] = matmul_4bit(
        #         bf_x, qweight[offsets[i]:offsets[i + 1]].t(), quant_states[i])

            # current_index += output_size

        # out = out.to(original_type)

        if bias is not None:
            out += bias

        return out
    
    def _apply_4bit_weight(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        # only load the bitsandbytes module when needed
        from bitsandbytes import matmul_4bit

        original_type = x.dtype
        bf_x = x.to(torch.bfloat16)

        qweight = layer.qweight
        quant_states = qweight.bnb_quant_state
        offsets = qweight.bnb_shard_offsets

        out_dim_0 = x.shape[0]
        out_dim_1 = sum(
            [quant_state[1].shape[0] for quant_state in quant_states.items()])
        out = torch.empty(out_dim_0,
                          out_dim_1,
                          dtype=torch.bfloat16,
                          device=x.device)

        current_index = 0
        for i in range(len(quant_states)):
            output_size = quant_states[i].shape[0]
            # It is more efficient to use out kwarg like
            # matmul_4bit(..., out = ...).  Infeasible now due to the bug
            # https://github.com/TimDettmers/bitsandbytes/issues/1235.
            # Need to change  after the bug is fixed.
            out[:, current_index:current_index + output_size] = matmul_4bit(
                bf_x, qweight[offsets[i]:offsets[i + 1]].t(), quant_states[i])

            current_index += output_size

        out = out.to(original_type)

        if bias is not None:
            out += bias

        return out

