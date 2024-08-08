import torch

def load_tensor(file_path):
    return torch.load(file_path)

def compare_tensors(tensor1, tensor2):
    result = {}

    # Compare shapes
    result['shape'] = {
        'tensor1_shape': tensor1.shape,
        'tensor2_shape': tensor2.shape,
        'equal': tensor1.shape == tensor2.shape
    }

    # Compare data types
    result['data_type'] = {
        'tensor1_dtype': tensor1.dtype,
        'tensor2_dtype': tensor2.dtype,
        'equal': tensor1.dtype == tensor2.dtype
    }

    # Compare values
    values_equal = torch.equal(tensor1, tensor2)
    differing_indices = None

    if not values_equal:
        differing_indices = (tensor1 != tensor2).nonzero(as_tuple=False)
        differing_count = differing_indices.size(0)
    else:
        differing_count = 0

    result['values'] = {
        'equal': values_equal,
        'differing_count': differing_count,
        'differing_indices': differing_indices
    }

    return result

def print_comparison(result):
    print("Comparison Result:")
    print(f"Shapes equal: {result['shape']['equal']}")
    print(f"Tensor 1 shape: {result['shape']['tensor1_shape']}")
    print(f"Tensor 2 shape: {result['shape']['tensor2_shape']}")
    print()
    print(f"Data types equal: {result['data_type']['equal']}")
    print(f"Tensor 1 data type: {result['data_type']['tensor1_dtype']}")
    print(f"Tensor 2 data type: {result['data_type']['tensor2_dtype']}")
    print()
    print(f"Values equal: {result['values']['equal']}")
    
    if not result['values']['equal']:
        print(f"Number of differing values: {result['values']['differing_count']}")
        print("Indices of differing values:")
        print(result['values']['differing_indices'])

def main():
    tensor1 = load_tensor('/home/ubuntu/8bit_log/B_transformer_499')
    tensor2 = load_tensor('/home/ubuntu/8bit_log/B_vllm_499')

    comparison_result = compare_tensors(tensor1, tensor2)
    print_comparison(comparison_result)

if __name__ == "__main__":
    main()
