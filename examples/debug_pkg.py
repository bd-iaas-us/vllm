import pickle
import torch

# Path to your .pkl file
file_path = "/tmp/err_execute_model_input_20240920-065350.pkl"  # Change this to the actual path if different

# Function to load and inspect the .pkl file
def load_and_inspect_pkl(file_path):
    try:
        # Load the contents of the .pkl file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Print the type of data to understand its structure
        print(f"Data type: {type(data)}\n")

        # Inspect contents based on the data type
        if isinstance(data, dict):
            print("Inspecting a dictionary of inputs:")
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key}: Tensor with shape {value.shape}, dtype: {value.dtype}, device: {value.device}")
                else:
                    print(f"{key}: {value}")
        elif isinstance(data, list):
            print("Inspecting a list of inputs:")
            for i, item in enumerate(data):
                if isinstance(item, torch.Tensor):
                    print(f"Item {i}: Tensor with shape {item.shape}, dtype: {item.dtype}, device: {item.device}")
                else:
                    print(f"Item {i}: {item}")
        else:
            # If it's another type, just print it
            print(f"Data: {data}")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred while loading the .pkl file: {e}")

# Run the inspection
load_and_inspect_pkl(file_path)
