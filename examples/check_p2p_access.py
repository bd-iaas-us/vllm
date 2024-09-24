import torch

def check_p2p_access():
    num_devices = torch.cuda.device_count()
    for i in range(num_devices):
        for j in range(num_devices):
            if i != j:
                can_access = torch.cuda.can_device_access_peer(i, j)
                if can_access:
                    print(f"Peer access supported between device {i} and {j}")
                else:
                    print(f"Peer access NOT supported between device {i} and {j}")

check_p2p_access()
