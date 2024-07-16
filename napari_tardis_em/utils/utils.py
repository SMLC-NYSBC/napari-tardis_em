#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2024                                                   #
#######################################################################
import torch


def get_list_of_device():
    devices = ["cpu"]

    # Check if CUDA (NVIDIA GPU) is available and list all available CUDA devices
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")

    # Check for MPS (Apple's Metal Performance Shaders) availability
    if torch.backends.mps.is_available():
        devices.append("mps")

    return devices
