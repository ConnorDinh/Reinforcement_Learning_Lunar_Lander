import torch
from src.network import QNetwork

def test_network():
    q_network = QNetwork(8, 4)
    input = torch.randn(1, 8)
    q_value = q_network.forward(input)
    print("Q-values: ", q_value)
    print("Output shape: ", q_value.shape)
    assert q_value.shape == (1,4)

def test_batch():
    q_network = QNetwork(8,4)
    input = torch.randn(64, 8)
    q_value = q_network.forward(input)
    assert q_value.shape == (64, 4)

test_network()
test_batch()