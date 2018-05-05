import torch


cnn_params = {
    'lr': 0.003,
    'dropout':0.25,
    'epochs': 1,
    'cuda': torch.cuda.is_available(),
    'channel_count': 512
}