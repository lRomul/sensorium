import torch

Inputs = tuple[torch.Tensor, torch.Tensor]  # (frames, behavior)
Responses = torch.Tensor  # responses
MouseSample = tuple[Inputs, Responses]  # ((frames, behavior), responses)
MiceSample = tuple[
    tuple[torch.Tensor, list[torch.Tensor]],  # (frames, list[behavior])
    tuple[list[torch.Tensor], torch.Tensor],  # (list[responses], mice_weights)
]
