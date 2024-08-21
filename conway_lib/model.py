# conway_lib/model.py

import torch
from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper
import os
import datetime

class ConwayModel:
    def __init__(self, max_length, num_words, device, model_name, dim=256, depth=12, heads=8, attn_dim_head=64, rot_pos=True):
        self.device = device
        self.model_name = model_name
        self.model, self.model_dir = self.get_model(max_length, num_words, dim, depth, heads, attn_dim_head, rot_pos)

    def get_model(self, max_length, num_words, dim, depth, heads, attn_dim_head, rot_pos):
        model = TransformerWrapper(
            num_tokens=num_words,
            max_seq_len=max_length,
            attn_layers=Decoder(
                dim=dim,
                depth=depth,
                heads=heads,
                attn_dim_head=attn_dim_head,
                rotary_pos_emb=rot_pos,
                attn_flash=True
            )
        )
        model = AutoregressiveWrapper(model)
        model.cuda()

        model_creation_time = datetime.datetime.now()
        model_creation_time_str = model_creation_time.strftime("%Y-%m-%d %H-%M-%S")
        print(f'Model "{self.model_name}" Created @ {model_creation_time_str} Eastern Time')

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model has {num_params} trainable parameters")

        model_dir = os.path.join("model_parameters", f"{self.model_name}_{model_creation_time_str}")
        os.makedirs(model_dir, exist_ok=True)

        model_info_file = os.path.join(model_dir, f"{self.model_name}_info.txt")
        with open(model_info_file, 'w') as f:
            f.write(f"Model Name: {self.model_name}\n")
            f.write(f"Model Created @ {model_creation_time_str} Eastern Time\n")
            f.write(f"Number of trainable parameters: {num_params}\n")
            f.write(f"Model Architecture:\n{model}\n")

        return model, model_dir
