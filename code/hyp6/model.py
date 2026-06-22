from pytorch_tabnet.tab_model import TabNetClassifier
import torch

def build_tabnet_model():
    model = TabNetClassifier(
                        n_d=8,
                        n_a=8,
                        n_steps=3,
                        gamma=1.3,
                        lambda_sparse=1e-3,
                        optimizer_fn=torch.optim.Adam,
                        optimizer_params=dict(lr=0.02),
                        scheduler_params={"step_size":10, "gamma":0.9},
                        scheduler_fn=torch.optim.lr_scheduler.StepLR,
                        mask_type='entmax',
                        device_name='cpu',
                        verbose= 1,
    )
    return model