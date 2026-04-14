import pytorch_lightning as pl
import torch
import torch.optim as optim

class SolarLightningModule(pl.LightningModule):
    def __init__(self, model, criterion, config):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.config = config
        
        # O Learning rate vem do yaml, usaremos ele no configure_optimizers
        self.learning_rate = config['learning_rate']

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx, phase="train"):
        x, y, mask, aux = batch
        active_mask = mask if self.config['use_mask'] else None
        is_physics = self.config.get('loss_function') in ['physics_loss', 'physics_loss_pers']
        
        out = self(x)
        
        if is_physics:
            if self.config.get('loss_function') == 'physics_loss_pers':
                loss, loss_dict = self.criterion(out, y, aux, mask=active_mask, x_past=x)
            else:
                loss, loss_dict = self.criterion(out, y, aux, mask=active_mask)
                
            # Logamos cada métrica física mapeada pela custom loss
            for key, val in loss_dict.items():
                self.log(f"{phase}_{key}", val, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        else:
            loss = self.criterion(out, y, mask=active_mask)
            
        # Loga a perca principal no prog_bar
        self.log(f"{phase}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, phase="val")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Futuramente, schedulers podem ser colocados aqui de forma muito natural
        return optimizer
