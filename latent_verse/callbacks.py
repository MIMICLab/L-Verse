from typing import Any, Optional, Tuple

import pytorch_lightning as pl
from pytorch_lightning import Callback

from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.utilities.distributed import rank_zero_only
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF





class ReconstructedImageLogger(Callback):
    def __init__(
        self,
        every_n_steps: int = 1000,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = True,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
        use_wandb: bool = False,
        multi_optim = False,
    ) -> None:
        """
        Args:
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``True``.
            norm_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """
        super().__init__()
        self.every_n_steps = every_n_steps
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value
        self.multi_optim = multi_optim
        self.use_wandb = use_wandb

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,        
    ) -> None:
        """Called when the train batch ends."""

        if batch_idx % self.every_n_steps == 0:
            if self.multi_optim:
                x = outputs[0]['x']
                xrec = outputs[0]['xrec']
            else:
                x = outputs['x']
                xrec = outputs['xrec']

            x_grid = torchvision.utils.make_grid(
                    tensor=x,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )           
            xrec_grid = torchvision.utils.make_grid(
                    tensor=xrec,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )  

            x_title = "train/input"
            trainer.logger.experiment.add_image(x_title, x_grid, global_step=trainer.global_step)
            xrec_title = "train/reconstruction"
            trainer.logger.experiment.add_image(xrec_title, xrec_grid, global_step=trainer.global_step)

    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the validation batch ends."""
        if batch_idx % self.every_n_steps == 0:   
            x = outputs['x']
            xrec = outputs['xrec']
            x_grid = torchvision.utils.make_grid(
                    tensor=x,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )           
            xrec_grid = torchvision.utils.make_grid(
                    tensor=xrec,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    value_range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )  
 
            x_title = "val/input"
            trainer.logger.experiment.add_image(x_title, x_grid, global_step=batch_idx)
            xrec_title = "val/reconstruction"
            trainer.logger.experiment.add_image(xrec_title, xrec_grid, global_step=batch_idx)

    def on_test_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,        
    ) -> None:
        """Called when the text batch ends."""              
        x = outputs['input_images']
        xrec = outputs['generated_images']

        x_grid = torchvision.utils.make_grid(
                tensor=x,
                nrow=self.nrow,
                padding=self.padding,
                normalize=self.normalize,
                value_range=self.norm_range,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            )           
        xrec_grid = torchvision.utils.make_grid(
                tensor=xrec,
                nrow=self.nrow,
                padding=self.padding,
                normalize=self.normalize,
                value_range=self.norm_range,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            )  

        x_title = "test/input"
        trainer.logger.experiment.add_image(x_title, x_grid, global_step=batch_idx)
        xrec_title = "test/reconstruction"
        trainer.logger.experiment.add_image(xrec_title, xrec_grid, global_step=batch_idx)
        
        pl_module.gather_and_save(outputs, batch_idx)
        return outputs