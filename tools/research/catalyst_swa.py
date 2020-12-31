# -*- coding: utf-8 -*-
"""
catalyst is a good framework for training, but very slow!!!
"""
import os
import torch
from catalyst import dl
from torch.nn import CrossEntropyLoss
from catalyst.contrib.nn.criterion.lovasz import LovaszLossMultiClass
import argparse

from torchseg.utils.configs.semanticseg_config import get_net,load_config
from torchseg.utils.torch_tools import get_loaders,get_ckpt_path,load_ckpt
from torchseg.utils.metrics import get_scores


class CustomRunner(dl.Runner):
    def __init__(self,config):
        super().__init__()
        self.config=config 
        if config.loss_type == 'cross_entropy':
            self.loss_fn=CrossEntropyLoss(ignore_index=config.ignore_index)
        elif config.loss_type == 'lovasz_softmax':
            self.loss_fn=LovaszLossMultiClass(ignore=config.ignore_index)
        else:
            raise NotImplementedError
        
    def predict_batch(self, batch):
        # model inference step
        return self.model(batch[0].to(self.device).float())

    def _handle_batch(self, batch):
        # model train/valid step
    
        outputs = self.model(batch[0].to(self.device).float())
        
        if isinstance(outputs, dict):
            prediction=outputs['seg']
        elif isinstance(outputs, torch.Tensor):
            prediction=outputs
        else:
            assert False, 'unexcepted outputs type %s' % type(outputs)
        
        
        labels=batch[1].to(self.device).long()
        loss = self.loss_fn(prediction,labels)
        
        np_prediction=torch.argmax(prediction, dim=1).data.cpu().numpy()
        np_labels=labels.data.cpu().numpy()
        scores=get_scores(label_trues=np_labels,label_preds=np_prediction)
        self.batch_metrics.update(
            {"loss": loss, 
             "acc": scores['Overall Acc'],
             "macc": scores['Mean Acc'], 
             "miou": scores['Mean IoU']}
        )

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

if __name__ == '__main__':
    parser=argparse.ArgumentParser(prog='catalyst swa experiment')
    parser.add_argument('--checkpoint_path',
                        help='checkpoint dir or txt/pkl path for model',
                        required=True)
    
    args=parser.parse_args()
    config=load_config(args.checkpoint_path)
    config.batch_size=5
    model = get_net(config)
    ckpt_path=get_ckpt_path(args.checkpoint_path)
    model = load_ckpt(model,ckpt_path)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    
    train_loader,val_loader=get_loaders(config)
    loaders = {
        "train": train_loader,
        "valid": val_loader,
    }
    
    runner = CustomRunner(config)
    # model training
    runner.train(
        model=model,
        optimizer=optimizer,
        loaders=loaders,
        logdir=os.path.expanduser('~/tmp/logs/catalyst'),
        num_epochs=5,
        verbose=True,
        load_best_on_end=True,
    )
    
    # model tracing
    traced_model = runner.trace(loader=loaders["valid"])