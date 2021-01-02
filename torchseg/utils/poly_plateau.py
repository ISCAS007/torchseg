from torch.optim.lr_scheduler import ReduceLROnPlateau as rop
from torch.optim.lr_scheduler import _LRScheduler
import warnings

class poly_rop(rop):
    def __init__(self,poly_max_iter=50,poly_power=0.9,**args):
        super().__init__(**args)
        self.poly_or_rop='rop'
        self.poly_offset=0
        self.poly_max_iter=poly_max_iter
        self.poly_power=poly_power
        
    def step(self,metrics,epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch + 1
            
        self.last_epoch = epoch
        
        if self.poly_or_rop=='rop':
            if self.is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.in_cooldown:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

            if self.num_bad_epochs > self.patience:
                self.old_lrs = [float(param_group['lr']) for param_group in self.optimizer.param_groups]
                
                # self._reduce_lr(epoch)
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0
                self.poly_or_rop='poly'
                self.poly_offset=self.last_epoch
                
        else:
            # poly mode, max_values = self.old_lrs, min_values = max(self.min_lrs, self.factor*self.old_lrs)
            iter = self.last_epoch - self.poly_offset
            if iter == self.poly_max_iter:
                self.poly_or_rop='rop'
            
            scale = (1 - iter/(1.0+self.poly_max_iter))**self.poly_power
            for i, param_group in enumerate(self.optimizer.param_groups):
                old_lr = self.old_lrs[i]
                new_lr = max(old_lr * scale, old_lr*self.factor, self.min_lrs[i])
                if old_lr - new_lr > self.eps:
                    param_group['lr'] = new_lr

    
class poly_lr_scheduler(_LRScheduler):
    def __init__(self,optimizer,max_iter=100,power=0.9,last_epoch=-1):
        super().__init__(optimizer,last_epoch)
        self.max_iter=max_iter
        self.power=power
        
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return self.base_lrs
        
        scale = (1 - self.last_epoch/(1.0+self.max_iter))**self.power
        return [group['lr'] * scale for group in self.base_lrs]
    
    def _get_closed_form_lr(self):
        return self.get_lr()