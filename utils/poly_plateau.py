from torch.optim.lr_scheduler import ReduceLROnPlateau as rop

class poly_rop(rop):
    def __init__(self,poly_max_iter,poly_power,**args):
        super().__init__(**args)
        self.poly_or_rop='rop'
        self.poly_offset=0
        self.poly_max_iter=50
        self.poly_power=0.9
        
    def step(self,metrics,epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
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