from torch.optim.lr_scheduler import OneCycleLR

class TrapezoidLRScheduler(OneCycleLR):
    """
    TrapezoidLRScheduler is a learning rate scheduler that uses the trapezoidal learning rate schedule. It is implemnted as a subclass of OneCycleLR with three_phase=True. An additional parameter pct_cooldown is added to specify the percentage of the total steps to be used as start of the cooldown phase. Additionally, the anneal_strategy parameter is set to 'linear' by default, instead of 'cos', which is the default in OneCycleLR.
    
    Args:
    - All arguments from OneCycleLR, except three_phase which is set to True.
    - pct_cooldown (float): The percentage of the total steps to be used as start of the cooldown phase. Default: 0.8
    - step_cooldown (int): Step to start the cooldown phase. If 0, use pct_cooldown to calculate this value. Default: 0.
    """
    
    def __init__(self,
                 optimizer,
                 max_lr,
                 total_steps=None,
                 epochs=None,
                 steps_per_epoch=None,
                 pct_start=0.3,
                 pct_cooldown=0.8,
                 step_cooldown=0,
                 anneal_strategy='linear',
                 cycle_momentum=True,
                 base_momentum=0.85,
                 max_momentum=0.95,
                 div_factor=25.,
                 final_div_factor=1e4,
                 three_phase=True,
                 last_epoch=-1,
                 verbose="deprecated"):
        
        super().__init__(optimizer,
                 max_lr,
                 total_steps,
                 epochs,
                 steps_per_epoch,
                 pct_start,
                 anneal_strategy,
                 cycle_momentum,
                 base_momentum,
                 max_momentum,
                 div_factor,
                 final_div_factor,
                 three_phase,
                 last_epoch,
                 verbose)

        assert three_phase == True, "TrapezoidLRScheduler only supports three_phase=True"
        
        cooldown_start_step = float(pct_cooldown * self.total_steps) - 1 if step_cooldown==0 else step_cooldown
        
        self._schedule_phases = [
                        { # rampup phase
                            'end_step': float(pct_start * self.total_steps) - 1,
                            'start_lr': 'initial_lr',
                            'end_lr': 'max_lr',
                            'start_momentum': 'max_momentum',
                            'end_momentum': 'base_momentum',
                        },
                        { # constant LR phase
                            'end_step': cooldown_start_step,
                            'start_lr': 'max_lr',
                            'end_lr': 'max_lr',
                            'start_momentum': 'base_momentum',
                            'end_momentum': 'base_momentum',
                        },
                        { # cooldown phase
                            'end_step': self.total_steps - 1,
                            'start_lr': 'max_lr',
                            'end_lr': 'min_lr',
                            'start_momentum': 'base_momentum',
                            'end_momentum': 'max_momentum',
                        },
                    ]