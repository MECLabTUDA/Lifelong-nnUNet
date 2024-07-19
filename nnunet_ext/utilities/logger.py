#https://github.com/EPFL-VILAB/MultiMAE/blob/66910f5b5ba236f5e731883db85fe4f24ee01106/utils/logger.py#L170
import sys, torch, warnings

try: 
    import wandb
except ImportError:
    pass


def try_get_id():
    if "wandb" in sys.modules:
        return wandb.util.generate_id()
    else:
        return "dummy_id"

class _WandbLogger(object):
    def __init__(self, args):
        self.step = 0
        if isinstance(args, dict):
            wandb.init(
                config=args,
                entity=args['wandb_entity'],
                project=args['wandb_project'],
                group=args.get('wandb_group', None),
                name=args.get('wandb_run_name', None)
            )
        else:
            wandb.init(
                config=args,
                entity=args.wandb_entity,
                project=args.wandb_project,
                group=getattr(args, 'wandb_group', None),
                name=getattr(args, 'wandb_run_name', None)
            )

    def set_step(self, step=None):
        self.step = step

    def increment_step(self):
        self.step += 1

    def update(self, metrics):
        log_dict = dict()
        for k, v in metrics.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            log_dict[k] = v
        wandb.log(log_dict, step=self.step)

    def flush(self):
        pass

class _EmptybLogger(object):
    def __init__(self, args=None):
        pass

    def set_step(self, step=None):
        pass

    def increment_step(self):
        pass

    def update(self, metrics=None):
        pass

    def flush(self):
        pass



if "wandb" in sys.modules:
    WandbLogger = _WandbLogger
else:
    WandbLogger = _EmptybLogger
    warnings.warn("wandb is not installed, therefore wandb logging is disabled")