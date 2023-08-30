import torch
from torch.utils.tensorboard import SummaryWriter

class IncrementalSummaryWriter(SummaryWriter):
    def __init__(self, log_dir=None, comment="", purge_step=None, max_queue=10, flush_secs=120, filename_suffix=""):
        super().__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
        self.last_epoch = -1
        self.data = {}
        self.flush_needed = False
        self._epoch_has_figure = False

    @torch.no_grad()
    def add_batch(self, tag, batch, global_step):
        if self.last_epoch == -1:
            self.last_epoch = global_step
        elif global_step != self.last_epoch:
            ## flush data
            for k, v in self.data.items():

                self.add_scalar(k, torch.mean(torch.cat(v)), self.last_epoch)

            self.data.clear()
            self.last_epoch = global_step
            self.flush_needed = True
            self._epoch_has_figure = False
            
        if tag in self.data.keys():
            self.data[tag].append(batch)
        else:
            self.data[tag] = [batch]

    def flush(self):
        self.flush_needed = False
        super().flush()

    def flush_if_needed(self):
        if self.flush_needed:
            self.flush()

    def epoch_has_figure(self):
        return self._epoch_has_figure

    def add_one_figure_per_epoch(self, tag, figure, global_step):
        self.add_figure(tag, figure, global_step)
        self._epoch_has_figure = True