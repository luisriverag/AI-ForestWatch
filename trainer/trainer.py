# Copyright (c) 2023, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from base import BaseTrainer
from torch.nn.utils import clip_grad_norm_
from utils import MetricTracker, inf_loop
from sklearn.metrics import classification_report
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, test_data_loader=None,
                 lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.test_data_loader = test_data_loader
        self.do_test = self.test_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns])
        
        # Initialize TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=os.path.join(config.models_dir, 'tensorboard_logs'))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        
        # Create progress bar for training
        pbar = tqdm(self.data_loader, desc=f'Training Epoch {epoch}', 
                   leave=False, ncols=100, unit='batch', 
                   disable=False, dynamic_ncols=True)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            out_x, softmaxed = self.model(data)
            pred = torch.argmax(softmaxed, dim=1)
            loss_target = target.clone()
            loss_target[loss_target != 0] -= 1
            loss_target = loss_target.squeeze(1)
            loss = self.criterion(softmaxed, loss_target)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), 0.05)
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(
                    met.__name__, met(softmaxed, loss_target))

            # Update progress bar with current loss
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})

            # Disable the original progress logging since we have tqdm
            # if batch_idx % self.log_step == 0:
            #     self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
            #         epoch,
            #         self._progress(batch_idx),
            #         loss.item()))

            if batch_idx == self.len_epoch:
                break
        
        pbar.close()
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(self.valid_data_loader)
            log.update(**{'val_'+k: v for k, v in val_log.items()})
        if self.do_test and epoch == self.config['trainer']['epochs']:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            self._resume_checkpoint(best_path)
            self.logger.info("Testing current best: model_best.pth ...")
            test_log = self._valid_epoch(self.test_data_loader)
            log.update(**{'test_'+k: v for k, v in test_log.items()})

        # Log metrics to TensorBoard
        self._log_to_tensorboard(log, epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, data_loader):
        """
        Validate after training an epoch

        :return: A log that contains information about validation
        """
        preds = np.array([])
        targets = np.array([])

        self.model.eval()
        self.valid_metrics.reset()
        
        # Create progress bar for validation
        pbar = tqdm(data_loader, desc='Validation', leave=False, ncols=100, unit='batch', 
                   dynamic_ncols=True)
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)

                out_x, softmaxed = self.model(data)
                pred = torch.argmax(softmaxed, dim=1)
                loss_target = target.clone()
                loss_target[loss_target != 0] -= 1
                loss_target = loss_target.squeeze(1)
                loss = self.criterion(softmaxed, loss_target)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(
                        met.__name__, met(softmaxed, loss_target))

                # Update progress bar with current loss
                pbar.set_postfix({'Loss': f'{loss.item():.6f}'})

                label_valid_indices = (target.view(-1) != 0)
                valid_pred = pred.view(-1)[label_valid_indices]
                valid_label = target.view(-1)[label_valid_indices] - 1
                preds = np.concatenate((preds, valid_pred.view(-1).cpu()), axis=0)
                targets = np.concatenate((targets, valid_label.view(-1).cpu()), axis=0)
        
        pbar.close()
        log = self.valid_metrics.result()
        log['classification_report'] = "\n" + classification_report(targets, preds, target_names=('Non-Forest', 'Forest'))
        return log

    def _log_to_tensorboard(self, log, epoch):
        """
        Log metrics to TensorBoard
        
        :param log: Dictionary containing metrics to log
        :param epoch: Current epoch number
        """
        for key, value in log.items():
            if isinstance(value, (int, float)) and key != 'epoch':
                self.tb_writer.add_scalar(key, value, epoch)
        
        # Log learning rate if available
        if self.lr_scheduler is not None:
            self.tb_writer.add_scalar('learning_rate', 
                                    self.lr_scheduler.get_last_lr()[0], epoch)
    
    def close_tensorboard(self):
        """
        Close TensorBoard writer
        """
        self.tb_writer.close()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
