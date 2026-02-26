# Copyright (c) 2023, Technische Universität Kaiserslautern (TUK) & National University of Sciences and Technology (NUST).
# All rights reserved.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import collections
import torch
import numpy as np
import itertools
import json
import os
from datetime import datetime
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def run_single_experiment(config, hyperparams, trial_id):
    """
    Run a single experiment with given hyperparameters
    
    :param config: Base configuration
    :param hyperparams: Dictionary of hyperparameters to test
    :param trial_id: Unique identifier for this trial
    :return: Dictionary with results
    """
    logger = config.get_logger(f'trial_{trial_id}')
    
    # Update config with hyperparameters
    for key, value in hyperparams.items():
        if key == 'se_reduction':
            config['arch']['args']['se_reduction'] = value
        elif key == 'se_flags':
            config['arch']['args']['se_flags'] = value
        elif key == 'lr':
            config['optimizer']['args']['lr'] = value
        elif key == 'batch_size':
            config['data_loader']['args']['batch_size'] = value
        elif key == 'topology':
            config['arch']['args']['topology'] = value
    
    try:
        # setup data_loader instances
        train_data_loader = config.init_obj('train_data_loader', module_data)
        val_data_loader = config.init_obj('train_data_loader', module_data, mode='val')
        
        # build model architecture
        model = config.init_obj('arch', module_arch)
        
        # prepare for (multi-device) GPU training
        device, device_ids = prepare_device(config['n_gpu'])
        model = model.to(device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        
        # get function handles of loss and metrics
        criterion = getattr(module_loss, config['loss'])
        metrics = [getattr(module_metric, met) for met in config['metrics']]
        
        # build optimizer, learning rate scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
        
        trainer = Trainer(model, criterion, metrics, optimizer,
                          config=config,
                          device=device,
                          data_loader=train_data_loader,
                          valid_data_loader=val_data_loader,
                          test_data_loader=None,
                          lr_scheduler=lr_scheduler)
        
        # Train the model
        trainer.train()
        
        # Get final validation results
        val_log = trainer._valid_epoch(val_data_loader)
        
        # Close tensorboard writer
        trainer.close_tensorboard()
        
        return {
            'trial_id': trial_id,
            'hyperparams': hyperparams,
            'val_loss': val_log['loss'],
            'val_accuracy': val_log.get('accuracy', 0.0),
            'val_f1': val_log.get('f1', 0.0),
            'status': 'completed'
        }
        
    except Exception as e:
        logger.error(f"Trial {trial_id} failed: {str(e)}")
        return {
            'trial_id': trial_id,
            'hyperparams': hyperparams,
            'val_loss': float('inf'),
            'val_accuracy': 0.0,
            'val_f1': 0.0,
            'status': 'failed',
            'error': str(e)
        }


def hyperparameter_tuning(config):
    """
    Simple hyperparameter tuning - just run multiple training sessions
    Each training automatically gets its own directory with timestamp
    """
    logger = config.get_logger('hyperparameter_tuning')
    
    # Define hyperparameter search space
    search_space = {
        'se_reduction': [1, 16],
        'se_flags': [
            {'input': False, 'encoder': True, 'decoder': True, 'bottleneck': False},
        ],
        'lr': [1e-5, 1e-4, 1e-3, 5e-3],  # Learning rates
        'weight_decay': [0, 1e-5, 1e-4],  # Weight decay (L2 regularization)
        'dropout': [0.3, 0.5],  # Dropout rates
    }
    
    # Generate all combinations
    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    all_combinations = list(itertools.product(*param_values))
    
    logger.info(f"Total combinations to test: {len(all_combinations)}")
    
    # Results storage
    results = []
    best_result = None
    best_val_loss = float('inf')
    
    # Run experiments
    for i, combination in enumerate(all_combinations):
        trial_id = f"trial_{i+1:03d}"
        hyperparams = dict(zip(param_names, combination))
        
        logger.info(f"Running {trial_id}: {hyperparams}")
        
        # Create descriptive run_id with hyperparameters
        timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
        se_red = hyperparams['se_reduction']
        se_flags = hyperparams['se_flags']
        flag_str = f"in{se_flags['input']}_enc{se_flags['encoder']}_dec{se_flags['decoder']}"
        
        # Add other hyperparameters to run_id
        lr_str = f"lr{hyperparams.get('lr', 'default')}"
        wd_str = f"wd{hyperparams.get('weight_decay', 'default')}"
        dropout_str = f"drop{hyperparams.get('dropout', 'default')}"
        
        custom_run_id = f"{timestamp}_se{se_red}_{flag_str}_{lr_str}_{wd_str}_{dropout_str}"
        
        # Create new config with custom run_id and hyperparameters
        import copy
        config_copy = copy.deepcopy(config.config)
        config_copy['arch']['args']['se_reduction'] = se_red
        config_copy['arch']['args']['se_flags'] = se_flags
        
        # Apply additional hyperparameters
        if 'lr' in hyperparams:
            config_copy['optimizer']['args']['lr'] = hyperparams['lr']
        if 'weight_decay' in hyperparams:
            config_copy['optimizer']['args']['weight_decay'] = hyperparams['weight_decay']
        if 'dropout' in hyperparams:
            # Note: dropout is handled in the model architecture, may need model-specific handling
            pass
        
        # Create new ConfigParser with custom run_id
        from parse_config import ConfigParser
        config_with_run_id = ConfigParser(config_copy, run_id=custom_run_id)
        
        # Run training - this automatically creates unique directories!
        try:
            # setup data_loader instances
            train_data_loader = config_with_run_id.init_obj('train_data_loader', module_data)
            val_data_loader = config_with_run_id.init_obj('train_data_loader', module_data, mode='val')
            
            # build model architecture
            model = config_with_run_id.init_obj('arch', module_arch)
            
            # prepare for (multi-device) GPU training
            device, device_ids = prepare_device(config_with_run_id['n_gpu'])
            model = model.to(device)
            if len(device_ids) > 1:
                model = torch.nn.DataParallel(model, device_ids=device_ids)
            
            # get function handles of loss and metrics
            criterion = getattr(module_loss, config_with_run_id['loss'])
            metrics = [getattr(module_metric, met) for met in config_with_run_id['metrics']]
            
            # build optimizer, learning rate scheduler
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = config_with_run_id.init_obj('optimizer', torch.optim, trainable_params)
            lr_scheduler = config_with_run_id.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
            
            trainer = Trainer(model, criterion, metrics, optimizer,
                              config=config_with_run_id,
                              device=device,
                              data_loader=train_data_loader,
                              valid_data_loader=val_data_loader,
                              test_data_loader=None,
                              lr_scheduler=lr_scheduler)
            
            # Train the model
            trainer.train()
            
            # Get results from the training
            result = {
                'trial_id': trial_id,
                'hyperparams': hyperparams,
                'status': 'completed',
                'model_dir': str(trainer.models_dir),  # Path to saved model
                'run_id': custom_run_id
            }
            
        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {str(e)}")
            result = {
                'trial_id': trial_id,
                'hyperparams': hyperparams,
                'status': 'failed',
                'error': str(e)
            }
        
        results.append(result)
        logger.info(f"Trial {trial_id} completed")
    
    # Save results summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"hyperparameter_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Check individual model directories in: saved/models/Landsat8_UNet/UNetSE/")


def main(config):
    if config.config.get('hyperparameter_tuning', {}).get('enabled', False):
        hyperparameter_tuning(config)
    else:
        # Original single experiment code
        logger = config.get_logger('train')

        # setup data_loader instances
        train_data_loader = config.init_obj('train_data_loader', module_data)
        val_data_loader = config.init_obj('train_data_loader', module_data, mode='val')
        test_data_loader = config.init_obj('train_data_loader', module_data, mode='test')

        # print data shapes
        print(f"Train data shape: {train_data_loader.dataset[0][0].shape}")
        print(f"Val data shape: {val_data_loader.dataset[0][0].shape}")
        print(f"Test data shape: {test_data_loader.dataset[0][0].shape}")

        # build model architecture, then print to console
        model = config.init_obj('arch', module_arch)
        logger.info(model)

        # prepare for (multi-device) GPU training
        device, device_ids = prepare_device(config['n_gpu'])
        model = model.to(device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        # get function handles of loss and metrics
        criterion = getattr(module_loss, config['loss'])
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        # if epochs != 20 then recalculate gamma in config.json via 0.1 ** (1. / epochs)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        trainer = Trainer(model, criterion, metrics, optimizer,
                          config=config,
                          device=device,
                          data_loader=train_data_loader,
                          valid_data_loader=val_data_loader,
                          test_data_loader=test_data_loader,
                          lr_scheduler=lr_scheduler)

        if config['trainer']['mode'] == 'train':
            trainer.train()
        else:
            log = trainer._valid_epoch(test_data_loader)
            for key, value in log.items():
                logger.info('    test_{:15s}: {}'.format(str(key), value))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='U-Net Forest Segmentation Trainer')
    args.add_argument('-c', '--config', default="./config.json", type=str,
                      help='config file path (default: ./config.json)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--epochs'], type=int, target='trainer;epochs'),
        CustomArgs(['--topology'], type=str, target='arch;args;topology'),
        CustomArgs(['--se_reduction'], type=int, target='arch;args;se_reduction'),
        CustomArgs(['--enable_hyperparameter_tuning'], type=bool, target='hyperparameter_tuning;enabled'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
