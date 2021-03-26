#!/usr/bin/env python
import wandb
from experiments.protonets_train_with_sweeps import train_sweep

if __name__ == '__main__':
    sweep_config = {
        'method': 'grid', #grid, random
        'metric': {
          'name': 'accuracy',
          'goal': 'maximize'
        },
        'parameters': {
            'lr': {
                'values': [0.0001, 0.00001, 0.000001]
            },
            'optimiser': {
                'values': ['adam', 'adamw']
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config, entity="zerodezibels", project="intent_classifier_protonet")
    wandb.agent(sweep_id, function=train_sweep)    


