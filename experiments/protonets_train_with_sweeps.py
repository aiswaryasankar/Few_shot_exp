
def train_sweep()


    from torch.optim import Adam
    from torch.utils.data import DataLoader
    import argparse

    from few_shot.datasets import OmniglotDataset, MiniImageNet, ClinicDataset
    from few_shot.models import XLNetForEmbedding
    from few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
    from few_shot.proto import proto_net_episode
    from few_shot.train_with_prints import fit
    from few_shot.callbacks import *
    from few_shot.utils import setup_dirs
    from few_shot.utils import get_gpu_info
    from config import PATH
    import wandb
    from transformers import AdamW

    gpu_dict =  get_gpu_info()
    print('Total GPU Mem: {} , Used GPU Mem: {}, Used Percent: {}'.format(gpu_dict['mem_total'], gpu_dict['mem_used'], gpu_dict['mem_used_percent']))


    setup_dirs()
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True


    ##############
    # Parameters #
    ##############
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='clinic150')
    parser.add_argument('--distance', default='l2')
    parser.add_argument('--n-train', default=5, type=int)
    parser.add_argument('--n-test', default=2, type=int)
    parser.add_argument('--k-train', default=5, type=int)
    parser.add_argument('--k-test', default=2, type=int)
    parser.add_argument('--q-train', default=5, type=int)
    parser.add_argument('--q-test', default=2, type=int)
    args = parser.parse_args()

    evaluation_episodes = 100
    episodes_per_epoch = 10

    if args.dataset == 'omniglot':
        n_epochs = 40
        dataset_class = OmniglotDataset
        num_input_channels = 1
        drop_lr_every = 20
    elif args.dataset == 'miniImageNet':
        n_epochs = 80
        dataset_class = MiniImageNet
        num_input_channels = 3
        drop_lr_every = 40
    elif args.dataset == 'clinic150':
        n_epochs = 5
        dataset_class = ClinicDataset
        num_input_channels = 150
        drop_lr_every = 2
    elif args.dataset == 'SNIPS':
        n_epochs = 5
        dataset_class = SNIPS
        num_input_channels = 150
        drop_lr_every = 2        
    else:
        raise(ValueError, 'Unsupported dataset')

    param_str = f'{args.dataset}_nt={args.n_train}_kt={args.k_train}_qt={args.q_train}_' \
                f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}'

    print(param_str)

    ###################
    # Create datasets #
    ###################
    train_df = dataset_class('train')
    train_taskloader = DataLoader(
        train_df,
        batch_sampler=NShotTaskSampler(train_df, episodes_per_epoch, args.n_train, args.k_train, args.q_train)
    )
    val_df = dataset_class('val')
    evaluation_taskloader = DataLoader(
        val_df,
        batch_sampler=NShotTaskSampler(val_df, episodes_per_epoch, args.n_test, args.k_test, args.q_test)
    )

    #train_iter = iter(train_taskloader)
    #train_taskloader = next(train_iter)

    #val_iter = iter(evaluation_taskloader)
    #evaluation_taskloader = next(val_iter)


    #########
    # Wandb #
    #########

    config_defaults = {
          'lr': 0.00001,
          'optimiser': 'adam',
          'batch_size': 16,
      }

    wandb.init(config=config_defaults)

    #########
    # Model #
    #########

    torch.cuda.empty_cache()





    try:
        print('Before Model Move') 
        gpu_dict =  get_gpu_info()
        print('Total GPU Mem: {} , Used GPU Mem: {}, Used Percent: {}'.format(gpu_dict['mem_total'], gpu_dict['mem_used'], gpu_dict['mem_used_percent']))
    except:
        pass
        
    #from transformers import XLNetForSequenceClassification, AdamW


    #model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=150)
    #model.cuda()    

    model = XLNetForEmbedding(num_input_channels)
    model.to(device, dtype=torch.double)

    #param_optimizer = list(model.named_parameters())
    #no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    #optimizer_grouped_parameters = [
    #                                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #                                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
    #]




    try:
        print('After Model Move') 
        gpu_dict =  get_gpu_info()
        print('Total GPU Mem: {} , Used GPU Mem: {}, Used Percent: {}'.format(gpu_dict['mem_total'], gpu_dict['mem_used'], gpu_dict['mem_used_percent']))
    except:
        pass

    wandb.watch(model)


    ############
    # Training #
    ############

        

    from transformers import AdamW

    print(f'Training Prototypical network on {args.dataset}...')
    if wandb.config.optimiser == 'adam':
        optimiser = Adam(model.parameters(), lr=wandb.config.lr)
    else:
        optimiser = AdamW(model.parameters(), lr=wandb.config.lr)

    #optimiser = AdamW(optimizer_grouped_parameters, lr=3e-5)
    #loss_fn = torch.nn.NLLLoss().cuda()

    #loss_fn = torch.nn.CrossEntropyLoss()

    #max_grad_norm = 1.0


    loss_fn = torch.nn.NLLLoss()


    def lr_schedule(epoch, lr):
        # Drop lr every 2000 episodes
        if epoch % drop_lr_every == 0:
            return lr / 2
        else:
            return lr


    callbacks = [
        EvaluateFewShot(
            eval_fn=proto_net_episode,
            num_tasks=evaluation_episodes,
            n_shot=args.n_test,
            k_way=args.k_test,
            q_queries=args.q_test,
            taskloader=evaluation_taskloader,
            prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
            distance=args.distance
        ),
        ModelCheckpoint(
            filepath=PATH + f'/models/proto_nets/{param_str}.pth',
            monitor=f'val_{args.n_test}-shot_{args.k_test}-way_acc'
        ),
        LearningRateScheduler(schedule=lr_schedule),
        CSVLogger(PATH + f'/logs/proto_nets/{param_str}.csv'),
    ]

    try:
        print('Before Fit') 
        print('optimiser :', optimiser )
        print('Learning Rate: ', wandb.config.lr)
        gpu_dict =  get_gpu_info()
        print('Total GPU Mem: {} , Used GPU Mem: {}, Used Percent: {}'.format(gpu_dict['mem_total'], gpu_dict['mem_used'], gpu_dict['mem_used_percent']))
    except:
        pass
        
    fit(
        model,
        optimiser,
        loss_fn,
        epochs=n_epochs,
        dataloader=train_taskloader,
        prepare_batch=prepare_nshot_task(args.n_train, args.k_train, args.q_train),
        callbacks=callbacks,
        metrics=['categorical_accuracy'],
        fit_function=proto_net_episode,
        fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': True,
                             'distance': args.distance},
      )    



