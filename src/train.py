from pathlib import Path
import argparse
import shutil
import traceback

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from models import get_model

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['srcnn','fsrcnn','edsr','srgan'], 
                         help='select model', required=True)
    parser.add_argument('--gpus', type=str, default='0',help='number of gpu')
    parser.add_argument('--seed', type=int, default=0, help='seed number')
    parser.add_argument('--logdir', type=str, default='./logs', help='log directory')
    parser.add_argument('--epochs', type=int, default=1000, help='max_epochs')
    return parser

def main():
    parser  = get_arguments()
    opt, _  = parser.parse_known_args()
    pl.seed_everything(opt.seed)

    Model = get_model(opt.model)
    # add model specific arguments to original parser
    parser = Model.add_model_specific_args(parser)
    opt    = parser.parse_args()
    # WandB Logger
    wandb_logger = WandbLogger(
                        project='sr',
                        offline = True,
    )
    wandb_logger.log_hyperparams(dict(opt.__dict__))
    logpath = Path("%s/%s/%s" %(opt.logdir, opt.model, wandb_logger.version))
    logpath.mkdir(parents=True, exist_ok=True)
 
    model = Model(opt)
    # define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath = logpath,
        monitor='val_loss',
        save_weights_only=True,
        mode="min",
    )
    ealry_stop_callback = EarlyStopping(
        patience=20,
        min_delta=0.00,
        monitor='val_loss',
        verbose=False,
        mode="min",
    )
    # define trainer
    trainer = pl.Trainer(
        callbacks=[ealry_stop_callback, checkpoint_callback],
        logger= wandb_logger,
        gpus= [int(i) for i in opt.gpus.split(',')],
        max_epochs=opt.epochs,
        #log_every_n_steps=10,
        #check_val_every_n_epoch=10,
    )
    # start training
    ierror=False
    try:
        trainer.fit(model)
    except KeyboardInterrupt as e:
        traceback.print_stack()
        traceback.print_exc()
        ierror=True
    finally:
        if ierror:
            shutil.rmtree(logpath)
            print('Stop Running : Remove  %s' %(logpath))

if __name__ == "__main__":
    main()
