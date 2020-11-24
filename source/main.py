from hparams import hyperparams
print(hyperparams.path_dataset_common)

from depen import *
from model import Multi_Synth_pl
from datasets import Common_pl_dataset

from pytorch_model_summary import summary
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

def main(path):
    seed_e(42)
    comet_logger = CometLogger(
    save_dir='log/',
    api_key="23CU99n7TeyZdPeegNDlQ5aHf",
    project_name="sound-proj",
    workspace="etzelkut",
    # rest_api_key=os.environ["COMET_REST_KEY"], # Optional
    # experiment_name="default" # Optional
    )
    dataset_pl = Common_pl_dataset(hyperparams)
    dataset_pl.prepare_data()
    dataset_pl.setup()
    train_loader = dataset_pl.train_dataloader()
    steps_per_epoch = int(len(train_loader))
    print(steps_per_epoch)
    model = Multi_Synth_pl(hyperparams, steps_per_epoch)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
                                    monitor='val_loss',
                                    save_last=True, 
                                      dirpath= os.path.join(path, "/checkpoints"),
                                      filename='sample_model_{epoch}'
                                      )
    trainer = Trainer(callbacks=[checkpoint_callback, lr_monitor],
                    logger=comet_logger,
                    gpus=1,
                    profiler=True,
                    #auto_lr_find=True, #set hparams
                    #gradient_clip_val=0.5,
                    check_val_every_n_epoch=3,
                    #early_stop_callback=True,
                    max_epochs = hyperparams.epochs,
                    #min_epochs=400,
                    progress_bar_refresh_rate = 0,
                    deterministic=True,)
    trainer.fit(model, dataset_pl)
    checkpoint_name = os.path.join(os.path.join(path, "/checkpoints"), 'manual_save' + str(228) + '.ckpt')
    trainer.save_checkpoint(checkpoint_name)
    trainer.test()
    return trainer


