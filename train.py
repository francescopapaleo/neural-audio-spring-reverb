from pathlib import Path
import torch
import pytorch_lightning as pl

from argparse import ArgumentParser

from tcn import TCNModel
from utils.data import SpringDataset

def train():

    torch.backends.cudnn.benchmark = True

    train_configs = [
        {"name" : "TCN",
        "nparams" : 2,
        "nblocks" : 5,
        "dilation_growth" : 10,
        "kernel_size" : 1,
        "causal" : False,
        "train_fraction" : 1.0,
        "train_loss" : "esr",
        "batch_size" : 8,
        "max_epochs" : 100,
        "lr" : 0.01,
        "cond_dim": 1,
        "nchannels": 32,
        "length": 88800,
        "mix": 100,
        "width": 21,
        "stereo": False,
        "tail": True,
        "channel_width": 32,
        "channel_growth": 1,
        "batch_norm": False
        },
    ]

    n_configs = len(train_configs)


    for idx, tconf in enumerate(train_configs):

        parser = ArgumentParser()
        # add PROGRAM level args
        parser.add_argument('--model_type', type=str, default='tcn', help='tcn or lstm')
        parser.add_argument('--root_dir', type=str, default='./data/plate-spring/spring')
        parser.add_argument('--preload', action="store_true")
        parser.add_argument('--sample_rate', type=int, default=16000)
        parser.add_argument('--shuffle', type=bool, default=True)
        parser.add_argument('--train_fraction', type=float, default=1.0)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--num_workers', type=int, default=8)

        args = parser.parse_args()
        print(args)
        
        if "max_epochs" in tconf:
            args.max_epochs = tconf["max_epochs"]
        else:
            args.max_epochs = 60

        print(f"* Training config {idx+1}/{n_configs}")
        print(tconf)
        
        pl.seed_everything(42)

        print(f"* Training config {idx+1}/{n_configs}")

        specifier =  f"{idx+1}-{tconf['name']}"
        specifier += "__causal" if tconf['causal'] else "__noncausal"
        specifier += f"__{tconf['nblocks']}-{tconf['dilation_growth']}-{tconf['kernel_size']}"
        specifier += f"__fraction-{tconf['train_fraction']}-bs{tconf['batch_size']}"
        
        if "train_loss" in tconf:
            args.train_loss = tconf["train_loss"]
            specifier += f"__loss-{tconf['train_loss']}"

        args.default_root_dir = Path("logs", specifier)
        print(args.default_root_dir)
        
        # create logger
        csv_logger = pl.loggers.CSVLogger(args.default_root_dir, name=specifier)
        tb_logger = pl.loggers.TensorBoardLogger(args.default_root_dir, name=specifier)

        print(args.max_epochs)
        # create trainer
        trainer = pl.Trainer(detect_anomaly=True,
                             max_epochs=args.max_epochs,
                            default_root_dir=args.default_root_dir,
                            log_every_n_steps=1,
                            logger = [csv_logger, tb_logger], 
                            check_val_every_n_epoch=1,
                            accelerator="auto")
        
        trainer.logger._log_graph = True
        trainer.logger._default_hp_metric = True

        print(f"* Training config {idx+1}/{n_configs}")

        print("## Loading data...")
        dataset = SpringDataset(root_dir=args.root_dir, split='train')
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train, valid = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, num_workers=args.num_workers )
        val_dataloader = torch.utils.data.DataLoader(valid, batch_size=args.batch_size, num_workers=args.num_workers)

        print("## Creating model...")
        dict_args = vars(args)
        dict_args["nparams"] = 2

        # if tconf["model_type"] == 'tcn':
        dict_args["nblocks"] = tconf["nblocks"]
        dict_args["dilation_growth"] = tconf["dilation_growth"]
        dict_args["kernel_size"] = tconf["kernel_size"]
        dict_args["causal"] = tconf["causal"]
        if "channel_width" in tconf:
            dict_args["channel_width"] = tconf["channel_width"]
        if "channel_growth" in tconf:
            dict_args["channel_growth"] = tconf["channel_growth"]
        model = TCNModel(**dict_args)
        model = model.float()
        
        print("## Summary") 

        print("## Training")
        trainer.fit(model, train_dataloader, val_dataloader)
        # automatically restores model, epoch, step, LR schedulers, etc...
        # trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")

if __name__ == "__main__":
    train()