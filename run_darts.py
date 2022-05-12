from nord.design.differenetiable.darts.darts import main
import argparse

# nohup python -u run_darts_nccl.py --master_port 12355 --master_address 127.0.0.1  --world_size 1 --rank 0 --batch_size 64 --layers 8 --init_channels 16 > outs.txt 2>&1 &

# nohup python -u run_darts.py  --batch_size 64 --layers 8 --init_channels 16 --seed 1337 --epochs 50 > outs.txt 2>&1 &
parser = argparse.ArgumentParser("cifar")
parser.add_argument(
    "--data", type=str, default="../data", help="location of the data corpus"
)
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument(
    "--learning_rate", type=float, default=0.025, help="init learning rate"
)
parser.add_argument(
    "--learning_rate_min", type=float, default=0.001, help="min learning rate"
)
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight_decay", type=float,
                    default=3e-4, help="weight decay")
parser.add_argument("--report_freq", type=float,
                    default=50, help="report frequency")
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
parser.add_argument(
    "--epochs", type=int, default=50, help="num of training epochs"  # 50
)
parser.add_argument(
    "--init_channels", type=int, default=8, help="num of init channels"
)  # 8
parser.add_argument("--layers", type=int, default=16,
                    help="total number of layers")
parser.add_argument(
    "--model_path", type=str, default="saved_models", help="path to save the model"
)
parser.add_argument("--cutout", action="store_true",
                    default=False, help="use cutout")
parser.add_argument("--cutout_length", type=int,
                    default=16, help="cutout length")
parser.add_argument(
    "--drop_path_prob", type=float, default=0.3, help="drop path probability"
)
parser.add_argument("--save", type=str, default="EXP", help="experiment name")
parser.add_argument("--seed", type=int, default=2, help="random seed")
parser.add_argument("--grad_clip", type=float,
                    default=5, help="gradient clipping")
parser.add_argument(
    "--train_portion", type=float, default=0.5, help="portion of training data"
)
parser.add_argument(
    "--unrolled",
    action="store_true",
    default=False,
    help="use one-step unrolled validation loss",
)
parser.add_argument(
    "--arch_learning_rate",
    type=float,
    default=3e-4,
    help="learning rate for arch encoding",
)
parser.add_argument(
    "--arch_weight_decay",
    type=float,
    default=1e-3,
    help="weight decay for arch encoding",
)


if __name__ == "__main__":
    # mymain(parser, 2)
    # mymain(parser)
    main(parser)
