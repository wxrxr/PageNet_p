import argparse

def default_parser():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--config', type=str, required=True)
    parser.add_argument("--config", type=str, default="/home/wxr/PycharmProjects/PageNet/PageNet-pytorch/configs/casia-hwdb.yaml")
    #parser.add_argument("--config", type=str, default="/home/wxr/PycharmProjects/PageNet/PageNet-pytorch/configs/scut-hccdoc.yaml")
    #parser.add_argument("--config", type=str, default="/home/wxr/PycharmProjects/PageNet/PageNet-pytorch/configs/mthv2.yaml")
    return parser