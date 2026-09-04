import argparse

import torch
import torch.nn as nn
import model.detector
import utils.utils


def fuse_conv_and_batch_norm(module):
    """Fuse adjacent Conv2d/BatchNorm2d pairs in an evaluation model."""
    for child in module.children():
        fuse_conv_and_batch_norm(child)

    if isinstance(module, nn.Sequential):
        layers = list(module.children())
        for index in range(len(layers) - 1):
            if isinstance(layers[index], nn.Conv2d) and isinstance(layers[index + 1], nn.BatchNorm2d):
                module[index] = torch.nn.utils.fuse_conv_bn_eval(
                    layers[index], layers[index + 1]
                )
                module[index + 1] = nn.Identity()

if __name__ == '__main__':
    #指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='', 
                        help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='', 
                        help='The path of the .pth model to be transformed')

    parser.add_argument('--output', type=str, default='./model.onnx', 
                        help='The path where the onnx model is saved')

    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True, True).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))
    #sets the module in eval node
    model.eval()
    fuse_conv_and_batch_norm(model)

    test_data = torch.rand(1, 3, cfg["height"], cfg["width"], device=device)
    torch.onnx.export(model,                    #model being run
                     test_data,                 # model input (or a tuple for multiple inputs)
                     opt.output,               # where to save the model (can be a file or file-like object)
                     export_params=True,        # store the trained parameter weights inside the model file
                     opset_version=9,           # the ONNX version to export the model to
                     do_constant_folding=True,  # whether to execute constant folding for optimization
                     dynamo=False)              # use the legacy exporter to preserve opset 9

    

