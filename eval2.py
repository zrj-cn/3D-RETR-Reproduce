"""
本文件的中文注释由@zrj-cn添加
"""
# 导入 Python 标准库中的 argparse 模块，用于解析命令行参数
import argparse
import os
from pprint import PrettyPrinter

import numpy as np
# 导入 PyTorch Lightning 库，用于简化 PyTorch 的训练过程
import pytorch_lightning as pl
from torch.utils.data import DataLoader
# 从 tqdm 库中导入 tqdm 类，用于在循环中显示进度条
from tqdm import tqdm

# 从 src.data 包中导入自定义的 ShapeNetDataset等类
from src.data import ShapeNetDataset, ShuffleDataset, transforms, normalize
# 从 src.data.binvox_rw 模块中导入 Voxels 类
from src.data.binvox_rw import Voxels
# 从 src.image2voxel 模块中导入 Image2Voxel 类，用于将图片转为体素
from src.image2voxel import Image2Voxel
# 从 src.utils 包中导入 load_config 函数
from src.utils import load_config


def save_binvox(voxel, dest, translate, scale):
    """
    将体素数据保存为binvox格式，然后调用write方法将其保存到指定文件
    参数：
        voxel (numpy.ndarray): 体素数据。
        dest (str): 保存路径。
        translate (tuple): 平移向量。
        scale (float): 缩放比例。
    """
    binvox = Voxels(voxel, voxel.shape, translate, scale, 'xyz')
    binvox.write(open(dest, 'wb'))


def to_numpy(image):
    """
    将PIL图像转换为numpy数组，并进行归一化
    参数：
        image:PIL图像对象。
    返回：
        numpy.ndarray: 转换后的numpy数组。
    """
    image.convert("RGB")
    # 转换并归一化
    return [np.asarray(image, dtype=np.float32) / 255]


if __name__ == '__main__':
    """
    创建一个命令行参数解析器，用于解析命令行参数
    """
    parser = argparse.ArgumentParser(description='Train transformer conditioned on image inputs')
    parser.add_argument('--annot_path', type=str, required=True,
                        help='Path to "ShapeNet.json"')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the voxel models')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input images')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for dataloader')
    parser.add_argument('--seed', type=int, default=0,
                        help='Manual seed for python, numpy and pytorch')
    parser.add_argument('--split', type=str, default='val',
                        help='"train", "test", or "val"')
    parser.add_argument("--transformer_config", type=str, default=None,
                        help='Path to the image2voxel config file')
    parser.add_argument("--background", type=int, nargs=3, default=(0, 0, 0),
                        help='The (R, G, B) color for the image background')
    parser.add_argument("--beam", type=int, default=1,
                        help='Number of beams for generation')
    parser.add_argument("--view_num", type=int, default=1,
                        help='Number of views for the image input')
    parser.add_argument("--threshold", type=float, default=0.5,
                        help='Threshold for deciding voxel occupancy')
    parser.add_argument("--predict", action='store_true',
                        help='Predict and save results')
    parser.add_argument("--save_path", type=str, default=None,
                        help='Path to save the prediction')
    # 将 PyTorch Lightning 的训练器参数添加到解析器中
    parser = pl.Trainer.add_argparse_args(parser)
    # 解析命令行参数
    args = parser.parse_args()

    if args.resume_from_checkpoint is None:
        raise ValueError('No checkpoint specified')

    # 使用 PrettyPrinter 格式化打印命令行参数，缩进为 4 个空格
    pp = PrettyPrinter(indent=4)
    pp.pprint(vars(args))


    pl.seed_everything(args.seed)

    # 定义图像的变换操作
    image_trans = transforms.Compose([
        # numpy化
        to_numpy,
        # 中心裁剪
        transforms.CenterCrop((224, 224), (128, 128)),
        # 添加随机背景
        transforms.RandomBackground(((240, 240), (240, 240), (240, 240))),
        # 转换为 PyTorch 的张量
        transforms.ToTensor(),
        # 取张量的第一个元素
        lambda x: x[0],
        # 进行归一化操作
        normalize
    ])

    # 数据集参数
    dataset_params = {
        'annot_path': args.annot_path,
        'model_path': args.model_path,
        'image_path': args.image_path
    }
    # 创建 ShapeNet 数据集
    dataset = ShapeNetDataset(
        **dataset_params,
        image_transforms=image_trans,
        split=args.split,
        mode='first',
        background=args.background,
        view_num=args.view_num
    )

    # 对数据集进行随机重排
    dataset = ShuffleDataset(dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )

    # 将图像加载到体素的配置文件
    transformer_config = load_config(args.transformer_config)
    pp.pprint(transformer_config)
    model = Image2Voxel.load_from_checkpoint(
        threshold=args.threshold,
        checkpoint_path=args.resume_from_checkpoint,
        **transformer_config
    )

    # 创建PyTorch Lightning训练器
    trainer = pl.Trainer.from_argparse_args(args, logger=False)
    # 如果指定了--predict参数，则进行预测并保存结果
    if args.predict:
        # 检查是否指定了保存预测结果的路径，如果没有则抛出异常
        if args.save_path is None:
            raise ValueError('save_path is not specified')

        prediction = trainer.predict(model, loader)
        for pred_dict in tqdm(prediction):
            for i in range(len(pred_dict['generation'])):
                tax_path = os.path.join(args.save_path, pred_dict['taxonomy_id'][i], pred_dict['model_id'][i])

                if not os.path.isdir(tax_path):
                    os.makedirs(tax_path)

                voxel = pred_dict['generation'][i][0].cpu().numpy()
                # 保存体素数据为binvox格式
                save_binvox(
                    voxel.astype(np.bool),
                    os.path.join(tax_path, 'prediction.binvox'),
                    pred_dict['translate'][i].cpu().numpy(),
                    pred_dict['scale'][i].cpu().numpy(),
                )
    # 没指定预测的情况
    else:
        trainer.test(model, loader)