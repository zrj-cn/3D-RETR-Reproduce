# 3D-RETR: End-to-End Single and Multi-View 3D Reconstruction with Transformers (BMVC 2021)


### **Zai Shi***, **Zhao Meng***, **Yiran Xing**, **Yunpu Ma**, **Roger Wattenhofer**   

∗The first two authors contribute equally to this work

[[BMVC (with presentation)](https://www.bmvc2021-virtualconference.com/conference/papers/paper_1112.html)]
[[Paper](https://www.bmvc2021-virtualconference.com/assets/papers/1112.pdf)]
[[Supplementary](https://www.bmvc2021-virtualconference.com/assets/supp/1112_supp.pdf)]

 
![image](https://user-images.githubusercontent.com/14837467/137624930-96072863-a32b-431f-ab20-985ffd1e51f4.png)

## Citation
```
@inproceedings{3d-retr,
  author    = {Zai Shi, Zhao Meng, Yiran Xing, Yunpu Ma, Roger Wattenhofer},
  title     = {3D-RETR: End-to-End Single and Multi-View3D Reconstruction with Transformers},
  booktitle = {BMVC},
  year      = {2021}
}
```

## Create Environment

```
git clone git@github.com:FomalhautB/3D-RETR.git
cd 3D-RETR
conda env create -f config/environment.yaml
conda activate 3d-retr
```

## Prepare Data

### ShapeNet

Download the [Rendered Images](http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz) and [Voxelization (32)](http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz) and decompress them into `$SHAPENET_IMAGE` and `$SHAPENET_VOXEL`

## Train

Here is an example of reproducing the result of the single view 3D-RETR-B on the ShapeNet dataset:

```
python train.py \
    --model image2voxel \
    --transformer_config config/3d-retr-b.yaml \
    --annot_path data/ShapeNet.json \
    --model_path $SHAPENET_VOX \
    --image_path $SHAPENET_IMAGES \
    --gpus 1 \
    --precision 16 \
    --deterministic \
    --train_batch_size 16 \
    --val_batch_size 16 \
    --num_workers 4 \
    --check_val_every_n_epoch 1 \
    --accumulate_grad_batches 1 \
    --view_num 1 \
    --sample_batch_num 0 \
    --loss_type dice \
```

## Attention

The content of the above document is from the original author. The following are the precautions for the reproduction process:
- Before reproducing the project, please read the reproduction guide to ensure that you can quickly solve the problems I encountered previously.
- Due to time constraints, I can only provide the project reproduction guide in Chinese. Please forgive me. 

以上文档内容来源于原作者，以下为复现过程的注意事项
- 在进行项目复现之前请阅读复现指南内容，以确保能快速解决我先前遇到的问题
- 出于时间原因，我仅能提供中文版的项目复现指南，请见谅