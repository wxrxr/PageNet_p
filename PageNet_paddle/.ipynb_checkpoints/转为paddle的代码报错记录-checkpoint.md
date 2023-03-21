# 转为paddle的代码报错记录

- [环境]()
- [使用的数据集]()
- [paddle代码的报错信息]()
- [PageNet_paddle的网络结构]()
- [PageNet_pytorch的网络结构]()
- [casia-hwdb.pth，pytorch的权重转paddle的权重]()


## 环境
```
GCC: gcc-7.5.0
cuda-10.1
cudnn-7.6.5
pytorch-1.7.0+cu101
paddlepaddle-gpu 2.3.2.post101
```

## 使用的数据集
- ICDAR2013 Competition Dataset


## paddle代码报错信息
- pytorch代码可以正常运行，修改为paddle的代码报错。 paddlepaddle-gpu-2.2.2.post101报以下同样的错误。 paddlepaddle-gpu-2.1.3.post101不支持识别省略号。(paddle和pytorch装在同一个虚拟环境，pycharm中debug调试会出现不兼容的现象，使用只有paddle的虚拟环境可正常debug。-->~~paddle的代码从进入val.py这行语句开始，无法再debug出过程中的信息，但pytorch的代码可以： pred_det_rec, pred_read_order, pred_sol, pred_eol = model(images)~~)
- 问题还没找到解决方法，大概可以简化为
```
import torch
x = torch.tensor([[1], [2], [3]])
print(x.size())
print('x.expand(3, 4):', x.expand(3, 4))
print('x.expand(-1, 4):', x.expand(-1, 4))
print('\n####################################################\n')

import paddle
x = paddle.to_tensor([[1], [2], [3]])
#x = x.numpy()
#print(x.size())
print('x.expand(3, 4):', x.expand(3, 4))
print('x.expand(-1, 4):', x.expand(-1, 4))




运行后得：
torch.Size([3, 1])
x.expand(3, 4): tensor([[1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3]])
x.expand(-1, 4): tensor([[1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3]])

####################################################

/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/tensor/creation.py:130: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. 
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  if data.dtype == np.object:
Traceback (most recent call last):
  File "/home/wxr/PycharmProjects/PageNet/PageNet_paddle/1.py", line 12, in <module>
    print('x.expand(3, 4):', x.expand(3, 4))
  File "/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/tensor/manipulation.py", line 1869, in expand
    return _C_ops.expand_v2(x, 'shape', shape)
ValueError: (InvalidArgument) expand_v2(): argument (position 2) must be list or tuple, but got int (at /paddle/paddle/fluid/pybind/op_function.h:438)

```

```
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer1.0.bn1._mean. backbone.layer1.0.bn1._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer1.0.bn1._variance. backbone.layer1.0.bn1._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer1.0.bn2._mean. backbone.layer1.0.bn2._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer1.0.bn2._variance. backbone.layer1.0.bn2._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer1.0.downsample.1._mean. backbone.layer1.0.downsample.1._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer1.0.downsample.1._variance. backbone.layer1.0.downsample.1._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer1.1.bn1._mean. backbone.layer1.1.bn1._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer1.1.bn1._variance. backbone.layer1.1.bn1._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer1.1.bn2._mean. backbone.layer1.1.bn2._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer1.1.bn2._variance. backbone.layer1.1.bn2._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer2.0.bn1._mean. backbone.layer2.0.bn1._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer2.0.bn1._variance. backbone.layer2.0.bn1._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer2.0.bn2._mean. backbone.layer2.0.bn2._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer2.0.bn2._variance. backbone.layer2.0.bn2._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer2.0.downsample.1._mean. backbone.layer2.0.downsample.1._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer2.0.downsample.1._variance. backbone.layer2.0.downsample.1._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer2.1.bn1._mean. backbone.layer2.1.bn1._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer2.1.bn1._variance. backbone.layer2.1.bn1._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer2.1.bn2._mean. backbone.layer2.1.bn2._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer2.1.bn2._variance. backbone.layer2.1.bn2._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer3.0.bn1._mean. backbone.layer3.0.bn1._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer3.0.bn1._variance. backbone.layer3.0.bn1._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer3.0.bn2._mean. backbone.layer3.0.bn2._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer3.0.bn2._variance. backbone.layer3.0.bn2._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer3.0.downsample.1._mean. backbone.layer3.0.downsample.1._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer3.0.downsample.1._variance. backbone.layer3.0.downsample.1._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer3.1.bn1._mean. backbone.layer3.1.bn1._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer3.1.bn1._variance. backbone.layer3.1.bn1._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer3.1.bn2._mean. backbone.layer3.1.bn2._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer3.1.bn2._variance. backbone.layer3.1.bn2._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer4.0.bn1._mean. backbone.layer4.0.bn1._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer4.0.bn1._variance. backbone.layer4.0.bn1._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer4.0.bn2._mean. backbone.layer4.0.bn2._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer4.0.bn2._variance. backbone.layer4.0.bn2._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer4.0.downsample.1._mean. backbone.layer4.0.downsample.1._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer4.0.downsample.1._variance. backbone.layer4.0.downsample.1._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer4.1.bn1._mean. backbone.layer4.1.bn1._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer4.1.bn1._variance. backbone.layer4.1.bn1._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer4.1.bn2._mean. backbone.layer4.1.bn2._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for backbone.layer4.1.bn2._variance. backbone.layer4.1.bn2._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.box_convs.0.bn._mean. srm_rom_feat.box_convs.0.bn._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.box_convs.0.bn._variance. srm_rom_feat.box_convs.0.bn._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.box_convs.1.bn._mean. srm_rom_feat.box_convs.1.bn._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.box_convs.1.bn._variance. srm_rom_feat.box_convs.1.bn._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.box_convs.2.bn._mean. srm_rom_feat.box_convs.2.bn._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.box_convs.2.bn._variance. srm_rom_feat.box_convs.2.bn._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.dis_convs.0.bn._mean. srm_rom_feat.dis_convs.0.bn._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.dis_convs.0.bn._variance. srm_rom_feat.dis_convs.0.bn._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.dis_convs.1.bn._mean. srm_rom_feat.dis_convs.1.bn._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.dis_convs.1.bn._variance. srm_rom_feat.dis_convs.1.bn._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.dis_convs.2.bn._mean. srm_rom_feat.dis_convs.2.bn._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.dis_convs.2.bn._variance. srm_rom_feat.dis_convs.2.bn._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.cls_convs.0.bn._mean. srm_rom_feat.cls_convs.0.bn._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.cls_convs.0.bn._variance. srm_rom_feat.cls_convs.0.bn._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.cls_convs.1.bn._mean. srm_rom_feat.cls_convs.1.bn._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.cls_convs.1.bn._variance. srm_rom_feat.cls_convs.1.bn._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.cls_convs.2.bn._mean. srm_rom_feat.cls_convs.2.bn._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.cls_convs.2.bn._variance. srm_rom_feat.cls_convs.2.bn._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.box2dis_conv.bn._mean. srm_rom_feat.box2dis_conv.bn._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.box2dis_conv.bn._variance. srm_rom_feat.box2dis_conv.bn._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.cls2dis_conv.bn._mean. srm_rom_feat.cls2dis_conv.bn._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.cls2dis_conv.bn._variance. srm_rom_feat.cls2dis_conv.bn._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.rom_convs.0.bn._mean. srm_rom_feat.rom_convs.0.bn._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.rom_convs.0.bn._variance. srm_rom_feat.rom_convs.0.bn._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.rom_convs.1.bn._mean. srm_rom_feat.rom_convs.1.bn._mean is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1492: UserWarning: Skip loading for srm_rom_feat.rom_convs.1.bn._variance. srm_rom_feat.rom_convs.1.bn._variance is not found in the provided dict.
  warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
  0%|          | 0/300 [00:03<?, ?it/s]
Traceback (most recent call last):
  File "/home/wxr/PycharmProjects/PageNet/PageNet_paddle/main.py", line 29, in <module>
    main(cfg)
  File "/home/wxr/PycharmProjects/PageNet/PageNet_paddle/main.py", line 19, in main
    validate(model, val_dataloader, converter, cfg)
  File "/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/decorator.py", line 232, in fun
    return caller(func, *(extras + args), **kw)
  File "/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/base.py", line 354, in _decorate_function
    return func(*args, **kwargs)
  File "/home/wxr/PycharmProjects/PageNet/PageNet_paddle/engine/val.py", line 32, in validate
    pred_det_rec, pred_read_order, pred_sol, pred_eol = model(images)
  File "/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py", line 930, in __call__
    return self._dygraph_call_func(*inputs, **kwargs)
  File "/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py", line 915, in _dygraph_call_func
    outputs = self.forward(*inputs, **kwargs)
  File "/home/wxr/PycharmProjects/PageNet/PageNet_paddle/model/pagenet.py", line 19, in forward
    output = self.predictor(box_feat, dis_feat, cls_feat, rom_feat)
  File "/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py", line 930, in __call__
    return self._dygraph_call_func(*inputs, **kwargs)
  File "/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py", line 915, in _dygraph_call_func
    outputs = self.forward(*inputs, **kwargs)
  File "/home/wxr/PycharmProjects/PageNet/PageNet_paddle/model/predictor.py", line 23, in forward
    grids_x, grids_y = self.get_anchor_coordinates(box_feat.shape[2], box_feat.shape[3])
  File "/home/wxr/PycharmProjects/PageNet/PageNet_paddle/model/predictor.py", line 51, in get_anchor_coordinates
    grids_x = grids_x.expand(nGh, -1)  #nGh=79
  File "/home/wxr/miniconda3/envs/pagenet-10.1/lib/python3.7/site-packages/paddle/tensor/manipulation.py", line 2007, in expand
    return _C_ops.expand_v2(x, 'shape', shape)
ValueError: (InvalidArgument) expand_v2(): argument (position 2) must be list or tuple, but got int (at /paddle/paddle/fluid/pybind/op_function_common.cc:336)


Process finished with exit code 1

```

## PageNet_paddle的网络结构
- 和PageNet_pytorch的网络结构看起来差不多，但没看出来有什么问题
```
PageNet(
  (backbone): Backbone(
    (layer1): Sequential(
      (0): BasicBlock(
        (conv1): Conv2D(1, 64, kernel_size=[3, 3], stride=[2, 2], padding=1, data_format=NCHW)
        (bn1): BatchNorm2D(num_features=64, momentum=0.1, epsilon=1e-05)
        (relu): LeakyReLU(negative_slope=0.1)
        (conv2): Conv2D(64, 64, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn2): BatchNorm2D(num_features=64, momentum=0.1, epsilon=1e-05)
        (downsample): Sequential(
          (0): Conv2D(1, 64, kernel_size=[1, 1], stride=[2, 2], data_format=NCHW)
          (1): BatchNorm2D(num_features=64, momentum=0.1, epsilon=1e-05)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2D(64, 64, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn1): BatchNorm2D(num_features=64, momentum=0.1, epsilon=1e-05)
        (relu): LeakyReLU(negative_slope=0.1)
        (conv2): Conv2D(64, 64, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn2): BatchNorm2D(num_features=64, momentum=0.1, epsilon=1e-05)
      )
    )
    (layer2): Sequential(
      (0): BasicBlock(
        (conv1): Conv2D(64, 128, kernel_size=[3, 3], stride=[2, 2], padding=1, data_format=NCHW)
        (bn1): BatchNorm2D(num_features=128, momentum=0.1, epsilon=1e-05)
        (relu): LeakyReLU(negative_slope=0.1)
        (conv2): Conv2D(128, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn2): BatchNorm2D(num_features=128, momentum=0.1, epsilon=1e-05)
        (downsample): Sequential(
          (0): Conv2D(64, 128, kernel_size=[1, 1], stride=[2, 2], data_format=NCHW)
          (1): BatchNorm2D(num_features=128, momentum=0.1, epsilon=1e-05)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2D(128, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn1): BatchNorm2D(num_features=128, momentum=0.1, epsilon=1e-05)
        (relu): LeakyReLU(negative_slope=0.1)
        (conv2): Conv2D(128, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn2): BatchNorm2D(num_features=128, momentum=0.1, epsilon=1e-05)
      )
    )
    (layer3): Sequential(
      (0): BasicBlock(
        (conv1): Conv2D(128, 256, kernel_size=[3, 3], stride=[2, 2], padding=1, data_format=NCHW)
        (bn1): BatchNorm2D(num_features=256, momentum=0.1, epsilon=1e-05)
        (relu): LeakyReLU(negative_slope=0.1)
        (conv2): Conv2D(256, 256, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn2): BatchNorm2D(num_features=256, momentum=0.1, epsilon=1e-05)
        (downsample): Sequential(
          (0): Conv2D(128, 256, kernel_size=[1, 1], stride=[2, 2], data_format=NCHW)
          (1): BatchNorm2D(num_features=256, momentum=0.1, epsilon=1e-05)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2D(256, 256, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn1): BatchNorm2D(num_features=256, momentum=0.1, epsilon=1e-05)
        (relu): LeakyReLU(negative_slope=0.1)
        (conv2): Conv2D(256, 256, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn2): BatchNorm2D(num_features=256, momentum=0.1, epsilon=1e-05)
      )
    )
    (layer4): Sequential(
      (0): BasicBlock(
        (conv1): Conv2D(256, 512, kernel_size=[3, 3], stride=[2, 2], padding=1, data_format=NCHW)
        (bn1): BatchNorm2D(num_features=512, momentum=0.1, epsilon=1e-05)
        (relu): LeakyReLU(negative_slope=0.1)
        (conv2): Conv2D(512, 512, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn2): BatchNorm2D(num_features=512, momentum=0.1, epsilon=1e-05)
        (downsample): Sequential(
          (0): Conv2D(256, 512, kernel_size=[1, 1], stride=[2, 2], data_format=NCHW)
          (1): BatchNorm2D(num_features=512, momentum=0.1, epsilon=1e-05)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2D(512, 512, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn1): BatchNorm2D(num_features=512, momentum=0.1, epsilon=1e-05)
        (relu): LeakyReLU(negative_slope=0.1)
        (conv2): Conv2D(512, 512, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn2): BatchNorm2D(num_features=512, momentum=0.1, epsilon=1e-05)
      )
    )
  )
  (srm_rom_feat): SRMROMFeat(
    (box_convs): Sequential(
      (0): CBL(
        (conv): Conv2D(512, 256, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn): BatchNorm2D(num_features=256, momentum=0.1, epsilon=1e-05)
        (leakyrelu): LeakyReLU(negative_slope=0.1)
      )
      (1): CBL(
        (conv): Conv2D(256, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn): BatchNorm2D(num_features=128, momentum=0.1, epsilon=1e-05)
        (leakyrelu): LeakyReLU(negative_slope=0.1)
      )
      (2): CBL(
        (conv): Conv2D(128, 64, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn): BatchNorm2D(num_features=64, momentum=0.1, epsilon=1e-05)
        (leakyrelu): LeakyReLU(negative_slope=0.1)
      )
    )
    (dis_convs): Sequential(
      (0): CBL(
        (conv): Conv2D(512, 256, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn): BatchNorm2D(num_features=256, momentum=0.1, epsilon=1e-05)
        (leakyrelu): LeakyReLU(negative_slope=0.1)
      )
      (1): CBL(
        (conv): Conv2D(256, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn): BatchNorm2D(num_features=128, momentum=0.1, epsilon=1e-05)
        (leakyrelu): LeakyReLU(negative_slope=0.1)
      )
      (2): CBL(
        (conv): Conv2D(128, 64, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn): BatchNorm2D(num_features=64, momentum=0.1, epsilon=1e-05)
        (leakyrelu): LeakyReLU(negative_slope=0.1)
      )
    )
    (cls_convs): Sequential(
      (0): CBL(
        (conv): Conv2D(512, 512, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn): BatchNorm2D(num_features=512, momentum=0.1, epsilon=1e-05)
        (leakyrelu): LeakyReLU(negative_slope=0.1)
      )
      (1): CBL(
        (conv): Conv2D(512, 512, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn): BatchNorm2D(num_features=512, momentum=0.1, epsilon=1e-05)
        (leakyrelu): LeakyReLU(negative_slope=0.1)
      )
      (2): CBL(
        (conv): Conv2D(512, 1024, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn): BatchNorm2D(num_features=1024, momentum=0.1, epsilon=1e-05)
        (leakyrelu): LeakyReLU(negative_slope=0.1)
      )
    )
    (box2dis_conv): CBL(
      (conv): Conv2D(64, 64, kernel_size=[1, 1], data_format=NCHW)
      (bn): BatchNorm2D(num_features=64, momentum=0.1, epsilon=1e-05)
      (leakyrelu): LeakyReLU(negative_slope=0.1)
    )
    (cls2dis_conv): CBL(
      (conv): Conv2D(1024, 64, kernel_size=[1, 1], data_format=NCHW)
      (bn): BatchNorm2D(num_features=64, momentum=0.1, epsilon=1e-05)
      (leakyrelu): LeakyReLU(negative_slope=0.1)
    )
    (rom_convs): Sequential(
      (0): CBL(
        (conv): Conv2D(1088, 64, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn): BatchNorm2D(num_features=64, momentum=0.1, epsilon=1e-05)
        (leakyrelu): LeakyReLU(negative_slope=0.1)
      )
      (1): CBL(
        (conv): Conv2D(64, 64, kernel_size=[3, 3], padding=1, data_format=NCHW)
        (bn): BatchNorm2D(num_features=64, momentum=0.1, epsilon=1e-05)
        (leakyrelu): LeakyReLU(negative_slope=0.1)
      )
    )
  )
  (predictor): Predictor(
    (box_fc): Linear(in_features=64, out_features=4, dtype=float32)
    (dis_fc): Linear(in_features=64, out_features=1, dtype=float32)
    (cls_fc): Sequential(
      (0): Dropout(p=0.5, axis=None, mode=upscale_in_train)
      (1): Linear(in_features=1024, out_features=7356, dtype=float32)
    )
    (read_order_fc): Linear(in_features=64, out_features=4, dtype=float32)
    (sol_fc): Linear(in_features=64, out_features=1, dtype=float32)
    (eol_fc): Linear(in_features=64, out_features=1, dtype=float32)
  )
)
```

## PageNet_pytorch的网络结构
```
PageNet(
  (backbone): Backbone(
    (layer1): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        (relu): LeakyReLU(negative_slope=0.1, inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        (downsample): Sequential(
          (0): Conv2d(1, 64, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        (relu): LeakyReLU(negative_slope=0.1, inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      )
    )
    (layer2): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        (relu): LeakyReLU(negative_slope=0.1, inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        (downsample): Sequential(
          (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        (relu): LeakyReLU(negative_slope=0.1, inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      )
    )
    (layer3): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        (relu): LeakyReLU(negative_slope=0.1, inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        (downsample): Sequential(
          (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        (relu): LeakyReLU(negative_slope=0.1, inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      )
    )
    (layer4): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        (relu): LeakyReLU(negative_slope=0.1, inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        (downsample): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        (relu): LeakyReLU(negative_slope=0.1, inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      )
    )
  )
  (srm_rom_feat): SRMROMFeat(
    (box_convs): Sequential(
      (0): CBL(
        (conv): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        (leakyrelu): LeakyReLU(negative_slope=0.1, inplace=True)
      )
      (1): CBL(
        (conv): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        (leakyrelu): LeakyReLU(negative_slope=0.1, inplace=True)
      )
      (2): CBL(
        (conv): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        (leakyrelu): LeakyReLU(negative_slope=0.1, inplace=True)
      )
    )
    (dis_convs): Sequential(
      (0): CBL(
        (conv): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        (leakyrelu): LeakyReLU(negative_slope=0.1, inplace=True)
      )
      (1): CBL(
        (conv): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        (leakyrelu): LeakyReLU(negative_slope=0.1, inplace=True)
      )
      (2): CBL(
        (conv): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        (leakyrelu): LeakyReLU(negative_slope=0.1, inplace=True)
      )
    )
    (cls_convs): Sequential(
      (0): CBL(
        (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        (leakyrelu): LeakyReLU(negative_slope=0.1, inplace=True)
      )
      (1): CBL(
        (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        (leakyrelu): LeakyReLU(negative_slope=0.1, inplace=True)
      )
      (2): CBL(
        (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        (leakyrelu): LeakyReLU(negative_slope=0.1, inplace=True)
      )
    )
    (box2dis_conv): CBL(
      (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (leakyrelu): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (cls2dis_conv): CBL(
      (conv): Conv2d(1024, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (leakyrelu): LeakyReLU(negative_slope=0.1, inplace=True)
    )
    (rom_convs): Sequential(
      (0): CBL(
        (conv): Conv2d(1088, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        (leakyrelu): LeakyReLU(negative_slope=0.1, inplace=True)
      )
      (1): CBL(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        (leakyrelu): LeakyReLU(negative_slope=0.1, inplace=True)
      )
    )
  )
  (predictor): Predictor(
    (box_fc): Linear(in_features=64, out_features=4, bias=False)
    (dis_fc): Linear(in_features=64, out_features=1, bias=False)
    (cls_fc): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=1024, out_features=7356, bias=False)
    )
    (read_order_fc): Linear(in_features=64, out_features=4, bias=False)
    (sol_fc): Linear(in_features=64, out_features=1, bias=False)
    (eol_fc): Linear(in_features=64, out_features=1, bias=False)
  )
)

```


## casia-hwdb.pth，pytorch的权重转paddle的权重
- 运行pth2pdparams.py，对fc层的参数做了维度转换
```
name: backbone.layer1.0.conv1.weight, ori shape: (64, 1, 3, 3), new shape: (3, 3, 1, 64)
name: backbone.layer1.0.conv2.weight, ori shape: (64, 64, 3, 3), new shape: (3, 3, 64, 64)
name: backbone.layer1.1.conv1.weight, ori shape: (64, 64, 3, 3), new shape: (3, 3, 64, 64)
name: backbone.layer1.1.conv2.weight, ori shape: (64, 64, 3, 3), new shape: (3, 3, 64, 64)
name: backbone.layer2.0.conv1.weight, ori shape: (128, 64, 3, 3), new shape: (3, 3, 64, 128)
name: backbone.layer2.0.conv2.weight, ori shape: (128, 128, 3, 3), new shape: (3, 3, 128, 128)
name: backbone.layer2.1.conv1.weight, ori shape: (128, 128, 3, 3), new shape: (3, 3, 128, 128)
name: backbone.layer2.1.conv2.weight, ori shape: (128, 128, 3, 3), new shape: (3, 3, 128, 128)
name: backbone.layer3.0.conv1.weight, ori shape: (256, 128, 3, 3), new shape: (3, 3, 128, 256)
name: backbone.layer3.0.conv2.weight, ori shape: (256, 256, 3, 3), new shape: (3, 3, 256, 256)
name: backbone.layer3.1.conv1.weight, ori shape: (256, 256, 3, 3), new shape: (3, 3, 256, 256)
name: backbone.layer3.1.conv2.weight, ori shape: (256, 256, 3, 3), new shape: (3, 3, 256, 256)
name: backbone.layer4.0.conv1.weight, ori shape: (512, 256, 3, 3), new shape: (3, 3, 256, 512)
name: backbone.layer4.0.conv2.weight, ori shape: (512, 512, 3, 3), new shape: (3, 3, 512, 512)
name: backbone.layer4.1.conv1.weight, ori shape: (512, 512, 3, 3), new shape: (3, 3, 512, 512)
name: backbone.layer4.1.conv2.weight, ori shape: (512, 512, 3, 3), new shape: (3, 3, 512, 512)

```


