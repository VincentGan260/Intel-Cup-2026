有的数据集选用val（验证集）比较好，因为val中的数量相对较少。下面的数据集应该都还行，我只是粗看。需要自己仔细看一下，
尤其注意label是不是能够同时支持目标检测和语义分割的测试！！！
重点聚焦于复杂场景、多样天气等等，当然基础的也要有！可以分类，比如晴天、阴天、雾天；复杂场景、简单场景；白天、夜晚等等。

## BDD100K
http://bdd-data.berkeley.edu/download.html

## KITTI
http://www.cvlibs.net/datasets/kitti/raw_data.php

## Cityscapes
https://www.cityscapes-dataset.com/downloads/

## CamVid
https://www.kaggle.com/datasets/carlolepelaars/camvid


视觉部分线路：
先从公开数据集中筛出有标注的可用部分，分别测试目标检测和语义分割模型的基础能力；
确定模型后，不做人行道精细分割，而是提取可通行区域、目标位置和目标大小等视觉特征
规则生成低中高视觉风险；
少量人工标注风险等级用于验证和调阈值；
最后实路测试时再接入雷达修正综合风险。