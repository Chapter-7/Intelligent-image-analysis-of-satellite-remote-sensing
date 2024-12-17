# Intelligent-image-analysis-of-satellite-remote-sensing
卫星遥感智能影像分析

## 文件
### make_data
通过图片操作对数据操作，包括：
将几何坐标转换为图像像素坐标，基于图像大小和最大边界值进行比例缩放。

从网格大小的DataFrame中获取指定图像的最大横坐标（xmax）和最小纵坐标（ymin）。

从WKT格式数据中提取特定图像和类别的多边形列表（如果有）。

将多边形列表转换为图像的轮廓表示，包括外部轮廓和内部孔洞的坐标。

根据轮廓绘制一个二值掩码图像，将外部区域填充为类值（如1），内部孔洞填充为0。

生成特定图像和类别的掩码图像，通过组合上述函数实现。

加载遥感图像的16波段（M）数据，并调整其通道维度。

### model
定义了，unet以及deeplabv模型，具体可以看赛题简介

### train
训练模型文件

### plot
作为测试文件，可以输入图片进行修改并测试
