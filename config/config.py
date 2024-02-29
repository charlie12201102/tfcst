class DefaultConfig():
    # backbone
    # 是否预训练   不使用
    pretrained = False
    #参数冻结   不使用
    freeze_stage_1 = False
    freeze_bn = False

    # 检测层的输出通道数  设置的越大、网络性能越优，但也容易过拟合，参数量过大
    #轻量化设计 64   提升性能 128或者256
    fpn_out_channels = 64

    # head
    # 类别数量
    class_num = 1
    # 默认即可
    use_GN_head = True
    # 默认即可
    prior = 0.01
    
    # 是否添加中心度分支
    add_centerness = True
    # 中心度分支与回归分支并行？（否则与分类分支并行）
    cnt_on_reg = True

    # training
    #检测层的下采样次数
    strides = [4,8]
    #每个检测层中目标大小的回归限制    低层预测点目标、较高层预测斑状目标
    limit_range = [[1, 16],[16, 48]]

    # inference
    # 置信度得分阈值
    score_threshold = 0.5
    # NMS中IOU阈值
    nms_iou_threshold = 0.1
    # 最大候选数量
    max_detection_boxes_num = 100

    # FCOS中正样本的选择：目标中心的某个邻域大小
    sample_radiu_ratio = 1.0
    lambda_ctn = 1.0