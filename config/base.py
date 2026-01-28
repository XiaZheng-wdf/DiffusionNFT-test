"""
DiffusionNFT 训练配置基类

此文件定义了训练的默认配置参数，包括：
- 通用设置：运行名称、随机种子、日志目录等
- 预训练模型配置
- 采样配置：推理步数、引导强度、批量大小等
- 训练配置：学习率、优化器参数、梯度累积等
- 奖励函数配置

所有具体的训练任务配置（如 nft.py）都应继承此基础配置并进行覆盖。
"""

import ml_collections


def get_config():
    """
    获取默认的训练配置。
    
    返回:
        config: ml_collections.ConfigDict 对象，包含所有训练参数
    """
    config = ml_collections.ConfigDict()

    ##########################################
    #              通用设置
    ##########################################
    
    # 运行名称，用于 wandb 日志记录和检查点保存
    # 如果为空，将根据当前时间自动生成
    config.run_name = ""
    
    # 调试模式开关
    # 开启时会跳过某些耗时操作（如保存检查点、评估）
    config.debug = False

    # 随机种子，用于实验可复现性
    # 注意：分布式训练时每个 GPU 的种子会自动偏移
    config.seed = 42
    
    # 顶层日志目录，检查点将保存在此目录下
    config.logdir = "logs"
    
    # 训练的总 epoch 数
    # 每个 epoch 包括：从模型采样 + 在采样数据上训练
    config.num_epochs = 100000
    
    # 保存检查点的频率（每隔多少个 epoch 保存一次）
    config.save_freq = 30
    
    # 评估的频率（每隔多少个 epoch 评估一次）
    config.eval_freq = 10
    
    # 混合精度训练设置
    # 可选值: "fp16"（半精度）, "bf16"（脑浮点）, "no"（全精度）
    # 半精度可以显著加速训练
    config.mixed_precision = "fp16"
    
    # 是否在 Ampere GPU 上启用 TF32
    # 可以加速矩阵运算，对精度影响很小
    config.allow_tf32 = True
    
    # 从检查点恢复训练
    # 可以是精确的检查点目录（如 checkpoint_50），
    # 或包含多个检查点的目录（将使用最新的）
    # 注意：use_lora 必须与保存检查点时的设置一致
    config.resume_from = ""
    
    # 是否使用 LoRA（低秩适应）进行微调
    # 开启后只训练少量参数，节省显存
    config.use_lora = True
    
    # 数据集路径
    config.dataset = ""
    
    # 生成图像的分辨率（正方形，如 512 表示 512x512）
    config.resolution = 768

    ##########################################
    #           预训练模型配置
    ##########################################
    config.pretrained = pretrained = ml_collections.ConfigDict()
    
    # 基础模型路径
    # 可以是本地目录路径，或 HuggingFace 模型仓库名称
    # 例如: "stabilityai/stable-diffusion-3.5-medium"
    pretrained.model = ""
    
    # 模型版本（用于 HuggingFace 模型）
    # 留空则使用默认版本
    pretrained.revision = ""

    ##########################################
    #             采样配置
    ##########################################
    config.sample = sample = ml_collections.ConfigDict()
    
    # 采样器的推理步数
    # 步数越多质量越高，但速度越慢
    sample.num_steps = 40
    
    # 评估时的推理步数
    # 通常与训练时相同或更多
    sample.eval_num_steps = 40
    
    # 无分类器引导（CFG）权重
    # 1.0 表示不使用引导；越大引导越强，图像越符合文本但多样性降低
    sample.guidance_scale = 4.5
    
    # 训练采样时的批量大小（每个 GPU）
    sample.train_batch_size = 1
    
    # 每个 prompt 生成的图像数量
    # GRPO 算法需要多张图像来估计策略梯度
    sample.num_image_per_prompt = 1
    
    # 测试/评估时的批量大小（每个 GPU）
    sample.test_batch_size = 1
    
    # 每个 epoch 的采样批次数
    # 每个 epoch 的总样本数 = num_batches_per_epoch * batch_size * num_gpus
    sample.num_batches_per_epoch = 2
    
    # 是否使用全局标准差来归一化奖励
    # True: 使用整个 batch 的 std
    # False: 按 prompt 分组计算 std（不再使用，GDPO 采用独立归一化）
    sample.global_std = True
    
    # 噪声水平（用于 Flow Matching）
    # 控制采样过程中的噪声强度
    sample.noise_level = 1.0

    ##########################################
    #             训练配置
    ##########################################
    config.train = train = ml_collections.ConfigDict()
    
    # 训练时的批量大小（每个 GPU）
    train.batch_size = 1
    
    # 学习率
    train.learning_rate = 3e-4
    
    # Adam 优化器 beta1 参数（一阶矩估计的指数衰减率）
    train.adam_beta1 = 0.9
    
    # Adam 优化器 beta2 参数（二阶矩估计的指数衰减率）
    train.adam_beta2 = 0.999
    
    # 权重衰减（L2 正则化系数）
    train.adam_weight_decay = 1e-4
    
    # Adam epsilon（数值稳定性常数）
    train.adam_epsilon = 1e-8
    
    # 梯度累积步数
    # 有效批量大小 = batch_size * num_gpus * gradient_accumulation_steps
    train.gradient_accumulation_steps = 1
    
    # 最大梯度范数（用于梯度裁剪，防止梯度爆炸）
    train.max_grad_norm = 1.0
    
    # 内部 epoch 数量
    # 每个外部 epoch 采样一次，然后在采样数据上训练 num_inner_epochs 次
    train.num_inner_epochs = 1
    
    # 优势（advantage）裁剪范围 [-adv_clip_max, adv_clip_max]
    # 防止极端奖励导致训练不稳定
    train.adv_clip_max = 5
    
    # 时间步训练比例（0.0 ~ 1.0）
    # 小于 1.0 时只训练部分时间步，可加速训练但降低梯度估计精度
    train.timestep_fraction = 0.99
    
    # KL 散度正则化系数（beta）
    # 控制新策略与参考策略的偏离程度
    train.beta = 0.0001
    
    # 预训练的 LoRA 权重路径（可选）
    # 用于在已有 LoRA 基础上继续训练
    train.lora_path = None
    
    # 是否使用 EMA（指数移动平均）来稳定训练
    train.ema = True

    ##########################################
    #          Prompt 函数配置
    ##########################################
    # Prompt 函数名称，参见 prompts.py 中的可用函数
    # 决定如何加载和处理训练数据
    config.prompt_fn = ""
    
    # 传递给 prompt 函数的额外参数
    config.prompt_fn_kwargs = {}

    ##########################################
    #          奖励函数配置
    ##########################################
    # 奖励函数配置字典
    # 格式: {reward_name: weight, ...}
    # 例如: {"pickscore": 1.0, "aesthetic": 0.5}
    # GDPO 会对每个奖励独立归一化，然后按权重加权求和
    config.reward_fn = ml_collections.ConfigDict()
    
    # 模型保存目录
    config.save_dir = ""

    ##########################################
    #        Per-Prompt 统计跟踪
    ##########################################
    # 是否按 prompt 跟踪奖励统计信息（均值和标准差）
    # 注意：GDPO 实现后此选项已废弃，优势计算改为 batch 内独立归一化
    config.per_prompt_stat_tracking = True

    return config
