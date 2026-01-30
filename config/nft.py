"""
DiffusionNFT NFT 训练配置

此文件定义了基于 Stable Diffusion 3 的 NFT（Neural Flow Transport）训练配置。
包含多种预设配置函数，支持不同的奖励函数组合：
- sd3_ocr: OCR 文字渲染奖励
- sd3_geneval: GenEval 组合生成奖励  
- sd3_pickscore: PickScore 人类偏好奖励
- sd3_hpsv2: HPSv2 人类偏好奖励
- sd3_multi_reward: 多奖励组合（GDPO）

使用方法:
    python train.py --config=config/nft.py:sd3_pickscore
"""

import imp
import os

# 加载基础配置模块
# 注意: imp 模块已被废弃，但在此保留以兼容性
base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def get_config(name):
    """
    根据配置名称获取对应的配置对象。
    
    参数:
        name: 配置函数名称，如 "sd3_pickscore"
        
    返回:
        config: 对应的配置对象
    """
    return globals()[name]()


def _get_config(base_model="sd3", n_gpus=1, gradient_step_per_epoch=1, dataset="pickscore", reward_fn={}, name=""):
    """
    内部配置生成函数，用于创建 NFT 训练配置。
    
    参数:
        base_model: 基础模型类型，目前仅支持 "sd3" (Stable Diffusion 3)
        n_gpus: 使用的 GPU 数量，用于自动计算批量大小
        gradient_step_per_epoch: 每个 epoch 的梯度更新次数
        dataset: 数据集名称，可选 "pickscore", "ocr", "geneval"
        reward_fn: 奖励函数配置字典，格式 {name: weight}
        name: 配置名称，用于日志和保存路径
        
    返回:
        config: 完整的训练配置对象
    """
    # 获取基础配置
    config = base.get_config()
    
    # 参数验证
    assert base_model in ["sd3"], f"不支持的基础模型: {base_model}，目前仅支持 sd3"
    assert dataset in ["pickscore", "ocr", "geneval"], f"不支持的数据集: {dataset}"

    # 设置基础模型类型和数据集路径
    config.base_model = base_model
    config.dataset = os.path.join(os.getcwd(), f"dataset/{dataset}")
    
    # SD3 特定配置
    if base_model == "sd3":
        # HuggingFace 模型仓库路径
        config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
        
        # 训练时使用较少的步数以加速采样
        config.sample.num_steps = 20
        
        # 评估时使用更多步数以获得更好的质量
        config.sample.eval_num_steps = 40
        
        # 无分类器引导强度
        config.sample.guidance_scale = 4.5
        
        # 训练图像分辨率，默认是512
        config.resolution = 512
        
        # KL 散度正则化系数
        config.train.beta = 0.0001
        
        # Flow Matching 噪声水平（0.7 降低训练难度）
        config.sample.noise_level = 0.7
        
        # 初始批量大小（后续会自动调整）
        bsz = 8

    # GRPO 算法需要多张图像来估计同一 prompt 的策略梯度
    config.sample.num_image_per_prompt = 8
    
    # 每个 epoch 的 prompt 组数
    # 总样本数 = num_groups * num_image_per_prompt
    num_groups = 48

    # 自动计算合适的批量大小
    # 需要满足以下条件：
    # 1. 总样本数能被 (GPU数 * batch_size) 整除
    # 2. (batch_size * GPU数) 能被 num_image_per_prompt 整除
    # 3. 每 epoch 的批次数能被梯度更新次数整除
    while True:
        if bsz < 1:
            assert False, "无法找到合适的批量大小，请调整 n_gpus 或 num_groups"
        if (
            num_groups * config.sample.num_image_per_prompt % (n_gpus * bsz) == 0
            and bsz * n_gpus % config.sample.num_image_per_prompt == 0
        ):
            n_batch_per_epoch = num_groups * config.sample.num_image_per_prompt // (n_gpus * bsz)
            if n_batch_per_epoch % gradient_step_per_epoch == 0:
                config.sample.train_batch_size = bsz
                config.sample.num_batches_per_epoch = n_batch_per_epoch
                config.train.batch_size = config.sample.train_batch_size
                config.train.gradient_accumulation_steps = (
                    config.sample.num_batches_per_epoch // gradient_step_per_epoch
                )
                break
        bsz -= 1

    # 测试批量大小设置
    # 数据集大小: ocr=1018, geneval=2212, pickscore=2048
    # 设置合适的 batch size 使得样本数尽量能被 GPU 数整除
    # 避免多卡训练时填充最后一个不完整的 batch
    config.sample.test_batch_size = 14 if dataset == "geneval" else 16
    if n_gpus > 32:
        # 大规模分布式时减小测试批量大小以节省显存
        config.sample.test_batch_size = config.sample.test_batch_size // 2

    # 根据数据集选择 prompt 加载函数
    # geneval: 使用 GenevalPromptDataset（包含元数据）
    # 其他: 使用 TextPromptDataset（纯文本）
    config.prompt_fn = "geneval" if dataset == "geneval" else "general_ocr"

    # 运行名称和保存目录
    config.run_name = f"nft_{base_model}_{name}"
    config.save_dir = f"logs/nft/{base_model}/{name}"
    
    # 设置奖励函数配置
    # GDPO 会对字典中的每个奖励独立归一化后加权求和
    config.reward_fn = reward_fn

    # NFT 算法特有参数
    # decay_type: 控制 old_model 更新策略
    # 0: 不更新 (flat=0, uprate=0, uphold=0)
    # 1: 缓慢更新 (flat=0, uprate=0.001, uphold=0.5)
    # 2: 延迟更新 (flat=75, uprate=0.0075, uphold=0.999)
    config.decay_type = 1
    
    #! beta 参数: 控制正负样本预测的混合比例
    # positive_prediction = beta * forward + (1-beta) * old
    # implicit_negative = (1+beta) * old - beta * forward
    config.beta = 1.0
    
    # 优势处理模式
    # "all": 使用所有优势值
    # "positive_only": 只使用正优势
    # "negative_only": 只使用负优势
    # "binary": 二值化优势 (+1/-1)
    config.train.adv_mode = "all"

    # 训练时的采样设置
    # 不使用 CFG（guidance_scale=1.0）以加速采样
    config.sample.guidance_scale = 1.0
    
    # 使用确定性采样（同一噪声得到相同结果）
    config.sample.deterministic = True
    
    # 使用 DPM2 求解器
    config.sample.solver = "dpm2"
    
    return config


##########################################
#          预设配置函数
##########################################

def sd3_ocr():
    """
    OCR 文字渲染优化配置
    
    使用 OCR 分数作为奖励，训练模型生成更清晰准确的文字。
    适用于需要在图像中渲染文字的场景。
    """
    reward_fn = {
        "ocr": 1.0,  # OCR 奖励权重
    }
    config = _get_config(
        base_model="sd3", 
        n_gpus=8, 
        gradient_step_per_epoch=2, 
        dataset="ocr", 
        reward_fn=reward_fn, 
        name="ocr"
    )
    
    # OCR 任务使用较大的 beta 和延迟衰减
    config.beta = 0.1
    config.decay_type = 2
    
    return config


def sd3_geneval():
    """
    GenEval 组合生成优化配置
    
    使用 GenEval 评估作为奖励，训练模型提升组合生成能力。
    评估维度包括：颜色、形状、位置、数量等属性的准确性。
    """
    reward_fn = {
        "geneval": 1.0,  # GenEval 奖励权重
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="geneval",
        reward_fn=reward_fn,
        name="geneval",
    )
    return config


def sd3_pickscore():
    """
    PickScore 人类偏好优化配置
    
    使用 PickScore 作为奖励，训练模型生成更符合人类偏好的图像。
    PickScore 基于人类偏好数据训练，能评估图像的美观程度和文本对齐度。
    """
    reward_fn = {
        "pickscore": 1.0,  # PickScore 奖励权重
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn=reward_fn,
        name="pickscore",
    )
    return config


def sd3_hpsv2():
    """
    HPSv2（Human Preference Score v2）优化配置
    
    使用 HPSv2 作为奖励，这是另一个人类偏好评分模型。
    与 PickScore 类似，但使用不同的训练数据和模型架构。
    """
    reward_fn = {
        "hpsv2": 1.0,  # HPSv2 奖励权重
    }
    config = _get_config(
        base_model="sd3", 
        n_gpus=8, 
        gradient_step_per_epoch=1, 
        dataset="pickscore", 
        reward_fn=reward_fn, 
        name="hpsv2"
    )
    return config


def sd3_multi_reward():
    """
    多奖励联合优化配置（GDPO）
    
    使用 GDPO（分组解耦策略优化）同时优化多个奖励目标。
    每个奖励会先在 batch 内独立归一化，然后按权重加权求和。
    
    当前配置使用四个奖励：
    - pickscore: 人类偏好评分
    - hpsv2: 另一个人类偏好评分  
    - clipscore: CLIP 文本-图像对齐分数
    - ocr: OCR 文字渲染准确度
    
    权重都设为 1.0，意味着四个奖励同等重要。
    可以调整权重来侧重某些目标。
    """
    reward_fn = {
        "pickscore": 1.0,   # PickScore 权重
        "hpsv2": 1.0,       # HPSv2 权重
        "clipscore": 1.0,   # CLIP Score 权重
        "ocr": 1.0,         # OCR 文字渲染奖励权重
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=1,
        gradient_step_per_epoch=1,
        dataset="ocr",  # 使用 OCR 数据集
        reward_fn=reward_fn,
        name="multi_reward",
    )
    
    #! 多奖励时使用更多采样步数以确保质量
    config.sample.num_steps = 25
    
    #! 使用较小的 beta 值以平衡多个奖励，越小信号强度越高
    config.beta = 0.1
    
    return config
