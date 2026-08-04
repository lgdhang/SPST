import torch
import glob
import os
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm # 用于显示进度
from torchvision.utils import save_image



# 归一化参数定义
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 定义 VGG 特征层名称
CONTENT_LAYERS = ['relu4_1', 'relu5_1']
STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1']
# STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

# 学习率
LEARNING_RATE = 1e-4
# 批量大小
BATCH_SIZE = 1          # 8 ！！！！！！！！！！！！
# 总迭代次数 (例如 100,000 次)
TOTAL_ITERATIONS = 1000
# 特征图尺寸 (例如 VGG relu4_1 对应的 28x28 或 64x64)
FEAT_SIZE = 28 
# DINO Token 数量
DINO_N_TOKENS = 256 
# DINO 维度
DINO_DIM = 384
# VGG 维度 (Bottleneck)
VGG_DIM = 512

# 保存路径
OUTPUT_DIR = 'ablation\\base_dgsa1'
CHECKPOINT_INTERVAL = 10000
IMAGE_SAVE_INTERVAL = 500

ALPHA_C = 1.0
ALPHA_S = 5.0
ALPHA_DINO = 0.1

def calc_mean_std(feat: torch.Tensor):
    """
    计算特征张量在通道维度上的均值和标准差。
    
    Args:
        feat: 输入特征张量 [B, C, H, W]
        
    Returns:
        mean: [B, C]
        std: [B, C]
    """
    size = feat.size()
    assert (len(size) == 4)
    B, C = size[:2]
    # 计算空间维度上的元素数量
    spatial_size = size[2] * size[3]
    
    # 将特征重塑为 [B, C, H*W]
    feat_view = feat.view(B, C, -1)
    
    # 沿着空间维度 (dim=2) 计算均值
    mean = feat_view.mean(dim=2) # [B, C]
    
    # 沿着空间维度 (dim=2) 计算标准差 (std)
    # 默认 torch.std 采用 Bessel's correction (ddof=1)
    std = feat_view.std(dim=2) # [B, C]
    
    # 为防止出现除以零或数值不稳定，可以添加一个极小的 epsilon
    # std = std + 1e-6 
    
    return mean, std

# decoder 模块：将风格化后的特征图还原为图像
decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'), # 上采样
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'), # 上采样
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'), # 上采样
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)), # 最终输出3通道图像
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

def get_common_transform(size=512, crop_size=224):
    """
    获取 VGG 和 DINO 网络通用的图像预处理流水线。
    
    Args:
        size: 图像将被缩放和裁剪到的最终尺寸。
    """
    common_transforms = transforms.Compose([
        # 1. 缩放和裁剪 (确保统一输入尺寸)
        # 常见做法是先缩放短边，再随机裁剪，这里简化为 Resize + RandomCrop
        transforms.Resize(size),
        #transforms.RandomCrop(crop_size), # 训练时可用 RandomCrop，这里用 CenterCrop 简化
        
        # 2. 转换为张量
        transforms.ToTensor(),
        
        # 3. 归一化
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return common_transforms

class StyleContentDataset(Dataset):
    def __init__(self, content_dir, style_dir, transform=None):
        self.content_paths = sorted(glob.glob(os.path.join(content_dir, '*')))
        self.style_paths = sorted(glob.glob(os.path.join(style_dir, '*')))
        self.transform = transform
        
        # 确保数据不为空
        if len(self.content_paths) == 0 or len(self.style_paths) == 0:
            raise ValueError(f"Data not found. Content: {len(self.content_paths)}, Style: {len(self.style_paths)}")
        
        # 数据集长度定义为较长的那个，保证一个epoch能遍历完主要数据
        self.len_content = len(self.content_paths)
        self.len_style = len(self.style_paths)
        self.length = max(self.len_content, self.len_style)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 内容图片：顺序读取 (如果越界则取模)
        c_idx = idx % self.len_content
        c_path = self.content_paths[c_idx]
        
        # 风格图片：随机读取 (或者也取模，这里为了风格多样性选择随机)
        # s_idx = idx % self.len_style 
        s_idx = random.randint(0, self.len_style - 1)
        s_path = self.style_paths[s_idx]

        try:
            c_img = Image.open(c_path).convert('RGB')
            s_img = Image.open(s_path).convert('RGB')
            
            if self.transform:
                c_img = self.transform(c_img)
                s_img = self.transform(s_img)
                
            return c_img, s_img
        except Exception as e:
            print(f"Error loading image: {c_path} or {s_path}, {e}")
            # 遇到错误随机返回一张替代
            return self.__getitem__(random.randint(0, self.length - 1))

class SkipVGGDecoder(nn.Module):
    """
    风格迁移解码器，将 [B, 512, 28, 28] 的瓶颈特征解码为 [B, 3, 224, 224] 的图像。
    结构基于 VGG 风格的 Conv + ReflectionPad + Upsample 序列。
    """
    def __init__(self):
        super(SkipVGGDecoder, self).__init__()
        # 使用 nn.ReflectionPad2d 作为成员，在 forward 中重复使用
        self.pad = nn.ReflectionPad2d((1, 1, 1, 1)) 
        
        # 1. Block 1 (Input 512 -> 256, Upsample 1)
        self.conv1 = nn.Conv2d(512, 256, (3, 3))
        self.relu1 = nn.ReLU() 
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest') # 28x28 -> 56x56
        
        # --- Skip Connection Point 1 (relu3-1, 256ch) ---
        # Decoded(256ch) + Skip(256ch) = 512ch
        self.conv2 = nn.Conv2d(256 + 256, 256, (3, 3)) # **输入通道数修改: 512**
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(256, 256, (3, 3))
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256, 256, (3, 3))
        self.relu4 = nn.ReLU()
        
        # 2. Block 2 (256 -> 128, Upsample 2)
        self.conv5 = nn.Conv2d(256, 128, (3, 3))
        self.relu5 = nn.ReLU()
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest') # 56x56 -> 112x112
        
        # --- Skip Connection Point 2 (relu2-1, 128ch) --- 
        # Decoded(128ch) + Skip(128ch) = 256ch
        self.conv6 = nn.Conv2d(128 + 128, 128, (3, 3)) # **输入通道数修改: 256**
        self.relu6 = nn.ReLU()
        
        # 3. Block 3 (128 -> 64, Upsample 3)
        self.conv7 = nn.Conv2d(128, 64, (3, 3))
        self.relu7 = nn.ReLU()
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest') # 112x112 -> 224x224
        
        # --- Skip Connection Point 3 (relu1-1, 64ch) --- 
        # Decoded(64ch) + Skip(64ch) = 128ch
        self.conv8 = nn.Conv2d(64 + 64, 64, (3, 3)) # **输入通道数修改: 128**
        self.relu8 = nn.ReLU()
        
        # Final Block
        self.conv9 = nn.Conv2d(64, 3, (3, 3))

    def forward(self, x, content_feats):
        # content_feats[2]: relu3-1 (256ch)
        # content_feats[1]: relu2-1 (128ch)
        # content_feats[0]: relu1-1 (64ch)

        # Block 1 (512 -> 256, Upsample 1)
        x = self.pad(x)
        x = self.relu1(self.conv1(x)) # 修复错误，使用 self.relu1
        x = self.up1(x) 
        
        # --- Skip 1: relu3-1 (256ch) ---
        x = torch.cat([x, content_feats['relu3_1']], dim=1) # x: 512ch
        
        x = self.pad(x)
        x = self.relu2(self.conv2(x)) 
        x = self.pad(x)
        x = self.relu3(self.conv3(x))
        x = self.pad(x)
        x = self.relu4(self.conv4(x))
        
        # Block 2 (256 -> 128, Upsample 2)
        x = self.pad(x)
        x = self.relu5(self.conv5(x))
        x = self.up2(x) 
        
        # --- Skip 2: relu2-1 (128ch) ---
        x = torch.cat([x, content_feats['relu2_1']], dim=1) 
        
        x = self.pad(x)
        x = self.relu6(self.conv6(x))
        
        # Block 3 (128 -> 64, Upsample 3)
        x = self.pad(x)
        x = self.relu7(self.conv7(x))
        x = self.up3(x) 
        
        # --- Skip 3: relu1-1 (64ch) ---
        x = torch.cat([x, content_feats['relu1_1']], dim=1) 
        
        # Final Block (128 -> 64 -> 3)
        x = self.pad(x)
        x = self.relu8(self.conv8(x))
        
        x = self.pad(x)
        x = self.conv9(x)
        
        return x

class VGGDecoder(nn.Module):
    """
    风格迁移解码器，将 [B, 512, 28, 28] 的瓶颈特征解码为 [B, 3, 224, 224] 的图像。
    结构基于 VGG 风格的 Conv + ReflectionPad + Upsample 序列。
    """
    def __init__(self, pretrain="decoder_init.pth"):
        super(VGGDecoder, self).__init__()
        
        self.decoder = nn.Sequential(
            # --- Block 1 (Input 512, Upsample to 56x56) ---
            # Input: [B, 512, 28, 28]
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='nearest'), # 28x28 -> 56x56
            
            # Conv Layers after first upsample
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(inplace=True),
            
            # --- Block 2 (Output 128, Upsample to 112x112) ---
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='nearest'), # 56x56 -> 112x112
            
            # Conv Layers after second upsample
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(inplace=True),
            
            # --- Block 3 (Output 3, Upsample to 224x224) ---
            nn.Upsample(scale_factor=2, mode='nearest'), # 112x112 -> 224x224
            
            # Final Conv Layers
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)), # Output: [B, 3, 224, 224]
        )
        if pretrain is not None:
            self.decoder.load_state_dict(torch.load(pretrain))
            print("Load pretrain decoder.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 瓶颈特征 [B, 512, 28, 28]
        
        Returns:
            图像张量 [B, 3, 224, 224]
        """
        return self.decoder(x)

class VGG19Encoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.layer_ids = {
            "relu1_1": 3,           # torch.Size([1, 64, 224, 224])
            "relu2_1": 10,          # torch.Size([1, 128, 112, 112])
            "relu3_1": 17,          # torch.Size([1, 256, 56, 56])
            "relu4_1": 30,          # torch.Size([1, 512, 28, 28])
            "relu5_1": 43           # torch.Size([1, 512, 14, 14])
        }
    
    def forward(self, x):
        feats = {}
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in self.layer_ids.values():
                for name, idx in self.layer_ids.items():
                    if idx == i:
                        feats[name] = x.clone()
        
        return feats

class DINOv2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.dinov2 = torch.hub.load(
                'facebookresearch/dinov2', 
                'dinov2_vits14',
            )
            print(f"成功从本地缓存加载 DINOv2 结构: dinov2_vits14")
        except Exception as e:
            # 如果本地缓存文件损坏或结构无法实例化，可能会抛出异常
            print(f"错误: 无法从本地缓存加载 DINOv2 结构")
            print(f"原始错误信息: {e}")
            self.dinov2 = None
            raise RuntimeError("模型结构加载失败。")
    
    def forward(self, x):
        out = self.dinov2.forward_features(x)["x_norm_patchtokens"] # torch.Size([1, 256, 384])
        return out

class DinoGuidedStyleAttention(nn.Module):
    """
    DINO-Guided Style Attention (DGSA) Module.
    
    论文创新模块：利用 DINO 特征作为语义锚点，将风格特征精确映射到内容结构上，
    并注入到 VGG 特征流中。
    """
    def __init__(self, dino_dim=384, vgg_dim=512, num_heads=8):
        super(DinoGuidedStyleAttention, self).__init__()
        self.dino_dim = dino_dim
        self.vgg_dim = vgg_dim
        self.num_heads = num_heads
        self.scale = (dino_dim // num_heads) ** -0.5

        # --- 1. 注意力投影层 ---
        # 我们保持 DINO 维度进行注意力计算，以保留预训练的语义能力
        self.q_proj = nn.Linear(dino_dim, dino_dim)
        self.k_proj = nn.Linear(dino_dim, dino_dim)
        self.v_proj = nn.Linear(dino_dim, dino_dim)

        # --- 2. 特征融合/适配层 ---
        # 将对齐后的 DINO 特征 (384) 投影到 VGG 维度 (512)
        self.output_proj = nn.Sequential(
            nn.Conv2d(dino_dim, vgg_dim, kernel_size=1),
            nn.InstanceNorm2d(vgg_dim), # 风格迁移常用 IN
            nn.ReLU(inplace=True)
        )

        # --- 3. 最终融合卷积 (残差融合) ---
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(vgg_dim * 2, vgg_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(vgg_dim),
            nn.ReLU(inplace=True)
        )
    
    def _reshape_tokens_to_map(self, tokens: torch.Tensor) -> torch.Tensor:
        """辅助函数: [B, N, C] -> [B, C, H, W]"""
        B, N, C = tokens.shape
        H = W = int(N ** 0.5)
        # Transpose to [B, C, N] then reshape
        return tokens.transpose(1, 2).reshape(B, C, H, W)
    
    def forward(
            self, 
            content_vgg: torch.Tensor,
            content_dino: torch.Tensor,
            style_dino: torch.Tensor
        ):
        """
        Args:
            content_vgg:  [B, 512, H_vgg, W_vgg] (e.g., 28x28)
            content_dino: [B, N, 384] (Content Structure, 已移除 CLS token)
            style_dino:   [B, N, 384] (Style Texture, 已移除 CLS token)
            
        Returns:
            stylized_features: [B, 512, H_vgg, W_vgg] - 注入了语义风格的特征
            attention_map:     [B, N_content, N_style] - 用于可视化的注意力矩阵
        """
        B, N_c, D = content_dino.shape
        _, N_s, _ = style_dino.shape

        # --- Step 1: 语义对齐 (Cross-Attention) ---
        # Query 来自 Content (我长什么样？)
        # Key/Value 来自 Style (你有什么风格？)
        q = self.q_proj(content_dino).view(B, N_c, self.num_heads, D // self.num_heads).transpose(1, 2)
        k = self.k_proj(style_dino).view(B, N_s, self.num_heads, D // self.num_heads).transpose(1, 2)
        v = self.v_proj(style_dino).view(B, N_s, self.num_heads, D // self.num_heads).transpose(1, 2)

        # 计算注意力分数: (Q @ K.T)
        # Shape: [B, Heads, N_c, N_s]
        attn_score = (q @ k.transpose(-2, -1)) * self.scale
        attn_map = attn_score.softmax(dim=-1) # 获取注意力权重

        # 风格重组 (Warping): (Attn @ V)
        # 这一步将风格图的特征 "移动" 到了内容图对应的位置
        aligned_style = (attn_map @ v).transpose(1, 2).reshape(B, N_c, D)

        # --- Step 2: 空间恢复与上采样 ---
        # 将 Token 序列变回 2D 特征图
        # Shape: [B, 384, 16, 16] (假设 DINO 是 16x16 patch)
        aligned_style_map = self._reshape_tokens_to_map(aligned_style)

        # 上采样到 VGG 特征图大小 (e.g., 28x28)
        # 使用 bilinear 插值以获得平滑的梯度
        aligned_style_upsampled = F.interpolate(
            aligned_style_map, 
            size=content_vgg.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )

        # --- Step 3: 特征注入 (Feature Injection) ---
        # 投影到 VGG 通道数: [B, 384, 28, 28] -> [B, 512, 28, 28]
        projected_style = self.output_proj(aligned_style_upsampled)

        # 融合方式：级联 (Concatenation) + 卷积
        # 相比直接相加，Concat 允许网络学习如何保留原始内容和注入新风格的最佳比例
        combined = torch.cat([content_vgg, projected_style], dim=1) # [B, 1024, 28, 28]
        stylized_features = self.fusion_conv(combined) + content_vgg # [B, 512, 28, 28]

        # 论文可视化点：返回平均后的 Attention Map (对 Head 维度取平均)
        # [B, N_c, N_s]
        vis_attn_map = attn_map.mean(dim=1)

        return stylized_features, vis_attn_map

class DinoStructuralLoss(nn.Module):
    """
    基于 DINOv2 特征的结构损失。
    计算内容图像和生成图像的 DINO Patch Tokens 之间的 MSE 距离。
    """
    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.mse_loss = nn.MSELoss()
        print(f"DinoStructuralLoss: 激活，权重为 {loss_weight}")

    def forward(self, 
                generated_dino_tokens: torch.Tensor, 
                content_dino_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            generated_dino_tokens: 生成图像的 DINO 特征 [B, N, D]
            content_dino_tokens: 内容图像的 DINO 特征 [B, N, D]
            
        Returns:
            l_dino_struct: DINO 结构损失 (标量)
        """
        
        # 1. 确保特征不包含 CLS Token (如果包含，则需要切片移除)
        # 假设 DINO Encoder 返回的已经是 Patch Tokens [B, N, D]
        # 如果 DINO Encoder 返回 [B, N+1, D]，则需要使用 generated_dino_tokens[:, 1:, :]
        
        # 2. 计算结构损失 (MSE Loss)
        # 损失鼓励生成的图像特征在语义空间上与内容图像特征完全匹配。
        l_dino_struct = self.mse_loss(generated_dino_tokens, content_dino_tokens)
        
        # 3. 应用权重 (在返回给总损失函数时应用)
        return self.loss_weight * l_dino_struct

class LossNetwork(nn.Module):
    """
    用于风格迁移的损失网络。
    计算基于特征 Instance Normalization 的内容损失，和基于均值/方差的风格损失。
    """
    def __init__(self, content_layers: list = CONTENT_LAYERS, style_layers: list = STYLE_LAYERS):
        super().__init__()
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.dino_structural_loss = DinoStructuralLoss(loss_weight=1.0)
        print("LossNetwork: 损失计算初始化完成。")

    def forward(
            self, 
            generated_vgg_features: torch.Tensor, 
            content_vgg_features: torch.Tensor, 
            style_vgg_features: torch.Tensor,
            # 新增 DINO 特征输入
            generated_dino_tokens: torch.Tensor,
            content_dino_tokens: torch.Tensor
        ):
        
        # --- 1. 内容损失 (Content Loss) ---
        L_content = torch.tensor(0.0).to(generated_vgg_features['relu4_1'].device)
        
        for layer_name in self.content_layers:
            feat_g = generated_vgg_features[layer_name]
            feat_c = content_vgg_features[layer_name]
            
            # 1.1. (关键要求) 对 Content 特征和 Generated 特征进行 Instance Normalization
            # 采用自身的均值和方差进行归一化
            mean_g, std_g = calc_mean_std(feat_g)
            mean_c, std_c = calc_mean_std(feat_c)
            
            # 将 mean/std 扩展回 [B, C, 1, 1] 方便广播
            mean_g = mean_g.view(mean_g.size(0), mean_g.size(1), 1, 1)
            std_g = std_g.view(std_g.size(0), std_g.size(1), 1, 1) + 1e-6
            mean_c = mean_c.view(mean_c.size(0), mean_c.size(1), 1, 1)
            std_c = std_c.view(std_c.size(0), std_c.size(1), 1, 1) + 1e-6

            # 执行 IN: (feat - mean) / std
            normalized_g = (feat_g - mean_g) / std_g
            normalized_c = (feat_c - mean_c) / std_c
            
            # 1.2. 计算 MSE Loss
            L_content += F.mse_loss(normalized_g, normalized_c)
            
        # 1.3. 平均内容损失 (除以层数)
        L_content /= len(self.content_layers)


        # --- 2. 风格损失 (Style Loss) ---
        L_style = torch.tensor(0.0).to(generated_vgg_features['relu4_1'].device)
        
        for layer_name in self.style_layers:
            feat_g = generated_vgg_features[layer_name]
            feat_s = style_vgg_features[layer_name]

            # 2.1. 计算 Generated 图像特征的均值和标准差
            mean_g, std_g = calc_mean_std(feat_g) # [B, C]
            
            # 2.2. 计算 Style 图像特征的均值和标准差
            mean_s, std_s = calc_mean_std(feat_s) # [B, C]
            
            # 2.3. 计算均值和标准差的 MSE 损失
            L_style += F.mse_loss(mean_g, mean_s)
            L_style += F.mse_loss(std_g, std_s)
        
        # 2.4. 平均风格损失 (除以层数和度量次数 (2))
        L_style /= len(self.style_layers) * 2

        # --- 3. DINO 结构损失 ---
        L_dino_struct = self.dino_structural_loss(generated_dino_tokens, content_dino_tokens)

        return L_content, L_style, L_dino_struct

def train_style_transfer(
    content_images,
    style_images, 
    encoder_vgg: VGG19Encoder, 
    encoder_dino: DINOv2Encoder, 
    dgsa_module: DinoGuidedStyleAttention, 
    decoder: VGGDecoder, 
    loss_network: LossNetwork,
    device: torch.device,
    save_name
):
    # 1. 设置优化器
    # 仅优化 Decorder 和 DGSA 模块
    trainable_params = list(decoder.parameters()) + list(dgsa_module.parameters())
    optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE)

    # 2. 冻结编码器参数
    encoder_vgg.eval()
    encoder_dino.eval()
    for param in encoder_vgg.parameters():
        param.requires_grad = False
    for param in encoder_dino.parameters():
        param.requires_grad = False
    
    # 3. 设置训练模式
    decoder.train()
    dgsa_module.train()

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    # A. 提取 VGG 特征 (Content & Style Loss)
    with torch.no_grad():
        content_vgg_features = encoder_vgg(content_images)
        style_vgg_features = encoder_vgg(style_images)

    # B. 提取 DINO 特征 (Style Attention Guide)
    # 注意: DINOV2Encoder 应该只返回 Patch Tokens (256个)，或者在 DGSA 内部处理 CLS Token
    with torch.no_grad():
        content_dino_output = encoder_dino(content_images)
        style_dino_output = encoder_dino(style_images)

    # 使用 tqdm 跟踪进度
    pbar = tqdm(range(1, TOTAL_ITERATIONS + 1), desc="Training Style Transfer")

    for iteration in pbar:

        optimizer.zero_grad()
        
        # C. 瓶颈层：DINO 引导的风格注意力融合
        # 假设 DGSA 接收 VGG relu4_1 特征进行注入
        bottleneck_vgg_feat = content_vgg_features['relu4_1']

        fused_bottleneck_feat, _ = dgsa_module(
            content_vgg = bottleneck_vgg_feat,
            content_dino = content_dino_output,
            style_dino = style_dino_output
        )

        # D. 解码
        generated_images = decoder(fused_bottleneck_feat, content_vgg_features)

        # E. 计算损失
        # 提取生成图像的 VGG 特征
        generated_vgg_features = encoder_vgg(generated_images)

        # E. 提取生成图像的 DINO 特征 (用于结构损失)
        with torch.no_grad():
             generated_dino_output = encoder_dino(generated_images) # 新增行
        
        # 损失计算
        content_loss, style_loss, dino_struct_loss = loss_network(
            generated_vgg_features, 
            content_vgg_features, 
            style_vgg_features,
            # 新增参数
            generated_dino_output,
            content_dino_output
        )

        # 最终损失 (需要根据经验设置权重，这里简化为 1.0 和 10.0)
        alpha_c = ALPHA_C
        alpha_s = ALPHA_S
        alpha_dino = ALPHA_DINO
        total_loss = alpha_c * content_loss + alpha_s * style_loss + alpha_dino * dino_struct_loss

        # --- 反向传播 ---
        total_loss.backward()
        optimizer.step()

        # --- 进度和保存 ---
        # 更新进度条显示
        pbar.set_postfix({
            'L_total': f'{total_loss.item():.3f}', 
            'L_c': f'{content_loss.item():.2f}', 
            'L_s': f'{style_loss.item():.2f}',
            'L_dino': f'{dino_struct_loss.item():.2f}'
        })

        # 保存中间结果图像
        if iteration % IMAGE_SAVE_INTERVAL == 0:
            # 保存原图和风格化图像 (移除归一化以便查看)
            def denormalize(img):
                mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(device)
                std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(device)
                img = img * std + mean
                return torch.clamp(img, 0, 1)

            save_path = os.path.join(OUTPUT_DIR, str(iteration) + save_name)
            # 拼接 Content, Style, Generated 图像
            display_img = torch.cat([
                denormalize(content_images[0]), 
                denormalize(style_images[0]), 
                denormalize(generated_images[0])
            ], dim=2) # 沿宽度拼接
            
            save_image(display_img, save_path)

        # 保存模型检查点
        if iteration % CHECKPOINT_INTERVAL == 0:
            torch.save({
                'dgsa': dgsa_module.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': iteration,
            }, os.path.join(OUTPUT_DIR, f'ckpt_{iteration:05d}.pth'))


if __name__ == "__main__":

    # 0. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 实例化所有模块
    print("Initialise VGG.")
    weights_file = "vgg_normalised.pth"
    vgg.load_state_dict(torch.load(weights_file))
    vgg_encoder = VGG19Encoder(vgg).to(device)
    
    # dino 特征
    print("Initialise DINOv2.")
    dinov2_encoder = DINOv2Encoder().to(device)

    print("Initialise dgsa, decoder and loss net.")
    dgsa_module = DinoGuidedStyleAttention(
        dino_dim=DINO_DIM,
        vgg_dim=VGG_DIM,
    ).to(device)
    decoder_module = SkipVGGDecoder().to(device)
    loss_module = LossNetwork().to(device)

    # dataset
    content_dir = "ablation\\contents1"                                #!!!!!!!!!!
    style_dir = "ablation\\styles1"                                    #!!!!!!!!!!                     
    tranform = get_common_transform(size=224)

    content_list = sorted(os.listdir(content_dir))
    style_list = sorted(os.listdir(style_dir))

    i = 4   # 0-7
    content = content_list[4]   
    style = style_list[1]       

    content_path = content_dir + "\\" + content
    c_img = Image.open(content_path).convert('RGB')
    content_t = tranform(c_img).unsqueeze(0).to(device)

    style_path = style_dir + "\\" + style
    s_img = Image.open(style_path).convert('RGB')
    style_t = tranform(s_img).unsqueeze(0).to(device)

    print(f"conent/style: {content}/{style}") 

    def generate_output_filename(file1: str, file2: str) -> str:

        # 方法1: 使用 os.path (推荐，兼容性最好)
        name1 = os.path.splitext(os.path.basename(file1))[0]
        name2 = os.path.splitext(os.path.basename(file2))[0]
    
        return f"{name1}_{name2}.jpg"

    save_name = generate_output_filename(content, style)

    train_style_transfer(
        content_t,
        style_t, 
        vgg_encoder, 
        dinov2_encoder, 
        dgsa_module, 
        decoder_module, 
        loss_module,
        device,
        save_name,
    )
    print("\n--- 训练完成 ---")

