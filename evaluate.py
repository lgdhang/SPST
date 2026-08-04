import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import argparse
from tqdm import tqdm
import torch.nn.functional as F

# ==========================================
# 1. 辅助函数：计算均值和标准差
# ==========================================
def calc_mean_std(feat, eps=1e-5):
    """
    计算特征图的通道均值和标准差
    Input: [B, C, H, W]
    Output: Mean [B, C, 1, 1], Std [B, C, 1, 1]
    """
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

# ==========================================
# 2. VGG Encoder (提取内容和风格特征)
# ==========================================
class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练的 VGG19
        vgg = models.vgg19(pretrained=True).features
        
        # 定义需要提取的层
        # relu1_1 (idx 1), relu2_1 (idx 6), relu3_1 (idx 11), relu4_1 (idx 20)
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        
        for x in range(2):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg[x])
            
        # 冻结参数
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h1 = self.slice1(x) # relu1_1
        h2 = self.slice2(h1) # relu2_1
        h3 = self.slice3(h2) # relu3_1
        h4 = self.slice4(h3) # relu4_1
        return h1, h2, h3, h4

# ==========================================
# 3. DINOv2 Encoder (提取结构/语义特征)
# ==========================================
class DINOEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载 DINOv2 Small (ViT-S/14)
        print("Loading DINOv2 (ViT-S/14)...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 提取特征 (包含 CLS token 和 Patch tokens)
        # return dict_keys(['x_norm_clstoken', 'x_norm_patchtokens'])
        features_dict = self.model.forward_features(x)
        # 我们将 CLS token 和 Patch tokens 拼接作为完整的特征表达
        # Patch tokens: [B, N, C], CLS: [B, 1, C] -> Concat: [B, N+1, C]
        return torch.cat((features_dict['x_norm_clstoken'].unsqueeze(1), 
                          features_dict['x_norm_patchtokens']), dim=1)


def create_stylized_filenames(pairs):
    """
    将内容文件名和风格文件名（去掉原始后缀）以下划线连接，并添加 .jpg 后缀。
    
    Args:
        pairs: 包含 (内容文件名, 风格文件名) 元组的列表。
        
    Returns:
        新的风格化结果文件名列表。
    """
    new_filenames = []
    
    for content_file, style_file in pairs:
        # 1. 提取不带扩展名的基础文件名
        # os.path.splitext('02.jpg')[0] -> '02'
        content_base = os.path.splitext(content_file)[0]
        style_base = os.path.splitext(style_file)[0]
        
        # 2. 连接并添加 '.jpg' 后缀
        new_name = f"{content_base}_{style_base}.png"            # change jpg and png
        new_filenames.append(new_name)
        
    return new_filenames

# ==========================================
# 4. 评估主程序
# ==========================================
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on {device}...")

    # --- 初始化模型 ---
    vgg_net = VGGEncoder().to(device)
    dino_net = DINOEncoder().to(device)
    
    mse_loss = nn.MSELoss()

    # --- 数据预处理 ---
    # VGG 和 DINO 都需要 ImageNet 标准化
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # 统一大小，便于 DINO 处理
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # --- 获取文件列表 ---
    # 假设 results 文件夹中的文件名与 contents 和 styles 对应
    sc_pairs_14 = [
        ["02.jpg", "07.png"],
        ["10.jpg", "06.png"],
        ["avril.jpg", "33.jpg"],
        ["cornell.jpg", "la_muse.jpg"],
        ["figures.jpg", "mondrian.jpg"],
        ["golden_gate.jpg", "rain_princess.jpg"],
        ["RoundBales.jpg", "16.png"],
        ["sailboat.jpg", "sketch.jpg"]
    ]

    valid_files_cs = create_stylized_filenames(sc_pairs_14)
    
    print(f"Found {len(valid_files_cs)} content images to evaluate.")

    # --- 累加器 ---
    total_content_loss = 0.0
    total_style_loss = 0.0
    total_dino_loss = 0.0
    count = 0

    for fname in tqdm(valid_files_cs):
        # 构建路径
        pc = sc_pairs_14[count][0]
        ps = sc_pairs_14[count][1]
        path_c = os.path.join(args.content_dir, pc)
        path_s = os.path.join(args.style_dir, ps)
        path_g = os.path.join(args.results_dir, fname) # Generated/Stylized

        if not (os.path.exists(path_c) and os.path.exists(path_s)):
            print(f"Warning: Missing content or style for {fname}, skipping.")
            continue

        # 加载图片
        try:
            img_c = transform(Image.open(path_c).convert('RGB')).unsqueeze(0).to(device)
            img_s = transform(Image.open(path_s).convert('RGB')).unsqueeze(0).to(device)
            img_g = transform(Image.open(path_g).convert('RGB')).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue

        with torch.no_grad():
            # === 1. VGG Feature Extraction ===
            # Content Features
            _, _, _, f_c_relu4_1 = vgg_net(img_c)
            
            # Style Features
            f_s_relu1_1, f_s_relu2_1, f_s_relu3_1, _ = vgg_net(img_s)
            
            # Generated Features
            f_g_relu1_1, f_g_relu2_1, f_g_relu3_1, f_g_relu4_1 = vgg_net(img_g)

            # === 2. Calculate Content Loss (VGG relu4_1) ===
            # Loss between Result and Content
            l_content = mse_loss(f_g_relu4_1, f_c_relu4_1)

            # === 3. Calculate Style Loss (VGG relu1_1, 2_1, 3_1 Stats) ===
            # Loss between Result and Style statistics
            l_style = 0.0
            # Layer 1_1
            gm1, gs1 = calc_mean_std(f_g_relu1_1)
            sm1, ss1 = calc_mean_std(f_s_relu1_1)
            l_style += mse_loss(gm1, sm1) + mse_loss(gs1, ss1)
            
            # Layer 2_1
            gm2, gs2 = calc_mean_std(f_g_relu2_1)
            sm2, ss2 = calc_mean_std(f_s_relu2_1)
            l_style += mse_loss(gm2, sm2) + mse_loss(gs2, ss2)
            
            # Layer 3_1
            gm3, gs3 = calc_mean_std(f_g_relu3_1)
            sm3, ss3 = calc_mean_std(f_s_relu3_1)
            l_style += mse_loss(gm3, sm3) + mse_loss(gs3, ss3)

            # === 4. Calculate Framework Loss (DINOv2) ===
            # Structure Loss between Result and Content
            feat_dino_c = dino_net(img_c)
            feat_dino_g = dino_net(img_g)
            l_dino = mse_loss(feat_dino_g, feat_dino_c)

            # === Accumulate ===
            total_content_loss += l_content.item()
            total_style_loss += l_style.item()
            total_dino_loss += l_dino.item()
            count += 1

    if count == 0:
        print("No valid image pairs found.")
        return

    # --- 输出平均结果 ---
    avg_content = total_content_loss / count
    avg_style = total_style_loss / count
    avg_dino = total_dino_loss / count

    print("\n" + "="*40)
    print(f"Evaluation Results (Avg over {count} images)")
    print("="*40)
    print(f"VGG Content Loss (Relu4_1) : {avg_content:.4f}")
    print(f"VGG Style Loss (Stats 1,2,3): {avg_style:.4f}")
    print(f"DINO Structure Loss        : {avg_dino:.4f}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Style Transfer Quality")
    parser.add_argument('--content_dir', type=str, default='content', help='Path to content images')
    parser.add_argument('--style_dir', type=str, default='style', help='Path to style images')
    parser.add_argument('--results_dir', type=str, default='results\\StyleFormer', help='Path to stylized result images')  # change
    
    args = parser.parse_args()
    
    # 检查路径是否存在
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' not found.")
    else:
        evaluate(args)