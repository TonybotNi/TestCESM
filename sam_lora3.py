# Sheng Wang at Apr 6 2023
# What a time to be alive (first half of 2023)

from segment_anything.modeling import Sam
from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam
from safetensors import safe_open
from safetensors.torch import save_file
from timm.models.vision_transformer import VisionTransformer as timm_ViT
# from modules.FADC import AdaptiveDilatedConv

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x):
        batch, c, h, w = x.size()
        r_size = x.size()
        # 进行傅里叶变换，返回复数结果
        ffted = torch.fft.fft2(x, norm="ortho")
        ffted_real = ffted.real  # 提取实部
        ffted_imag = ffted.imag  # 提取虚部
        # 将复数分解成两个通道：实部和虚部拼接
        ffted = torch.cat([ffted_real, ffted_imag], dim=1)
        # 卷积操作
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))
        # 将卷积后的实部和虚部重新分离
        ffted_real, ffted_imag = torch.chunk(ffted, 2, dim=1)
        ffted = torch.complex(ffted_real, ffted_imag)  # 重新组合为复数
        # 进行逆傅里叶变换
        output = torch.fft.ifft2(ffted, s=(h, w), norm="ortho").real  # 只保留实部输出
        return output

class _LoRA_qkv_conv_(nn.Module):
    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        FU_q: nn.Module,
        FU_v: nn.Module,
        conv_q: nn.Module,
        conv_v: nn.Module
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.FU_q = FU_q
        self.FU_v = FU_v
        self.conv_q = conv_q
        self.conv_v = conv_v
    def forward(self, x):
        # x: [25, 14, 14, 768]; self.qkv: Linear(in_features=768, out_features=2304, bias=True)
        qkv = self.qkv(x)  # B,N,N,3*org_C
        
        cv_q = self.linear_a_q(x).permute(0, 3, 1, 2)
        cv_v = self.linear_a_v(x).permute(0, 3, 1, 2)
        
        mid_q = self.FU_q(cv_q).permute(0,2,3,1) + self.conv_q(cv_q).permute(0,2,3,1)
        mid_v = self.FU_v(cv_v).permute(0,2,3,1) + self.conv_v(cv_v).permute(0,2,3,1)
        
        mid_q += self.linear_a_q(x)
        mid_v += self.linear_a_v(x)
        
        new_q = self.linear_b_q(mid_q)
        new_v = self.linear_b_v(mid_v)
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim :] += new_v
        return qkv

class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        # x: [25, 14, 14, 768]; self.qkv: Linear(in_features=768, out_features=2304, bias=True)
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim :] += new_v
        return qkv


class LoRA(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def save_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """
        assert filename.endswith(".safetensors")
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}
        save_file(fc_tensors, filename)

    def load_fc_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith(".safetensors")
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        with safe_open(filename, framework="pt") as f:
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.
        
        save both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        conv_tensors = {f"w_conv_{i:03d}": self.w_FUs[i].conv_layer.weight for i in range(num_layer)}
        bn_tensors = {f"w_bn_{i:03d}": self.w_FUs[i].bn.weight for i in range(num_layer)}
        conv3_tensors = {f"w_conv3_{i:03d}": self.w_convs[i].weight for i in range(num_layer)}
        
        _in = self.lora_vit.head.in_features
        _out = self.lora_vit.head.out_features
        fc_tensors = {f"fc_{_in}in_{_out}out": self.lora_vit.head.weight}
        
        merged_dict = {**a_tensors, **b_tensors, **conv_tensors, **bn_tensors, **conv3_tensors, **fc_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\
            
        load both lora and fc parameters.
        """

        assert filename.endswith(".safetensors")

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = Parameter(saved_tensor)
                
            for i, w_FU in enumerate(self.w_FUs):
                saved_key = f"w_conv_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_FU.conv_layer.weight = Parameter(saved_tensor)
            
            for i, w_FU in enumerate(self.w_FUs):
                saved_key = f"w_bn_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_FU.bn.weight = Parameter(saved_tensor)
            for i, w_conv in enumerate(self.w_convs):
                saved_key = f"w_conv3_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_conv.weight = Parameter(saved_tensor)
                
            _in = self.lora_vit.head.in_features
            _out = self.lora_vit.head.out_features
            saved_key = f"fc_{_in}in_{_out}out"
            try:
                saved_tensor = f.get_tensor(saved_key)
                self.lora_vit.head.weight = Parameter(saved_tensor)
            except ValueError:
                print("this fc weight is not for this model")

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
        for conv in self.convs_3:
            nn.init.zeros_(conv.weight)
        for conv in self.convs_1:
            nn.init.zeros_(conv.weight)


class LoRA_Sam(LoRA):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, sam_model: Sam, r: int, lora_layer=None):
        
        
        super(LoRA_Sam, self).__init__()

        assert r > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.blocks)))
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        self.w_FUs = []

        # lets freeze first
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            
            conv_q = nn.Conv2d(r, r, kernel_size=3, padding=1, bias=False)
            conv_v = nn.Conv2d(r, r, kernel_size=3, padding=1, bias=False)
            # conv_q_1 = nn.Conv2d(r, r, kernel_size=1, padding=0, bias=False)
            # conv_v_1 = nn.Conv2d(r, r, kernel_size=1, padding=0, bias=False)
            FU_q = FourierUnit(r, r)
            FU_v = FourierUnit(r, r)
            
            # conv_q = AdaptiveDilatedConv(in_channels=r, out_channels=r, kernel_size=3)
            # conv_v = AdaptiveDilatedConv(in_channels=r, out_channels=r, kernel_size=3)
            
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            self.w_FUs.append(FU_q)
            self.w_FUs.append(FU_v)
            self.w_convs.append(conv_q)
            self.w_convs.append(conv_v)
            
            blk.attn.qkv = _LoRA_qkv_conv_(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                FU_q,
                FU_v,
                conv_q,
                conv_v,
            )
        self.reset_parameters()
        # self.sam = sam_model
        self.lora_vit = sam_model

