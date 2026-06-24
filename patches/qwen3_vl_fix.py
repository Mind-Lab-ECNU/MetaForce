"""
Patch for Qwen3-VL device mismatch issue
"""
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl

# 保存原始方法
_original_fast_pos_embed_interpolate = modeling_qwen3_vl.Qwen3VLVisionTransformer.fast_pos_embed_interpolate

def patched_fast_pos_embed_interpolate(self, grid_thw):
    """
    修复版本：确保所有 tensor 在同一设备上
    """
    # ✅ 确保 grid_thw 在正确设备上
    if isinstance(grid_thw, torch.Tensor):
        device = next(self.parameters()).device
        grid_thw = grid_thw.to(device)
    
    # 调用原始方法
    return _original_fast_pos_embed_interpolate(self, grid_thw)

# 应用 patch
modeling_qwen3_vl.Qwen3VLVisionTransformer.fast_pos_embed_interpolate = patched_fast_pos_embed_interpolate

print("✅ Qwen3-VL device mismatch patch applied")
