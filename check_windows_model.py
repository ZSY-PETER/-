# -*- coding: utf-8 -*-
"""Diagnose Windows inference dependencies without running training or Ollama."""
from __future__ import annotations
from bridge_paths import default_checkpoint, default_vmamba_models
import sys
from pathlib import Path


def main():
    checkpoint=default_checkpoint()
    vmamba_path=default_vmamba_models()
    print('Python:', sys.version.replace('\n',' '))
    print('Python executable:', sys.executable)
    print('权重:', checkpoint, '| exists:', checkpoint.is_file())
    print('官方 VMamba:', vmamba_path, '| exists:', (vmamba_path/'vmamba.py').is_file())
    missing = []
    for name in ('torch','torchvision','segmentation_models_pytorch','numpy','cv2','PIL','scipy','skimage','docx'):
        try:
            __import__(name)
            print('[OK]',name)
        except Exception as exc:
            missing.append(name)
            print('[FAILED]', name, ':', type(exc).__name__,str(exc)[:200])
    if not checkpoint.is_file() or not (vmamba_path/'vmamba.py').is_file() or missing:
        print('\n请先补齐上述缺失文件或依赖；模型尚未尝试加载。')
        return 1
    try:
        from bridge_model import build_model
        import torch
        ckpt=torch.load(str(checkpoint),map_location='cpu',weights_only=False)
        print('Checkpoint experiment:', ckpt.get('experiment'))
        print('Checkpoint epoch:', ckpt.get('epoch'))
        print('Checkpoint Val mIoU:', ckpt.get('best_val_miou'))
        m=build_model(str(vmamba_path))
        m.load_state_dict(ckpt['model_state_dict'], strict=True)
        print('[PASS] 官方 VMamba 已导入，模型权重严格加载成功。')
        print('CUDA available:',torch.cuda.is_available())
        print('注意：这只检查构建和加载，尚未验证 Windows CUDA 推理算子。')
        return 0
    except Exception as exc:
        print('[FAILED] VMamba/模型加载:',type(exc).__name__,str(exc)[:700])
        print('需要检查 VMamba 依赖、CUDA 编译算子及 SMP 版本；不能假定 Windows 可直接运行。')
        return 1


if __name__=='__main__':
    raise SystemExit(main())
