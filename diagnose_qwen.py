# -*- coding: utf-8 -*-
"""Windows-local smoke test for the already-installed Qwen3 model.

No ChromaDB connection, model training, file scanning, or full batch inference.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from pipeline_core import run_local_qwen


def main():
    ap = argparse.ArgumentParser(description='只测试本地 Qwen3 结构化 JSON 输出')
    ap.add_argument('--model', default='qwen3:8b')
    ap.add_argument('--url', default='http://127.0.0.1:11434')
    ap.add_argument('--output', default=str(Path(__file__).parent / 'qwen_smoke_debug.json'))
    args = ap.parse_args()
    derived = {
        'calibration': {'mode': 'simulation', 'mm_per_pixel': 0.02, 'verified': False,
                        'is_real_physical_measurement': False},
        'pixel_metrics': {'note': '仅供接口连通性测试，没有真实病害测量值'},
        'physical_metrics': {'note': '本测试不提供毫米尺寸'},
    }
    context = {'project_name': '本地 Qwen 连通测试', 'component_type': '未指定',
               'is_real_inspection': False}
    try:
        reply = run_local_qwen(derived, context, [], model=args.model, url=args.url,
                               diagnostic_path=args.output)
    except Exception as exc:
        print('[QWEN TEST FAILED]', exc, flush=True)
        raise SystemExit(1)
    print('[QWEN TEST PASSED] 模型：', reply['source'])
    for key in ('summary', 'evidence_analysis', 'limitations', 'suggestions'):
        print('  ', key, '=', reply[key])
    print('诊断文件：', args.output)


if __name__ == '__main__':
    main()
