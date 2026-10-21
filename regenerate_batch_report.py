# -*- coding: utf-8 -*-
"""Generate the single combined Word from a saved batch_summary.json.

No image scanning, GPU inference, or recomputation of masks is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from batch_runner import FIELDS, parser as batch_parser
from batch_report import create_batch_report
from utf8_process import set_utf8_stdio


def get_parser():
    ap=argparse.ArgumentParser(description='使用现有 batch_summary.json 直接生成唯一 Word；不重新分割图像')
    ap.add_argument('--summary',required=True,help='已有批量结果目录中的 batch_summary.json')
    ap.add_argument('--rag-mode',choices=['live','skip'],default='live')
    ap.add_argument('--llm-mode',choices=['live','off'],default='live')
    ap.add_argument('--retrieval-strategy',choices=['auto','exact'],default='auto')
    ap.add_argument('--rag-script',default=None)
    ap.add_argument('--db-dir',default=None)
    ap.add_argument('--ollama-model',default=None)
    ap.add_argument('--ollama-url',default=None)
    ap.add_argument('--project',default=None)
    ap.add_argument('--structure',default=None)
    ap.add_argument('--component',default=None)
    ap.add_argument('--component-id',default=None)
    ap.add_argument('--batch-metadata-csv',default=None)
    ap.add_argument('--standard-id',default=None)
    ap.add_argument('--clause-id',default=None)
    return ap


def main(argv=None):
    set_utf8_stdio()
    options=get_parser().parse_args(argv)
    summary_path=Path(options.summary).expanduser().resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(f'找不到批量统计 JSON：{summary_path}')
    manifest=json.loads(summary_path.read_text(encoding='utf-8-sig'))
    if not isinstance(manifest.get('rows'),list) or not manifest.get('root'):
        raise ValueError('这不是有效的 batch_summary.json：缺少 rows 或 root')
    batch_dir=summary_path.parent
    a=batch_parser().parse_args(['--root',str(manifest['root']), '--action','report',
                                 '--out',str(batch_dir.parent)])
    a.scale_mode=manifest.get('scale_mode','simulation')
    if a.scale_mode=='simulation':
        a.demo_mm_per_px=manifest.get('demo_mm_per_px')
        if not isinstance(a.demo_mm_per_px,(int,float)) or a.demo_mm_per_px<=0:
            raise ValueError('原批次缺少有效的示例尺度系数 demo_mm_per_px')
    for key in ('rag_mode','llm_mode','retrieval_strategy','rag_script','db_dir',
                'ollama_model','ollama_url','project','structure','component',
                'component_id','batch_metadata_csv','standard_id','clause_id'):
        val=getattr(options,key,None)
        if val is not None:
            setattr(a,key,val)
    existing=[r for r in manifest['rows'] if r.get('status') in ('completed','resumed')
              and r.get('quant_json') and Path(r['quant_json']).is_file()]
    print('[RECOVER] 读取原批次：'+str(batch_dir),flush=True)
    print(f'[RECOVER] 原记录 {len(manifest["rows"])} 条；已有量化 JSON {len(existing)} 条；不重新执行模型。',flush=True)
    if not existing:
        raise ValueError('没有找到可用量化 JSON，确认当前电脑上的输出目录尚未移动或删除')
    doc,updated=create_batch_report(a,batch_dir,manifest)
    updated['combined_report']['recovered_without_inference']=True
    tmp=batch_dir/'batch_summary.tmp.json'
    tmp.write_text(json.dumps(updated,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(summary_path)
    csvpath=batch_dir/'batch_summary.csv'
    with csvpath.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS)
        w.writeheader()
        for r in updated['rows']:
            w.writerow({k:r.get(k,'') for k in FIELDS})
    print('BATCH_REPORT_DOCX='+str(doc),flush=True)
    print(f'[RECOVER] 完成：已有 {len(existing)} 条量化结果已汇总到同一份 Word。',flush=True)
    if updated.get('combined_report',{}).get('status') == 'partial':
        print('[RECOVER PARTIAL] Word 已生成，但智能分析尚未完成，请检查 qwen_debug.json。',flush=True)
        return 4
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:
        print('[RECOVER ERROR] '+str(exc),file=sys.stderr,flush=True)
        sys.exit(1)
