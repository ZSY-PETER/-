# -*- coding: utf-8 -*-
"""Recursive image batch scheduler for Bridge AI V2.3.

Uses the unchanged single-image run_pipeline.py in one child per image to isolate CUDA
resources and failures. Outputs a resumable manifest and summary. This is a
sequential scheduler, not parallel GPU inference.
"""
from __future__ import annotations
from bridge_paths import default_checkpoint, default_vmamba_models

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

from utf8_process import utf8_child_env, set_utf8_stdio

BASE = Path(__file__).resolve().parent
IMAGE_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'})
SKIP_DIRS = frozenset({
    '.git', '.venv', 'venv', '__pycache__', 'node_modules',
    'train_mask', 'val_mask', 'test_mask', 'train_masks', 'val_masks', 'test_masks',
    'masks', 'mask', 'labels', 'label', 'annotations', 'annotation',
    'test_mark_color', 'train_mark_color', 'val_mark_color',
})
MASK_NAME_RE = re.compile(r'(?:^|[_\-.])(mask|label|labels|gt|overlay|contour|prediction|pred_mask|heatmap)(?:[_\-.]|$)', re.I)


def safe_name(text: str, length: int = 36) -> str:
    """Use ASCII paths for outputs to avoid Windows path/tool incompatibilities."""
    s = re.sub(r'[^A-Za-z0-9_-]+', '_', str(text)).strip('_')[:length]
    return s or 'images'


def is_inside(p: Path, parent: Path) -> bool:
    try:
        p.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def discover_images(root: Path, *, exclusions: Iterable[Path] = (), include_masks: bool = False):
    """Stable recursive traversal. Never follows linked directories. Returns (images, stats)."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError('批量根目录不存在或不是文件夹：' + str(root))
    excluded = [Path(x).expanduser().resolve() for x in exclusions]
    images = []
    stats = {'visited_files': 0, 'unsupported_files': 0, 'excluded_masks': 0,
             'excluded_directories': 0, 'unreadable_directories': 0, 'eligible_images': 0}

    def on_error(exc):
        stats['unreadable_directories'] += 1
        print('[BATCH WARNING] 无法读取子目录：' + str(exc), flush=True)

    for folder, dirnames, filenames in os.walk(root, topdown=True, followlinks=False, onerror=on_error):
        folder_path = Path(folder)
        filtered_dirs = []
        for d in sorted(dirnames, key=str.casefold):
            candidate = folder_path / d
            if any(is_inside(candidate, x) for x in excluded):
                stats['excluded_directories'] += 1
                continue
            if not include_masks and d.lower() in SKIP_DIRS:
                stats['excluded_directories'] += 1
                continue
            if candidate.is_symlink():
                stats['excluded_directories'] += 1
                continue
            filtered_dirs.append(d)
        dirnames[:] = filtered_dirs
        if any(is_inside(folder_path, x) for x in excluded):
            continue
        for filename in sorted(filenames, key=str.casefold):
            p = folder_path / filename
            if p.is_symlink():
                continue
            stats['visited_files'] += 1
            if p.suffix.lower() not in IMAGE_EXTS:
                stats['unsupported_files'] += 1
                continue
            if not include_masks and MASK_NAME_RE.search(p.stem):
                stats['excluded_masks'] += 1
                continue
            images.append(p)
    stats['eligible_images'] = len(images)
    return images, stats


def case_id_for(relative: Path, prefix: str = 'DEMO') -> str:
    path_string = relative.as_posix()
    digest = hashlib.sha256(path_string.encode('utf-8')).hexdigest()[:10]
    label = safe_name(relative.stem, 28)
    return f'{safe_name(prefix, 15)}-{label}-{digest}'


def batch_root_for(out: Path, input_root: Path) -> Path:
    input_root = Path(input_root).expanduser().resolve()
    fingerprint = hashlib.sha256(str(input_root).casefold().encode('utf-8')).hexdigest()[:8]
    return Path(out).expanduser().resolve() / f'BATCH_{safe_name(input_root.name, 28)}_{fingerprint}'


def make_fingerprint(path: Path, args, pipeline_script: Path) -> str:
    st = path.stat()
    def signature_file(filename):
        if not filename:
            return None
        p=Path(filename)
        if not p.is_file():
            return [str(p), 'not_found']
        s=p.stat()
        return [str(p.resolve()), s.st_size, s.st_mtime_ns]
    fields = {
        'source': [str(path.resolve()), st.st_size, st.st_mtime_ns],
        'pipeline_version': signature_file(pipeline_script),
        'implementation': [signature_file(BASE/name) for name in (
            'bridge_model.py','bridge_geometry.py','run_bridge_quantifier.py',
            'report_v2.py','pipeline_core.py','batch_runner.py','batch_report.py',
            'dynamic_rag.py','presentation_groups.py','bridge_paths.py')],
        'chroma_sqlite': signature_file(
            (Path(args.db_dir)/'chroma_db'/'chroma.sqlite3')
            if (Path(args.db_dir)/'chroma_db'/'chroma.sqlite3').is_file()
            else (Path(args.db_dir)/'chroma.sqlite3')) if args.rag_mode=='live' else None,
        'checkpoint': signature_file(args.checkpoint),
        'vmamba_models': str(Path(args.vmamba_models).expanduser()),
        'parameters': {k: getattr(args, k, None) for k in (
            'action', 'scale_mode', 'demo_mm_per_px', 'rag_mode', 'rag_script',
            'rag_json', 'db_dir', 'llm_mode', 'ollama_model', 'ollama_url',
            'project', 'structure', 'component', 'component_id', 'location',
            'orientation', 'crack_type', 'standard_id', 'clause_id','retrieval_strategy','batch_metadata_csv')},
        'rag_script_file': signature_file(args.rag_script),
        'image_metadata_csv': signature_file(args.batch_metadata_csv),
        'rag_json_file': signature_file(args.rag_json) if args.rag_mode == 'file' else None,
    }
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False, sort_keys=True, default=str).encode('utf-8')).hexdigest()


def make_command(a, pipeline: Path, image: Path, item_root: Path, cid: str):
    display_image = image
    extra = []
    if a.action=='report':
        quant_dir=item_root/cid/'segmentation'/image.stem
        q=quant_dir/'09_quantification.json'
        if not q.is_file():
            raise FileNotFoundError('该图片没有已有量化 JSON，先运行 quantify：'+str(q))
        display_image=quant_dir/'01_center_crop.png'
        extra=['--quant-json',str(q),'--overlay',str(quant_dir/'04_overlay.png')]
    # Batch reporting happens ONCE at the end. Never create individual Word files.
    child_action = 'quantify'
    cmd=[sys.executable, '-u', str(pipeline), '--action', child_action,
         '--image', str(display_image), '--case-id', cid, '--out', str(item_root), *extra]
    keys=['checkpoint','vmamba_models','scale_mode','rag_mode','rag_script','rag_json',
          'db_dir','llm_mode','ollama_model','ollama_url','project','structure',
          'component','component_id','location','orientation','crack_type',
          'standard_id','clause_id','retrieval_strategy']
    for k in keys:
        val=getattr(a,k,None)
        if val is not None and str(val).strip():
            cmd += ['--'+k.replace('_','-'),str(val)]
    if a.scale_mode=='simulation':
        cmd += ['--demo-mm-per-px',str(a.demo_mm_per_px)]
    return cmd


def run_child(cmd, log_path: Path) -> tuple[int, str]:
    error_lines=[]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w',encoding='utf-8',errors='replace') as handle:
        with subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT, text=True,
                              encoding='utf-8',errors='replace',bufsize=1,
                              env=utf8_child_env()) as child:
            for line in child.stdout:
                line=line.rstrip('\r\n')
                print('  '+line,flush=True)
                handle.write(line+'\n')
                handle.flush()
                if '[ERROR]' in line or 'Traceback' in line:
                    error_lines.append(line.strip())
            rc=child.wait()
    return rc, '; '.join(error_lines[-3:]) if error_lines else ''


FIELDS=['relative_path','source_path','case_id','status','elapsed_s','crack_pixels',
        'quant_json','report_docx','log_file','error','source_modified_ns',
        'component_type','detected_damage_types','rag_group_ids']


def write_summary(batch_dir: Path, rows, stats, args):
    batch_dir.mkdir(parents=True,exist_ok=True)
    summary = {
        'schema_version': 'batch-1.1', 'root': str(Path(args.root).resolve()),
        'action': args.action, 'scale_mode':args.scale_mode, 'demo_mm_per_px':args.demo_mm_per_px if args.scale_mode=='simulation' else None,
        'processing_notice': '演示尺度非真实物理测量；批量汇总不代表整桥评价。',
        'eligible_images': stats['eligible_images'],
        'completed': sum(x['status']=='completed' for x in rows),
        'resumed': sum(x['status']=='resumed' for x in rows),
        'failed': sum(x['status']=='failed' for x in rows),
        'processed_count': len(rows), 'scan':stats,
        'rows': rows,
    }
    json_tmp=batch_dir/'batch_summary.tmp.json'
    json_tmp.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    json_tmp.replace(batch_dir/'batch_summary.json')
    csv_tmp=batch_dir/'batch_summary.tmp.csv'
    with csv_tmp.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader()
        for x in rows:
            writer.writerow({k:x.get(k,'') for k in FIELDS})
    csv_tmp.replace(batch_dir/'batch_summary.csv')
    return summary


def parser():
    ap=argparse.ArgumentParser(description='递归批量处理图片，全部输出合并到一个Word报告')
    ap.add_argument('--root',required=True,help='递归遍历的根文件夹')
    ap.add_argument('--out',default=str(BASE/'outputs'))
    ap.add_argument('--action',choices=['quantify','report','all'],default='all')
    ap.add_argument('--resume',action='store_true',help='跳过输入与参数都未变化且结果完整的条目')
    ap.add_argument('--include-masks',action='store_true',help='将名称含 Mask 的 PNG 也按照片送入模型（通常不建议）')
    ap.add_argument('--pipeline-script',default=str(BASE/'run_pipeline.py'),help=argparse.SUPPRESS)
    ap.add_argument('--checkpoint',default=str(default_checkpoint()))
    ap.add_argument('--vmamba-models',default=str(default_vmamba_models()))
    ap.add_argument('--scale-mode',choices=['simulation','none'],default='simulation')
    ap.add_argument('--demo-mm-per-px',type=float,default=0.02)
    ap.add_argument('--rag-mode',choices=['live','file','skip'],default='live')
    ap.add_argument('--rag-script',default=str(BASE/'rag_query.py'))
    ap.add_argument('--retrieval-strategy',choices=['auto','exact'],default='auto')
    ap.add_argument('--batch-metadata-csv',default='',help='可选：relative_path,component_type,component_id,location,orientation,crack_type')
    ap.add_argument('--rag-json',default='')
    ap.add_argument('--db-dir',default=str(Path.home()/'Desktop/新建文件夹 (2)'))
    ap.add_argument('--llm-mode',choices=['live','off'],default='live')
    ap.add_argument('--ollama-model',default='qwen3:8b')
    ap.add_argument('--ollama-url',default='http://127.0.0.1:11434')
    ap.add_argument('--case-id',default='DEMO')
    ap.add_argument('--project',default='桥梁病害批量检测')
    ap.add_argument('--structure',default='公路桥梁')
    ap.add_argument('--component',default='')
    ap.add_argument('--component-id',default='')
    ap.add_argument('--location',default='')
    ap.add_argument('--orientation',default='')
    ap.add_argument('--crack-type',default='')
    ap.add_argument('--standard-id',default='')
    ap.add_argument('--clause-id',default='')
    return ap


def main(argv=None):
    set_utf8_stdio()
    a=parser().parse_args(argv)
    if a.scale_mode=='simulation' and not (0<float(a.demo_mm_per_px)<=1000):
        raise ValueError('模拟比例尺须大于0，且不大于1000 mm/px')
    root=Path(a.root).expanduser().resolve()
    out=Path(a.out).expanduser().resolve()
    batch_dir=batch_root_for(out, root)
    # Avoid reading our own outputs or report images even if root includes Desktop.
    excluded=[out, BASE/'outputs',batch_dir]
    if out==root:
        raise ValueError('输出目录不能与图片根目录完全相同；请把输出放在根目录以外或其专属子目录')
    paths,stats=discover_images(root,exclusions=excluded,include_masks=a.include_masks)
    print('BATCH_FOLDER='+str(batch_dir),flush=True)
    print('BATCH_TOTAL='+str(len(paths)),flush=True)
    print(f'[BATCH] 根目录：{root}；扫描文件 {stats["visited_files"]}；可处理图片 {len(paths)}；其他文件 {stats["unsupported_files"]}；过滤Mask {stats["excluded_masks"]}',flush=True)
    if not paths:
        write_summary(batch_dir,[],stats,a)
        print('[BATCH] 没有找到可处理的图片（支持 jpg/jpeg/png/bmp/tif/tiff/webp）',flush=True)
        return 1
    pipeline=Path(a.pipeline_script).resolve()
    if not pipeline.is_file():
        raise FileNotFoundError('单图入口不存在：'+str(pipeline))
    rows=[]
    for index,img in enumerate(paths,1):
        rel=img.relative_to(root)
        cid=case_id_for(rel,a.case_id)
        item_dir=batch_dir/'items'/cid
        q=item_dir/'segmentation'/img.stem/'09_quantification.json'
        doc=item_dir/'report'/'07_bridge_inspection_demo.docx'
        state_file=item_dir/'batch_item_state.json'
        fingerprint=make_fingerprint(img,a,pipeline)
        status='completed'; err=''; elapsed=0.0
        print(f'BATCH_PROGRESS={index}/{len(paths)}',flush=True)
        print(f'[BATCH] ({index}/{len(paths)}) {rel.as_posix()}',flush=True)
        # Per-picture Word is intentionally eliminated in batch mode.
        required=[q]
        if a.resume and state_file.is_file() and all(p.is_file() for p in required):
            try:
                saved=json.loads(state_file.read_text(encoding='utf-8'))
                if saved.get('fingerprint')==fingerprint and saved.get('action')==a.action:
                    status='resumed'
                    print('[BATCH] 跳过：输入及配置未变化，已有完整输出',flush=True)
            except (OSError,ValueError):
                pass
        if status!='resumed':
            start=time.perf_counter()
            try:
                item_dir.mkdir(parents=True, exist_ok=True)
                if a.action == 'report':
                    # Reuse existing quantified JSON without loading the segmentation model.
                    print('[BATCH] 复用量化结果，不重新推理',flush=True)
                else:
                    cmd=make_command(a,pipeline,img,batch_dir/'items',cid)
                    rc,err=run_child(cmd,item_dir/'batch_console.log')
                    if rc:
                        raise RuntimeError(f'单图程序返回码={rc}；{err}')
                missing=[str(p) for p in required if not p.is_file()]
                if missing:
                    raise FileNotFoundError('子进程返回成功但缺失结果文件：'+', '.join(missing))
                state_file.write_text(json.dumps({'fingerprint':fingerprint,'action':a.action,
                    'relative_path':rel.as_posix(),'source':str(img),'updated_at':datetime.now().isoformat()},
                    ensure_ascii=False,indent=2),encoding='utf-8')
            except Exception as exc:
                status='failed'
                err=str(exc)
                print(f'[BATCH ERROR] {rel.as_posix()} -> {err}',flush=True)
            finally:
                elapsed=round(time.perf_counter()-start,3)
        crack_px=''
        if q.is_file() and status!='failed':
            try:
                data=json.loads(q.read_text(encoding='utf-8-sig'))
                crack=next((c for c in data.get('classes',[]) if c.get('code')=='crack'),{})
                crack_px=crack.get('retained_pixels_after_filter','')
            except (OSError,ValueError):
                pass
        row={'relative_path':rel.as_posix(),'source_path':str(img),'case_id':cid,
             'status':status,'elapsed_s':elapsed,'crack_pixels':crack_px,
             'quant_json':str(q) if q.is_file() and status!='failed' else '',
             'report_docx':'',  # Whole batch shares the combined report; no per-image DOCX.
             'log_file':str(item_dir/'batch_console.log'), 'error':err,
             'source_modified_ns':img.stat().st_mtime_ns}
        rows.append(row)
        summary=write_summary(batch_dir,rows,stats,a)
        print('BATCH_ITEM='+json.dumps({'index':index,'total':len(paths),'relative_path':rel.as_posix(),
                                       'status':status,'quant_json':row['quant_json'],
                                       'report_docx':row['report_docx']},ensure_ascii=False),flush=True)
    report_error = ''
    if a.action in ('all', 'report'):
        if any(r.get('quant_json') for r in rows):
            try:
                print('[BATCH] 全部逐图任务结束，开始合并唯一 Word 总报告...',flush=True)
                from batch_report import create_batch_report
                combined_doc, updated = create_batch_report(a,batch_dir,summary)
                summary=updated
                tmp=batch_dir/'batch_summary.tmp.json'
                tmp.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
                tmp.replace(batch_dir/'batch_summary.json')
                # Rebuild the CSV after joining per-image damage / component / evidence groups.
                with (batch_dir/'batch_summary.csv').open('w',encoding='utf-8-sig',newline='') as fh:
                    w=csv.DictWriter(fh,fieldnames=FIELDS);w.writeheader()
                    for record in summary['rows']:
                        w.writerow({field:record.get(field,'') for field in FIELDS})
                print('BATCH_REPORT_DOCX='+str(combined_doc),flush=True)
            except Exception as exc:
                report_error=str(exc)
                print('[BATCH ERROR] 合并报告失败：'+report_error,flush=True)
                summary['combined_report']={'status':'failed','error':report_error}
                tmp=batch_dir/'batch_summary.tmp.json'
                tmp.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
                tmp.replace(batch_dir/'batch_summary.json')
        else:
            report_error='没有任何可读取的量化 JSON；无法创建合并 Word'
            print('[BATCH ERROR] '+report_error,flush=True)
    print('BATCH_SUMMARY_JSON='+str(batch_dir/'batch_summary.json'),flush=True)
    print('BATCH_DONE='+json.dumps({k:summary[k] for k in ('completed','resumed','failed','eligible_images')},ensure_ascii=False),flush=True)
    if not report_error and summary.get('combined_report',{}).get('status') == 'partial':
        print('[BATCH PARTIAL] Word 已生成，但 RAG 或 Qwen 没有全部完成；请查看 report/qwen_debug.json 和 06_ai_analysis.json',flush=True)
        return 4
    return 3 if report_error else (2 if summary['failed'] else 0)


if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('[BATCH ERROR] '+str(exc),flush=True)
        sys.exit(1)
