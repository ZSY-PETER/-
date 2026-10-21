# -*- coding: utf-8 -*-
"""Windows/CLI entry for segmentation -> quantified JSON -> candidate RAG -> Qwen -> DOCX."""
from __future__ import annotations
from bridge_paths import default_checkpoint, default_vmamba_models
import argparse
import hashlib
import json
import re
import subprocess
import sys
import os
import shutil
from datetime import datetime
from pathlib import Path

from utf8_process import utf8_child_env, set_utf8_stdio
import pipeline_core as core
from report_v2 import build_report
from dynamic_rag import plan_requests, summarize_group_evidence

BASE = Path(__file__).resolve().parent


def log(s):
    print(s, flush=True)


def safe_dir(s):
    v = re.sub(r'[^a-zA-Z0-9_\-]', '_', s or 'DEMO-CASE')
    return v[:60].strip('_') or 'DEMO-CASE'


def launch(cmd, label, logfile):
    log(f'[{label}] 开始执行 ...')
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('w', encoding='utf-8') as output:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace', bufsize=1,
                              env=utf8_child_env()) as proc:
            for line in proc.stdout:
                output.write(line)
                output.flush()
                print(line.rstrip(), flush=True)
            returncode = proc.wait()
    if returncode:
        raise RuntimeError(f'{label} 执行失败：code={returncode}；详细日志 {logfile}')
    log(f'[{label}] 完成')


def get_context(a):
    if a.engineering_json:
        context = core.read_json(Path(a.engineering_json))
    else:
        context = {
            'case_id': a.case_id,
            'project_name': a.project,
            'structure_type': a.structure,
            'component_type': a.component,
            'component_id': a.component_id,
            'component_location': a.location,
            'component_type_source': '人工填写的演示信息，待核实',
            'damage_type': '裂缝',
            'crack_orientation': a.orientation,
            'crack_type': a.crack_type,
            'standard_id': a.standard_id,
            'clause_id': a.clause_id,
            'is_real_inspection': False,
        }
    if context.get('is_real_inspection') is True:
        raise ValueError('本工程输出属于初赛演示，不支持将其标记为正式现场检验')
    return context


def run_quant(a, job_dir):
    if not a.image and not a.mask:
        raise ValueError('图像处理需要选择原图（有模型推理）或灰度类别 Mask（跳过模型推理）')
    cmd = [sys.executable, str(BASE/'run_bridge_quantifier.py')]
    if a.image:
        cmd += ['--image', str(Path(a.image).resolve())]
    if a.mask:
        cmd += ['--mask', str(Path(a.mask).resolve())]
    else:
        cmd += ['--checkpoint', a.checkpoint, '--vmamba-models', a.vmamba_models]
    cmd += ['--out', str(job_dir/'segmentation')]
    launch(cmd, '分割/像素量化', job_dir/'segmentation_console.log')
    stem = Path(a.image or a.mask).stem
    folder = job_dir/'segmentation'/stem
    p = folder/'09_quantification.json'
    if not p.is_file():
        raise FileNotFoundError(f'分割或量化程序未输出所需文件：{p}')
    log('QUANT_JSON=' + str(p))
    return p, folder/'01_center_crop.png', folder/'04_overlay.png'


def verify_cache(rag, context):
    saved = rag.get('engineering_inputs') or {}
    asked = str(context.get('clause_id') or '').strip()
    saved_clause = str(rag.get('requested_clause_id') or '').strip()
    if asked and saved_clause != asked:
        raise ValueError('缓存 RAG 的目标条款与当前输入不一致，不应复用；请改用实时检索')
    requested_std = re.sub(r'\W', '', str(context.get('standard_id') or '')).upper()
    cached_std = re.sub(r'\W', '', str(rag.get('requested_standard_id') or '')).upper()
    if requested_std and requested_std != cached_std:
        raise ValueError('缓存 RAG 标准编号与工程设定不一致')
    component = saved.get('component_type')
    if component and context.get('component_type') and component != context['component_type']:
        raise ValueError('缓存 RAG 构件类型与本次演示不符')
    if rag.get('success') is not True:
        raise ValueError('缓存 RAG 检索失败，不得假装已找到规范')


def run_report(a, job_dir, quant_path, image=None, overlay=None):
    context = get_context(a)
    dst = job_dir/'report'
    dst.mkdir(parents=True, exist_ok=True)
    log('[1/5] 读取并校验量化 JSON：' + str(quant_path))
    quant = core.read_json(quant_path)
    scale = float(a.demo_mm_per_px) if a.scale_mode == 'simulation' else None
    derived = core.derive_quantification(quant, a.scale_mode, scale)
    derived['source_quantification_sha256'] = hashlib.sha256(Path(quant_path).read_bytes()).hexdigest()
    derived['engineering_context'] = context
    core.write_json(dst/'01_engineering_measurements.json', derived)
    core.write_json(dst/'02_engineering_context.json', context)

    log('[2/5] 规范 RAG：' + a.rag_mode)
    # The automatic single-image entry is also damage-aware, not fixed to crack/9.1.2.
    from dynamic_rag import detected_damages, DAMAGE_LABELS
    detected = detected_damages(derived)
    if a.retrieval_strategy == "auto":
        context["damage_type"] = "、".join(DAMAGE_LABELS[c] for c in detected) if detected else "未检出四类病害"
    q = core.compose_rag_input(context) if a.retrieval_strategy=='exact' else None
    if a.retrieval_strategy=='auto':
        groups=plan_requests([{'case_id':context.get('case_id') or 'single',
                               'measurements':derived,'engineering_context':context}],strategy='auto')
        core.write_json(dst/'03_rag_request.json',{'retrieval_strategy':'auto','groups':groups})
        evidence=[];group_results=[]
        for group in groups:
            rid=group['group_id']
            qfile=dst/f'03_{rid}_request.json';rfile=dst/f'04_{rid}_result.json'
            core.write_json(qfile,group['request'])
            rag_group={'success':False,'results':[],'context':'','retrieval_mode':'skipped'}
            if a.rag_mode=='live':
                cmd=[sys.executable,str(Path(a.rag_script)),str(qfile),str(rfile)]
                if a.db_dir:cmd.extend(['--db',a.db_dir])
                launch(cmd,'BridgeRAG 动态 '+rid,dst/f'rag_{rid}.log')
                rag_group=core.read_json(rfile)
            elif a.rag_mode=='file':
                raise ValueError('自动模式不可复用旧单条款 RAG 缓存，请改用 live 或 skip')
            core.write_json(rfile,rag_group)
            evidence.extend(summarize_group_evidence(group,rag_group,core))
            group_results.append({'group_id':rid,'damage_type':group['damage_type'],
                                  'result_file':str(rfile), 'candidate_count':len(rag_group.get('results') or [])})
        rag={'success':any(g['candidate_count']>0 for g in group_results),
             'retrieval_mode':'skipped_demo' if a.rag_mode=='skip' else 'auto_grouped','groups':group_results,'results':[]}
    else:
        core.write_json(dst/'03_rag_request.json',q)
        if a.rag_mode == 'skip':
            rag = {'success': False, 'results': [], 'context': '', 'retrieval_mode': 'skipped_demo'}
            evidence = []
        elif a.rag_mode == 'file':
            if not a.rag_json or not Path(a.rag_json).is_file():
                raise FileNotFoundError('缓存检索模式必须选择既有有效的 RAG 结果 JSON')
            rag = core.read_json(Path(a.rag_json))
            verify_cache(rag, context)
            evidence = core.evidence_summary(rag)
        else:
            rag_script = Path(a.rag_script or (BASE/'rag_query.py'))
            cmd = [sys.executable, str(rag_script), str(dst/'03_rag_request.json'), str(dst/'04_rag_result.json')]
            if a.db_dir:cmd += ['--db', a.db_dir]
            launch(cmd, 'BridgeRAG V2', dst/'rag_console.log')
            rag = core.read_json(dst/'04_rag_result.json')
            evidence = core.evidence_summary(rag)
    core.write_json(dst/'04_rag_result.json', rag)
    core.write_json(dst/'05_evidence_summary.json', evidence)

    log('[3/5] Qwen 分析：' + a.llm_mode)
    if a.llm_mode == 'live':
        if not evidence:
            raise ValueError('RAG 未执行或无证据，无法使用 Qwen 生成规范分析；请选择 RAG 实时/缓存模式')
        analysis = core.run_local_qwen(derived, context, evidence, model=a.ollama_model, url=a.ollama_url)
    else:
        analysis = core.deterministic_analysis(derived, context, evidence)
        if not evidence:
            analysis['evidence_analysis'] = '本次未运行规范 RAG；未提供任何规范引用，不能开展规范适用性评定。'
    core.write_json(dst/'06_ai_analysis.json', analysis)

    log('[4/5] 输出工程风格 Word 报告')
    docx_path = dst/'07_bridge_inspection_demo.docx'
    build_report(docx_path, derived, context, evidence, analysis, image=image if image and Path(image).is_file() else None,
                 overlay=overlay if overlay and Path(overlay).is_file() else None, rag_state=a.rag_mode)
    log('[5/5] 所有阶段处理完成。')
    log('RESULT_JSON=' + str(dst/'01_engineering_measurements.json'))
    log('RESULT_DOCX=' + str(docx_path))
    log('RESULT_DIR=' + str(dst))
    return docx_path


def main(argv=None):
    ap=argparse.ArgumentParser(description='Bridge Contest GUI V2 pipeline / 不构成正式工程检测')
    ap.add_argument('--action', choices=['quantify','report','all'], default='report')
    ap.add_argument('--quant-json', default=str(BASE/'examples/09_quantification.json'))
    ap.add_argument('--engineering-json', default='')
    ap.add_argument('--image', default='')
    ap.add_argument('--mask', default='')
    ap.add_argument('--overlay', default='')
    ap.add_argument('--checkpoint', default=str(default_checkpoint()))
    ap.add_argument('--vmamba-models', default=str(default_vmamba_models()))
    ap.add_argument('--out', default=str(BASE/'outputs'))
    ap.add_argument('--scale-mode', choices=['simulation','none'], default='simulation')
    ap.add_argument('--demo-mm-per-px', type=float, default=None)
    ap.add_argument('--rag-mode', choices=['live','file','skip'], default='live')
    ap.add_argument('--rag-script', default=str(BASE/'rag_query.py'))
    ap.add_argument('--retrieval-strategy', choices=['auto','exact'], default='auto')
    ap.add_argument('--rag-json', default='')
    ap.add_argument('--db-dir', default=str(Path.home()/'Desktop/新建文件夹 (2)'))
    ap.add_argument('--llm-mode', choices=['live','off'], default='live')
    ap.add_argument('--ollama-model', default='qwen3:8b')
    ap.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    ap.add_argument('--case-id', default='DEMO-CASE-0067')
    ap.add_argument('--project', default='桥梁病害智能检测与辅助分析')
    ap.add_argument('--structure', default='公路桥梁')
    ap.add_argument('--component', default='')
    ap.add_argument('--component-id', default='')
    ap.add_argument('--location', default='')
    ap.add_argument('--orientation', default='')
    ap.add_argument('--crack-type', default='')
    ap.add_argument('--standard-id', default='')
    ap.add_argument('--clause-id', default='')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args(argv)
    if a.check:
        for filename in ('report_v2.py','pipeline_core.py','rag_query.py','run_bridge_quantifier.py'):
            print(filename, 'OK' if (BASE/filename).is_file() else 'MISSING')
        print('Python:',sys.version.split()[0])
        print('ChromaDB目录存在:',Path(a.db_dir).is_dir())
        print('Checkpoint存在:',Path(a.checkpoint).is_file())
        print('VMamba源码存在:',(Path(a.vmamba_models)/'vmamba.py').is_file())
        return 0
    c = get_context(a)
    job_dir = Path(a.out).expanduser().resolve() / safe_dir(str(c.get('case_id')))
    job_dir.mkdir(parents=True,exist_ok=True)
    log('项目输出目录：'+str(job_dir))
    q = Path(a.quant_json).expanduser().resolve()
    img = a.image or str(BASE/'examples/test_0067_center_crop.png')
    over = a.overlay or str(BASE/'examples/test_0067_overlay.png')
    if a.action in ('quantify','all'):
        q, img, over = run_quant(a,job_dir)
    if a.action in ('report','all'):
        if not q.is_file():
            raise FileNotFoundError('缺少量化 JSON：'+str(q))
        run_report(a,job_dir,q,img,over)
    return 0


if __name__=='__main__':
    set_utf8_stdio()
    try:
        sys.exit(main())
    except Exception as exc:
        log('[ERROR] '+str(exc))
        sys.exit(1)
