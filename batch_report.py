# -*- coding: utf-8 -*-
"""Consolidated one-DOCX batch report. Does not generate per-image DOCX files.

The generated Word is a technical demonstration, NOT an inspection certificate.
Individual source masks, quantifications and photos remain in items/ for audit.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

import pipeline_core as core
from utf8_process import utf8_child_env
from inspection_synthesis import annotate_records, meaningful, n as val_n
from presentation_groups import grouped_findings, DISPLAY_NAMES, present_damage_codes
from dynamic_rag import (DAMAGE_LABELS, detected_damages, load_image_metadata,
                         engineering_for_image, plan_requests, summarize_group_evidence)


def _txt(x, default='未提供'):
    return str(x) if x is not None and str(x).strip() else default


def _num(x, digits=2, unit=''):
    if not isinstance(x, (int,float)) or isinstance(x,bool) or not math.isfinite(x):
        return '—'
    return f'{x:.{digits}f}{unit}'


def collect_measurements(rows, scale_mode: str, scale: float | None):
    """Return one measurement for each valid row and failures separately."""
    records = []
    for row in rows:
        item = dict(row)
        item['measurements'] = None
        q = Path(str(row.get('quant_json') or ''))
        if row['status'] in ('completed','resumed') and q.is_file():
            try:
                quant = core.read_json(q)
                item['measurements'] = core.derive_quantification(
                    quant, scale_mode, scale if scale_mode=='simulation' else None)
                item['measurements']['source_quantification_sha256'] = hashlib.sha256(q.read_bytes()).hexdigest()
            except (OSError,ValueError,TypeError,KeyError) as exc:
                item['status'] = 'failed'
                item['error'] = '量化 JSON 无法汇总：' + str(exc)
        records.append(item)
    return records


def engineering_context(a):
    return {
        'project_name': a.project, 'structure_type': a.structure,
        'component_type': a.component, 'component_id': a.component_id,
        'component_location': a.location, 'crack_orientation': a.orientation,
        'crack_type': a.crack_type, 'damage_type': '裂缝',
        'standard_id': a.standard_id, 'clause_id': a.clause_id,
        'component_type_source': '批量统一填写；不代表每张图的构件身份已核实',
        'is_real_inspection': False,
    }


def obtain_batch_analysis(a, output_dir: Path, records):
    """Dynamically plan a lookup for each (observed damage, declared component).

    Each successful group has its own query/results pair; exactly ONE Word and
    at most one Qwen narrative are produced after aggregating all groups.
    """
    context = engineering_context(a)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = load_image_metadata(getattr(a, 'batch_metadata_csv',''),
                                  [x.get('relative_path','') for x in records])
    for row in records:
        row['engineering_context'] = engineering_for_image(row, context, mapping)
    annotate_records(records)
    requested_strategy=getattr(a,'retrieval_strategy','auto')
    strategy=requested_strategy
    fallback_reason=''
    # An exact-clause query is inappropriate if its identifiers are absent or
    # when a heterogeneous batch would force unrelated diseases onto one clause.
    if strategy=='exact':
        if not str(a.standard_id or '').strip() or not str(a.clause_id or '').strip():
            fallback_reason='精确检索缺少标准编号或目标条款'
        else:
            detected_codes={c for row in records for c in detected_damages(row.get('measurements'))}
            if len(detected_codes)>1:
                fallback_reason='同批图片包含多种病害，不能全部套用一个指定条款'
        if fallback_reason:
            strategy='auto'
            print('[BATCH RAG] '+fallback_reason+'；已改用 auto 动态规范检索。',flush=True)
    groups=plan_requests(records,strategy=strategy,standard_id=a.standard_id,clause_id=a.clause_id,top_k=5)
    # The remaining report metadata uses the actual strategy, not the stale UI value.
    a.retrieval_strategy=strategy
    if a.rag_mode=='file' and groups:
        raise ValueError('批量动态/分组报告不允许复用单一旧 RAG 缓存。请选择 live 或 skip。')
    print(f'[RAG V2.6] 动态规划：{len(groups)} 组（按病害类别和已提供的构件信息分组）；策略={strategy}',flush=True)
    for group in groups:
        print(f"[RAG V2.6] {group['group_id']} {group['damage_type']} | 构件={group['component_type']} | 图片数={len(group['image_case_ids'])}",flush=True)
    core.write_json(output_dir/'03_rag_request.json',{
        'schema_version':'batch-rag-request-2.7','retrieval_strategy':strategy,
        'requested_strategy':requested_strategy,
        'strategy_fallback_reason':fallback_reason or None,
        'note':'auto 模式不指定 standard_id / clause_id；模型只识别病害种类，不能确定构件身份',
        'groups':groups})
    evidence=[];errors=[]
    grouped_dir=output_dir/'rag_by_group'
    grouped_dir.mkdir(parents=True,exist_ok=True)
    from run_pipeline import launch
    for group in groups:
        key=group['group_id'];qfile=grouped_dir/f'{key}_request.json';rfile=grouped_dir/f'{key}_result.json'
        core.write_json(qfile,group['request'])
        rag={'success':False,'results':[],'context':'','retrieval_mode':'skipped'}
        if a.rag_mode=='skip':
            group['status']='skipped'
        elif a.rag_mode=='file':
            # An old single-clause cache MUST NOT be silently reused for other types.
            group['status']='cache_incompatible'
            errors.append('动态多病害检索不支持复用旧单条款缓存，请选择 RAG=live 或 skip')
        else:
            try:
                cmd=[__import__('sys').executable,str(Path(a.rag_script)),str(qfile),str(rfile)]
                if a.db_dir:cmd += ['--db',a.db_dir]
                launch(cmd,'动态规范检索 '+key,grouped_dir/f'{key}_console.log')
                rag=core.read_json(rfile)
                if not rag.get('success') or not rag.get('results'):
                    raise RuntimeError('没有返回候选规范')
                group['status']='retrieved'
            except Exception as exc:
                group['status']='failed'
                errors.append(f'{key} {group["damage_type"]}：{exc}')
                rag={'success':False,'results':[],'context':'','retrieval_mode':'failed','error':str(exc)}
                print('[RAG V2.4 WARNING] '+str(exc),flush=True)
        core.write_json(rfile,rag)
        group['retrieval_mode']=rag.get('retrieval_mode','')
        group['result_file']=str(rfile)
        group['candidate_count']=len(rag.get('results') or [])
        evidence.extend(summarize_group_evidence(group,rag,core))
    core.write_json(output_dir/'04_rag_result.json',{
        'schema_version':'batch-rag-results-2.6','retrieval_strategy':strategy,
        'groups':[{'group_id':g['group_id'],'damage_type':g['damage_type'],
                   'component_type':g['component_type'],'status':g['status'],
                   'candidate_count':g['candidate_count'],'result_file':g['result_file'],
                   'image_case_ids':g['image_case_ids']} for g in groups],
        'errors':errors})
    core.write_json(output_dir/'05_evidence_summary.json',evidence)
    total=len(records)
    successful=[x for x in records if x.get('measurements') is not None]
    by_damage={code:sum(code in x.get('detected_damage_codes',[]) for x in successful)
               for code in DAMAGE_LABELS}
    by_display={code:sum(code in present_damage_codes(x.get('detected_damage_codes', [])) for x in successful)
                for code in DISPLAY_NAMES}
    demo_observations=[]
    for row in successful[:24]:
        for f in grouped_findings(row.get('inspection_findings',[])):
            demo_observations.append({'image':row.get('relative_path'),
                                      'damage':f['damage_type'],
                                      'metrics':f['measurement_text'][:180],
                                      'demo_attention_level':f['demo_attention_level']})
    aggregate={
        'calibration':{'mode':a.scale_mode,'source_type':'assumed_uniform_scale' if a.scale_mode=='simulation' else 'no_scale',
                       'mm_per_pixel':a.demo_mm_per_px if a.scale_mode=='simulation' else None,
                       'verified':False,'is_real_physical_measurement':False},
        'pixel_metrics':{'input_images':total,'successfully_quantified_images':len(successful),
                         'images_with_predicted_crack_pixels':by_damage['crack'],
                         'images_by_damage_class':by_damage,
                         'images_by_display_group':by_display,
                         'note':'每类统计检测阳性图片数；不表示独立病害条数或整桥评级'},
        'physical_metrics':{'note':'假设比例尺转换只作为演示，不用于实际工程阈值判定',
                            'demo_observations':demo_observations[:40]},
    }
    short_evidence=[]
    # Distinct types get representation, but do not overflow the local 8B context window.
    for group in groups:
        items=[v for v in evidence if v['group_id']==group['group_id']]
        for ev in items[:2]:
            short_evidence.append({**ev,'excerpt_for_llm':ev.get('excerpt_for_llm','')[:450]})
    llm_context={**context,'component_type':'逐图见批次工程明细，可能不同',
                 'retrieval_strategy':strategy,'evidence_group_count':len(groups),
                 'detection_classes':[DISPLAY_NAMES[c] for c in DISPLAY_NAMES if by_display[c]],
                 'damage_type':'、'.join(DISPLAY_NAMES[c] for c in DISPLAY_NAMES if by_display[c]) or '未检出目标病害'}
    if a.llm_mode=='live' and short_evidence:
        try:
            analysis=core.run_local_qwen(aggregate,llm_context,short_evidence,
                                         model=a.ollama_model,url=a.ollama_url,
                                         diagnostic_path=output_dir/'qwen_debug.json')
        except Exception as exc:
            errors.append('Qwen 生成失败：'+str(exc))
            analysis=core.deterministic_analysis(aggregate,llm_context,evidence)
            analysis['summary']='Qwen 调用失败，下面为规则回退说明，非大模型生成。'
            print('[BATCH WARNING] '+str(exc),flush=True)
    else:
        if a.llm_mode=='live' and not short_evidence:
            errors.append('Qwen 未运行：无可用规范候选证据')
        analysis=core.deterministic_analysis(aggregate,llm_context,evidence)
    analysis['llm_status'] = ('success' if str(analysis.get('source', '')).startswith('ollama:')
                              else 'failed' if a.llm_mode == 'live' else 'disabled')
    if analysis.get('source')=='rule_based_no_llm':
        class_text='、'.join(f'{DISPLAY_NAMES[c]} {by_display[c]}张' for c in DISPLAY_NAMES if by_display[c]) or '未检出目标病害'
        analysis['summary']=f'批量导入{total}张图片，成功解析{len(successful)}张；各类阳性图片数：{class_text}。此为图像计数，不是独立病害条数。'
        analysis['evidence_analysis']=('已按实际阳性类别分别检索规范候选，所有条款仍需核实构件适用性和原版表格；不得据此正式分级。'
                                       if evidence else '未获得可用规范证据，不执行规范阈值判定。')
        analysis['limitations']='构件身份按逐图 CSV 或统一输入，仅为人工提供的信息；模拟毫米尺寸不是实测。'
        analysis['suggestions']='逐图核实构件编号、病害性状和尺度参考；复核规范原件后再提出正式维修方案。'
    analysis['batch_processing_notice']='每种实际检出的病害按构件分别检索；各图定量和演示关注等级由程序直接生成。'
    if errors:analysis['pipeline_error']='；'.join(dict.fromkeys(errors))
    core.write_json(output_dir/'06_ai_analysis.json',analysis)
    core.write_json(output_dir/'07_per_image_diagnoses.json',[{'case_id':r.get('case_id'), 'relative_path':r.get('relative_path'), 'findings':r.get('inspection_findings',[]), 'presentation_findings':grouped_findings(r.get('inspection_findings',[]))} for r in records])
    return context,evidence,analysis,'；'.join(dict.fromkeys(errors))


def _small_figure(row, target:Path):
    """Generate a width-bounded original/overlay contact sheet with Pillow.

    Avoid embedding huge originals at full resolution; image bytes are recompressed
    locally without changing source images.
    """
    from PIL import Image,ImageOps,ImageDraw
    q = Path(str(row.get('quant_json') or ''))
    qdir=q.parent
    candidates=[Path(str(row.get('image_path') or '')) if row.get('image_path') else qdir/'01_center_crop.png',
                Path(str(row.get('overlay_path') or '')) if row.get('overlay_path') else qdir/'04_overlay.png']
    if not candidates[0].is_file():
        candidates[0]=Path(row['source_path'])
    available=[]
    for p in candidates:
        if p.is_file():
            try:
                with Image.open(p) as obj:
                    rgb=ImageOps.exif_transpose(obj).convert('RGB')
                    rgb.thumbnail((800,470), Image.Resampling.LANCZOS)
                    available.append(rgb.copy())
            except (OSError,ValueError):
                continue
    if not available:
        return None
    canvas=Image.new('RGB',(1600,525),(255,255,255))
    draw=ImageDraw.Draw(canvas)
    labels=['输入观察图','AI 分割叠加'] if len(available)>1 else ['检测图像']
    for i,img in enumerate(available[:2]):
        x=15+800*i+(800-img.width)//2
        y=30+(470-img.height)//2
        canvas.paste(img,(x,y))
        # Captions are added in Word; avoid PIL default-font glyph boxes for Chinese.
    if len(available)>1:
        draw.line([(800,0),(800,525)],fill=(205,215,220),width=2)
    target.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(target,'JPEG',quality=75,optimize=True)
    return target


def build_consolidated_docx(path:Path, manifest:dict, records:list, context:dict, evidence:list,
                             analysis:dict, *, rag_mode:str, embed_images:bool=True):
    """Engineer-facing one-document report, with source-aware measurements and references.

    Presentation is concise; unknown evidence never becomes a fabricated clause.
    All report figures remain generated from the input images, not reference PDFs.
    """
    from docx import Document
    from docx.shared import Cm, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc=Document()
    sec=doc.sections[0]
    sec.page_height,sec.page_width=Cm(29.7),Cm(21)
    sec.top_margin,sec.bottom_margin=Cm(1.6),Cm(1.45)
    sec.left_margin=sec.right_margin=Cm(1.85)
    sec.header_distance,sec.footer_distance=Cm(0.75),Cm(0.8)
    navy=RGBColor(29,69,95)
    muted=RGBColor(89,105,115)
    for style,size,bold in [('Normal',9,False),('Title',20,True),('Heading 1',13,True),('Heading 2',10.5,True)]:
        fs=doc.styles[style]
        fs.font.name='Microsoft YaHei';fs.font.size=Pt(size);fs.font.bold=bold
        fs._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
        if style!='Normal':fs.font.color.rgb=navy
    doc.styles['Normal'].paragraph_format.space_after=Pt(4)
    h=sec.header.paragraphs[0]
    h.text='BRIDGE AI     桥梁表观病害智能检测报告'
    h.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    for run in h.runs:run.font.size=Pt(8);run.font.color.rgb=muted
    footer=sec.footer.paragraphs[0]
    footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run('桥梁病害智能检测报告  ·  第 ')
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
    for run in footer.runs:run.font.size=Pt(8);run.font.color.rgb=muted

    def p(text='',keep=False):
        pp=doc.add_paragraph(str(text));pp.paragraph_format.keep_with_next=keep
        return pp
    def heading(text,level=1):
        pp=doc.add_heading(text,level);pp.paragraph_format.keep_with_next=True
        return pp
    def table(heads,rows):
        if not rows:return
        tt=doc.add_table(rows=1,cols=len(heads))
        tt.style='Table Grid';tt.alignment=WD_TABLE_ALIGNMENT.CENTER
        tt.autofit=True
        trpr=tt.rows[0]._tr.get_or_add_trPr()
        f=OxmlElement('w:tblHeader');f.set(qn('w:val'),'true');trpr.append(f)
        for cell,v in zip(tt.rows[0].cells,heads):cell.text=str(v)
        for ri,row in enumerate(rows):
            cc=tt.add_row().cells
            for cell,val in zip(cc,row):cell.text=str(val)
        for ri,row in enumerate(tt.rows):
            for cell in row.cells:
                cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                for pp in cell.paragraphs:
                    pp.paragraph_format.space_after=Pt(2)
                    for run in pp.runs:
                        run.font.name='Microsoft YaHei';run.font.size=Pt(8)
                        run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
                        if ri==0:run.bold=True
        doc.add_paragraph().paragraph_format.space_after=Pt(0)

    valid=[r for r in records if isinstance(r.get('measurements'),dict)]
    failed=[r for r in records if not isinstance(r.get('measurements'),dict)]
    findings=[(r,f) for r in valid for f in grouped_findings(r.get('inspection_findings', []))]
    simulated=manifest.get('scale_mode')=='simulation'
    ttl=doc.add_paragraph(style='Title');ttl.alignment=WD_ALIGN_PARAGRAPH.CENTER
    ttl.add_run('桥梁表观病害智能检测与处置分析报告')
    pp=p('一个工程 · 多张图片 · 统一结果',keep=True)
    pp.alignment=WD_ALIGN_PARAGRAPH.CENTER
    facts=[('工程名称',context.get('project_name')),('桥梁类型',context.get('structure_type')),
           ('构件类型',context.get('component_type')),('构件编号',context.get('component_id')),
           ('输入图片',len(records)),('完成检测',len(valid)),('处理异常',len(failed)),
           ('检出病害记录',len(findings)),('报告生成时间',datetime.now().strftime('%Y-%m-%d %H:%M'))]
    table(['项目','结果'],[(k,str(v)) for k,v in facts if meaningful(v) or isinstance(v,int)])
    if simulated:
        p('尺寸数据说明：输入示例换算系数为 '+str(manifest.get('demo_mm_per_px'))+
          ' mm/px；毫米及平方毫米数值为软件换算结果，不属于现场实测。')
    else:
        p('尺寸数据说明：本批次没有物理尺度输入，报告使用像素单位。')

    heading('一、病害识别统计')
    p('展示口径：背景＋裂缝＋剥落＋钢筋相关病害。钢筋相关病害由锈蚀与钢筋外露两类汇总；模型仍为原始五分类，细分数据保留在原始量化 JSON。')
    counters={code:sum(any(f.get('damage_code')==code for f in grouped_findings(row.get('inspection_findings',[]))) for row in valid)
              for code in DISPLAY_NAMES}
    table(['检测项目','阳性图片数'],[(DISPLAY_NAMES[c],str(k)) for c,k in counters.items() if k>0] or [('目标病害','0')])
    p('以上按图片统计。同一病害跨图重复出现时，不作为新的独立病害计数。')

    heading('二、病害定量结果与处置建议')
    rows=[]
    for i,r in enumerate(records,1):
        if not r.get('measurements'):
            rows.append((f'P-{i:03}',str(r.get('relative_path') or ''),'处理异常','','','详见批量任务日志'))
            continue
        fs=grouped_findings(r.get('inspection_findings', []))
        if not fs:
            rows.append((f'P-{i:03}',str(r.get('relative_path') or ''),'未检出目标病害','','','—'))
            continue
        for f in fs:
            rows.append((f'P-{i:03}',str(r.get('relative_path') or ''),
                         f['damage_type'],f.get('measurement_text') or '',
                         (f.get('demo_attention_level') or '—'), f.get('suggestion') or ''))
    table(['图号','来源图片','检出病害','长度 / 宽度 / 面积','关注级别','处置建议'],rows)
    p('关注级别：高 / 中 / 低由程序规则生成，用于排查优先级，不代表桥梁技术状况评定等级。')

    heading('三、逐图检测记录与影像')
    figures=path.parent/'_report_figures'
    for i,r in enumerate(records,1):
        heading(f'照片 P-{i:03}  {r.get("relative_path") or "检测图片"}',2)
        ctx=r.get('engineering_context') or {}
        labels=[v for v in (ctx.get('component_type'),ctx.get('component_id'),ctx.get('component_location')) if meaningful(v)]
        if labels:p('构件：'+' / '.join(map(str,labels)))
        fs=grouped_findings(r.get('inspection_findings', []))
        for f in fs:
            p(f['damage_type']+'：'+(f.get('measurement_text') or '')+'。处置建议：'+(f.get('suggestion') or ''),keep=embed_images)
        if not fs and r.get('measurements'):p('本图未检出目标病害。')
        if not r.get('measurements'):p('本图未完成检测，具体错误见批量运行记录。')
        if embed_images and r.get('measurements'):
            try:
                fig=_small_figure(r,figures/f'P-{i:03}.jpg')
                if fig is not None:
                    pic=doc.add_paragraph()
                    pic.alignment=WD_ALIGN_PARAGRAPH.CENTER
                    pic.add_run().add_picture(str(fig),width=Cm(15.7))
                    kinds=[]
                    if r.get('source_path') or r.get('image_path'):kinds.append('输入图像')
                    if r.get('overlay_path') or r.get('quant_json'):kinds.append('AI 分割图')
                    p('照片 P-'+f'{i:03}'+'：'+'与'.join(kinds or ['检测图像']).replace('与AI','与 AI')+'。')
            except (OSError,ValueError) as exc:
                p('影像文件读取失败：'+str(exc))

    heading('四、规范资料与处置分析')
    # Avoid invented clause numbers. If clause is missing, omit it rather than claiming verification.
    source_rows=[];seen=set()
    for ev in evidence:
        damage=ev.get('damage_type') or ''
        standard=(ev.get('standard_id') or ev.get('file_name') or '').strip()
        clause=str(ev.get('clause_id') or '').strip()
        if clause in ('未核实','未知','未确认','待核实','候选条款'):clause=''
        if not standard or not re.search(r'(JTG|CJJ|GB|JTJ|DB\s*\d|\.txt$|\.pdf$)',standard,re.I):
            # Keep raw retrieval JSON for traceability; omit non-identifying placeholders from Word.
            continue
        key=(damage,standard,clause)
        if key in seen:continue
        seen.add(key)
        label=damage if damage in DAMAGE_LABELS.values() else str(damage)
        # Keep a real source name. Do not render empty clause numbers or dash placeholders.
        source_text = standard + (' · 第'+clause+'条' if clause else '')
        source_rows.append((label,source_text))
        if len(source_rows)>=24:break
    if source_rows:
        table(['病害类型','检索关联规范资料'],source_rows)
        p('资料说明：规范来源由检索程序记录；现场适用性以标准正式文本和构件资料为准。')
    else:
        p('本批次未取得规范检索文本。下列处置建议按病害类型给出，不作为规范符合性结论。')

    heading('五、智能分析与建议')
    # Structured measurements above are the source of truth; free text never changes them.
    overview=analysis.get('summary')
    if meaningful(overview):
        overview=str(overview)
        if analysis.get('source')=='rule_based_no_llm' and '假设比例尺' in overview:
            overview=f'已处理 {len(valid)} 张图片，取得 {len(findings)} 项图像病害记录。'
        p('总体检测情况：'+overview)
    # Deterministic per-damage action table remains present even if Ollama fails or is disabled.
    per_damage=[]
    for code,count in counters.items():
        if count:
            sugg=next((f['suggestion'] for r,f in findings if f['damage_code']==code),'')
            per_damage.append((DISPLAY_NAMES[code],str(count),sugg))
    table(['病害类别','涉及图片','处置方向'],per_damage)
    if analysis.get('source') not in (None,'rule_based_no_llm'):
        for key,label in [('evidence_analysis','AI 规范分析'),('suggestions','AI 处置分析')]:
            if meaningful(analysis.get(key)):p(label+'：'+str(analysis[key]))
    if analysis.get('llm_status') == 'failed':
        p('智能分析模块运行未完成；本次表格的量化结果来自图像算法，处置方向来自预置规则。')
    p('报告范围：本结果针对所输入的图像与构件资料，不包含整桥评分或结构安全结论。')

    doc.core_properties.author='Bridge AI'
    doc.core_properties.title='桥梁病害智能检测与处置分析报告'
    doc.core_properties.subject='Image-based AI inspection report; example calibration values are not field measurements'
    doc.save(path)
    return path

def create_batch_report(a,batch_dir:Path,manifest:dict):
    report_dir=batch_dir/'report'
    report_dir.mkdir(parents=True,exist_ok=True)
    scale=float(a.demo_mm_per_px) if a.scale_mode=='simulation' else None
    entries=collect_measurements(manifest['rows'],a.scale_mode,scale)
    # Unknown/corrupt JSON entries remain explicitly failed, not silently omitted.
    context,evidence,analysis,analysis_error=obtain_batch_analysis(a,report_dir,entries)
    manifest['combined_report']={
        'path':str(report_dir/'00_batch_bridge_report.docx'),
        'rag_mode':a.rag_mode,'llm_mode':a.llm_mode,
        'llm_source':analysis.get('source'),
        'llm_status':analysis.get('llm_status'),
        'status':'partial' if analysis_error else 'completed',
        'warning':analysis_error,
        'images_in_report':len(entries),
        'retrieval_strategy':getattr(a,'retrieval_strategy','auto'),
        'dynamic_group_count':len({gid for x in entries for gid in x.get('rag_groups',[])}),
        'valid_quantifications':sum(x.get('measurements') is not None for x in entries),
    }
    core.write_json(report_dir/'01_batch_merged_measurements.json',{
        'schema_version':'batch-measurements-1.0',
        'root':manifest['root'],'calibration_mode':a.scale_mode,
        'mm_per_pixel_assumed':scale,'engineering_context':context,
        'entries':entries,
    })
    result=build_consolidated_docx(Path(manifest['combined_report']['path']),manifest,entries,context,
                                   evidence,analysis,rag_mode=a.rag_mode)
    # Copy per-image dynamic retrieval provenance into the public manifest.
    for raw,enriched in zip(manifest['rows'],entries):
        raw['component_type']=(enriched.get('engineering_context') or {}).get('component_type','未核实')
        raw['detected_damage_types']='、'.join(DAMAGE_LABELS.get(c,c) for c in enriched.get('detected_damage_codes',[])) or '未检出'
        raw['rag_group_ids']=','.join(enriched.get('rag_groups',[]))
    return result, manifest
