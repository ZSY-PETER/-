# -*- coding: utf-8 -*-
"""Multi-disease dynamic normative RAG planning.

Never infers a bridge component from image pixels or image filename.
Results are candidate sources only, not verified code applicability.
"""
from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path

DAMAGE_LABELS = {'crack':'裂缝','spalling':'混凝土剥落',
                 'corrosion':'表面锈蚀','exposed_rebar':'钢筋外露'}
TERMS = {
 'crack':'混凝土裂缝 裂纹 裂缝长度 宽度 病害分级 检查与处置',
 'spalling':'混凝土剥落 破损 掉角 面积 评定与养护修补',
 'corrosion':'混凝土表面锈迹 钢筋锈蚀 表观检查 技术状况评定 养护',
 'exposed_rebar':'钢筋外露 露筋 混凝土保护层损坏 评定及修补',
}
UNKNOWN={'','未知','未核实','待核实','待确认','未确认','不确定','盖梁/系梁'}


def member_name(value):
    s=str(value or '').strip()
    if s in UNKNOWN or any(t in s for t in ['未核实','待核实','演示设定']):
        return ''
    return s


def detected_damages(measurements):
    if not isinstance(measurements,dict):return []
    result=[]
    if (measurements.get('pixel_metrics') or {}).get('pixel_area_px2',0)>0:
        result.append('crack')
    for c in measurements.get('other_class_pixel_metrics') or []:
        if c.get('code') in DAMAGE_LABELS and (c.get('retained_pixels_after_filter') or 0)>0:
            result.append(c['code'])
    return list(dict.fromkeys(result))


def normpath(s):
    return str(s or '').strip().replace('\\','/').casefold()


def load_image_metadata(csv_path, paths):
    if not csv_path:return {}
    f=Path(csv_path).expanduser()
    if not f.is_file():raise FileNotFoundError('逐图构件 CSV 不存在：'+str(f))
    valid={normpath(p) for p in paths}
    found={}
    with f.open('r',encoding='utf-8-sig',newline='') as fh:
        r=csv.DictReader(fh)
        if not {'relative_path','component_type'}.issubset(set(r.fieldnames or [])):
            raise ValueError('逐图构件 CSV 必须包含 relative_path,component_type')
        for n,obj in enumerate(r,2):
            key=normpath(obj.get('relative_path'))
            if not key:continue
            if key in found:raise ValueError(f'CSV 第{n}行重复 relative_path: {key}')
            if key not in valid:raise ValueError(f'CSV 第{n}行指定的图片不在根目录扫描结果中：{key}')
            found[key]={k:str(v or '').strip() for k,v in obj.items() if k!='relative_path' and k}
    return found


def engineering_for_image(row, base, meta):
    vals=meta.get(normpath(row.get('relative_path')))
    context=dict(base)
    if vals is None:
        context['component_type_source']='批量统一输入或未知，未逐图核实'
    else:
        context.update({
            'component_type':vals.get('component_type') or '未核实',
            'component_id':vals.get('component_id') or '未核实',
            'component_location':vals.get('location') or vals.get('component_location') or '未核实',
            'crack_orientation':vals.get('orientation') or '未核实',
            'crack_type':vals.get('crack_type') or '未核实',
            'component_type_source':'逐图 CSV 人工填写，未独立核实'})
    return context


def make_request(code,component='',*,strategy='auto',standard_id='',clause_id='',top_k=5):
    if code not in DAMAGE_LABELS:raise ValueError('未知病害类别：'+str(code))
    if strategy not in ('auto','exact'):raise ValueError('未知检索策略：'+str(strategy))
    comp=member_name(component)
    if strategy=='exact':
        std=str(standard_id or '').strip(); clause=str(clause_id or '').strip()
        if not std or not clause:raise ValueError('精确模式必须指定标准编号和条款')
        query=f'公路桥梁 {comp or "构件未知"} {DAMAGE_LABELS[code]} {std} 第{clause}条 评定依据'
    else:
        query=f'公路桥梁 {comp+" " if comp else "构件未确认 "}{TERMS[code]} 适用构件 相关检测评定及处置要求'
    payload={'query':query,'top_k':int(top_k), 'component_type':comp or '未核实',
             'damage_type':DAMAGE_LABELS[code], 'retrieval_strategy':strategy}
    if strategy=='exact':payload.update({'standard_id':std,'clause_id':clause})
    return payload


def plan_requests(records,*,strategy='auto',standard_id='',clause_id='',top_k=5):
    groups=OrderedDict()
    for row in records:
        codes=detected_damages(row.get('measurements'))
        row['detected_damage_codes']=codes
        component=member_name((row.get('engineering_context') or {}).get('component_type'))
        row['confirmed_component_for_query']=component or None
        row['rag_groups']=[]
        for code in codes:
            key=(code,component)
            if key not in groups:
                groups[key]={'group_id':f'R{len(groups)+1:03}', 'damage_code':code,
                             'damage_type':DAMAGE_LABELS[code],
                             'component_type':component or '未核实',
                             'component_identity_verified':False,
                             'request':make_request(code,component,strategy=strategy,
                                 standard_id=standard_id,clause_id=clause_id,top_k=top_k),
                             'image_case_ids':[]}
            g=groups[key]
            if row.get('case_id') not in g['image_case_ids']:
                g['image_case_ids'].append(row.get('case_id'))
            row['rag_groups'].append(g['group_id'])
    return list(groups.values())


def summarize_group_evidence(group,rag,core):
    if rag.get('success') is not True or not rag.get('results'):
        return []
    try: items=core.evidence_summary(rag)
    except (ValueError,RuntimeError):return []
    result=[]
    for item in items:
        ev=dict(item)
        ev.update({'group_id':group['group_id'],'damage_code':group['damage_code'],
                   'damage_type':group['damage_type'],'target_component':group['component_type'],
                   'image_case_ids':group['image_case_ids'],
                   'applicability_status':'requires_engineering_review'})
        result.append(ev)
    return result
