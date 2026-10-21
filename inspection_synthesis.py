# -*- coding: utf-8 -*-
"""Create auditable, non-normative demo-priority findings from measured image masks.

The student competition prototype has NO validated real-world calibration; example
mm/mm2 figures and attention levels are demonstration-only. No thresholds are
claimed to be JTG normative criteria. Never infer a bridge position from images.
"""
from __future__ import annotations
import math
from dynamic_rag import DAMAGE_LABELS, detected_damages

PLACEHOLDERS={'','未核实','未提供','未知','待核实','待确认','未确认','不确定','/', '无'}


def meaningful(value):
    if value is None or str(value).strip() in PLACEHOLDERS:return False
    s=str(value)
    return not any(k in s for k in ('实际桥位未核实','演示构件观察面','待核实','人工填写','未提供','未经核实'))


def n(value,digits=2,unit=''):
    if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value):return None
    return f'{value:.{digits}f}{unit}'


def priority(code, width_mm, crop_percent):
    """Software-demo attention bands, NOT an engineering code assessment."""
    if code=='exposed_rebar':return '高'
    if code=='crack':
        if width_mm is None:return '—'
        return '高' if width_mm>=0.30 else '中' if width_mm>=0.15 else '低'
    if crop_percent is None:return '—'
    if code=='spalling':return '高' if crop_percent>=10 else '中' if crop_percent>=2 else '低'
    return '高' if crop_percent>=5 else '中' if crop_percent>=1 else '低'


SUGGESTIONS={
    'crack':'记录裂缝宽度与发展情况；结合构件条件选择表面封闭、注浆或持续监测。',
    'spalling':'测定剥落范围，清理松动混凝土，恢复表面防护层及损坏区域。',
    'corrosion':'检查渗水和锈蚀位置；清理锈蚀表面并采取防护措施，必要时开展钢筋检测。',
    'exposed_rebar':'检查钢筋状态和保护层损失，进行钢筋防护与混凝土保护层修补。',
}


def findings_for_measurement(measured, case_id=''):
    codes=detected_damages(measured)
    z=measured.get('pixel_metrics') or {}
    physical=measured.get('physical_metrics') or {}
    cal=measured.get('calibration') or {}
    scale=cal.get('mm_per_pixel') if cal.get('mode')=='simulation' else None
    others={c.get('code'):c for c in measured.get('other_class_pixel_metrics') or []}
    found=[]
    for code in codes:
        display=[];metric_values={}
        if code=='crack':
            count=z.get('candidate_connected_regions')
            if isinstance(count,int) and count>0:display.append(f'{count} 个分割连通区域')
            length_px=z.get('visible_main_path_length_px_approx')
            width_px=z.get('normal_width_mean_px_approx')
            max_px=z.get('normal_width_max_px_approx')
            area_px=z.get('pixel_area_px2')
            length_mm=physical.get('visible_main_path_length_mm_demo')
            width_mm=physical.get('normal_width_mean_mm_demo')
            max_mm=physical.get('normal_width_max_mm_demo')
            projected_mm2=physical.get('projected_crack_area_mm2_demo')
            for label,val,prec,unit in [('可见长度',length_mm,2,' mm'),('平均宽度',width_mm,3,' mm'),('最大宽度',max_mm,3,' mm'),('投影面积',projected_mm2,2,' mm²')]:
                text=n(val,prec,unit)
                if text:display.append(label+' '+text)
            if not scale:
                for label,val,dig,unit in [('可见长度',length_px,1,' px'),('平均宽度',width_px,2,' px'),('投影面积',area_px,0,' px²')]:
                    v=n(val,dig,unit)
                    if v:display.append(label+' '+v)
            if z.get('visible_only_not_entire_crack'):display.append('图像边缘处裂缝被截断，所列长度为图内长度')
            crop=z.get('observed_crop_percent')
            metric_values={'length_px':length_px,'width_mean_px':width_px,'width_max_px':max_px,'area_px2':area_px,
                'length_mm_demo':length_mm,'width_mean_mm_demo':width_mm,'width_max_mm_demo':max_mm,'area_mm2_demo':projected_mm2}
        else:
            item=others.get(code) or {}
            area_px=item.get('retained_pixels_after_filter')
            crop=item.get('retained_area_percent_of_crop')
            area_mm2=round(area_px*scale**2,5) if scale and isinstance(area_px,(int,float)) else None
            if area_mm2 is not None:display.append('投影面积 '+n(area_mm2,2,' mm²'))
            else:
                if n(area_px,0,' px²'):display.append('像素面积 '+n(area_px,0,' px²'))
            if n(crop,2,'%'):display.append('观察图占比 '+n(crop,2,'%'))
            metric_values={'area_px2':area_px,'area_mm2_demo':area_mm2,'crop_percent':crop}
            width_mm=None
        display=[v for v in display if v and 'None' not in v]
        level=priority(code, width_mm if code=='crack' else None, crop)
        found.append({'image_case_id':case_id,'damage_code':code,'damage_type':DAMAGE_LABELS[code],
            'measurement_text':'；'.join(display),'measurements':metric_values,
            'demo_attention_level':level,'suggestion':SUGGESTIONS[code],
            'measurements_are_demo':bool(scale),'normative_rating_verified':False})
    return found


def annotate_records(records):
    for row in records:
        m=row.get('measurements')
        row['inspection_findings']=findings_for_measurement(m,row.get('case_id','')) if isinstance(m,dict) else []
    return records
