# -*- coding: utf-8 -*-
"""Presentation-only grouping: corrosion + exposed_rebar -> steel_related_damage.

Native five-class model masks, labels, geometry, per-class metadata and RAG requests
remain unchanged. This does NOT merge training labels or change original metrics.
"""
from __future__ import annotations

DISPLAY_NAMES = {
    'crack': '裂缝',
    'spalling': '混凝土剥落',
    'steel_related_damage': '钢筋相关病害（锈蚀／外露）',
}
RAW_STEEL = ('corrosion', 'exposed_rebar')


def grouped_quantification(raw_report):
    classes = {c.get('code'): c for c in raw_report.get('classes', [])}
    entries = []
    for group_code, source_codes in (
        ('crack', ('crack',)),
        ('spalling', ('spalling',)),
        ('steel_related_damage', RAW_STEEL),
    ):
        items = [classes.get(c) or {} for c in source_codes]
        pixels = sum(int(x.get('retained_pixels_after_filter') or 0) for x in items)
        raw_pixels = sum(int(x.get('raw_predicted_pixels_in_roi') or 0) for x in items)
        base = raw_report.get('roi_observed_pixel_count') or 0
        entries.append({
            'code': group_code,
            'name': DISPLAY_NAMES[group_code],
            'source_class_ids': [x['class_id'] for x in items if 'class_id' in x],
            'retained_pixels_after_filter': pixels,
            'raw_predicted_pixels_in_roi': raw_pixels,
            'retained_area_percent': round(100.0 * pixels / base, 4) if base else None,
            'positive': pixels > 0,
        })
    return entries


def grouped_findings(raw_findings):
    """Merge report rows while retaining original findings as provenance."""
    raw_findings = list(raw_findings or [])
    answer = [dict(f) for f in raw_findings if f.get('damage_code') not in RAW_STEEL]
    steel = [f for f in raw_findings if f.get('damage_code') in RAW_STEEL]
    if steel:
        parts = []
        total_area_px = 0
        total_area_mm = 0.0
        has_mm = False
        for finding in steel:
            metrics = finding.get('measurements') or {}
            px = metrics.get('area_px2')
            mm = metrics.get('area_mm2_demo')
            if isinstance(px, (int, float)):
                total_area_px += int(px)
                parts.append(f"{finding.get('damage_type', finding.get('damage_code'))} {int(px)} px²")
            else:
                parts.append(str(finding.get('damage_type', finding.get('damage_code'))))
            if isinstance(mm, (int, float)):
                has_mm = True
                total_area_mm += float(mm)
        high = any(f.get('demo_attention_level') == '高' for f in steel)
        answer.append({
            'image_case_id': steel[0].get('image_case_id', ''),
            'damage_code': 'steel_related_damage',
            'damage_type': DISPLAY_NAMES['steel_related_damage'],
            'source_codes': [f['damage_code'] for f in steel],
            'measurement_text': f"组合像素面积 {total_area_px} px²；分项：" + '；'.join(parts),
            'measurements': {
                'area_px2': total_area_px,
                'area_mm2_demo': round(total_area_mm, 5) if has_mm else None,
            },
            'demo_attention_level': '高' if high else steel[0].get('demo_attention_level', '—'),
            'suggestion': '核查钢筋锈蚀与外露及混凝土保护层状态，必要时人工复检。',
            'measurements_are_demo': any(f.get('measurements_are_demo') for f in steel),
            'normative_rating_verified': False,
        })
    return answer


def present_damage_codes(raw_codes):
    values=[]
    for code in raw_codes:
        name = 'steel_related_damage' if code in RAW_STEEL else code
        if name not in values: values.append(name)
    return values
