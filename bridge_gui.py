# -*- coding: utf-8 -*-
"""Windows Tk/ttk front-end for Bridge Contest V2.2. Resizable, threaded, logged."""
from __future__ import annotations
from bridge_paths import default_checkpoint, default_vmamba_models
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
import tkinter as tk
from utf8_process import utf8_child_env, set_utf8_stdio
from tkinter import ttk, filedialog, messagebox

BASE = Path(__file__).resolve().parent
HOME = Path.home()
FONT = 'Microsoft YaHei' if sys.platform == 'win32' else 'DejaVu Sans'
COLOR_NAVY = '#153B52'
COLOR_MAIN = '#0D607B'
COLOR_TEXT = '#253F50'
COLOR_MUTED = '#647989'


def dpi_fix():
    if sys.platform == 'win32':
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


class BridgeGUI(tk.Tk):
    def __init__(self):
        dpi_fix()
        super().__init__()
        self.title('Bridge AI  ·  桥梁病害智能量化与规范辅助报告  V3.0（最终整合版）')
        self.geometry('1330x835')
        self.minsize(1000, 620)
        self.configure(bg='#EDF3F6')
        self.protocol('WM_DELETE_WINDOW', self.close)
        self._setup_style()
        self.running_proc = None
        self.worker = None
        self.events = queue.Queue()
        self.photo_ref = None
        self.image_current = None
        self._resize_call = None
        self.last_batch_dir = None
        self.last_batch_counts = None
        self.batch_resume = tk.BooleanVar(value=True)
        self.batch_include_masks = tk.BooleanVar(value=False)
        self.vars = {}
        defaults = {
            'image': str(BASE/'examples/test_0067_center_crop.png'),
            'batch_root': str(HOME/'Desktop/HRCDS/test_image'), 'batch_action': 'all',
            'batch_summary_path': '',
            'mask': '',
            'quant_json': str(BASE/'examples/09_quantification.json'),
            'overlay': str(BASE/'examples/test_0067_overlay.png'),
            'checkpoint': str(default_checkpoint()),
            'vmamba_models': str(default_vmamba_models()),
            'out': str(BASE/'outputs'),
            'db_dir': str(HOME/'Desktop/新建文件夹 (2)'),
            'rag_script': str(BASE/'rag_query.py'),
            'rag_json': str(HOME/'Desktop/bridge_rag_v2_check_output.json'),
            'scale_mode': 'simulation', 'demo_mm_per_px': '0.02',
            'rag_mode': 'live', 'llm_mode': 'live',
            'ollama_url': 'http://127.0.0.1:11434', 'ollama_model': 'qwen3:8b',
            'case_id': 'DEMO-CASE-0067', 'project': '桥梁表观病害智能检测与辅助分析',
            'structure': '公路桥梁', 'component': '',
            'component_id': '', 'location': '', 'orientation': '', 'crack_type': '',
            'standard_id': '', 'clause_id': '',
            'retrieval_strategy': 'auto', 'batch_metadata_csv': ''
        }
        for key, value in defaults.items():
            self.vars[key] = tk.StringVar(value=value)
        self.status = tk.StringVar(value='就绪 · 示例数据已载入，选择运行模式后开始')
        self.progress_mode = tk.StringVar(value='尚未运行')
        self.batch_scan_text=tk.StringVar(value='尚未扫描。将递归查找所有子目录内的图片。')
        self._create_layout()
        self.after(100, self._poll)
        self.after(250, self._refresh_preview)
        self._log('V3.0 GUI 已启动：动态病害检索、具体量化指标、逐图处置建议；整批唯一 Word。')
        self._log('注意：示例尺度计算值与现场实测数据分开记录。')
        self._log('类别说明：原始模型为5分类；汇总展示为背景＋裂缝＋剥落＋钢筋相关病害。')

    def _setup_style(self):
        st=ttk.Style(self)
        st.theme_use('clam')
        st.configure('.', font=(FONT, 10))
        st.configure('TFrame', background='#F7FAFB')
        st.configure('TLabel', foreground=COLOR_TEXT, background='#F7FAFB')
        st.configure('Title.TLabel', font=(FONT,17,'bold'), foreground='#FFFFFF', background=COLOR_NAVY)
        st.configure('Subtitle.TLabel', font=(FONT,9), foreground='#C8E0EB', background=COLOR_NAVY)
        st.configure('Section.TLabel', font=(FONT,11,'bold'), foreground=COLOR_NAVY, background='#F7FAFB')
        st.configure('Dim.TLabel', font=(FONT,9), foreground=COLOR_MUTED, background='#F7FAFB')
        st.configure('TButton', padding=(9,6), background='#EDF2F6', foreground=COLOR_TEXT)
        st.map('TButton', background=[('active','#DCECF2')])
        st.configure('Accent.TButton', padding=(12,9), background=COLOR_MAIN, foreground='white',font=(FONT,10,'bold'))
        st.map('Accent.TButton', background=[('active','#107A96'),('disabled','#9BACB6')])
        st.configure('Warn.TButton', padding=(10,9),background='#DCE4E9')
        st.configure('TEntry', padding=(7,4))
        st.configure('TNotebook', background='#F7FAFB')
        st.configure('TNotebook.Tab', padding=(13,8))
        st.configure('Treeview', rowheight=30, font=(FONT,9))
        st.configure('Treeview.Heading', font=(FONT,9,'bold'))

    def _create_layout(self):
        banner=tk.Frame(self,bg=COLOR_NAVY,height=77)
        banner.pack(side='top',fill='x')
        banner.pack_propagate(False)
        tk.Label(banner,text='BRIDGE AI   /   桥梁表观病害智能评估',font=(FONT,17,'bold'),bg=COLOR_NAVY,fg='white').pack(anchor='w',padx=22,pady=(9,0))
        tk.Label(banner,text='V3.0  ·  批量识别 → 几何量化 → 尺度换算 → 规范检索 → 智能分析 → 一份 Word',font=(FONT,9),bg=COLOR_NAVY,fg='#C2D8E5').pack(anchor='w',padx=23,pady=(0,5))

        paned=ttk.Panedwindow(self,orient='horizontal')
        paned.pack(fill='both',expand=True,padx=11,pady=9)
        self.paned=paned
        left=ttk.Frame(paned)
        right=ttk.Frame(paned)
        paned.add(left, weight=4)
        paned.add(right, weight=7)
        self._form_side(left)
        self._preview_side(right)

        statusbar=tk.Frame(self,bg='#DBE6EC',height=32)
        statusbar.pack(side='bottom',fill='x')
        tk.Label(statusbar,textvariable=self.status,anchor='w',font=(FONT,9),bg='#DBE6EC',fg=COLOR_NAVY).pack(side='left',fill='x',expand=True,padx=12)
        tk.Label(statusbar,text='软件辅助分析 · 数据来源可追溯',font=(FONT,9,'bold'),bg='#DBE6EC',fg='#965737').pack(side='right',padx=13)

    def _form_side(self, parent):
        parent.columnconfigure(0,weight=1)
        parent.rowconfigure(0,weight=1)
        canvas=tk.Canvas(parent,bg='#F7FAFB',highlightthickness=0)
        scroll=ttk.Scrollbar(parent,orient='vertical',command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.grid(row=0,column=0,sticky='nsew')
        scroll.grid(row=0,column=1,sticky='ns')
        f=ttk.Frame(canvas,padding=(14,11))
        self.inner=f
        cw=canvas.create_window((0,0),window=f,anchor='nw')
        f.bind('<Configure>',lambda e:canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',lambda e:canvas.itemconfigure(cw,width=e.width))
        def wheel(e):
            canvas.yview_scroll(int(-1*(e.delta/120)), 'units')
        canvas.bind('<Enter>',lambda e:canvas.bind_all('<MouseWheel>',wheel))
        canvas.bind('<Leave>',lambda e:canvas.unbind_all('<MouseWheel>'))
        f.columnconfigure(1,weight=1)
        self._section('A / 输入数据与模型')
        self._entry('原始图片', 'image', file=True, patterns=[('图片','*.png *.jpg *.jpeg *.bmp')])
        self._entry('现有 Mask（可选）', 'mask', file=True, patterns=[('类别ID Mask','*.png')])
        self._entry('量化 JSON', 'quant_json', file=True, patterns=[('JSON','*.json')])
        self._entry('叠加图（报告）', 'overlay', file=True, patterns=[('图片','*.png *.jpg *.jpeg')])
        self._entry('模型权重(.pth)', 'checkpoint', file=True, patterns=[('PyTorch','*.pth *.pt')])
        self._entry('官方 VMamba 目录', 'vmamba_models', folder=True)
        self._section('A2 / 批量文件夹（递归遍历子目录）')
        self._entry('图片根目录', 'batch_root', folder=True)
        self._combo('批量执行模式', 'batch_action', ['all','quantify','report'])
        self._entry('逐图构件CSV(可选)', 'batch_metadata_csv',file=True,patterns=[('CSV','*.csv')])
        self._entry('已有批次汇总JSON', 'batch_summary_path',file=True,patterns=[('JSON','*.json')])
        ttk.Label(f,text='all：逐图量化→整批唯一 Word；report：已有量化→整批唯一 Word；quantify：只算 JSON',style='Dim.TLabel',wraplength=415).grid(row=self._nextrow(),column=0,columnspan=3,sticky='w',pady=(1,4))
        ttk.Checkbutton(f,text='跳过已完成且输入/配置未改变的图片（断点续跑）',variable=self.batch_resume).grid(row=self._nextrow(),column=0,columnspan=3,sticky='w',pady=(4,1))
        ttk.Checkbutton(f,text='也把 Mask/GT/label 图片作为输入（通常不要勾选）',variable=self.batch_include_masks).grid(row=self._nextrow(),column=0,columnspan=3,sticky='w',pady=(1,3))
        ttk.Label(f,textvariable=self.batch_scan_text,style='Dim.TLabel',wraplength=415).grid(row=self._nextrow(),column=0,columnspan=3,sticky='w',pady=(1,6))
        self._section('B / 工程信息（选填）')
        for lab,k in [('工程名称','project'),('案例编号','case_id'),('桥梁类型','structure'),('构件类别','component'),('构件编号','component_id')]:
            self._entry(lab,k)
        self._section('C / 尺度与规范配置')
        self._combo('物理尺度', 'scale_mode', ['simulation','none'])
        self._entry('示例换算系数(mm/px)','demo_mm_per_px')
        self._combo('RAG检索模式','rag_mode',['live','file','skip'])
        self._combo('规范检索策略','retrieval_strategy',['auto','exact'])
        ttk.Label(f,text='auto：按检测类型动态选规范，不锁定条款；exact：手动指定标准+条款。批量建议 auto+live。',style='Dim.TLabel',wraplength=410).grid(row=self._nextrow(),column=0,columnspan=3,sticky='w')
        self._combo('Qwen生成模式','llm_mode',['live','off'])
        self._entry('ChromaDB目录','db_dir',folder=True)
        self._entry('RAG检索脚本','rag_script',file=True,patterns=[('Python','*.py')])
        self._entry('离线检索缓存','rag_json',file=True,patterns=[('JSON','*.json')])
        self._entry('标准编号','standard_id')
        self._entry('目标条款','clause_id')
        self._entry('Ollama模型','ollama_model')
        self._entry('Ollama地址','ollama_url')
        self._entry('输出目录','out',folder=True)
        ttk.Label(f,text='状态说明：simulation = 示例尺度换算；none = 仅像素\nlive = 调用本机服务；file = 已有检索结果；skip = 不检索',style='Dim.TLabel').grid(row=self._nextrow(),column=0,columnspan=3,sticky='w',pady=9)

        actions=ttk.Frame(parent,padding=(8,9))
        actions.grid(row=1,column=0,columnspan=2,sticky='ew')
        for col in range(2):actions.columnconfigure(col,weight=1)
        self.buttons=[]
        for col,row,label,action,style in [(0,0,'① 单图分割 / 量化','quantify','TButton'),(1,0,'② 单图 RAG / 出报告','report','TButton'),(0,1,'单图一键完整流程','all','Accent.TButton'),(1,1,'停止当前任务','cancel','Warn.TButton'),(0,2,'③ 扫描根目录','scan_batch','TButton'),(1,2,'④ 批量处理 / 合并Word','batch','Accent.TButton'),(0,3,'⑤ 已有批次重新出Word','recover','TButton')]:
            b=ttk.Button(actions,text=label,style=style,command=(self.cancel if action=='cancel' else self.scan_batch if action=='scan_batch' else lambda a=action:self.run_action(a)))
            b.grid(row=row,column=col,sticky='ew',padx=4,pady=4)
            if action!='cancel':self.buttons.append(b)
        self.progress=ttk.Progressbar(actions,mode='indeterminate')
        self.progress.grid(row=4,column=0,columnspan=2,sticky='ew',padx=5,pady=(8,2))

    def _nextrow(self):
        if not hasattr(self,'formrow'):self.formrow=0
        r=self.formrow;self.formrow+=1;return r

    def _section(self,heading):
        f=self.inner
        row=self._nextrow()
        ttk.Label(f,text=heading,style='Section.TLabel').grid(row=row,column=0,columnspan=3,sticky='w',pady=(17,8))
        ttk.Separator(f).grid(row=self._nextrow(),column=0,columnspan=3,sticky='ew',pady=(0,4))

    def _entry(self, label, key, file=False, folder=False, patterns=None):
        row=self._nextrow()
        ttk.Label(self.inner,text=label,width=17).grid(row=row,column=0,sticky='w',padx=(0,7),pady=4)
        entry=ttk.Entry(self.inner,textvariable=self.vars[key])
        entry.grid(row=row,column=1,sticky='ew',padx=3,pady=4)
        if file or folder:
            ttk.Button(self.inner,text='…',width=3,command=lambda:self._browse(key,folder,patterns)).grid(row=row,column=2,padx=2)

    def _combo(self,label,key,opts):
        row=self._nextrow()
        ttk.Label(self.inner,text=label,width=17).grid(row=row,column=0,sticky='w',pady=4)
        ttk.Combobox(self.inner,textvariable=self.vars[key],values=opts,state='readonly').grid(row=row,column=1,sticky='ew',padx=3,pady=4)

    def _browse(self,key,folder,patterns):
        current=self.vars[key].get()
        where=str(Path(current).parent if current else HOME/'Desktop')
        if folder:
            chosen=filedialog.askdirectory(initialdir=where)
        else:
            chosen=filedialog.askopenfilename(initialdir=where,filetypes=patterns or [('所有文件','*.*')])
        if chosen:
            self.vars[key].set(chosen)
            if key=='batch_root':self.after(75,self.scan_batch)

    def _preview_side(self,parent):
        notebook=ttk.Notebook(parent)
        notebook.pack(fill='both',expand=True)
        self.tabs={}
        for name in ['图像预览','指标概览','运行日志']:
            tab=ttk.Frame(notebook,padding=12)
            notebook.add(tab,text=name)
            self.tabs[name]=tab
        p=self.tabs['图像预览']
        p.rowconfigure(1,weight=1);p.columnconfigure(0,weight=1)
        toolbar=ttk.Frame(p)
        toolbar.grid(row=0,column=0,sticky='ew',pady=(0,8))
        ttk.Label(toolbar,text='图片查看').pack(side='left')
        self.image_select=ttk.Combobox(toolbar,values=['原图','叠加图'],state='readonly',width=14)
        self.image_select.current(1)
        self.image_select.pack(side='left',padx=10)
        self.image_select.bind('<<ComboboxSelected>>',lambda e:self._refresh_preview())
        ttk.Button(toolbar,text='刷新',command=self._refresh_preview).pack(side='left')
        ttk.Button(toolbar,text='打开输出目录',command=self.open_output).pack(side='right')
        ttk.Button(toolbar,text='打开最新 Word',command=self.open_word).pack(side='right',padx=(0,8))
        self.image_canvas=tk.Canvas(p,bg='#E4EDF1',highlightthickness=0)
        self.image_canvas.grid(row=1,column=0,sticky='nsew')
        self.image_canvas.bind('<Configure>',lambda e:self._refresh_preview())
        self.preview_msg=ttk.Label(p,text='病害类别与像素尺寸由程序计算，毫米数值根据输入尺度换算。',style='Dim.TLabel')
        self.preview_msg.grid(row=2,column=0,sticky='w',pady=(7,0))

        summary=self.tabs['指标概览']
        summary.rowconfigure(0,weight=1);summary.columnconfigure(0,weight=1)
        self.stat_tree=ttk.Treeview(summary,columns=('item','val','status'),show='headings')
        for k,title,width in [('item','指标',225),('val','当前数据',165),('status','解释/状态',280)]:
            self.stat_tree.heading(k,text=title)
            self.stat_tree.column(k,width=width,anchor='w',stretch=True)
        self.stat_tree.grid(row=0,column=0,sticky='nsew')
        ys=ttk.Scrollbar(summary,orient='vertical',command=self.stat_tree.yview)
        self.stat_tree.configure(yscrollcommand=ys.set)
        ys.grid(row=0,column=1,sticky='ns')
        ttk.Button(summary,text='从量化 JSON 读取指标',command=self.refresh_stats).grid(row=1,column=0,sticky='w',pady=9)

        logs=self.tabs['运行日志']
        logs.rowconfigure(0,weight=1);logs.columnconfigure(0,weight=1)
        self.log_box=tk.Text(logs,wrap='word',bg='#122331',fg='#D9E9F2',font=('Consolas',10),relief='flat',padx=13,pady=11)
        self.log_box.grid(row=0,column=0,sticky='nsew')
        scrollbar=ttk.Scrollbar(logs,command=self.log_box.yview)
        scrollbar.grid(row=0,column=1,sticky='ns')
        self.log_box.configure(yscrollcommand=scrollbar.set)
        self.log_box.tag_configure('error',foreground='#FC9696')
        self.log_box.tag_configure('success',foreground='#89E9B1')
        self.log_box.tag_configure('info',foreground='#D9E9F2')
        ttk.Button(logs,text='清空日志',command=lambda:self.log_box.delete('1.0','end')).grid(row=1,column=0,sticky='w',pady=7)
        self.refresh_stats()

    def _log(self, s, tag='info'):
        self.log_box.insert('end',f'{time.strftime("%H:%M:%S")}  {s}\n',tag)
        if int(self.log_box.index('end-1c').split('.')[0]) > 18000:
            self.log_box.delete('1.0','3000.0')
        self.log_box.see('end')

    def _refresh_preview(self):
        try:
            from PIL import Image, ImageTk
        except ImportError:
            self.image_canvas.delete('all')
            self.image_canvas.create_text(50,50,text='预览需要 Pillow：pip install pillow',anchor='nw')
            return
        key='overlay' if self.image_select.get()=='叠加图' else 'image'
        path=Path(self.vars[key].get())
        if not path.is_file():
            self.image_canvas.delete('all')
            self.image_canvas.create_text(50,50,text='尚未选择有效图片：'+str(path.name),anchor='nw',fill='#3D596C')
            return
        try:
            with Image.open(path) as opened: rgb=opened.convert('RGB')
            w=max(100,self.image_canvas.winfo_width()-18);h=max(100,self.image_canvas.winfo_height()-18)
            rgb.thumbnail((w,h),Image.Resampling.LANCZOS)
            self.photo_ref=ImageTk.PhotoImage(rgb)
            self.image_canvas.delete('all')
            self.image_canvas.create_image(self.image_canvas.winfo_width()//2,self.image_canvas.winfo_height()//2,image=self.photo_ref,anchor='center')
        except Exception as ex:
            self._log('图片预览失败：'+str(ex),'error')

    def refresh_stats(self, path=None):
        p=Path(path or self.vars['quant_json'].get())
        self.stat_tree.delete(*self.stat_tree.get_children())
        if not p.is_file():return
        try:
            j=json.loads(p.read_text(encoding='utf-8-sig'))
            crack=next((x for x in j.get('classes',[]) if x.get('code')=='crack'),{})
            g=crack.get('crack_overall_geometry') or {}
            top=g.get('refined_topology') or {}
            nm=(g.get('width_px_normal_approx') or {}).get('summary') or {}
            items=[
                ('JSON Schema',j.get('schema_version'),'保留原始输入'),
                ('观测图片尺寸',str(j.get('image_size_wh')),'中心裁剪范围'),
                ('裂缝候选连通域数',crack.get('candidate_connected_region_count'),'不等于人工核查条数'),
                ('裂缝面积 / px²',crack.get('retained_pixels_after_filter'),'观察区投影面积'),
                ('裁剪画面占比 / %',crack.get('retained_area_percent'),'不是构件面积比'),
                ('主路径长度 / px',top.get('longest_endpoint_geodesic_length_px_approx'),'可见路径'),
                ('网络骨架长度 / px',top.get('pruned_graph_network_length_px_approx'),'不是主裂缝长度'),
                ('平均宽度 / px',nm.get('mean'),'法线估计'),
                ('最大宽度 / px',nm.get('max'),'法线估计'),
                ('毫米标定',j.get('calibration',{}).get('physical_units_available'),'False 表示原输入未标定')]
            # 仅展示层合并锈蚀与钢筋外露；原始 mask、类 ID、量化 JSON 保持五分类。
            from presentation_groups import grouped_quantification
            grouped = grouped_quantification(j)
            for group in grouped:
                items.append((group['name']+' / px²',
                              group['retained_pixels_after_filter'],
                              '来自原始五分类，未修改模型权重'))
            items.append(('类别展示说明', '背景＋3类病害',
                          '锈蚀/外露合并仅用于汇总，不代表识别精度提高'))
            for a,b,c in items:self.stat_tree.insert('', 'end', values=(a,b,c))
        except Exception as ex:
            self._log('解析量化 JSON 失败：'+str(ex),'error')

    def _args(self, action):
        if action == 'recover':
            cmd=[sys.executable, '-u', str(BASE/'regenerate_batch_report.py'),
                 '--summary', self.vars['batch_summary_path'].get().strip()]
            for key in ('rag_mode','llm_mode','retrieval_strategy','rag_script',
                        'db_dir','ollama_model','ollama_url','project','structure',
                        'component','component_id','batch_metadata_csv',
                        'standard_id','clause_id'):
                val=self.vars[key].get().strip()
                if val and not (key=='rag_mode' and val=='file'):
                    cmd.extend(['--'+key.replace('_','-'),val])
            return cmd
        is_batch = (action == 'batch')
        if is_batch:
            a=[sys.executable,'-u',str(BASE/'batch_runner.py'), '--root', self.vars['batch_root'].get().strip(),
               '--action', self.vars['batch_action'].get()]
        else:
            a=[sys.executable,'-u',str(BASE/'run_pipeline.py'),'--action',action]
        keys=['image','mask','quant_json','overlay','checkpoint','vmamba_models','out',
              'db_dir','rag_script','rag_json','scale_mode','rag_mode','llm_mode',
              'ollama_model','ollama_url','case_id','project','structure','component',
              'component_id','standard_id','clause_id',
              'retrieval_strategy','batch_metadata_csv']
        if is_batch:
            keys=[key for key in keys if key not in ('image','mask','quant_json','overlay')]
        else:
            keys=[key for key in keys if key!='batch_metadata_csv']
        for key in keys:
            val=self.vars[key].get().strip()
            if not val:continue
            a.extend(['--'+key.replace('_','-'),val])
        if self.vars['scale_mode'].get()=='simulation':
            a.extend(['--demo-mm-per-px',self.vars['demo_mm_per_px'].get().strip()])
        if is_batch:
            if self.batch_resume.get():a.append('--resume')
            if self.batch_include_masks.get():a.append('--include-masks')
        return a

    def scan_batch(self):
        if self.worker and self.worker.is_alive():
            messagebox.showwarning('任务运行中','请先等待当前批量/单图任务完成。')
            return
        root_str=self.vars['batch_root'].get().strip()
        if not root_str or not Path(root_str).is_dir():
            messagebox.showerror('无效目录','请先选择存在的图片根目录。')
            return
        include_masks=self.batch_include_masks.get()
        out=self.vars['out'].get().strip()
        self.batch_scan_text.set('正在后台扫描子目录……')
        def scan():
            try:
                from batch_runner import discover_images
                images,stats=discover_images(Path(root_str),exclusions=[Path(out), BASE/'outputs'],include_masks=include_masks)
                self.events.put(('batch_scan', (len(images),stats, [x.name for x in images[:3]])))
            except Exception as ex:self.events.put(('batch_scan_error',str(ex)))
        threading.Thread(target=scan,daemon=True).start()

    def run_action(self,action):
        if self.worker and self.worker.is_alive():
            messagebox.showwarning('任务运行中','已有任务正在执行，请等待或停止。')
            return
        if action=='recover':
            summary=Path(self.vars['batch_summary_path'].get().strip())
            if not summary.is_file() or summary.name!='batch_summary.json':
                messagebox.showerror('缺少批次记录','请选择原批次输出中的 batch_summary.json 文件。')
                return
            if self.vars['rag_mode'].get() == 'file':
                messagebox.showerror('批量检索配置','请使用 RAG=live 或 skip，不要选 file。')
                return
            if self.vars['retrieval_strategy'].get() == 'exact' and (
                    not self.vars['standard_id'].get().strip() or
                    not self.vars['clause_id'].get().strip()):
                self.vars['retrieval_strategy'].set('auto')
                self._log('缺少精确条款编号，已使用 auto 动态检索。')
        if action=='batch' and not Path(self.vars['batch_root'].get().strip()).is_dir():
            messagebox.showerror('缺少文件夹','请选择有效的批量图片根目录。')
            return
        if action=='batch' and self.vars['batch_action'].get() in ('all','report'):
            if self.vars['retrieval_strategy'].get()=='exact' and (
                    not self.vars['standard_id'].get().strip() or
                    not self.vars['clause_id'].get().strip()):
                self.vars['retrieval_strategy'].set('auto')
                self._log('[BATCH RAG] 精确检索未提供完整标准和条款；已自动切换 auto，按病害动态检索。')
            if not messagebox.askyesno('批量报告的工程条件',
                '批量模式自动根据每张图片检出的病害分组检索规范。\n'
                '构件类别如未知请留空；请勿对不同构件强行设置同一条款。\n\n'
                '确定生成整批唯一 Word 检测分析报告吗？'):
                return
        if action=='report' and not Path(self.vars['quant_json'].get()).is_file():
            messagebox.showerror('缺少输入','量化 JSON 不存在，请先运行①，或选择现有 JSON。')
            return
        if self.vars['scale_mode'].get()=='simulation':
            try:
                v=float(self.vars['demo_mm_per_px'].get())
                if not (0<v<=1000):raise ValueError()
            except ValueError:
                messagebox.showerror('比例尺无效','尺度换算系数必须大于0且不超过1000 mm/px。')
                return
        cmd=self._args(action)
        self._log('\n'+'='*60)
        self._log(f'启动任务：{action}；RAG={self.vars["rag_mode"].get()}；LLM={self.vars["llm_mode"].get()}')
        if action in ('batch','recover'):
            self.last_docx=''
            self.last_batch_dir=(Path(self.vars['batch_summary_path'].get()).parent if action=='recover' else None)
            self.last_batch_counts=None
            self.progress.configure(mode='determinate',maximum=1,value=0)
            self._log('批量模式：'+self.vars['batch_action'].get()+'；逐张量化，整批只生成一份 Word；失败不影响下一张。')
        else:
            self.last_batch_dir=None
            self.last_batch_counts=None
            self.progress.configure(mode='indeterminate')
        for b in self.buttons:b.state(['disabled'])
        if action not in ('batch','recover'):self.progress.start(12)
        self.status.set('执行中：'+action)
        def work():
            try:
                p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                   bufsize=1,text=True,encoding='utf-8',errors='replace',
                                   env=utf8_child_env())
                self.running_proc=p
                for line in p.stdout:
                    self.events.put(('line',line.rstrip('\n')))
                rc=p.wait()
                self.events.put(('done',rc))
            except Exception as ex:
                self.events.put(('failed',str(ex)))
            finally:
                self.running_proc=None
        self.worker=threading.Thread(target=work,daemon=True)
        self.worker.start()

    def _poll(self):
        try:
            while True:
                kind,value=self.events.get_nowait()
                if kind=='batch_scan':
                    count,scan_stats,names=value
                    self.batch_scan_text.set(f'已找到 {count} 张图片；其他文件 {scan_stats["unsupported_files"]}；排除 Mask/GT {scan_stats["excluded_masks"]}；跳过目录 {scan_stats["excluded_directories"]}')
                    self._log(f'[BATCH SCAN] 合格图片={count}；预览：{", ".join(names)}','success')
                elif kind=='batch_scan_error':
                    self.batch_scan_text.set('扫描失败：'+str(value))
                    self._log('扫描失败：'+str(value),'error')
                elif kind=='line':
                    if value.startswith('BATCH_FOLDER='):
                        self.last_batch_dir=Path(value.split('=',1)[1]);self._log('批量输出目录：'+str(self.last_batch_dir))
                    elif value.startswith('BATCH_TOTAL='):
                        self.batch_total=int(value.split('=',1)[1]);self.progress.configure(maximum=max(1,self.batch_total),value=0)
                        self.status.set(f'批量处理准备：{self.batch_total} 张图片')
                    elif value.startswith('BATCH_PROGRESS='):
                        n,t=value.split('=',1)[1].split('/')
                        self.status.set(f'批量处理中：第 {n}/{t} 张')
                        self.progress.configure(maximum=max(1,int(t)),value=max(0,int(n)-1))
                    elif value.startswith('BATCH_ITEM='):
                        item=json.loads(value.split('=',1)[1])
                        self.progress.configure(value=item['index'])
                        self._log(f'[{item["index"]}/{item["total"]}] {item["relative_path"]} → {item["status"]}',
                                  'error' if item['status']=='failed' else 'success')
                        if item.get('quant_json'):
                            p=Path(item['quant_json']);self.vars['quant_json'].set(str(p))
                            self.vars['image'].set(str(p.parent/'01_center_crop.png'))
                            self.vars['overlay'].set(str(p.parent/'04_overlay.png'))
                            self.refresh_stats(p)
                            self._refresh_preview()
                        if item.get('report_docx'):self.last_docx=item['report_docx']
                    elif value.startswith('BATCH_REPORT_DOCX='):
                        self.last_docx=value.split('=',1)[1]
                        self._log('整批唯一 Word 已生成：'+self.last_docx,'success')
                    elif value.startswith('BATCH_DONE='):
                        self.last_batch_counts=json.loads(value.split('=',1)[1]);self._log('批量统计：'+str(self.last_batch_counts),'success')
                    elif value.startswith('BATCH_SUMMARY_JSON='):
                        self._log('批量 CSV/JSON 清单已保存：'+value.split('=',1)[1],'success')
                    else:
                        self._log(value,'error' if value.startswith(('[ERROR]','[BATCH ERROR]')) else 'success' if value.startswith(('RESULT_','[OK]')) else 'info')
                        if value.startswith('QUANT_JSON=') and self.last_batch_dir is None:
                            p=value.split('=',1)[1]
                            self.vars['quant_json'].set(p)
                            q=Path(p).parent
                            self.vars['image'].set(str(q/'01_center_crop.png'))
                            self.vars['overlay'].set(str(q/'04_overlay.png'))
                            self.refresh_stats(p)
                            self._refresh_preview()
                        if value.startswith('RESULT_DOCX='):
                            self.last_docx=value.split('=',1)[1]
                elif kind in ('done','failed'):
                    self.progress.stop()
                    for b in self.buttons:b.state(['!disabled'])
                    if kind=='failed':
                        self.status.set('执行失败：'+str(value))
                        self._log('启动失败：'+str(value),'error')
                    elif value==0:
                        if self.last_batch_counts is not None:
                            c=self.last_batch_counts
                            self.status.set(f'批量完成：新处理 {c["completed"]}，跳过 {c["resumed"]}，失败 {c["failed"]}')
                        else:self.status.set('完成：输出文件已保存')
                        self._log('任务成功完成。','success')
                    elif value==4:
                        self.status.set('部分完成：Word 已保存，智能分析模块未全部成功')
                        self._log('本批图像与 Word 已保存；RAG/Qwen 部分未完成，请检查报告目录中的调试 JSON。','error')
                    else:
                        if self.last_batch_counts is not None:
                            c=self.last_batch_counts
                            self.status.set(f'批量部分完成：成功 {c["completed"]}、跳过 {c["resumed"]}、失败 {c["failed"]}')
                        else:self.status.set('执行失败：请阅读日志')
                        self._log(f'子进程返回码={value}，请检查前面的错误信息。','error')
        except queue.Empty:
            pass
        self.after(100,self._poll)

    def cancel(self):
        p=self.running_proc
        if p and p.poll() is None:
            try:
                if sys.platform=='win32':
                    subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True,timeout=10)
                else:p.terminate()
            except Exception as ex:self._log('终止失败：'+str(ex),'error')
            self._log('已请求停止进程（已生成的阶段文件不会自动删除）','error')

    def open_word(self):
        p=Path(getattr(self,'last_docx',''))
        if not p.is_file() and self.last_batch_dir is not None:
            combined=self.last_batch_dir/'report'/'00_batch_bridge_report.docx'
            if combined.is_file():p=combined
        if not p.is_file():
            case=self.vars['case_id'].get()
            import re
            case=re.sub(r'[^a-zA-Z0-9_\-]','_',case)[:60].strip('_') or 'DEMO-CASE'
            p=Path(self.vars['out'].get())/case/'report'/'07_bridge_inspection_demo.docx'
        if not p.is_file():
            messagebox.showinfo('尚无报告','尚未生成 Word 报告，请先执行②、单图完整流程，或批量 all/report。')
            return
        if sys.platform=='win32':os.startfile(str(p))
        elif sys.platform=='darwin':subprocess.Popen(['open',str(p)])
        else:subprocess.Popen(['xdg-open',str(p)])

    def open_output(self):
        if self.last_batch_dir is not None and self.last_batch_dir.is_dir():
            p=self.last_batch_dir
            if sys.platform=='win32':os.startfile(str(p))
            elif sys.platform=='darwin':subprocess.Popen(['open',str(p)])
            else:subprocess.Popen(['xdg-open',str(p)])
            return
        p=Path(self.vars['out'].get())
        case=self.vars['case_id'].get()
        import re
        case=re.sub(r'[^a-zA-Z0-9_\-]','_',case)[:60].strip('_') or 'DEMO-CASE'
        p=p/case
        if not p.is_dir():p=Path(self.vars['out'].get())
        if not p.is_dir():
            messagebox.showinfo('尚无输出','输出目录尚不存在，请先运行任务。')
            return
        if sys.platform=='win32':os.startfile(str(p))
        elif sys.platform=='darwin':subprocess.Popen(['open',str(p)])
        else:subprocess.Popen(['xdg-open',str(p)])

    def close(self):
        if self.running_proc and self.running_proc.poll() is None:
            if not messagebox.askyesno('任务运行中','任务尚未完成，确认终止并退出吗？'):return
            self.cancel()
        self.destroy()


def main():
    app=BridgeGUI();app.mainloop()


if __name__=='__main__':
    main()
