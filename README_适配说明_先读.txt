Bridge AI Final V3 — Epoch27 兼容适配版（保留原 Tkinter UI 布局）
===============================================================

一、安装/迁移（Windows）
1. 为避免覆盖原版本，先备份 C:\Users\z8466\Desktop\Bridge_AI_Final_V3。
2. 将本压缩包中的文件覆盖复制到原 Bridge_AI_Final_V3；已有的 examples/、outputs/、tests/、知识库文件夹等请原样保留。
3. 将 AutoDL 上验证成功的、大小约 309.08 MiB 的 VMAMBA_BASELINE Epoch27 权重放到：
   C:\Users\z8466\Desktop\Bridge_AI_Final_V3\models\best_model_epoch27.pth
   代码包不包含此 309MiB 权重，也不包含官方 VMamba 源码。
4. 默认 VMamba 代码目录是 C:\Users\z8466\Desktop\VMamba-main\classification\models。
   如果位置不同，可在 GUI 左侧“官方 VMamba 目录”选择实际目录。
5. 双击「检查环境_Windows.cmd」。如果提示缺依赖，使用项目实际 Python 环境安装。
   如 Windows 无法编译/导入 selective_scan 等算子，须另行配置，不保证 Windows 原生运行。
6. 双击「启动GUI_Windows.cmd」，选择原图先运行单图流程。启动正常但分割失败时，检查错误日志。

二、类别规则
- 模型仍是五分类：0背景、1裂缝、2剥落、3锈蚀、4钢筋外露。
- 汇总展示为：背景、裂缝、剥落、钢筋相关病害（锈蚀＋外露）。
- 原始 class-ID mask、每类 JSON、单类二值图和原始动态 RAG 检索不变。
- 04_overlay.png / 05_contours.png 是四类展示图；03_color_mask.png、04_overlay_five_class.png 仍保留五类可视化；合并仅用于统计、显示和 Word。
- 合并类别后的评价指标不能冒充原五分类 mIoU，也不代表原模型召回率得到改善。

三、修改摘要
- 修复 rag_query(3).py 的文件名为 rag_query.py，避免程序无法定位 RAG 脚本。
- 统一 GUI / CLI 默认权重路径优先读取 models/best_model_epoch27.pth，保留 Desktop/best_model.pth 备用。
- 增加 Windows 离线检查脚本，验证 PyTorch、VMamba 导入与严格参数加载。
- 增加 4 类展示数据；并保留五分类模型/真实类别 ID，避免误改模型训练标注。
- 启动 .cmd 完全 ASCII，减少 Windows 批处理编码故障。

四、限制
此工程仅包含你本次上传的源文件及新增脚本；无法在此环境验证 Windows VMamba CUDA 算子、
本地 ChromaDB、Ollama、你的 GPU 驱动，也不能宣称本机完整流程已经测试通过。
之前文件夹中如果有其他未上传资产，请不要删除。项目的 Word 为比赛辅助演示材料。
