# ☀️ 阳光学院校园小助手 · 项目骨架

本项目目标：

> 用户上传一张阳光学院校园照片，系统识别这是哪里，并给出简要介绍。
> 进阶版本可以基于阳光学院相关文档，做简单 RAG 问答（聊天机器人）。

---

## 📁 推荐目录结构

```bash
project/
├── data/
│   ├── images/                 # 校园图片（按类别建子文件夹）
│   │   ├── 图书馆/
│   │   ├── 体育场/
│   │   └── ...
│   └── class_map.json          # 训练脚本自动生成：类别名 -> index
├── docs/                       # 阳光学院相关资料，用于 RAG
│   ├── campus_guide.md
│   └── building_info.txt
├── models/
│   └── best_model.pth          # 训练好的图像分类模型
├── rag/
│   ├── rag_build_index.py      # 构建文档向量索引
│   └── rag_answer.py           # 简单检索 + LLM 回答
├── src/
│   ├── train_model.py          # 图像分类训练脚本
│   ├── inference.py            # 单张图片预测地点
│   └── app.py                  # Gradio 前端 Demo
└── README.md
```

---

## 🚀 使用步骤

1. **准备数据**
   - 在 `data/images/` 下按类别建文件夹。每个文件夹名称就是结果名称，如启智楼，正德楼。
   可从教学楼名称中找名字，还需要加上图书馆，东门，南门，体育场等。
   图片可借鉴https://www.xxpie.com/m/album?id=67b2ff89126715761c69a3b2&source=SHARE_LINK&r=6119
   如不够则需自己照相
     - `data/images/启智楼/xxx.jpg`
     - `data/images/正德楼/yyy.jpg`
   - 确保每一类有若干张不同角度、不同时间的照片。

2. **训练模型**
   ```bash
   cd src
   python train_model.py
   ```
   - 会在 `models/` 下生成 `best_model.pth`
   - 会在 `data/` 下生成 `class_map.json`（类别映射）。

3. **测试单张图片推理**
   ```bash
   python inference.py
   ```
   - 在脚本中修改 `test_path` 为某一张校园图片路径，查看输出结果。

4. **准备 RAG 资料（可选 / 进阶）**
   - 在 `docs/` 下放入若干 `.txt` 或 `.md` 文档，内容可以是：
     - 校园总体介绍
     - 各教学楼、图书馆、体育场、食堂的介绍
   - 然后运行：
     ```bash
     cd rag
     python rag_build_index.py
     ```

5. **运行前端 Demo**
   ```bash
   cd src
   python app.py
   ```
   - 浏览器中打开链接，上传图片测试。

---

## 📝 作业扩展建议（可写进课程说明）

- **基础要求**
  - 至少支持 12 个不同地点的分类。
  - 提供每个地点的中文简介，可以写死在字典里。
  - 能通过前端应用完成“上传图片 → 显示地点 + 简介”。

- **进阶要求（高分项）**
  - 完成基于文档的 RAG 问答：允许用户针对某地点提问。
  - 分析模型在不同类别上的表现，给出混淆矩阵与误分类示例截图。
  - 支持 top-k 预测或图像检索（显示相似图库图片）。

- **创新加分**
  - 支持中英双语介绍。
  - 增加“路线建议”“附近设施推荐”等功能（可以简单 template 实现）。
  - 优化界面 UI/UX，让助手更像一个真实产品。

---
