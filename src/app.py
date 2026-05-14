import gradio as gr
import os
import sys

# 把项目根目录加入 sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from rag.rag_answer import rag_answer
from inference import predict

# 基础版本：地点 -> 简介
LOCATION_INTRO = {
    "正德楼": "阳光学院行政综合楼，主要用于学校行政办公，是学校管理层的所在地。",
    "博文楼": "文学院所在地，以文理科教学为主，楼内设有多媒体教室、学术研讨室。",
    "博雅楼": "主要用于文科教学，特别是人文社科类课程，配备先进教学设备。",
    "商德楼": "商学院所在地，专注于商科教育，设有模拟实验室、案例研讨室。",
    "知行楼": "主要用于工科教学，设有各类实验室和实训基地。",
    "数德楼": "理工科教学楼，主要用于数学、物理等基础学科的教学。",
    "润知楼": "信息实验楼，配备先进的计算机设备和网络设施。",
    "启智楼": "计算机科实验楼，拥有高性能计算设备和专业实验室。",
    "致远楼": "人文社科楼，注重学生综合素质的培养。",
    "载德楼": "主要用于文科类教学，建筑设计现代美观。",
    "雅颂楼": "艺术相关学科的教学场所，设有排练室、展示空间等。",
    "图书馆": "阳光学院图书馆，共7层，藏书丰富，是学生自习和查阅资料的主要场所。",
    "体育场": "阳光学院田径场，设有标准跑道、足球场和各类运动设施。",
    "东门": "学校的主要入口之一，交通便利，周边配套设施完善。",
    "南门": "学校的另一个主要入口，临近多个教学区，设计美观。"
}


def pipeline(image):
    """上传图片 → 预测地点 + 返回简介。"""
    try:
        location, prob, intro = predict(image)
        
        # 特殊处理：如果是不在识别类型中的情况
        if intro == "不在识别的类型中":
            return f"📸 识别结果\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n❓ {intro}\n\n💡 提示：当前系统仅支持识别以下地点\n{', '.join(LOCATION_INTRO.keys())}"
        
        # 美化输出格式
        confidence_emoji = "🟢" if prob > 0.8 else "🟡" if prob > 0.6 else "🟠"
        confidence_text = "高" if prob > 0.8 else "中" if prob > 0.6 else "低"
        
        result = f"""
🎯 识别结果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 地点名称：{location}

{confidence_emoji} 置信度：{prob:.2%} ({confidence_text})

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📖 地点简介
{intro}
        """
        return result.strip()
    except Exception as e:
        return f"❌ 识别失败\n\n错误信息：{str(e)}\n\n💡 建议：请确保模型文件已正确加载，或尝试上传更清晰的校园建筑照片。"


def chat(query, location):
    """基于 RAG 的提问接口。"""
    try:
        q = f"关于阳光学院的 {location}：{query}"
        answer = rag_answer(q)
        return f"🤖 AI 回答\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n{answer}"
    except Exception as e:
        return f"❌ 查询失败\n\n错误信息：{str(e)}\n\n💡 请确保 RAG 索引已构建"


def build_demo():
    # 准备示例图片路径
    example_images = []
    example_dir = os.path.join(PROJECT_ROOT, "data", "images")
    
    # 收集一些示例图片
    for location in ["体育场", "图书馆", "东门"]:
        location_path = os.path.join(example_dir, location)
        if os.path.exists(location_path):
            images = [f for f in os.listdir(location_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            if images:
                example_images.append([os.path.join(location_path, images[0])])
    
    # iOS 风格的自定义 CSS
    ios_css = """
    /* iOS 风格全局样式 -  */
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif !important;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
        padding: 12px !important;
    }
    
    /* 只对特定的功能块应用卡片样式，避免影响标题区域 */
    .gr-group .block {
        background: rgba(255, 255, 255, 0.92) !important;
        backdrop-filter: blur(15px) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08) !important;
        padding: 10px !important;
        margin: 5px 0 !important;
    }
    
    /* 确保 HTML 块不受影响 */
    .gr-html {
        background: none !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    
    /* 紧凑按钮 */
    .primary {
        background: linear-gradient(135deg, #007AFF 0%, #0051D5 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        margin: 5px 0 !important;
        box-shadow: 0 2px 10px rgba(0, 122, 255, 0.2) !important;
    }
    
    .secondary {
        background: rgba(142, 142, 147, 0.1) !important;
        color: #007AFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        margin: 5px 0 !important;
    }
    
    /* 紧凑输入框 */
    input, textarea, .input-text {
        background: rgba(255, 255, 255, 0.85) !important;
        border: 1px solid rgba(0, 0, 0, 0.08) !important;
        border-radius: 10px !important;
        padding: 10px 12px !important;
        font-size: 14px !important;
        margin: 5px 0 !important;
    }
    
    /* 紧凑标签 */
    label {
        font-weight: 600 !important;
        color: #1d1d1f !important;
        font-size: 13px !important;
        margin: 8px 0 4px 0 !important;
    }
    
    /* 紧凑图片区域 */
    .image-container {
        border-radius: 14px !important;
        border: 2px dashed rgba(0, 122, 255, 0.2) !important;
        background: rgba(255, 255, 255, 0.6) !important;
        margin: 5px 0 !important;
    }
    
    /* 紧凑示例 */
    .examples {
        border-radius: 12px !important;
        margin: 8px 0 !important;
    }
    
    /* 移除多余间距 */
    .row {
        margin: 0 !important;
        gap: 10px !important;
    }
    
    .column {
        padding: 0 !important;
    }
    
    /* 紧凑提示框 */
    .gr-textbox {
        margin: 5px 0 !important;
    }
    """
    
    with gr.Blocks(title="阳光学院校园智能助手") as demo:
        
        # 注入 iOS 风格 CSS
        gr.HTML(f"""
            <style>
            {ios_css}
            </style>
        """)
        
        # 卡片1：标题 + 图片识别功能合并卡片
        with gr.Group():
            gr.HTML("""
                <div style="text-align: center; margin: 10px 0 20px 0; padding: 20px; 
                            background: rgba(255, 255, 255, 0.8); 
                            backdrop-filter: blur(20px);
                            border-radius: 18px;
                            border: 1px solid rgba(255, 255, 255, 0.4);
                            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);">
                    <div style="font-size: 2.5em; margin-bottom: 8px;">☀️</div>
                    <h1 style="background: linear-gradient(135deg, #007AFF 0%, #0051D5 100%); 
                               -webkit-background-clip: text; 
                               -webkit-text-fill-color: transparent; 
                               font-size: 1.8em; 
                               font-weight: 700; 
                               margin: 0;
                               letter-spacing: -0.5px;">
                        阳光学院校园智能助手
                    </h1>
                    <p style="color: #86868b; font-size: 0.95em; margin: 8px 0 15px 0;">
                        基于 AI 的校园图片识别与智能问答系统
                    </p>
                    
                    <!-- 分隔线 -->
                    <div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(0, 122, 255, 0.2), transparent); margin: 15px 0;"></div>
                    
                    <div style="text-align: center; padding: 12px; margin-bottom: 15px;
                                background: linear-gradient(135deg, rgba(0, 122, 255, 0.1) 0%, rgba(0, 81, 213, 0.05) 100%);
                                border-radius: 12px;">
                        <h2 style="color: #007AFF; font-size: 1.3em; font-weight: 700; margin: 0;">
                            📸 校园地点识别
                        </h2>
                        <p style="color: #86868b; font-size: 0.85em; margin: 5px 0 0 0;">
                            上传校园照片，AI 智能识别地点并提供详细介绍
                        </p>
                    </div>
                </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(
                        type="filepath", 
                        label="上传图片",
                        height=280
                    )
                    btn = gr.Button("🔍 开始识别", variant="primary", size="lg")
                    
                    # 示例图片整合卡片
                    if example_images:
                        gr.HTML("""
                            <div style="text-align: center; padding: 12px; margin: 10px 0 5px 0;
                                        background: rgba(255, 149, 0, 0.05); 
                                        border-radius: 10px; 
                                        border-left: 3px solid #FF9500;">
                                <div style="color: #FF9500; font-size: 12px; font-weight: 600; margin-bottom: 8px;">
                                    ✨ 点击示例快速体验
                                </div>
                            </div>
                        """)
                        gr.Examples(
                            examples=example_images,
                            inputs=img_input,
                            label="",
                            examples_per_page=3
                        )
                
                with gr.Column(scale=1):
                    txt_output = gr.Textbox(
                        label="识别结果", 
                        lines=10,
                        placeholder="识别结果将在这里显示..."
                    )
                    
                    # 提示信息
                    gr.HTML("""
                        <div style="margin-top: 10px; padding: 12px;
                                    background: rgba(0, 122, 255, 0.05); 
                                    border-radius: 10px; 
                                    border-left: 3px solid #007AFF;">
                            <div style="color: #007AFF; font-size: 12px; font-weight: 600; margin-bottom: 5px;">
                                💡 使用提示
                            </div>
                            <div style="color: #1d1d1f; font-size: 11px; line-height: 1.5;">
                                支持 JPG、PNG 格式 • 建议上传清晰照片
                            </div>
                            <div style="color: #FF9500; font-size: 12px; font-weight: 600; margin: 8px 0 5px 0;">
                                📍 可识别地点
                            </div>
                            <div style="color: #1d1d1f; font-size: 11px; line-height: 1.4;">
                                正德楼 • 博文楼 • 商德楼 • 知行楼 • 润知楼 • 启智楼 • 致远楼 • 载德楼 • 雅颂楼 • 图书馆 • 体育场 • 东门 • 南门
                            </div>
                        </div>
                    """)
            
            btn.click(pipeline, inputs=[img_input], outputs=[txt_output])
        
        # 卡片2：智能问答功能卡片
        with gr.Group():
            gr.HTML("""
                <div style="text-align: center; padding: 12px; margin: 20px 0 15px 0;
                            background: linear-gradient(135deg, rgba(52, 199, 89, 0.1) 0%, rgba(48, 176, 199, 0.05) 100%);
                            border-radius: 12px;">
                    <h2 style="color: #34C759; font-size: 1.3em; font-weight: 700; margin: 0;">
                        💬 智能问答助手
                    </h2>
                    <p style="color: #86868b; font-size: 0.85em; margin: 5px 0 0 0;">
                        针对特定地点进行深度提问，获取更详细的信息
                    </p>
                </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    location_box = gr.Textbox(
                        label="🏛️ 地点名称", 
                        placeholder="例如：图书馆、体育场、正德楼"
                    )
                    query_box = gr.Textbox(
                        label="❓ 您的问题", 
                        placeholder="例如：开放时间是什么？有哪些功能？",
                        lines=3
                    )
                    ask_btn = gr.Button("💬 提问", variant="primary", size="lg")
                    
                    # 提问技巧
                    gr.HTML("""
                        <div style="margin-top: 10px; padding: 12px;
                                    background: rgba(52, 199, 89, 0.05); 
                                    border-radius: 10px; 
                                    border-left: 3px solid #34C759;">
                            <div style="color: #34C759; font-size: 12px; font-weight: 600; margin-bottom: 5px;">
                                💡 提问技巧
                            </div>
                            <div style="color: #1d1d1f; font-size: 11px; line-height: 1.5;">
                                • 先输入地点名称（如：图书馆）<br/>
                                • 再输入具体问题<br/>
                                • 支持开放时间、功能介绍等查询
                            </div>
                        </div>
                    """)
                
                with gr.Column(scale=1):
                    rag_out = gr.Textbox(
                        label="🤖 AI 回答", 
                        lines=8,
                        placeholder="AI 回答将在这里显示..."
                    )
                    
                    
            
            ask_btn.click(chat, inputs=[query_box, location_box], outputs=[rag_out])
        


    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=8080,
        share=False,
        show_error=True
    )