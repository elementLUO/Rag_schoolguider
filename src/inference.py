import json
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# 定义地点中文简介字典，避免循环导入
LOCATION_INTRO = {
    "正德楼": "阳光学院行政综合楼，主要用于学校行政办公，是学校管理层的所在地。建筑风格现代，设施完善，为学校各项工作的协调与管理提供重要保障。",
    "博文楼": "主要用于文科教学，特别是人文社科类课程。楼内配备先进的教学设备，是培养学生人文素养和社会责任感的重要场所。",
    "商德楼": "商学院所在地，专注于商科教育。设有模拟实验室、案例研讨室等专业设施，为商科学生提供实践学习的平台。",
    "润知楼": "主要用于工科教学，楼内设有各类实验室和实训基地。注重理论与实践相结合，培养学生的动手能力和创新思维。",
    "知行楼": "理工科教学楼，主要用于数学、物理等基础学科的教学，还有计算机相关的理论知识的教学。楼内环境安静，适合学生进行专注的学习和研究。",
    "启智楼": "计算机科实验楼，拥有高性能计算设备和专业实验室。主要用于计算机科学、人工智能等前沿领域的教学与研究。",
    "致远楼": "人文社科楼，注重学生综合素质的培养。楼内设有各种活动室和交流空间，促进学生之间的思想碰撞和文化交流。",
    "载德楼": "主要用于建筑类教学，建筑设计现代美观。楼内教学设施完善，为师生提供舒适的教学和学习环境。",
    "雅颂楼": "艺术相关学科的教学场所，环境优雅，氛围浓厚。楼内设有排练室、展示空间等，为艺术专业学生提供创作和展示的平台。",
    "图书馆": "阳光学院图书馆，共7层，藏书丰富，是学生自习和查阅资料的主要场所。馆内设有各类阅览室、电子资源区和学术交流空间，为全校师生提供优质的文献资源和学习环境。",
    "体育场": "阳光学院田径场，设有标准跑道、足球场和各类运动设施。用于体育课、课外锻炼和校运会等活动，是学生进行体育锻炼和团队活动的重要场所。",
    "东门": "学校的主要入口之一，交通便利，周边配套设施完善。是师生进出校园的重要通道，也是校园对外展示形象的窗口。",
    "南门": "学校的另一个主要入口，临近多个教学区。设计美观，是校园景观的重要组成部分，为师生提供便捷的通行条件。"
}

MODEL_PATH = "./models/best_model.pth"
CLASS_MAP_PATH = "./data/class_map.json"


def load_class_map(path: str = CLASS_MAP_PATH):
    with open(path, "r", encoding="utf8") as f:
        class_map = json.load(f)
    # 反转：index -> class_name
    inv_map = {v: k for k, v in class_map.items()}
    return inv_map


def load_model(model_path: str = MODEL_PATH, num_classes: int = 11):
    """加载训练好的 ResNet18 模型。"""
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, num_classes)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict(image_path: str):
    """输入图片路径，返回 (预测地点名, 置信度, 地点简介)。"""
    # 1. 读图 & 预处理
    img = Image.open(image_path).convert("RGB")
    img_tensor = _preprocess(img).unsqueeze(0)  # [1, 3, 224, 224]

    # 2. 加载类别映射和模型
    inv_map = load_class_map()  # load_class_map已经返回index -> class_name的映射
    class_map = {v: k for k, v in inv_map.items()}  # 创建class_name -> index的映射
    num_classes = len(class_map)
    model = load_model(num_classes=num_classes)

    # 3. 推理
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        top_prob, top_cls = probs.max(dim=1)
    
    # 获取预测的地点名
    top_cls_idx = int(top_cls)
    print(f"预测的类别索引：{top_cls_idx}")
    print(f"可用的类别索引：{list(inv_map.keys())}")
    
    # 检查预测结果和置信度
    if top_cls_idx in inv_map:
        location_name = inv_map[top_cls_idx]
        confidence = float(top_prob)
        
        # 1. 检查置信度是否低于阈值
        # 如果置信度过低，认为不在识别的类型中
        if confidence < 0.6:
            return "未知建筑", confidence, "不在识别的类型中"
            
        # 2. 检查是否在支持的建筑列表中
        if location_name not in LOCATION_INTRO:
            return "未知建筑", confidence, "不在识别的类型中"
    else:
        # 如果类别索引不在映射中，返回特殊标记
        return "未知建筑", float(top_prob), "不在识别的类型中"
    
    # 获取地点简介
    location_intro = LOCATION_INTRO.get(location_name, "暂时没有该地点的详细介绍")
    
    return location_name, float(top_prob), location_intro


if __name__ == "__main__":
    # 简单测试：手动填一张图片路径
    test_path = "./data/images/example.jpg"
    print(f"测试图片路径：{os.path.abspath(test_path)}")
    print(f"模型文件路径：{os.path.abspath(MODEL_PATH)}")
    print(f"类别映射文件路径：{os.path.abspath(CLASS_MAP_PATH)}")
    
    # 检查文件是否存在
    print(f"测试图片存在：{os.path.exists(test_path)}")
    print(f"模型文件存在：{os.path.exists(MODEL_PATH)}")
    print(f"类别映射文件存在：{os.path.exists(CLASS_MAP_PATH)}")
    
    try:
        location, prob, intro = predict(test_path)
        print(f"预测地点：{location} (置信度 {prob:.2f})")
        print(f"简介：{intro}")
    except Exception as e:
        import traceback
        print(f"测试失败：{e}")
        print("详细错误信息：")
        traceback.print_exc()
        print("请确保图片路径正确，并且模型文件已训练完成")
