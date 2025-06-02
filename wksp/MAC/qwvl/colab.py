import cv2
import json
import re
import os
import torch
import decord
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 加载模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="auto",
    # torch_dtype=torch.float16,
    device_map="auto"
)
print(model.hf_device_map)


# # 可选：启用 flash_attention_2（若硬件支持，取消注释）
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "/home/bd2/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# 加载处理器，设置像素范围
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    min_pixels=min_pixels,
    max_pixels=max_pixels
)

# 视频文件路径
video_path = "/content/qwvl/train0028.mp4"

# 验证视频并获取帧率
try:
    # cap = cv2.VideoCapture(video_path)
    # if not cap.isOpened():
    #     raise ValueError(f"无法打开视频文件：{video_path}")
    # fps = cap.get(cv2.CAP_PROP_FPS)  # 预期：30 fps
    # cap.release()
    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()  # 获取平均帧率
except Exception as e:
    print(f"访问视频时出错：{e}")
    exit(1)

print(f"视频帧率：{fps} fps")

# prompt = """
# 这是一个正常聊天的场景。你需要关注视频中的人物身体的所有细微变化，人物任何与上一帧不同的地方都需要仔细观察，请详细描述人物的细微动作（注意：微动作幅度极小），输出微动作所在部位及对微动作的详细描述。
# """
prompt = """
请详细描述视频信息，重点关注人物微动作（幅度极小时间极短）。以下为所有微动作类别：
0	shaking body
1	sitting straightly
2	shrugging
3	turning around
4	rising up
5	bowing head
6	head up
7	tilting head
8	turning head
9	nodding
10	shaking head
11	scratching arms
12	playing objects
13	putting hands together
14	rubbing hands
15	pointing oneself
16	clenching fist
17	stretching arms
18	retracting arms
19	waving
20	spreading hands
21	hands touching fingers
22	other finger movements
23	illustrative gestures
24	shaking legs
25	curling legs
26	spread legs
27	closing legs
28	crossing legs
29	stretching feet
30	retracting feet
31	tiptoe
32	scratching or touching neck
33	scratching or touching chest
34	scratching or touching back
35	scratching or touching shoulder
36	arms akimbo
37	crossing arms
38	playing or tidying hair
39	scratching or touching hindbrain
40	scratching or touching forehead
41	scratching or touching face
42	rubbing eyes
43	touching nose
44	touching ears
45	covering face
46	covering mouth
47	pushing glasses
48	patting legs
49	touching legs
50	scratching legs
51	scratching feet，输出格式为：人物编号+微动作+描述文本+微动作部位定位框@[时间段] 人物编号+对话@[时间段]
"""

# 准备推理消息
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "max_pixels": 360 * 420, # max_pixels
                "fps": fps,
            },
            {
                "type": "text",
                "text": prompt
            },
        ],
    }
]

# 准备推理输入
try:
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True) # process_vision_info() got an unexpected keyword argument 'return_video_kwargs
    # inputs = processor(
    # text=[text],
    # images=image_inputs,
    # videos=video_inputs,
    # fps=fps,
    # padding=True,
    # return_tensors="pt",
    # **video_kwargs,
    # )
    image_inputs, video_inputs = process_vision_info(messages)  #yyf
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
except Exception as e:
    print(f"处理输入时出错：{e}")
    exit(1)

inputs = inputs.to("cuda")
# inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


# 执行推理
try:
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
except Exception as e:
    print(f"推理过程中出错：{e}")
    exit(1)

print("推理结果：")
print(output_text[0])

output = output_text[0].strip()
json_output = []

# 首先尝试是否已经是合法 JSON 格式
try:
    json_output = json.loads(output)
except json.JSONDecodeError:
    # 尝试正则提取微动作信息（你原来的思路）
    if output == "[]":
        json_output = []
    elif output:
        pattern = r'(\d+)\s*[\-:]?\s*([^\s，]+)[，,:：]\s*([^@]+)?@\[([^\]]+)\]'
        matches = re.findall(pattern, output)
        for match in matches:
            person_id, action, description, time_range = match
            json_output.append({
                "person_id": person_id.strip(),
                "action": action.strip(),
                "description": description.strip() if description else "",
                "time": time_range.strip()
            })
    else:
        json_output = []

# 保存为 JSON 文件
os.makedirs("/content/qwvl", exist_ok=True)
output_path = "./qwvl/micro_actions_output.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(json_output, f, ensure_ascii=False, indent=2)

print(f"微动作 JSON 已保存至：{output_path}")
