import cv2
import sys
import json
import re
import os
import torch
import decord
from transformers import __version__ as transformers_version
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 视频文件路径
video_path = "/home/bd2/wksp/MAC/qwvl/train0028.mp4"
output_dir = "./video_frames_cut/frames_train0028"

def slice_video_to_frames(video_path: str, output_dir: str):
    """
    Splits the input video into individual frames (no skipping) and saves them as JPEGs.
    Prints the total number of frames extracted.
    Returns a list of file URI strings suitable for qwen2.5-vl message format.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    frame_paths = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        filename = f"frame{frame_idx:04d}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        # Convert to file URI
        frame_paths.append(f"file://{os.path.abspath(filepath)}")

    cap.release()

    print(f"Extracted {frame_idx} frames from {video_path}")
    return frame_paths



# Example usage:
# if __name__ == "__main__":
#     video_path = "train0028.mp4"
#     output_dir = "/video_frames_cut/frames_train0028"
#     frame_file_uris = slice_video_to_frames(video_path, output_dir)

    # # Build the messages list for qwen2.5-vl
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "video",
    #                 "video": frame_file_uris,
    #             },
    #             {
    #                 "type": "text",
    #                 "text": "请描述这个视频。"
    #             },
    #         ],
    #     }
    # ]

    # # Optionally, print or return messages
    # import pprint
    # pprint.pprint(messages)


# device = torch.device("cuda:5")
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "/home/bd2/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-3B-Instruct",
#     # torch_dtype="auto",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     # device_map="auto"
# ).to(device)
# # print(model.hf_device_map)
# # print("Current GPU:", device)
# print(model.config._attn_implementation)


# 可选：启用 flash_attention_2（若硬件支持，取消注释）
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/home/bd2/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# 加载处理器，设置像素范围
min_pixels = 256 * 28 * 28
max_pixels = 960 * 540 # 1980 * 1080 
processor = AutoProcessor.from_pretrained(
    "/home/bd2/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-3B-Instruct",
    min_pixels=min_pixels,
    max_pixels=max_pixels
)

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
# prompt = """
# 请以标准JSON格式输出视频中人物的微动作信息。
# 每个动作应包含以下字段：
# - person_id: 人物编号（字符串）
# - action: 微动作名称（参考如下列表）
# - description: 对该动作的描述（包含部位及动作细节）
# - time: 时间段（格式为 "开始时间-结束时间"）

# 输出格式示例：
# [
#   {
#     "person_id": "1",
#     "action": "sitting straightly",
#     "description": "人物坐在红色凳子上，身体保持直立。",
#     "time": "0-10"
#   },
#   ...
# ]

# 动作列表：
# 0	shaking body
# 1	sitting straightly
# 2	shrugging
# 3	turning around
# 4	rising up
# 5	bowing head
# 6	head up
# 7	tilting head
# 8	turning head
# 9	nodding
# 10	shaking head
# 11	scratching arms
# 12	playing objects
# 13	putting hands together
# 14	rubbing hands
# 15	pointing oneself
# 16	clenching fist
# 17	stretching arms
# 18	retracting arms
# 19	waving
# 20	spreading hands
# 21	hands touching fingers
# 22	other finger movements
# 23	illustrative gestures
# 24	shaking legs
# 25	curling legs
# 26	spread legs
# 27	closing legs
# 28	crossing legs
# 29	stretching feet
# 30	retracting feet
# 31	tiptoe
# 32	scratching or touching neck
# 33	scratching or touching chest
# 34	scratching or touching back
# 35	scratching or touching shoulder
# 36	arms akimbo
# 37	crossing arms
# 38	playing or tidying hair
# 39	scratching or touching hindbrain
# 40	scratching or touching forehead
# 41	scratching or touching face
# 42	rubbing eyes
# 43	touching nose
# 44	touching ears
# 45	covering face
# 46	covering mouth
# 47	pushing glasses
# 48	patting legs
# 49	touching legs
# 50	scratching legs
# 51	scratching feet
# """


# 准备推理消息
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "video",
#                 "video": video_path,
#                 "max_pixels": 360 * 420, # max_pixels
#                 "fps": fps,
#             },
#             {
#                 "type": "text",
#                 "text": prompt
#             },
#         ],
#     }
# ]

frame_file_uris = slice_video_to_frames(video_path, output_dir) #yyf

prompt = "You are an expert assistant specialized in micro-action recognition (MAR). Your task is to analyze video sequences from psychological interviews, where subtle and rapid human movements (micro-actions) occur naturally. These micro-actions involve various body parts (head, upper limbs, lower limbs, body-hand, head-hand, leg-hand) and are categorized into fine-grained actions (e.g., nodding, shaking legs) and coarse-grained body parts (e.g., head, lower limb). Your goal is to accurately identify the fine-grained micro-action and its corresponding coarse-grained body part from the input video frames. Analyze the following video sequence and identify the micro-action performed by the participant. The video consists of a series of frames capturing subtle movements during a psychological interview. Provide the fine-grained micro-action and its corresponding coarse-grained body part and the detailed description of micro-action in the format:\"Fine-grained action: {fine_label_name}\nCoarse-grained action: {coarse_label_name}\ndetailed description: {detailed_description}\". Focus on the spatiotemporal dynamics of the movements and consider the psychological context to enhance recognition accuracy."

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": frame_file_uris,
            },
            {
                "type": "text",
                "text": prompt,
            },
        ],
    }
] #yyf

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

# # 解析输出为 JSON
# output = output_text[0].strip()
# json_output = []

# try:
#     json_output = json.loads(output)
# except json.JSONDecodeError:
#     if output == "[]":
#         json_output = []
#     elif output:
#         pattern = r'- 动作：([^，]+)，部位：([^，]+)，时间：(\d+\.\d+)-(\d+\.\d+)\s*秒(?:，描述：([^\n]*))?'
#         matches = re.findall(pattern, output)
#         for match in matches:
#             action, body_part, start, end, desc = match
#             json_output.append({
#                 "action": action.strip(),
#                 "body_part": body_part.strip(),
#                 "time": f"{start}-{end}",
#                 "description": desc.strip() if desc else ""
#             })

# # 保存 JSON
# os.makedirs("./qwvl", exist_ok=True)
# with open("./qwvl/micro_actions_output.json", "w", encoding="utf-8") as f:
#     json.dump(json_output, f, ensure_ascii=False, indent=2)

# 获取输出文本
output = output_text[0].strip()
json_output = []

# 尝试解析 JSON
try:
    json_output = json.loads(output)
except json.JSONDecodeError:
    print("警告：模型未输出合法 JSON 格式，尝试正则提取...")
    # 可选：使用正则提取非 JSON 输出（如旧格式）
    pattern = r'(\d+)\s+([a-zA-Z_ ]+)[：:\s]+([^$$]+)(?:\s*$([\d\-\s]*)$)?'
    matches = re.findall(pattern, output)
    for match in matches:
        person_id, action, description, time_range = match
        json_output.append({
            "person_id": person_id.strip(),
            "action": action.strip(),
            "description": description.strip(),
            "time": time_range.strip() if time_range else ""
        })

# 保存为 JSON 文件
os.makedirs("./qwvl", exist_ok=True)
output_path = "./qwvl/micro_actions_output.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(json_output, f, ensure_ascii=False, indent=2)

print(f"微动作 JSON 已保存至：{output_path}")

# 收集环境和 GPU 信息
env_info = {
    # "python_version": sys.version,
    # "torch_version": torch.__version__,
    # "cuda_available": torch.cuda.is_available(),
    # "cuda_device_count": torch.cuda.device_count(),
    "python_version": sys.version,
    "torch_version": torch.__version__,
    "transformers_version": transformers_version,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "flash_attention_2_enabled": True,
    "environment_variables": dict(_os.environ),
    "cuda_devices": [],
    "current_device": None,
    "device_name": None,
    "model_config": {},
    "processor_config": {},
    "script_args": sys.argv,
    "working_directory": _os.getcwd(),
    "cuda_devices": [
        {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "capability": torch.cuda.get_device_capability(i),
            "total_memory": f"{torch.cuda.get_device_properties(i).total_memory / (1024 ** 2):.2f} MB"
        }
        for i in range(torch.cuda.device_count())
    ],
    "current_device": None,
    "device_name": None
}

if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    env_info["current_device"] = current_device
    env_info["device_name"] = torch.cuda.get_device_name(current_device)

# 保存为 JSON 文件
output_env_path = "./qwvl/environment_info2.json"
os.makedirs(os.path.dirname(output_env_path), exist_ok=True)
with open(output_env_path, "w", encoding="utf-8") as f:
    json.dump(env_info, f, ensure_ascii=False, indent=2)

print(f"环境信息已保存至：{output_env_path}")