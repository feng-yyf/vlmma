import os
import cv2
import torch
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 设置路径
model_path = "/home/bd2/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct"
video_path = "train0028.mp4"
clip_dir = "./clips"
output_json = "./qwvl/micro_actions_output.json"
os.makedirs(clip_dir, exist_ok=True)
os.makedirs("./qwvl", exist_ok=True)

# 加载模型和处理器
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    model_path,
    min_pixels=min_pixels,
    max_pixels=max_pixels
)

# 视频切片函数
def split_video(video_path, output_dir, clip_duration=1.5, stride=0.5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_len = int(clip_duration * fps)
    stride_len = int(stride * fps)
    idx = 0
    for start in range(0, total_frames - clip_len + 1, stride_len):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        out_path = os.path.join(output_dir, f"clip_{idx:04d}.mp4")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                              (int(cap.get(3)), int(cap.get(4))))
        for _ in range(clip_len):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()
        idx += 1
    cap.release()

# 推理函数
def infer_clip(clip_path, fps):
    prompt = """
请详细描述视频信息，重点关注人物微动作（幅度极小时间极短），所有微动作如下：0 shaking body ... 51 scratching feet，输出格式为：人物编号+微动作+描述文本+微动作部位定位框@[时间段] 人物编号+对话@[时间段]
"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": clip_path, "max_pixels": 360 * 420, "fps": fps},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    return output_text[0]

# 主流程
if __name__ == "__main__":
    print("开始切分视频...")
    split_video(video_path, clip_dir)

    print("开始推理每个视频片段...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    results = []
    for clip_file in sorted(os.listdir(clip_dir)):
        clip_path = os.path.join(clip_dir, clip_file)
        print(f"推理：{clip_path}")
        try:
            output = infer_clip(clip_path, fps)
            results.append({"clip": clip_file, "output": output})
        except Exception as e:
            print(f"推理失败 {clip_file}：{e}")

    print(f"保存结果到 {output_json}")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("处理完成！")
