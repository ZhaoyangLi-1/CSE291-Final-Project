import argparse
import torch
import os
import json
from torchmetrics.text.rouge import ROUGEScore
import torch.nn as nn
import random
from llava_run import gen_output, load_model

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
from Ours.utils import get_img_json_files, ensure_period, get_gpt_chat, format_output
from agi.utils.model_utils import ChatBot
from agi.utils.openai_utils import login_openai
import openai


def generate_prompt(task_desc, obj_desc=None, is_first=True):
    if is_first:
        prompt = "Analyze the attached image, which depicts a specific environment with various objects and elements. Identify and describe the spatial layout, including the positioning and characteristics of objects in the setting. Based on this detailed analysis, outline a comprehensive plan to accomplish the following high-level task: '{task_desc}' Break down this task into smaller, manageable sub-tasks, explicitly relating them to the objects and spatial layout identified in the image. For each sub-task, provide a detailed step-by-step guide, including how to utilize the objects and elements in the environment effectively, and the sequence in which these actions should be performed. Ensure that the plan accounts for potential variations or challenges in this setting, as observed in the image. Additionally, suggest alternative approaches for each sub-task, considering different scenarios or constraints present in the environment.".format(
            task_desc=task_desc
        )
    else:
        distil_prompt = "Imagine the environment that is similar to the one in the uploaded image, and the detailed plans and information of objects in the image also shown previously. What's more, we also know that available actions are: 'heat', 'clean', 'slice', 'put', toggle, 'pick up', and 'go to'.  Please generate a sequence of brief actions (3-4 words) relevant to a task, which should consist solely of a fine-grained sequence of actions."
        prompt = prompt + distil_prompt
    return prompt


def remove_from_substring_to_end(text, phrase="Alternative approaches:"):
    index = text.find(phrase)
    return text[:index].strip() if index != -1 else text


def main(args):
    tokenizer, model, image_processor, model_name = load_model(args)

    # gpt_model = "gpt-3.5-turbo-16k"
    # account = "gpt4-4"
    # login_openai(account)
    # ChatBot.init(gpt_model, url=21002, use_cpp=False)

    image_list, json_list = get_img_json_files(args.image_path)
    rouge = ROUGEScore()
    for idx in range(len(image_list)):
        print("Processing image {}".format(idx))
        log_list = []
        rouge_values = []

        with open(json_list[idx], "r") as f:
            anns = json.load(f)["anns"]
        args.image_file = image_list[idx]
        out_put_path = args.save_path + "/" + "output_{}".format(idx)
        if not os.path.exists(out_put_path):
            os.makedirs(out_put_path)
        for ann_idx, ann in enumerate(anns):
            print("Processing task {}".format(ann_idx))
            true_instructions = [
                f"{idx + 1}. {instruction.replace(',', '.')}"
                for idx, instruction in enumerate(ann["high_descs"])
            ]

            args.query = generate_prompt(ensure_period(ann["task_desc"]))
            response = gen_output(args, tokenizer, model, image_processor, model_name)
            response = remove_from_substring_to_end(response)
            print(response)
            args.query = generate_prompt(response, False)

            response = gen_output(args, tokenizer, model, image_processor, model_name)

            rouge_score = rouge(response, true_instructions)

            format_roge_score = {
                key: round(value.item(), 3) for key, value in rouge_score.items()
            }

            rouge_values.append(rouge_score)


            log = {
                "task_desc": [
                    ensure_period(task_desc, False)
                    for task_desc in args.query.split(". ")
                ],
                "output_plan": format_output(response),  # Convert Tensor to list
                "true_plan": true_instructions,  # Convert Tensor to list
                "rouge_score_main_plan": format_roge_score,  # Convert Tensor to list if it's a Tensor
            }

            log_list.append(log)

        with open(out_put_path + "/" + "output.json", "w") as f:
            json.dump(log_list, f, indent=4)

        image = Image.open(args.image_file)
        image.save(out_put_path + "/" + "input_image.jpg")
        fig_, _ = rouge.plot(rouge_values)
        fig_.savefig(out_put_path + "/" + "rouge_score_plot.png")

    # ChatBot.close()


if __name__ == "__main__":
    print("Running test_llava.py")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--image_file", type=str, required=None)
    parser.add_argument("--query", type=str, required=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--image_path", type=str, default="./new_data")
    parser.add_argument("--save_path", type=str, default="./experiment/output_13b")
    parser.add_argument(
        "--prompt_path", type=str, default="./data/json_2.1.0/valid_seen"
    )
    args = parser.parse_args()

    main(args)
