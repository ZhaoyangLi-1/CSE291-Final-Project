import argparse
import os
import json
import sys
import requests
from base64 import b64decode
from pickle import loads
from collections import deque
import time
from utils.utils import (
    get_image,
    refine_action,
    format_obs_task_desc,
    get_done_paths,
    format_admissible_commands,
    read_prompt,
    get_path_tasks,
)

os.environ["AGI_ROOT"] = "/home/zhaoyang/projects/neural-reasoning"
#os.environ["AGI_ROOT"] = "/root/neural-reasoning"
sys.path.append(os.path.join(os.environ["AGI_ROOT"]))
from agi.utils.openai_utils import get_total_money
from agi.utils.chatbot_utils import DecodingArguments, ChatBot

ALFWORLD_DATA = os.getenv("ALFWORLD_DATA")
ALFWORLD_SAVE = os.getenv("ALFWORLD_SAVE")
CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))

# Generate image informarion description
def geneare_image_infro_prompt():
    return "<image>\n" + basic_image_infor_prompt


# Generate action selection prompt
def generate_action_prompt(
    initial_obs, task_desc, admissible_commands, current_attempts, task_hint, image_desc
):
    if len(current_attempts) == 0:
        current_attempts = "Nothing"
    return "<image>\n" + basic_action_prompt.format(
        task_desc=task_desc,
        admissible_commands=admissible_commands,
        current_attempts=current_attempts,
        initial_obs=initial_obs,
        task_hint=task_hint,
        image_desc=image_desc
    )


# Generate current attempt memory _prompt
def generate_attempt_memory_prompt(
    initial_obs,
    task_desc,
    tried_actions,
    obs_list,
    images,
    previous_attempts,
    task_hint,
):
    if len(previous_attempts) == 0:
        previous_attempts = "No the summary of previously tried attempts."
    num = len(images)
    trajectory = "**Current Attempts:**\n"
    for i in range(num):
        tried_action = tried_actions[i]
        obs = obs_list[i]
        trajectory += f"Current Attempt {i+1}:\n- Action Taken: {tried_action}\n- Resulting Observation: {obs}\n- <image>\n"

    relfect_ptompt = trajectory + basic_gpt_reflect_prompt.format(
        initial_obs=initial_obs.lower(),
        task_desc=task_desc,
        num=num,
        previous_attempts=previous_attempts,
        task_hint=task_hint,
    )
    return relfect_ptompt


# Main function to run all scenes
def test_scenes(args):
    global basic_llm_prompt, basic_action_prompt, basic_gpt_reflect_prompt, task_hints_prompt, refine_prompt, basic_image_infor_prompt
    prompts, task_hints_prompt = read_prompt("prompts/mem_long_short_sep_prompts.json")
    basic_llm_prompt = prompts["long_memory_prompt"]
    basic_image_infor_prompt = prompts["basic_image_infor_prompt"]
    basic_action_prompt = prompts["basic_action_prompt"]
    basic_gpt_reflect_prompt = prompts["short_memory_prompt"]
    refine_prompt = prompts["refine_prompt"]

    env_url = "http://127.0.0.1:" + str(args.env_url)
    # initial VLM and LLM model
    print(f"VLM Model: {args.vlm_model}")

    # Setup VLM(gpt4-v) model as action selector and current attempts relfector
    short_memory_vlm_decoding_args = DecodingArguments(
        max_tokens=8192, n=1, temperature=0.5, image_detail="auto"
    )
    actor_vlm_model = ChatBot("gpt-4-vision-preview")
    reflect_vlm_model = ChatBot("gpt-4-vision-preview")

    # Setup LLM (gpt4) model for previous attempts relfector
    llm_query_decoding = DecodingArguments(max_tokens=32768, n=1, temperature=0.4)
    gpt_llm_model = ChatBot(args.llm_model)

    refine_query_decoding = DecodingArguments(max_tokens=4096, n=1, temperature=0.4)
    refine_llm_model = ChatBot(args.refine_llm_model)

    # recored the number of success of scenes
    num_succeess = 0

    # Check whether continue from last time
    save_path = os.path.join(ALFWORLD_SAVE, args.save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    succeed_num_save_path = os.path.join(
        save_path, f"succeed_num_begin_{args.begin_scene}.log"
    )
    done_json_paths = []
    no_head = False
    if os.path.exists(succeed_num_save_path):
        print("Continue from the last time")
        with open(succeed_num_save_path, "r") as f:
            lines = f.readlines()
            done_json_paths = get_done_paths(lines)
            for idx, line in enumerate(lines):
                if "SUCCEED" in line and "UNSUCCEED" not in line:
                    num_succeess += 1
                if idx == 0 and "Begin Scene:" not in line:
                    no_head = True
    else:
        no_head = True

    task_list_path = os.path.join(CURRENT_FOLDER, args.task_list_path)
    json_file_list = get_path_tasks(task_list_path)

    scenes_steps = args.total_scene // args.num_server
    json_file_list = sorted(
        json_file_list[
            args.begin_scene : min(args.begin_scene + scenes_steps, args.total_scene)
        ]
    )
    json_file_list = sorted(
        [item for item in json_file_list if item not in done_json_paths]
    )
    num_done = len(done_json_paths)

    if no_head:
        with open(succeed_num_save_path, "a") as succeed_num_f:
            succeed_num_f.write(
                f"Begin Scene: {args.begin_scene}     End Scene: {args.begin_scene+scenes_steps-1}      Number of Server: {args.num_server}\n"
            )
            succeed_num_f.close()

    # Set envrionment
    set_dic = {"env_type": "visual", "batch_size": 1}
    requests.post(env_url + "/set_environment", json=set_dic).text
    for scene_idx, json_file in enumerate(json_file_list):
        # Get task type
        with open(json_file, "r") as f_task:
            task_json = json.load(f_task)
            task_type = task_json["task_type"]
        task_hint = task_hints_prompt[task_type]
        # Setup configs
        # The current attempts
        current_attempts = ""
        previous_attempts = ""
        LLM_summary = ""
        refine_attempts=""
        max_step = 50
        succeed = False
        max_num = 3
        # Setup queue to store previous memory
        tried_actions = deque(maxlen=max_num)
        obs_list = deque(maxlen=max_num)
        images = deque(maxlen=max_num)

        # Setup Log path
        scene_idx = scene_idx + num_done + args.begin_scene
        print(f"Current scene: {scene_idx}")
        scene_save_root = os.path.join(save_path, f"scene_{scene_idx}")
        if not os.path.exists(scene_save_root):
            os.makedirs(scene_save_root)
        scene_save_path = os.path.join(scene_save_root, f"scene_{scene_idx}.log")
        with open(scene_save_path, "a") as scene_f:
            text = b64decode(
                eval(
                    requests.post(
                        env_url + "/reset", json={"json_file": json_file}
                    ).text
                )
            )

            obs, infos = loads(text)
            admissible_commands = format_admissible_commands(
                infos["admissible_commands"][0]
            )
            initial_obs, task_desc = format_obs_task_desc(obs)
            if initial_obs == "" or task_desc == "":
                continue
            scene_f.write(
                f"---------------------------------------------------------Scene: {scene_idx}---------------------------------------------------------\n"
            )
            scene_f.write(
                f"--------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"
            )
            scene_f.write(
                f"--------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"
            )

            print(f"Task Description: {task_desc}")
            for step in range(max_step):
                if step % max_num == 0 and step != 0:
                    start_time = time.time()
                    previous_attempts = (
                        previous_attempts
                        + f"Earlier Attempts {int((step + 1)/max_num)}:\n"
                        + current_attempts
                        + "\n\n"
                    )
                    llm_message = basic_llm_prompt.format(
                        previous_attempts=previous_attempts
                    )
                    LLM_summary = (
                        gpt_llm_model.call_model(
                            llm_message,
                            decoding_args=llm_query_decoding,
                            return_list=False,
                        )
                        .strip()
                        .replace("\n\n", "\n")
                    )
                    end_time = time.time()
                    scene_f.write(
                        f"\n**Earlier Memory Attempt Prompt:**\n{previous_attempts}\n\n"
                    )
                    scene_f.write(
                        f"Earlier Memory Attempt:\n{LLM_summary}\nEarlier Memory Attempt Running time: {end_time - start_time} seconds\n\n"
                    )

                image_root = os.path.join(scene_save_root, f"images")
                if not os.path.exists(image_root):
                    os.makedirs(image_root)
                image_path = os.path.join(image_root, f"scene_{scene_idx}_{step}.jpg")
                image = get_image(env_url, args.is_ins_seg)
                image.save(image_path, "JPEG")

                scene_f.write(
                    f"Step:--------------------------------------------------------------------------{step+1}--------------------------------------------------------------------------\n"
                )

                messages = {
                    "text": geneare_image_infro_prompt(),
                    "images": [image],
                }
                image_desc = actor_vlm_model.call_model(messages, decoding_args=short_memory_vlm_decoding_args, return_list=False)
                
                # Use VLM to get the current admissble action
                prompt = generate_action_prompt(
                    initial_obs,
                    task_desc,
                    admissible_commands,
                    refine_attempts,
                    task_hint,
                    image_desc
                )
                scene_f.write(f"VLM Prompt:\n{prompt}\n")

                messages = {
                    "text": prompt,
                    "images": [image],
                }

                start_time = time.time()
                for tried in range(5):
                    action_vlm_decoding_args = DecodingArguments(
                        max_tokens=8192,
                        n=1,
                        temperature=0.5 + (tried) * 0.1,
                        image_detail="auto",
                    )
                    action = actor_vlm_model.call_model(
                        messages,
                        decoding_args=action_vlm_decoding_args,
                        return_list=False,
                    ).strip()
                    # print(f"Original Response is: {action}")
                    scene_f.write(f"\nOriginal VLM Response:\n{action}\n\n")
                    action = refine_action(action).strip().replace("\n", "")
                    if "No action" != action:
                        break
                end_time = time.time()
                tried_actions.append(action)
                scene_f.write(
                    f"> Action: {action}\nRunning time: {end_time - start_time} seconds\n\n"
                )
                print(f">> Action is: {action}")

                # Interact with envrionment and get resulting observation for the chosen action
                text = b64decode(
                    eval(requests.post(env_url + "/step", json={"action": action}).text)
                )
                obs, _, done, infos = loads(text)
                obs, _, done, admissible_commands = (
                    obs[0],
                    infos["won"][0],
                    done[0],
                    infos["admissible_commands"][0],
                )
                obs_list.append(obs)
                actioned_image = get_image(env_url, args.is_ins_seg)
                images.append(actioned_image)

                admissible_commands = format_admissible_commands(admissible_commands)
                succeed = done
                goal_condition_success_rate = infos["goal_condition_success_rate"][0]

                # Log SUCCEED task and the Goal condition success rate if the task is SUCCEED
                if succeed:
                    image_path = os.path.join(
                        image_root, f"scene_{scene_idx}_{step+1}.jpg"
                    )
                    image = get_image(env_url, args.is_ins_seg)
                    image.save(image_path, "JPEG")
                    with open(succeed_num_save_path, "a") as succeed_num_f:
                        succeed_num_f.write(
                            f"{scene_idx} Path:{json_file}: SUCCEED, Goal condition success rate: {goal_condition_success_rate}\n"
                        )
                        succeed_num_f.close()
                    num_succeess += 1
                    print("SUCCEED")
                    scene_f.write("SUCCEED\n")
                    break

                # Using VLM to relfect recent summary and suggestion (Number of length: max_num)
                start_time = time.time()
                reflect_prompt = generate_attempt_memory_prompt(
                    initial_obs,
                    task_desc,
                    tried_actions,
                    obs_list,
                    images,
                    LLM_summary,
                    task_hint,
                )
                messages = {"text": reflect_prompt, "images": images}
                current_attempts = (
                    reflect_vlm_model.call_model(
                        messages,
                        decoding_args=short_memory_vlm_decoding_args,
                        return_list=False,
                    )
                    .strip()
                    .replace("\n\n", "\n")
                )
                # current_attempts = refine_short_mem(current_attempts)
                refine_attempts = refine_llm_model.call_model(
                    refine_prompt.format(origianl_reflection=current_attempts),
                    decoding_args=refine_query_decoding,
                    return_list=False,
                )
                end_time = time.time()
                scene_f.write(f"Current Attempts Reflect Prompt:\n{reflect_prompt}\n\n")
                scene_f.write(
                    f"Current Attempts Reflect:\n{current_attempts}\nCurrent Attempts Reflect Running time: {end_time - start_time} seconds\n\n"
                )
                scene_f.write(
                    f"Refined Attempts Reflect:\n{refine_attempts}\nCurrent Attempts Reflect Running time: {end_time - start_time} seconds\n\n"
                )

            # Log UNSUCCEED task and the Goal condition success rate if the task is UNSUCCEED
            if not succeed:
                with open(succeed_num_save_path, "a") as succeed_num_f:
                    print("UNSUCCEED")
                    succeed_num_f.write(
                        f"{scene_idx} Path:{json_file}: UNSUCCEED, Goal condition success rate: {goal_condition_success_rate}\n"
                    )
                    succeed_num_f.close()
                scene_f.write("UNSUCCEED\n")
        scene_f.close()

    total_money = get_total_money()
    print(f"Total Money Cost: {total_money}")

    # Compute the success rate of total sample scenes
    with open(succeed_num_save_path, "a") as succeed_num_f:
        succeed_num_f.write(f"\nTotal number: {len(scenes_steps)}\n")
        succeed_num_f.write(f"Total succeed number: {num_succeess}\n")
        succeed_num_f.write(f"Total Money Cost: {total_money}\n")

        succeed_num_f.close()

    requests.post(env_url + "/close", json={})


if __name__ == "__main__":
    print("Running alfworld_memory_cot_new.py")
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm_model", default="gpt-4-vision-preview", type=str)
    parser.add_argument("--llm_model", default="gpt-4-32k-0613", type=str)
    parser.add_argument("--refine_llm_model", default="gpt-4-0613", type=str)
    parser.add_argument(
        "--save_path",
        # default="gpt4_random_100_cot_ins_seg",
        default="test",
        type=str,
    )
    parser.add_argument(
        "--alfread-json-path",
        type=str,
        default="json_2.1.1/valid_train",
    )
    parser.add_argument(
        "--task_list_path",
        type=str,
        default="task_json.pkl",
    )
    parser.add_argument(
        "--env_url",
        type=str,
        default=3000,
    )
    parser.add_argument(
        "--is_ins_seg",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--begin_scene",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--num_server",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--total_scene",
        type=int,
        default=100,
    )
    args = parser.parse_args()
    test_scenes(args)
