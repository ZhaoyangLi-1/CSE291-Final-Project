import json
import os

basic_image_infor_prompt = """
Analyze the provided image and identify all visible objects. For each object, provide its instance name (e.g., cabinet 1, cabinet 2, pot 1) and precise location within the image (e.g., upper left, center, bottom right). Highlight the relationships between objects, if any, and note any interactions or notable features. The analysis should include a list of identified objects, categorized by type, along with a description of their spatial positioning and any relevant details about their appearance or actions within the scene.
"""

basic_action_prompt="""
Given an image depicting a complex scene with multiple elements, your objective is to analyze the scene of the image and the text of the description of the image about the specified task description and choose the appropriate action from a list of available actions to accomplish the task based the summary of feedback from prior attempts. There are two examples.

**Example 1:**
- Available Actions:(1): go to bed 1, (2): go to dresser 1, (3): go to drawer 1
- Response:[BEGIN](2): go to dresser 1[END]
**Example 2:**
- Available Actions:(1): go to garbagecan 1, (2): take handtowel 1 from handtowelholder 1, (3): take faucet 1 from handtowelholder 1, (4): take handtowel 2 from handtowelholder 1
- Response:[BEGIN](2): take handtowel 1 from handtowelholder 1[END]

**The Summary of Feedback from Prior Attempts:**
{current_attempts}

**Task Description:**
- The task at hand is "{task_desc}". {task_hint}

**Image Descrption:**
{image_desc}

**The Available Actions:**
[{admissible_commands}]

**Response:**
- Let's tink step-by-step.
- (i): [BEGIN][Selected Action][END]
"""

short_memory_prompt = """
Your expertise lies in dissecting actions, visual outcomes, textual observations, and previously tried attempts associated with specific instance objects (e.g., Cabinet 1, Cabinet 2, Key 1, Key 2, etc.) to enhance task efficiency and effectiveness.

**Current Attempts Formulation:**
- Number of Current Tried Attempts: {num}
- Each current attempt contains one taken action, one image of post-action, and one text of the resulting observation.

**Initial Task Overview:**
- Beginning Observation: {initial_obs}. 

**Previously Tried Attempts:**
{previous_attempts}

**The Given Task Description:** 
- Task: "{task_desc}" There is the task hint of this task. {task_hint}

**Response:**
- The analysis of each attempt and then suggested the next step focused on tracking and evaluating the progress of given task based on the current and previously tried Attempts.
"""

long_memory_prompt = """
You are an expert in summarizing a complex text that intricately details various efforts or actions taken within a specific context. The text uniquely identifies each action or effort through distinct instance objects (e.g., Cabinet 1, Cabinet 2, Key 1, Key 2, etc.), examining the results, impacts, and proposing future directions. 

**Text for Summarization:**
{previous_attempts}

**Response:** 
- The detailed summary.
"""

refine_prompt = """
Your task in extracting the suggestion of the next step forward for the given text.

**The Given Text:**
{origianl_reflection}
 
**Response:**
- The suggestion of the next step.
"""

def to_json():
    saved_basic_image_infor_prompt = basic_image_infor_prompt[1:]
    saved_basic_image_infor_prompt = saved_basic_image_infor_prompt[:-1]
    
    saved_action_prompt = basic_action_prompt[1:]
    saved_action_prompt = saved_action_prompt[:-1]
   
    saved_short_memory_prompt = short_memory_prompt[1:]
    saved_short_memory_prompt = saved_short_memory_prompt[:-1]
   
    saved_long_memory_prompt = long_memory_prompt[1:]
    saved_long_memory_prompt = saved_long_memory_prompt[:-1]
   
    saved_refine_prompt = refine_prompt[1:]
    saved_refine_prompt = saved_refine_prompt[:-1]
   
    json_dic = {
      "pick_and_place_simple": "The agent must find an object of the desired type, pick it up, find the correct location to place it, and put it down there. Remember that you must reach your destination before picking and puting.",
      "look_at_obj_in_light": "The agent must find an object of the desired type, locate and turn on a light source with the desired object in-hand. Remember that you must reach your destination before interacting with the object.",
      "pick_clean_then_place_in_recep": "The agent must find an object of the desired type, pick it up, go to a sink or a basin, clean the object with a sink or basin (The action is \"clean object with sink or a basin\"), then find the correct location to place it, and put it down there.",
      "pick_heat_then_place_in_recep": "The agent must find an object of the desired type, pick it up, go to a microwave, heat the object with the microwave (The action is \"heat object with microwave\"), then find the correct location to place it, and put it down there.",
      "pick_cool_then_place_in_recep": "The agent must find an object of the desired type, pick it up, go to a fridge, cool the object with the fridge (The action is \"cool object with microwave\), then find the correct location to place it, and put it down there.",
      "pick_two_obj_and_place": "The agent must find an object of the desired type, pick it up, find the correct location to place it, put it down there, then look for another object of the desired type, pick it up, return to previous location, and put it down there with the other object.",
      "basic_image_infor_prompt": saved_basic_image_infor_prompt,
      "basic_action_prompt": saved_action_prompt,
      "short_memory_prompt": saved_short_memory_prompt,
      "long_memory_prompt": saved_long_memory_prompt,
      "refine_prompt": saved_refine_prompt
    }
    json_str = json.dumps(json_dic, indent=4)
   
    file_path = "mem_long_short_sep_prompts.json"
   
    with open(file_path, 'w') as file:
        file.write(json_str)
   
    print(f'Dictionary has been saved to {file_path}')
   

if __name__ == "__main__":
    print("Running Save to JSON.")
    to_json()





