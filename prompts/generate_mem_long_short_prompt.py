# basic_action_prompt = """
# Given an image depicting a complex scene with multiple elements and a summary of feedback from prior attempts, your objective is to analyze the scene of the image about the specified task description and the summary of feedback from prior attempts, and then choose the appropriate action from a list of available actions to accomplish the task. There are two examples.

# **Example 1:**
# - Available Actions:(1): go to bed 1, (2): go to dresser 1, (3): go to drawer 1
# - Response:[BEGIN](2): go to dresser 1[END]
# **Example 2:**
# - Available Actions:(1): go to garbagecan 1, (2): take handtowel 1 from handtowelholder 1, (3): take faucet 1 from handtowelholder 1, (4): take handtowel 2 from handtowelholder 1
# - Response:[BEGIN](2): take handtowel 1 from handtowelholder 1[END]

# **Task Description:**
# - The task at hand is "{task_desc}". {task_hint}

# **The Summary of Feedback from Prior Attempts:**
# {current_attempts}

# **The Available Actions:**
# {admissible_commands}

# **Determine the appropriate action:**
# - Identify relevant objects in the image related to the task. Also consider the summary of feedback from prior attempts
# - Evaluate each action's potential to successfully complete the task.

# **Response:**
# - Let's tink step-by-step.
# - (i): [BEGIN][Selected Action][END]

# **Guidelines for Action Decision:**
# 1. Scrutinize the image to pinpoint objects and elements pertinent to the task.
# 2. Reflect on the summarized feedback from earlier attempts to identify pitfalls and successful strategies.
# 3. Assess the feasibility and effectiveness of each listed action within the context of the current scene and past insights.
# 4. Choose the most resoable action from the action selection list.
# """

# basic_action_prompt = """
# Leverage the visual context from an image and the thought to inform your decision-making process for task completion. Your role involves a detailed analysis of a scene captured in the image, relating it to the task at hand, and integrating lessons learned from previous efforts to identify the most viable action that will lead to success.

# **Task Overview:**
# - Objective: "{task_desc}". {task_hint}

# **Thought:**
# {current_attempts}

# **Action List:**
# {admissible_commands}

# **Response:**
# - (i): [Selected Action]", where 'i' is the numerical position of the chosen action in the list.
# """

basic_action_prompt = """
**Objective:** Achieve "{task_desc}". There is a hint about the given task. {task_hint}.

**Visual Context:** Given the image, closely examine all elements within to gather clues and insights relevant to the objective. Consider how the arrangement, interactions, and details of the scene relate to the task at hand.

**Reflective Thought:**
{current_attempts}. 

**Admissible Actions List:**
{admissible_commands}

**Output Examples:**
*Example 1*:
- Admissible Actions:(1): go to bed 1, (2): go to dresser 1, (3): go to drawer 1
- Response:[BEGIN](2): go to dresser 1[END]
*Example 2*:
- Admissible Actions:(1): go to garbagecan 1, (2): take handtowel 1 from handtowelholder 1, (3): take faucet 1 from handtowelholder 1, (4): take handtowel 2 from handtowelholder 1
- Response:[BEGIN](2): take handtowel 1 from handtowelholder 1[END]

**Response:**
- Choose an action from the admissible actions list directly contributing to the task objective based on the reflective thought and the visual context, are the form: [BEGIN](i): [Selected Action][END]
"""


# short_memory_prompt = """
# Your task is conducting a reflective analysis and next the suggestion of next step focused on tracking and evaluating the progress of various household tasks. Your expertise lies in dissecting actions, visual outcomes, and textual observations, which associate with specific instance objects (e.g., Cabinet 1, Cabinet 2, Key 1, Key 2, etc.), to enhance task efficiency and effectiveness.

# **Detailed Examination of Attempts:**
# - Number of Current Tried Attempts: {num}
# - For each current attempt, analyze the following:
#   - Action Taken
#   - Image Post-Action
  
# **Current Attempts**
# {attempt_history}

# **Summary of Previously Tried Attempts:**
# {previous_attempts}

# **The Given Task Description:** 
# - Task: "{task_desc}" There is the task hint of this task. {task_hint}

# **Response:**
# - Let's think step-by-step.
# """

short_memory_prompt="""
Interact with a household to solve a task. At begining, {initial_obs} You have {num} sets of current tried attempts, and each attempt consists of an action, an image of the action after it happened, and a resulting observation.

**You also have the summary of previously tried attempts:** 
{previous_attempts}

**For the task: '{task_desc}', you need to do**:
a. Analyze the impact of each current tried attempt including the image, resulting observation, and taken action on the task.
b. Combine the summary of previously tried attempts and the analysis of currently tried attempts to give a suggestion of the next step to complete the task.

**Response**: 
Let's think step-by-step.
"""


# long_memory_prompt = """
# You are an expert in summarizing a complex text that intricately details various efforts or actions taken within a specific context. The text uniquely identifies each action or effort through distinct instance objects (e.g., Cabinet 1, Cabinet 2, Key 1, Key 2, etc.), examining the results, impacts, and proposing future directions. 

# **Text for Summarization:**
# {previous_attempts}

# **Instructions:**
# - Focus on the identifier numbers (such as Cabinet 1, Cabinet 2) and their significance within the narrative.
# - Extract and synthesize key information, emphasizing:
#    - The outcomes of the actions associated with each instance object with identifier number.
#    - The collective impact of these actions, highlighting connections to the specific instance objects.
#    - Recommendations for future initiatives or strategies, with a particular emphasis on those related to the identified objects.

# **Response:** A detailed yet succinct summary. This summary should clearly articulate the outcomes and impacts linked to the respective instance objects and elucidate the text's main insights and recommendations.
# """

long_memory_prompt = """
Your expertise is required to distill complex narratives into concise summaries. These narratives detail various efforts, each uniquely associated with specific instance objects (e.g., Cabinet 1, Cabinet 2, Key 1, Key 2, etc.). The narratives explore the outcomes and implications of these efforts and suggest future directions.

**Narrative for Summarization (Tried Attempts):**
{previous_attempts}

**Summarization Objectives:**
- Pay special attention to unique identifiers (like Cabinet 1, Cabinet 2) and their roles in the narrative.
- Your summary should:
   1. Outline the results linked to actions involving each identified instance object.
   2. Discuss the combined effects of these actions, with a focus on their relation to the specific instance objects.
   3. Offer insights into prospective actions or strategies, especially those concerning the mentioned objects.

**Response:**
- Your summary should offer a clear and concise recapitulation of the actions' outcomes and their broader impacts. It should also provide strategic recommendations, all through the lens of the specific instance objects mentioned in the narrative.
"""



# refine_prompt = """
# Your task in extracting the suggestion of the next step forward for the given text.

# **The Given Text:**
# {origianl_reflection}
 
# **Response:**
# - The suggestion of the next step.
# """


import json

def to_json():
   saved_action_prompt = basic_action_prompt[1:]
   saved_action_prompt = saved_action_prompt[:-1]
   
   saved_short_memory_prompt = short_memory_prompt[1:]
   saved_short_memory_prompt = saved_short_memory_prompt[:-1]
   
   saved_long_memory_prompt = long_memory_prompt[1:]
   saved_long_memory_prompt = saved_long_memory_prompt[:-1]
   
   # saved_refine_prompt = refine_prompt[1:]
   # saved_refine_prompt = saved_refine_prompt[:-1]
   
   json_dic = {
      "pick_and_place_simple": "The agent must find an object of the desired type, pick it up, find the correct location to place it, and put it down there. Remember that you must reach your destination before picking and puting.",
      "look_at_obj_in_light": "The agent must find an object of the desired type, locate and turn on a light source with the desired object in-hand. Remember that you must reach your destination before interacting with the object.",
      "pick_clean_then_place_in_recep": "The agent must find an objectof the desired type, pick it up, go to a sink or a basin, clean the object with a sink or basin, then find the correct location to place it, and put it down there. Remember that you must reach the sink or basin before cleaning.",
      "pick_heat_then_place_in_recep": "The agent must find an object of the desired type, pick it up, go to a microwave, heat the object with the microwave, then find the correct location to place it, and put it down there. Remember that you must reach the microwave before heating.",
      "pick_cool_then_place_in_recep": "The agent must find an object of the desired type, pick it up, go to a fridge, cool the object with the fridge, then find the correct location to place it, and put it down there. Remember that you must reach the fridge before cooling.",
      "pick_two_obj_and_place": "The agent must find an object of the desired type, pick it up, find the correct location to place it, put it down there, then look for another object of the desired type, pick it up, return to previous location, and put it down there with the other object. Remember that you must reach your destination before picking and puting.",
      "basic_action_prompt": saved_action_prompt,
      "short_memory_prompt": saved_short_memory_prompt,
      "long_memory_prompt": saved_long_memory_prompt,
      # "refine_prompt": saved_refine_prompt
   }
   json_str = json.dumps(json_dic, indent=4)
   
   file_path = "mem_long_short_prompt.json"
   
   with open(file_path, 'w') as file:
      file.write(json_str)
   
   print(f'Dictionary has been saved to {file_path}')
   

if __name__ == "__main__":
   print("Running Save to JSON.")
   to_json()
