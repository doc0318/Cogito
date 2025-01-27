from openai import OpenAI
from human_eval.data import write_jsonl, read_problems
import time
import re
from typing import Optional
import os
import clean_data
from text_manager import getText, checklen
import json
from datetime import datetime
import tiktoken
from github import search_github_code, retry_request, extract_problems,extract_high_score_solution
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from test_in_GPTclient import run_test_humaneval
import http.client

# tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
# model = AutoModel.from_pretrained("microsoft/codebert-base")


client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="your open ai key",
)

text = []
role_solutions = {}
total_token = 0
task_file= 'HumanEvalWST.jsonl'
# task_file = 'mbppWST.jsonl'
# task_file = 'selected150.jsonl'
# task_file = "codeEval.jsonl"
# task_file = "codecontest_with_id.jsonl"
# solution_file = 'samples_1.jsonl'


def adjust_message_weight(previous_output, current_role, quality_score, traceback):
    # adjusted_message = adjust_for_task_type(current_role, task_type, previous_output)
    adjusted_message = previous_output
    weight = adjust_for_independence(current_role)
    importance_score = calculate_importance_score(quality_score, traceback) * weight

    return {
        'message': adjusted_message,
        # 'result':traceback,
        'importance_score': importance_score
    }


def adjust_for_task_type(current_role, task_type, previous_output):
    if task_type == 'complex_planning':
        adjusted_message = f"Planning stage message: {previous_output}"
    elif task_type == 'code_fix':
        adjusted_message = f"Fixing code based on test results: {previous_output}"
    else:
        adjusted_message = previous_output

    return adjusted_message


def adjust_for_independence(current_role):
    independence_factors = {
        'Testing': 1,
        'Implementation': 0.6,
        'Design': 0.4,
        'NCTesting': 1
    }


    role_weight = independence_factors.get(current_role)

    return role_weight


def calculate_importance_score(quality_score, result_traceback=None):

    if result_traceback.lower() == "passed":
        importance_score = 0.9 * quality_score
    else:
        importance_score = 0.2 * quality_score

    importance_score = min(max(importance_score, 0), 1)

    return importance_score


def next_role_decision(received_info):

    message = received_info['message']
    importance_score = received_info['importance_score']

    if importance_score > 0.09:
        print(f"High importance: Using historical context heavily. Message: {message}")
        process_message = message
    else:
        print(f"Low importance: Prioritizing new information. Message: {message}")
        process_message = generate_new_output()

    return process_message


def generate_new_output():
    return "Generated new output based on current task requirements."



class GroupMember:
    def __init__(self, name, role):
        self.name = name
        self.role = role



    def provide_solution(self, question, tb=None, implementation_solution=None, design_solution=None,
                         messages=None, text=None, prompt_name=None, task_id=None):
        if text is None:
            text = []
        if messages is None:
            messages = [{"role": "system", "content": "You are a professional code developer."}]


        if self.role == "Design":

            prompt = (
                f"Provide guided steps to solve the following problem and identify potential challenges.: {question}. "
                f"[requirement]: less text, don't give code")
            messages.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )


            return str(response.choices[0].message.content)




        elif self.role == "Implementation":
            examples = retry_request(lambda: search_github_code(query=question))
            test_case = extract_problems(task_id)


            current_question = question
            current_question = current_question.strip()
            reference = None
            reference_id = None
            max_similarity = 0.6
            # with open("D:\\PYTHON_FILE\\pythonProject1\\human-eval-v2-20210705.jsonl", 'r',encoding='utf-8') as file:
            #     for line in file:
            #         task = json.loads(line.strip())
            #         similarity = calculate_similarity(task['prompt'],current_question)
            #         if max_similarity < similarity < 1:
            #             reference_id = task['task_id']
            #
            # with open("D:\\PYTHON_FILE\\pythonProject1\\samples.jsonl", 'r', encoding='utf-8') as f:
            #     for line in f:
            #         task = json.loads(line.strip())
            #         if task['task_id'] == reference_id:
            #             reference = task.get('completion')

            prompt = (
                      f"As a code expert, according to the guidance:{design_solution}"
                      f"please provide a python solution to the following programming problem: {question}."
                      f"Ensure that the answer produced by your code matches the test cases in the examples:{test_case}"
                      # f"Some examples:{examples}"
                      # f"Similar passed tasks:{reference}"
                      # f"The function name must be the same as in the problem{prompt_name}" #APPS
                      f"[Important]only give the code and should not include any explanations or comments."
                      # f"[Important]:Use a function to solve the problem, ending with a return.All the code is inside the function."
                      # f"Make sure the function only requires a single string parameter."
                    )
            messages.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            implementation_code = str(response.choices[0].message.content)
            implementation_code = clean_completion(implementation_code)





            return implementation_code



        elif self.role == "Testing":
            max_attempts = 1
            attempts = 0
            result_traceback = None
            test_code = None
            test_case = extract_problems(task_id)

            while attempts < max_attempts:
                result_traceback = run_test_humaneval(task_file, implementation_solution, 3.0, task_id, problem=question)

                if isinstance(result_traceback, str) and result_traceback.lower() == "passed":
                    return implementation_solution


                prompt = (
                    f"According to the {question}, the code given is:{implementation_solution} "
                    f":Fix it using traceback:{result_traceback}. "
                    f"[Important]Only give code don't analyze and no annotation"
                    # f"Make sure the function name is the same as in the problem{prompt_name}"  #Apps
                    # f"[Important]:Use a function to solve the problem, ending with a return."
                    # f"Make sure the function only requires a single string parameter.All the code is inside the function."
                    # f"Only code no comments or other things" #codeEval
                )
                messages.append({"role": "user", "content": prompt})
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=messages
                )

                test_code = str(response.choices[0].message.content)
                # print(test_code)
                if not any(keyword in test_code for keyword in ["def", "return", "import", "class", "for", "if"]):
                    test_code = implementation_solution
                else:
                    test_code = clean_completion(test_code)

                if not test_code:
                    test_code = implementation_solution
                attempts += 1


            return test_code


class SequentialDiscussionGroup:
    def __init__(self, members_with_roles, common_member_name, common_member_initial_role):
        self.members = [GroupMember(name, role) for name, role in members_with_roles]
        self.common_member = GroupMember(common_member_name, common_member_initial_role)
        self.testing_solutions = []
        self.execution_result = None




    def provide_solution_with_role_change(self, question, task_id, text_1, prompt_name):
        global role_solutions
        if 'task_id' not in role_solutions:
            role_solutions = {
                'task_id': task_id,
                'Design': [],
                'Implementation': [],
                'Testing': [],
                'NCTesting':[]
            }



        # 1. Planner
        design_member = next(member for member in self.members if member.role == "Design")
        design_solution = None

        if self.common_member.role == "Design":
            merged_question = []
            for item in text_1:
                role = 'assistant' if item['role'].lower() != 'user' else item['role']
                merged_question.append({"role": role, "content": item["content"]})
            implementation_solution = role_solutions['Implementation']
            score = 0.3  #defaul
            if implementation_solution and implementation_solution[0].get('importance_score'):
                score = implementation_solution[0]['importance_score']
            if score > 0.03:
                messages = merged_question
            else:
                messages = []
            design_solution = design_member.provide_solution(messages = messages,task_id=task_id,question=question,prompt_name = prompt_name)
            print("com design")
            if design_solution:
                # solution = clean_completion(role_solutions['NCTesting'][-1])
                # result = run_test_humaneval(task_file, solution, 3.0, task_id, problem=question)
                # if result.lower() == "passed":
                #     importance_score = 0.9
                # else:
                #     importance_score = 0.2
                #     last_line = re.search(r"([A-Za-z]+Error.*)$", result, re.MULTILINE)
                #     result = last_line.group(1)
                # design_solution = adjust_message_weight(design_solution, 'Design', importance_score, result)
                design_solution = adjust_message_weight(design_solution, 'Design', 0.4, "passed")
                getText("Design", str(design_solution))
                # design_solution = tokenizer(design_solution, padding=True, truncation=True, return_tensors="pt")
                role_solutions['Design'].append(design_solution)

                # print("com_design:\n",role_solutions['Design'])
        else:
            design_solution = design_member.provide_solution(question=question,task_id=task_id,prompt_name = prompt_name)
            print("design")

            # design_solution = tokenizer(design_solution, padding=True, truncation=True, return_tensors="pt")
            # role_solutions['Design'].append(design_solution)


            # print("design:\n",role_solutions['Design'])


        # 2. Coder
        implementation_member = next(member for member in self.members if member.role == "Implementation")
        implementation_solution = None

        if self.common_member.role == "Implementation":
            merged_question = []
            for item in text_1:
                role = 'assistant' if item['role'].lower() != 'user' else item['role']
                merged_question.append({"role": role, "content": item["content"]})
            testing_solution = role_solutions['Testing']
            score = 0.3  #defaul
            if testing_solution and testing_solution[0].get('importance_score'):
                score = testing_solution[0]['importance_score']
            if score > 0:
              messages = merged_question
            else:
                messages = []
            implementation_solution = implementation_member.provide_solution(question=question, design_solution=design_solution, messages=messages,task_id=task_id,prompt_name = prompt_name)
            print("com imp")
            result = run_test_humaneval(task_file, implementation_solution, 3.0, task_id, problem=question)
            if result.lower() == "passed":
                importance_score = 100
            else:
                importance_score = 0.2
                # last_line = re.search(r"([A-Za-z]+Error.*)$", result, re.MULTILINE)
                # result = last_line.group(1)
            implementation_solution = adjust_message_weight(implementation_solution, 'Implementation', importance_score, result)
            getText("Implementation", str(implementation_solution))

                # implementation_solution = tokenizer(implementation_solution, padding=True, truncation=True, return_tensors="pt")
            role_solutions['Implementation'].append(implementation_solution)


        else:
            # messages = [
            #     {"role": "assistant", "content": design_solution}
            # ]
            implementation_solution = implementation_member.provide_solution(question=question,design_solution=design_solution,task_id=task_id,prompt_name = prompt_name)
            print("imp")
            # implementation_solution = tokenizer(implementation_solution, padding=True, truncation=True,return_tensors="pt")
            # role_solutions['Implementation'].append(implementation_solution)
            # print("imp:\n",role_solutions['Implementation'])



        # 3. Debugger
        testing_member = next(member for member in self.members if member.role == "Testing")
        testing_solution = None

        if self.common_member.role == "Testing":
            merged_question = []
            for item in text_1:
                role = 'assistant' if item['role'].lower() != 'user' else item['role']
                merged_question.append({"role": role, "content": item["content"]})
            messages = merged_question
            testing_solution = testing_member.provide_solution(messages = messages,question=question,implementation_solution=implementation_solution,task_id=task_id,prompt_name = prompt_name)
            print("com test")
            result = run_test_humaneval(task_file, testing_solution, 3.0, task_id, problem=question)
            if result.lower() == "passed":
                importance_score = 10000
            else:
                importance_score = 0.2
            testing_solution = adjust_message_weight(testing_solution, 'Testing', importance_score,  result)
            getText("Test", str(testing_solution))
                # testing_solution = tokenizer(testing_solution, padding=True, truncation=True,return_tensors="pt")
            role_solutions['Testing'].append(testing_solution)
                # print("com_test:\n",role_solutions['Testing'])

        else:
            print("test")
            imp_result = run_test_humaneval(task_file, implementation_solution, 3.0, task_id, problem=question)
            if imp_result.lower() == "passed":
                testing_solution = implementation_solution
                testing_solution = adjust_message_weight(testing_solution, 'NCTesting', 1000, "passed")
            else:
                testing_solution = testing_member.provide_solution(question=question,implementation_solution=implementation_solution,task_id=task_id,prompt_name = prompt_name)
                result = run_test_humaneval(task_file, testing_solution, 3.0, task_id, problem=question)
                if result.lower() == "passed":
                    importance_score = 10000
                else:
                    importance_score = 0.2
                testing_solution = adjust_message_weight(testing_solution, 'Testing', importance_score,  result)
            role_solutions['NCTesting'].append(testing_solution)


        checklen(text)
        return testing_solution

    def switch_common_member_role(self, common_member_roles):
        current_role = self.common_member.role
        current_index = common_member_roles.index(current_role)
        next_role = common_member_roles[(current_index + 1) % len(common_member_roles)]
        self.common_member.role = next_role
    def discuss_and_decide(self, question, directory, common_member_roles, task_id, text, prompt_name):
        iteration = 0
        decision = None
        global role_solutions

        while iteration < 1:
            decision = self.provide_solution_with_role_change(question=question,task_id=task_id,text_1=text, prompt_name = prompt_name)
            self.testing_solutions.append(decision)

            self.switch_common_member_role(common_member_roles)


            iteration += 1


        return decision
class SequentialDiscussionManager:
    def __init__(self, name, groups, common_member_name):
        self.name = name
        self.groups = groups
        self.common_member_name = common_member_name
        self.testing_solutions = []

    def conduct_sequential_discussions(self, question, directory, task_id, prompt_name):
        text = getText("user", question)
        for group in self.groups:
            group.discuss_and_decide(question, directory, ["Testing", "Implementation", "Design"], task_id, text, prompt_name)
            # print(text)
        #     self.testing_solutions.extend(group.testing_solutions)

        final_solution = self.old_fourth(question, text, task_id=task_id, prompt_name = prompt_name)
        text.clear()
        save_solutions_to_file(role_solutions, 'D:\PYTHON_FILE\pythonProject1\output_directory')
        role_solutions.clear()
        # for group in self.groups:
            # group.testing_solutions.clear()
        return final_solution


    def old_fourth(self, question, text_2, task_id, prompt_name):
        max_attempts = 5
        attempts = 0
        messages = []
        test_case = extract_problems(task_id)
        case = extract_high_score_solution(text_2)
        if case:
            return case
        for item in text_2:
            role = 'assistant' if item['role'].lower() != 'user' else item['role']
            messages.append({"role": role, "content": item["content"]})
        prompt = (
            f"According to the problem:{question}"
            f"Use the experience to give the code to solve it, make sure it will pass the text case:{test_case}"
            # f"Use the same function name in the problem{prompt_name}" #APPS
            f"[Important]:Only codes. No comments or annotation"
            # f"Use a function to solve the problem, ending with a return, and only require a single string parameter"
            # f"All the code is inside the function."
        )
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        solution = str(response.choices[0].message.content)
        role_solutions['NCTesting'].append(solution)
        first_solution = clean_completion(solution)
        result = run_test_humaneval(task_file, first_solution, 3.0, task_id, problem=question)
        # print(result)
        if result.lower() == "passed":
            final_solution = first_solution
            return final_solution
        else:
            while attempts < max_attempts:
                result = run_test_humaneval(task_file, first_solution, 3.0, task_id, problem=question)
                # print(result)

                combine_mess = f"{first_solution}{result}"
                role_solutions['NCTesting'].append(combine_mess)

                if result.lower() == "passed":
                    final_solution = clean_completion(first_solution)
                    break


                refer = retry_request(lambda: search_github_code(query=question, index = attempts))
                # refer = retry_request(lambda: search_github_code(query=question))

                if isinstance(refer, dict) and "error" in refer:
                    output = f"Error: {refer['error']}, Status Code: {refer.get('status_code', 'N/A')}"
                elif isinstance(refer, str):
                    output = f"Code: {refer}\n\n"
                elif isinstance(refer, list):
                    output = ""
                    for item in refer:
                        if isinstance(item, dict) and "code" in item:
                            output += f"Code: {item['code']}\n\n"
                        else:
                            output += "Invalid item in list.\n\n"

                messages = []
                for item in text_2:
                    role = 'assistant' if item['role'].lower() != 'user' else item['role']
                    messages.append({"role": role, "content": item["content"]})

                prompt = (
                    f"For this problem, {question}, your previous answer encountered an error: {first_solution}. "
                    f"Traceback: {result}. "
                    f"To proceed, ensure the new solution meets the following requirements:\n"
                    f"1. Is fundamentally different from the previous solution.\n"
                    f"2. Fixes the above error.\n"
                    f"3. Passes all the given test cases: {test_case}.\n\n"
                    f"Here are some examples: {output}. "
                    f"Hint: Try to explore different logic or structures, such as using loops, functions, or list comprehensions.\n\n"
                    f"[requirement]: Only codes. No comments or annotation"
                    # f"[requirement]: Only codes. Make only require a single string parameter"
                    # f"All the code is inside the function."
                    # f"Use the same function name in the problem{prompt_name}"
                    # f"code only require a single string parameter"  #codecontest
                )
                messages.append({"role": "user", "content": prompt})
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=messages
                )
                corr_solution = str(response.choices[0].message.content)
                if not any(keyword in corr_solution for keyword in ["def", "return", "import", "class", "for", "if"]):
                    corr_solution = first_solution
                else:
                    corr_solution = clean_completion(corr_solution)

                first_solution = corr_solution
                attempts += 1

            final_solution = first_solution
            if final_solution == None:
                final_solution = first_solution



        return final_solution



# def clean_completion(completion: str) -> str:
#     code_pattern = r'```.*?\n(.*?)```'
#     code_blocks = re.findall(code_pattern, completion, re.DOTALL)
#     return "\n".join(code_blocks)
def clean_completion(completion: str) -> str:
    completion = re.sub(r'.*python\n', '', completion, flags=re.IGNORECASE)
    completion = re.sub(r'```$', '', completion)
    return completion.strip()



def save_solutions_to_file(role_solutions, directory):
    os.makedirs(directory, exist_ok=True)

    filename = 'solutions.json'
    file_path = os.path.join(directory, filename)


    with open(file_path, 'a', encoding='utf-8') as file:
        if file.tell() > 0:
            file.write(",\n")
        json.dump(role_solutions, file, ensure_ascii=False, indent=4)
def generate_one_completion(prompt,prompt_name,groups, common_member_name, directory, task_id):
    discussion_manager = SequentialDiscussionManager(
    name="MyManager",
    groups=groups,
    common_member_name=common_member_name,
)

    final_decision = discussion_manager.conduct_sequential_discussions(prompt, directory, task_id, prompt_name = prompt_name)
    for group in groups:
        group.switch_common_member_role(["Implementation","Testing", "Design"])


    return final_decision
def load_problems(file_path):
    problems = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            task = json.loads(line.strip())
            task_id = task["task_id"]    #humaneval , mbpp
            # task_id = task["id"]     #APPS,CODECONTEST
            # task_id = task["src_uid"]   #codeveal
            problems[task_id] = task
    return problems

# def get_embeddings(text):
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).squeeze()
# def calculate_similarity(test1, test2):
#     emb1 = get_embeddings(test1)
#     emb2 = get_embeddings(test2)
#     return cosine_similarity([emb1.numpy()], [emb2.numpy()])[0][0]



if __name__ == '__main__':
    start_time = time.time()
    output_directory = r"D:\PYTHON_FILE\pythonProject1\output_directory"

    common_member_name = "MemberX"
    group1 = SequentialDiscussionGroup(
        members_with_roles=[
            ("Member1_A", "Design"),
            ("Member1_B", "Implementation"),
            ("Member1_C", "Testing")
        ],
        common_member_name=common_member_name,
        common_member_initial_role="Testing"
    )

    group2 = SequentialDiscussionGroup(
        members_with_roles=[
            ("Member2_A", "Design"),
            ("Member2_B", "Implementation"),
            ("Member2_C", "Testing")
        ],
        common_member_name=common_member_name,
        common_member_initial_role="Implementation"
    )

    group3 = SequentialDiscussionGroup(
        members_with_roles=[
            ("Member3_A", "Design"),
            ("Member3_B", "Implementation"),
            ("Member3_C", "Testing")
        ],
        common_member_name=common_member_name,
        common_member_initial_role="Design"
    )

    groups = [group1, group2, group3]
    problems = read_problems()
    num_samples_per_task = 1
    samples = []
    start_task = 0
    end_task = len(problems)


    with open("D:\PYTHON_FILE\\pythonProject1\\input.jsonl", "w", encoding="utf-8") as file:
        for task_id in list(problems.keys())[start_task:end_task]:
            for _ in range(num_samples_per_task):
                pro_name = ""
                completion = generate_one_completion(problems[task_id]["prompt"], pro_name, groups, common_member_name,
                                                     directory="D:\\PYTHON_FILE\\pythonProject1\\output_directory", task_id = task_id)
                sample = dict(task_id=task_id, completion=clean_completion(completion))
                samples.append(sample)


                file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                file.flush()

                print(f"Task {task_id} completed")

    # /--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------/
    #
    #
    # file_path = 'D:\PYTHON_FILE\pythonProject1\mbppWST.jsonl'
    # file_path = 'codeEval.jsonl'
    # file_path = 'codecontest_with_id.jsonl'
    # problems = load_problems(file_path)
    # groups = [group1, group2, group3]
    # num_samples_per_task = 1
    # samples = []
    # start_task = 55
    # end_task = len(problems)
    #
    # with open("D:\PYTHON_FILE\\pythonProject1\\input.jsonl", "w", encoding="utf-8") as file:
    #     for task_id in list(problems.keys())[start_task:end_task]:
    #         for _ in range(num_samples_per_task):
    #                 # completion = generate_one_completion(problems[task_id]["prompt"], groups, common_member_name,
    #                 #                                      directory=output_directory, task_id=task_id)         #MBPP
    #                 # pro_name = problems[task_id]["starter_code"]
    #                 pro_name = " "
    #                 completion = generate_one_completion(problems[task_id]["description"],pro_name,groups, common_member_name,
    #                                                      directory=output_directory, task_id=task_id)             #APPS
    #
    #                 sample = dict(task_id=task_id, completion=completion)
    #                 samples.append(sample)
    #                 file.write(json.dumps(sample, ensure_ascii=False) + "\n")
    #                 file.flush()
    #
    #                 print(f"Task {task_id} completed")




    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Used: {elapsed_time:.2f} Seconds")
    # clean_data.clean_data(input_file='input.jsonl', output_file='samples.jsonl')

