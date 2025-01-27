import concurrent.futures
import sys
import json
import os
import traceback
import time
from requests.exceptions import RequestException
import re
import ast






def load_jsonl(file_path: str):
    with open(file_path, 'r', encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def retry_request( func, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return func()
        except RequestException as e:
            print(f"Request failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise e

def extract_test_samples(prompt):
    test_samples = re.findall(r'>>> (.*?)\n\s*(.*?)\n', prompt)
    test_cases = []
    for sample in test_samples:
        function_call, expected_output = sample
        test_cases.append(f"assert candidate({function_call}) == {expected_output}")
    return "\n    ".join(test_cases)

def process_jsonl(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            entry = json.loads(line)
            prompt = entry["prompt"]
            extracted_tests = extract_test_samples(prompt)
            entry["test"] = f"\n\nMETADATA = {{\n    'author': 'jt',\n    'dataset': 'test'\n}}\n\n\ndef check(candidate):\n    {extracted_tests}\n"
            outfile.write(json.dumps(entry) + "\n")


#---------------------------------HumanEval------------------------------------------------------------------
def unsafe_execute(problem, problem_1, answer, timeout: float):
    try:
        # 获取代码及相关内容
        code = answer
        prompt = str(problem["prompt"])
        code = str(code)
        # test_setup_code = problem["test"]
        # test_case_list = problem_1["test_case_list"]
        test_sample_io = problem["sample_io"]
        test_sample_io = "\n    ".join(test_sample_io) if isinstance(test_sample_io, list) else test_sample_io
        # test_case_list = "\n    ".join(test_case_list) if isinstance(test_case_list, list) else test_case_list

        check_program = (
            prompt + code + "\n" +
            f"def check(candidate):\n" +
            f"    {test_sample_io}\n" +
            # f"    {test_case_list}\n" +
            f"check({problem['entry_point']})"
        )
        # print(check_program)


        exec_globals = {}

        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: exec(check_program, exec_globals))
            try:
                future.result(timeout=timeout)
                return {"status": "passed", "traceback": ""}
            except concurrent.futures.TimeoutError:
                return {"status": "timed out", "traceback": ""}
            except AssertionError as e:
                tb_str = traceback.format_exc()
                return {"status": f"failed: AssertionError: {e}", "traceback": tb_str}
            except Exception as e:
                tb_str = traceback.format_exc()
                return {"status": f"failed: {e}", "traceback": tb_str}
            finally:
                sys.stdout = old_stdout

    except Exception as e:
        tb_str = traceback.format_exc()
        return {"status": f"failed: {e}", "traceback": tb_str}



def run_test_humaneval(task_file: str, solution, timeout: float = 3.0, task_id_input = None, problem = None):
    tasks = load_jsonl(task_file)
    tasks_1 = load_jsonl('HumanEvalWST.jsonl')
    completion = None
    problem_1 = None

    for task in tasks:
        task_id = task["task_id"]
        if task_id == task_id_input:
            problem = task
            completion = solution
            break

    for t in tasks_1:
        task_id = t["task_id"]
        if task_id == task_id_input:
            problem_1 = t
            break



    result = unsafe_execute(problem, problem_1, completion, timeout)

    if result["status"] == "passed":
        return "passed"
    else:
        traceback_lines = result["traceback"].splitlines()
        last_three_lines = "\n".join(traceback_lines[-3:])
        return last_three_lines








#---------------------------------MBPP------------------------------------------------------------------
# def unsafe_execute(problem, answer, timeout: float):
#     try:
#         code = answer
#         prompt = str(problem["prompt"])
#         code = str(code)
#
#         test_sample_io = problem["sample_io"]
#         test_sample_io = "\n    ".join(test_sample_io) if isinstance(test_sample_io, list) else test_sample_io
#
#         check_program = (
#             prompt + code + "\n" +
#             f"def check(candidate):\n" +
#             f"    {test_sample_io}\n" +
#             f"check({problem['entry_point']})"
#         )
#         # print(check_program)
#
#
#         exec_globals = {}
#
#         old_stdout = sys.stdout
#         sys.stdout = open(os.devnull, 'w')
#
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             future = executor.submit(lambda: exec(check_program, exec_globals))
#             try:
#                 future.result(timeout=timeout)
#                 return {"status": "passed", "traceback": ""}
#             except concurrent.futures.TimeoutError:
#                 return {"status": "timed out", "traceback": ""}
#             except AssertionError as e:
#                 tb_str = traceback.format_exc()
#                 return {"status": f"failed: AssertionError: {e}", "traceback": tb_str}
#             except Exception as e:
#                 tb_str = traceback.format_exc()
#                 return {"status": f"failed: {e}", "traceback": tb_str}
#             finally:
#                 sys.stdout = old_stdout
#
#     except Exception as e:
#         tb_str = traceback.format_exc()
#         return {"status": f"failed: {e}", "traceback": tb_str}
#
#
#
# def run_test_humaneval(task_file: str, solution, timeout: float = 3.0, task_id_input = None, problem = None):
#     tasks = load_jsonl(task_file)
#     completion = None
#
#     for task in tasks:
#         task_id = task["task_id"]
#         if task_id == task_id_input:
#             problem = task
#             completion = solution
#             break
#
#
#     result = unsafe_execute(problem, completion, timeout)
#
#     if result["status"] == "passed":
#         return "passed"
#     else:
#         traceback_lines = result["traceback"].splitlines()
#         last_three_lines = "\n".join(traceback_lines[-3:])
#         return last_three_lines




#------------------------------APPS------------------------------------

def transform_sample_io(sample_io):
    inputs = []
    outputs = []

    for item in sample_io:
        input_data = item['input']

        # 如果 input 是字符串，则不作处理，直接作为字符串添加到 inputs
        if isinstance(input_data, str) and '\n' in input_data:
            input_list = input_data  # 保留原始字符串
        else:
            # 如果是其他类型（如列表），使用 ast.literal_eval 解析
            input_list = ast.literal_eval(input_data)

        output = item['output']

        inputs.append(input_list)
        outputs.append(output)

    return f"inputs = {inputs}", f"outputs = {outputs}"

# def unsafe_execute(problem, answer, timeout: float):
#     try:
#         # 获取代码及相关内容
#         code = answer
#         code = str(code)
#
#         test_setup_code = problem["sample_io"]
#
#         test_setup_code_inputs,test_setup_code_outputs = transform_sample_io(test_setup_code)
#
#         starter_code = problem["starter_code"]
#         if starter_code == "":
#             function_name = "solution"
#         else:
#             match = re.search(r"def\s+(\w+)\(", starter_code)
#         function_name = match.group(1)
#         match = re.search(r"def\s+(\w+)\(", answer)
#         function_name = match.group(1)
#
#
#
#
#
#
#         check_program = (
#                 code + "\n" +
#                 f"def check(candidate):\n"+
#                 f"    {test_setup_code_inputs}\n" +
#                 f"    {test_setup_code_outputs}\n" +
#                 f"    for i, test_input in enumerate(inputs):\n" +
#                 f"            expected_output = outputs[i]\n" +
#                 f"            result = candidate(test_input)\n" +
#                 f"            assert result == expected_output[0], f'Test case {{i + 1}} failed: expected {{expected_output[0]}}, got {{result}}'" + "\n" +
#                 f"check({function_name})"
#         )
#         print(check_program)
#
#
#         exec_globals = {}
#
#         old_stdout = sys.stdout
#         sys.stdout = open(os.devnull, 'w')
#
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             future = executor.submit(lambda: exec(check_program, exec_globals))
#             try:
#                 future.result(timeout=timeout)
#                 return {"status": "passed", "traceback": ""}
#             except concurrent.futures.TimeoutError:
#                 return {"status": "timed out", "traceback": ""}
#             except AssertionError as e:
#                 tb_str = traceback.format_exc()
#                 return {"status": f"failed: AssertionError: {e}", "traceback": tb_str}
#             except Exception as e:
#                 tb_str = traceback.format_exc()
#                 return {"status": f"failed: {e}", "traceback": tb_str}
#             finally:
#                 sys.stdout = old_stdout  # Restore original stdout
#
#     except Exception as e:
#         tb_str = traceback.format_exc()
#         return {"status": f"failed: {e}", "traceback": tb_str}


#----------------------------------------codeEval、codeContest----------------------------------------

#
#
#
# import sys
# import os
# import traceback
# import re
# import concurrent.futures
# import threading
#
# def unsafe_execute(problem, answer, timeout: float):
#     def execute_code(check_program, exec_globals):
#         exec(check_program, exec_globals)
#
#     try:
#         # 获取代码及相关内容
#         code = answer
#         code = str(code)
#
#         test_setup_code = problem["sample_io"]
#         test_setup_code_inputs, test_setup_code_outputs = transform_sample_io(test_setup_code)
#
#         match = re.search(r"def\s+(\w+)\(", answer)
#         function_name = match.group(1)
#
#         check_program = (
#                 code + "\n" +
#                 f"def check(candidate):\n" +
#                 f"    {test_setup_code_inputs}\n" +
#                 f"    {test_setup_code_outputs}\n" +
#                 f"    for i, test_input in enumerate(inputs):\n" +
#                 f"            expected_output = outputs[i]\n"
#                 f"            found = False\n" +
#                 f"            for value in expected_output:\n" +
#                 f"              result = str(candidate(test_input))\n" +
#                 f"              if result == value:\n" +
#                 f"                found = True\n" +
#                 f"                break\n" +
#                 f"            assert found, f'Test case {{i+1}} failed.  Expected {{value}}, got {{result}}'" + "\n" +
#                 f"check({function_name})"
#         )
#         print(check_program)
#
#         exec_globals = {}
#
#         old_stdout = sys.stdout
#         sys.stdout = open(os.devnull, 'w')
#
#         # Create a separate thread to run the code
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             future = executor.submit(execute_code, check_program, exec_globals)
#             try:
#                 # Set a timeout for the code execution
#                 future.result(timeout=timeout)  # Wait for the result with timeout
#                 return {"status": "passed", "traceback": ""}
#             except concurrent.futures.TimeoutError:
#                 # Timeout reached, stop execution
#                 return {"status": "timed out", "traceback": ""}
#             except AssertionError as e:
#                 tb_str = traceback.format_exc()
#                 return {"status": f"failed: AssertionError: {e}", "traceback": tb_str}
#             except Exception as e:
#                 tb_str = traceback.format_exc()
#                 return {"status": f"failed: {e}", "traceback": tb_str}
#             finally:
#                 sys.stdout = old_stdout  # Restore original stdout
#
#     except Exception as e:
#         tb_str = traceback.format_exc()
#         return {"status": f"failed: {e}", "traceback": tb_str}
#
#
# #
# #
# def run_test_humaneval(task_file: str, solution, timeout: float = 3.0, task_id_input = None, problem = None):
#     tasks = load_jsonl(task_file)
#     completion = None
#
#     for task in tasks:
#         task_id = task["id"]    #codeEval
#         # task_id = task["src_uid"]    #codecontest
#         if task_id == task_id_input:
#             problem = task
#             completion = solution
#             break
#
#
#     result = unsafe_execute(problem, completion, 3)
#
#     if result["status"] == "passed":
#         return "passed"
#     else:
#         traceback_lines = result["traceback"].splitlines()
#         last_three_lines = "\n".join(traceback_lines[-3:])
#         return last_three_lines





input_file = "input.jsonl"
output_file = "output.jsonl"

# process_jsonl(input_file, output_file)
