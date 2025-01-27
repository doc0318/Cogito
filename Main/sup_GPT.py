from test_MBPP import run_test_from_jsonl_MBPP
import re
import json
import clean_data
from openai import OpenAI
from github import search_github_code
from requests.exceptions import RequestException
import time
from memory_module import HippocampusStorage
from test_humaneval import run_test_from_jsonl_humaneval
import traceback

client = OpenAI(
    api_key="your open ai key",
)


class Supervisor:
    def __init__(self, name="Supervisor"):
        self.name = name
        self.client = client
        self.storage = HippocampusStorage()


    def test_code(self, code):
        try:
            exec(code)
            return {"success": True}
        except Exception as e:
            error_traceback = traceback.format_exc()
            return {"success": False, "error": str(e), "traceback": error_traceback}


    # def summerize_experience(self):
    #     messages = []
    #
    #     for idx, task_id in enumerate(unique_task_ids, start=1):
    #         successful_code = self.extract_completion(task_id, "D:\\PYTHON_FILE\\pythonProject1\\samples_1.jsonl")
    #         failed_code = self.extract_completion(task_id, "D:\\PYTHON_FILE\\pythonProject1\\samples.jsonl")
    #         combined_code = f"Task {idx}: Successful Code: {successful_code}\nFailed Code: {failed_code}\n"
    #
    #         lines = combined_code.split('\n')
    #         first_multiple_lines = lines[:3]  #
    #
    #     prompt = (
    #         f"Here are some examples of failures turned into successes.Analyze the reasons behind it and summarize the experience."
    #         f"{combined_code}"
    #
    #     )
    #     self.response = self.client.chat.completions.create(
    #         model="gpt-3.5-turbo",
    #         messages=[{"role": "user", "content": prompt}]
    #     )
    #
    #     answer = str(self.response.choices[0].message.content)
    #
    #     messages = [
    #         {"role":"user","content":prompt},
    #         {"role":"assistant", "content":answer}
    #     ]
    #
    #     return messages




    def fix_code(self, task_ids):
        messages = []
        # messages = self.summerize_experience()


        with open("D:\\PYTHON_FILE\\pythonProject1\\samples.jsonl", "r", encoding="utf-8") as file:
            data = [json.loads(line) for line in file]

        completed_tasks = set()
        temp_file = "D:\\PYTHON_FILE\\pythonProject1\\completed_tasks.json"
        try:
            with open(temp_file, "r", encoding="utf-8") as temp:
                completed_tasks = set(json.load(temp))
        except FileNotFoundError:
            pass

        for task_id in task_ids:
            if task_id in completed_tasks:
                # print(f"Task {task_id} already completed, skipping...")
                continue

            print(f"Processing task: {task_id}")
            try:

                question = self.extract_problems(task_id)
                completion = self.extract_completion(task_id)
                results = self.retry_request(lambda: search_github_code(query=question))

                similar_tasks = self. storage.add_task(task_id, prompt=question, entry_point="main")

                reference_code = []
                reference_traceback = []
                for task in similar_tasks:
                    if task.get("status") == "pass":
                        reference_code.append(task.get("solution"))
                    else:
                        reference_code.append()

                if reference_code:
                  print("Found successful solutions:", reference_code)
                else:
                  reference_code = None

                if "error" not in results:
                    output = ""
                    for result in results:
                        # output += f"Repository: {result['repo']}, Path: {result['path']}\n"
                        output += f"Code: {result['code']}\n\n"
                else:
                    output = f"Error: {results['error']}, Status Code: {results['status_code']}"


                traceback = traceback_dict.get(task_id, "")
                if any(task["CA2"].get("status") == "Error" and task["CA2"].get("traceback") for task in similar_tasks):
                    for task in similar_tasks:
                        if task.get("status") == "Error" and task.get("traceback"):
                            reference_traceback.append(task.get("traceback"))
                            print("reference traceback:",{reference_traceback})
                    prompt = (
                        f"For this problem, {question}, your previous answer encountered an error: {completion}."
                        f"Traceback: {traceback}"
                        f"Some similar error codes:{reference_code} and traceback:{reference_traceback}"
                        f"Some similar tasks had errors. Please analyze and revise your code accordingly."
                        f"Here are some examples: {output}. Ensure the updated code can pass all tests."
                        f"[requirement]: Only give the code, no comment and other things."
                    )
                else:
                    prompt = (
                        f"For this problem, {question}, your previous answer was incorrect: {completion}."
                        f"Traceback: {traceback}"
                        f"Similar passed tasks：{reference_code}"
                        f"Here are some examples: {output}. Please modify the code so that it can pass the tests."
                        f"You can modify the original one or completely use a new algorithm."
                        f"[requirement]: Only give the code, no comment and other things."
                    )

                messages.append({"role": "user", "content": prompt})

                self.response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )
                revised_code = str(self.response.choices[0].message.content)
                self.update_task(data, task_id, revised_code)

                test_results = self.test_code(clean_completion(revised_code))
                print(test_results)
                self.storage.update_task_and_solutions(
                    task_id=task_id,
                    new_solution=clean_completion(revised_code),
                    test_results=test_results,
                    final_solution=revised_code if test_results.get("success") else None,
                    validated_tests=test_results if test_results.get("success") else None
                )

                if test_results.get("success"):
                    completed_tasks.add(task_id)
                    with open(temp_file, "w", encoding="utf-8") as temp:
                      json.dump(list(completed_tasks), temp)

                if not test_results.get("success"):
                    error_traceback = test_results.get("traceback", "No traceback available")
                    print(f"Test failed for task {task_id}: {error_traceback}")
                    self.storage.add_error(task_id, traceback=error_traceback)

            except Exception as e:
                print(f"Error processing task {task_id}: {e}")
                self.storage.add_error(task_id, traceback=str(e))
                continue


            with open("D:\\PYTHON_FILE\\pythonProject1\\samples.jsonl", "w", encoding="utf-8") as file:
                for entry in data:
                    file.write(json.dumps(entry) + "\n")

        self.storage.save_data(r'D:\PYTHON_FILE\pythonProject1\progress.jsonl')


    def add_data(self, file_path):
        data = {
            "DG": self.DG,
            "CA3": self.CA3,
            "CA1": self.CA1,
            "CA2": self.CA2,
            "CA4": self.CA4
        }
        try:
            with open(file_path, 'a') as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            print(f"Failed to save data: {e}")


    def update_task(self, data, task_id, revised_code):
        for entry in data:
            if entry["task_id"] == task_id:
                entry["completion"] = revised_code
                break


def extract_completion(task_id):
    # with open("D:\\PYTHON_FILE\\pythonProject1\\samples.jsonl", 'r', encoding='utf-8') as f:
    with open("D:\\PYTHON_FILE\\pythonProject1\\samples_1.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            task = json.loads(line.strip())
            if task['task_id'] == task_id:
                return task['completion']
    return None


def extract_problems(task_id):
    # with open("D:\\PYTHON_FILE\\pythonProject1\\mbpp.jsonl", 'r', encoding='utf-8') as file:
    with open("D:\\PYTHON_FILE\\pythonProject1\\human-eval-v2-20210705.jsonl.jsonl", 'r', encoding='utf-8') as file:
        for line in file:
            task = json.loads(line.strip())
            if task['task_id'] == task_id:
                return task.get('prompt')
    return None



def clean_completion(completion: str) -> str:
    completion = re.sub(r'.*python\n', '', completion, flags=re.IGNORECASE)
    completion = re.sub(r'```$', '', completion)
    return completion.strip()

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



if __name__ == "__main__":
    supervisor = Supervisor()
    # problems_file = 'human-eval-v2-20210705.jsonl'
    problems_file = 'HumanEval_test_case_ET.jsonl'
    solutions_file = 'samples_1.jsonl'
    # task_ids, traceback_dict = run_test_from_jsonl_humaneval(problems_file, solutions_file, timeout=3.0)
    task_ids, traceback_dict = None


    # problems_file = 'mbpp.jsonl'
    # solutions_file = 'samples_1.jsonl'
    # solutions_file_1 = 'samples.jsonl'
    # task_ids, traceback_dict = run_test_from_jsonl_MBPP(problems_file, solutions_file, timeout=3.0)
    # other_task_ids, other_traceback_dict = run_test_from_jsonl_MBPP(problems_file, solutions_file_1, timeout=3.0)
    # unique_task_ids = list(set(other_task_ids)-set(task_ids))

    supervisor.fix_code(task_ids)
    # clean_data.clean_data(input_file='samples.jsonl', output_file='samples_1.jsonl')

    for task_id in task_ids:
        traceback = traceback_dict.get(task_id, "")
        supervisor.storage.add_error(task_id, traceback=traceback)
    supervisor.storage.save_ca2(r'D:\PYTHON_FILE\pythonProject1\progress.jsonl')




