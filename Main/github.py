import requests
import json
import base64
import time
from requests.exceptions import RequestException

def split_code_into_chunks(code, chunk_size=500):
    return [code[i:i + chunk_size] for i in range(0, len(code), chunk_size)]

def search_github_code(query, language="python", sort="stars", order="desc", max_results=10, index = 0):
    # GitHub API URL
    url = "https://api.github.com/search/code"

    headers = {
        'Authorization': 'your token key'
    }

    # 参数设置
    params = {
        "q": f"{query} language:{language}",
        "sort": sort,
        "order": order,
        "per_page": max_results
    }


    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
        code_results = []
        for item in data['items']:
            file_url = item['url']
            file_response = requests.get(file_url, headers=headers)
            if file_response.status_code == 200:
                file_data = file_response.json()
                encoded_content = file_data.get('content', '')

                if encoded_content:
                    decoded_content = base64.b64decode(encoded_content).decode('utf-8')
                else:
                    decoded_content = "No content available"


                decoded_content = decoded_content[:3000]
                code_chunks = split_code_into_chunks(decoded_content)
                if index < len(code_chunks):
                    return code_chunks[index]
                else:
                    return "Index out of range"

        return "No code found"
    elif response.status_code == 403:
        return {"error": "Forbidden (Check Token Permissions or Rate Limits)", "status_code": 403}
    else:
        return {"error": "Failed to retrieve data", "status_code": response.status_code}

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





# def extract_problems(task_id):
#     tasks = load_jsonl('HumanEvalWST.jsonl')
#     tasks_1 = load_jsonl('HumanEval_test_case_ET.jsonl')
#
#     for task in tasks:
#         id = task["task_id"]
#         if id == task_id:
#             problem = task['sample_io']
#             break
#
#     for t in tasks_1:
#         id_1 = t["task_id"]
#         if id_1 == task_id:
#             problem_1 = t['test_case_list']
#             break
#
#     combined = problem_1 + problem
#
#
#
#     # return "\n".join(combined)
#     return problem
#     return None, None



def extract_problems(task_id):
    # tasks = load_jsonl('mbppWST.jsonl')
    # tasks = load_jsonl('selected150.jsonl')
    # tasks = load_jsonl('codecontest_with_id.jsonl')
    # tasks = load_jsonl('codeEval.jsonl')
    tasks = load_jsonl('HumanEvalWST.jsonl')



    for task in tasks:
        id = task["task_id"]  #humaneval,mbpp
        # id = task["id"]  #apps,condecontest
        # id = task["src_uid"]  #codeEval
        if id == task_id:
            problem = task['sample_io']
            break


    # return "\n".join(combined)
    return problem
    return None, None



def load_jsonl(file_path: str):
    with open(file_path, 'r', encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def extract_high_score_solution(data):
    if isinstance(data, list):
        highest_score = 0.99
        high_score_solution = None

        for task_data in data:
            if isinstance(task_data, dict):
                for role, solutions in task_data.items():
                    if isinstance(solutions, list):
                        for solution in solutions:
                            if isinstance(solution, dict) and 'importance_score' in solution and 'message' in solution:
                                if solution['importance_score'] > highest_score:
                                    highest_score = solution['importance_score']
                                    high_score_solution = solution['message']

        return high_score_solution



