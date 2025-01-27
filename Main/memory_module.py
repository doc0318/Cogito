from collections import defaultdict
import os
import re
from openai import OpenAI
import time
import json
from typing import List, Dict, Tuple
from github import search_github_code
from requests.exceptions import RequestException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


prompt_file_path = "question file"
answer_file_path = "answer file"
class HippocampusStorage:
    def __init__(self,prompt_file_path = None):

        prompt_file_path = "D:\\PYTHON_FILE\\pythonProject1\\mbpp.jsonl"
        self.DG = {}
        self.CA1 = {}
        self.CA3 = {}
        self.CA4 = {}
        self.CA2 = {}
        if not self.DG:
            self.load_tasks(prompt_file_path)


    def add_task(self, task_id, prompt, entry_point):
        result = []
        prompt_keywords = prompt.split()
        similar_tasks = self.get_similar_tasks(prompt_keywords, None, str(task_id),0.7)
        if similar_tasks:
            print("Found Similar Tasks:", similar_tasks)
        else:
            print("No Similar Tasks Found.")

        return result


    def add_initial_solution(self, task_id, solution):
        self.CA1[task_id] = {"original_solution": solution}



    def add_refined_solution(self, task_id, refined_solution, test_results, traceback):
        if task_id not in self.CA3:
            self.CA3[task_id] = []

        current_version = len(self.CA3[task_id]) + 1

        self.CA3[task_id].append({
            "version": current_version,
            "original_solution":self.extract_completion(task_id),
            "refined_solution": refined_solution,
            "test_results": test_results,
            "traceback": traceback,
            "notes": "Refined based on previous feedback or error analysis."
        })



    def add_final_solution(self, task_id, final_solution, validated_tests):
        self.CA4[task_id] = {
            "final_solution": final_solution,
            "validated_tests": validated_tests,
            "notes": "Final solution that passed all tests."
        }



    def add_error(self, task_id, traceback):
        if task_id not in self.CA2:
            self.CA2[task_id] = []

        current_version = len(self.CA2[task_id]) + 1
        for solution in self.CA2[task_id]:
            if isinstance(solution, dict):
                solution["status"] = "Error"
                solution["traceback"] = traceback
                solution["notes"] = "Error encountered during testing or execution."

        self.CA2[task_id].append({
            "status": "Error",
            "traceback": traceback,
            "notes": "Error encountered during testing or execution."
        })

    def add_errors_from_test(self, task_ids, traceback_dict):
        for task_id in task_ids:
            traceback = traceback_dict.get(task_id, "No traceback available.")
            self.add_error(task_id, traceback)

    def get_common_errors(self):
        error_patterns = {}
        for task_id, error_info in self.CA3.items():
            for version in error_info.get("versions", []):
                error_message = version.get("traceback")
                if error_message:
                    if error_message not in error_patterns:
                        error_patterns[error_message] = {
                            "count": 0,
                            "examples": []
                        }
                    error_patterns[error_message]["count"] += 1
                    error_patterns[error_message]["examples"].append(task_id)
        return error_patterns

    def get_similar_tasks(self, prompt_keywords, personalized_keywords=None, task_id_to_ignore=None,
                          similarity_threshold=0.8):
        similar_tasks = []
        all_keywords = " ".join(prompt_keywords)
        if personalized_keywords:
            all_keywords += " " + " ".join(personalized_keywords)

        vectorizer = TfidfVectorizer().fit([task_info["prompt"] for task_info in self.DG.values()])

        all_keywords_vector = vectorizer.transform([" ".join(prompt_keywords + (personalized_keywords or []))])

        for task_id, task_info in self.DG.items():
            if task_id_to_ignore is not None and task_id == task_id_to_ignore:
                continue

            task_vector = vectorizer.transform([task_info["prompt"]])
            similarity = cosine_similarity(all_keywords_vector, task_vector).flatten()[0]

            if similarity_threshold <= similarity < 1:
                similar_tasks.append({
                    "task_id": task_id,
                    "prompt": task_info["prompt"],
                    "solution": self.extract_completion(task_id),
                    "similarity": similarity
                })

        similar_tasks.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_tasks

    def update_task_and_solutions(self, task_id, new_solution, test_results, final_solution=None, validated_tests=None):
        self.add_initial_solution(task_id, new_solution)
        traceback = test_results.get("traceback")

        if traceback:
            self.add_error(task_id, traceback)
        else:
            self.add_refined_solution(task_id, new_solution, test_results, traceback)

        if final_solution:
            self.add_final_solution(task_id, final_solution, validated_tests)

    def process_tasks_with_errors(self, task_ids, traceback_dict):
        self.add_errors_from_test(task_ids, traceback_dict)
        return self.get_common_errors()

    def save_data(self, file_path):
        data = {
            "DG": self.DG,
            "CA3": self.CA3,
            "CA1": self.CA1,
            "CA4": self.CA4
        }
        try:
            with open(file_path, 'a') as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            print(f"Failed to save data: {e}")


    def save_ca2(self, file_path):
        data = {
            "CA2": self.CA2,
        }
        try:
            with open(file_path, 'a') as file:
                json.dump(data, file, indent=4)
        except Exception as e:
            print(f"Failed to save data: {e}")

    def load_data(self, file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                self.DG = data.get("DG", {})
                self.CA3 = data.get("CA3", {})
                self.CA1 = data.get("CA1", {})
                self.CA4 = data.get("CA4", {})
        except Exception as e:
            print(f"Failed to load data: {e}")

    def load_tasks(self, prompt_file_path):
        with open(prompt_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                task = json.loads(line.strip())
                task_id = task['task_id']
                task_prompt = task.get('text')
                entry_point = task.get('entry_point')

                self.DG[task_id] = {
                    "prompt": task_prompt,
                    "entry_point": entry_point
                }

    def extract_completion(self, task_id):
        with open("D:\\PYTHON_FILE\\pythonProject1\\samples.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                task = json.loads(line.strip())
                if task['task_id'] == task_id:
                    return task['completion']
        return None


    def load_personalized_data(self, file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)

            personalized_keywords = []
            for item in data:
                if 'keywords' in item:
                    personalized_keywords.extend(item['keywords'])
                elif 'content' in item:
                    personalized_keywords.extend(item['content'].split())
            return personalized_keywords
        except Exception as e:
            print(f"Failed to load personalized data: {e}")
            return []

    def get_final_solution(self, task_id):
        return self.CA4.get(task_id, None)

    def get_refined_solutions(self, task_id):
        return self.CA3.get(task_id, {}).get("versions", [])

    def get_task(self, task_id):
        return self.DG.get(task_id, None)

    def get_initial_solution(self, task_id):
        return self.CA1.get(task_id, None)












