import json
import re


def clean_data(input_file='input.jsonl', output_file='samples.jsonl'):
    pattern = re.compile(r'```python\s*\n?', re.DOTALL)

    with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile, open(output_file, 'w', encoding='utf-8', errors='ignore') as outfile:
        success_count = 0
        error_count = 0

        for line in infile:
            try:
                data = json.loads(line.strip())
                if 'completion' in data:
                    original_completion = data['completion']

                    cleaned_completion = pattern.sub('', original_completion).strip()

                    if cleaned_completion.endswith('```'):
                        cleaned_completion = cleaned_completion[:-3]

                    data['completion'] = cleaned_completion

                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    success_count += 1

            except json.JSONDecodeError as e:
                error_count += 1
            except Exception as e:
                error_count += 1

    print(f"Done{success_count} rows, Error:{error_count} rwo. Output at {output_file}")


if __name__ == "__main__":
    clean_data()
