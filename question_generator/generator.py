import json
import random
import http.client
from tqdm import tqdm
import jsonlines
import time
import re

# SQL query elements
class SqlQuery:
    sel_op = 'SELECT'
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']
    where_op = 'WHERE'
    and_op = 'AND' 
    operators = ['MAX', 'MIN', 'COUNT', 'SUM', 'AVG', '=', '>', '<']

# Template strings
template_sql = '''Please generate a question whose answer can be found in the given rows.
(1) The question should not be too general and should not be answerable by other similar tables.
{condition_1}{condition_2}'''
template_synonyms = '''Please identify all entities in the question and find synonyms of extracted entities or nicknames if exist. Please respond using JSON: {{entity_1: list of its synonyms, entity_2: list of its synonyms}}'''
template_rewrite = '''Please replace the names of entities with their synonyms where applicable in the question and rephrase the question with different styles and lengths, while preserving the original meaning. (A question could also be a sentence, phrase or keyword combination). Only output just one rewritten question.'''
template_title = '''Please add the provided table title to the question where necessary, ensuring sufficient context for better understanding but you cannot directly copy the title to the question. Please output the rewritten question or the original if no change is made.'''
template_table_quality_check = '''Given a question and a table, please determine if the question can be accurately and uniquely answered using the given table, and not answerable using other similar tables. Please only output 'Yes' or 'No'.'''

class ContextLengthExceededError(Exception):
    def __init__(self, response):
        self.response = response

class GPTInference:
    def __init__(self, api_key, model="gpt-3.5-turbo", host="YOUR GPT SERVICE HOST"):
        self.api_key = api_key
        self.model = model
        self.host = host
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Accept': 'application/json',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json'
        }

    def generate_response(self, messages, stream=False, max_retries=6):
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": stream
        })

        retries = 0
        while retries < max_retries:
            try:
                conn = http.client.HTTPSConnection(self.host)
                conn.request("POST", "/v1/chat/completions", payload, self.headers)
                res = conn.getresponse()
                data = res.read()
                conn.close()

                response = json.loads(data.decode("utf-8"))

                if 'choices' in response and len(response['choices']) > 0:
                    return response['choices'][0]['message']['content']
                elif 'error' in response:
                    error_code = response['error'].get('code', '')
                    if error_code == 'context_length_exceeded':
                        raise ContextLengthExceededError(response)
                    else:
                        print(response)
                        raise Exception(f"API error: {response['error']['message']}")
                else:
                    print(response)
                    raise KeyError("'choices' not found in response")

            except (ContextLengthExceededError, KeyError, json.JSONDecodeError, http.client.HTTPException) as e:
                if isinstance(e, ContextLengthExceededError):
                    raise e  # Reraise the exception to handle it in generate_question
                print(f"Error: {e}. Retrying... ({retries+1}/{max_retries})")
                retries += 1
                time.sleep(5)

        raise Exception(f"Failed to generate response after {max_retries} retries.")

def read_tables_jsonl(file_path):
    tables = []
    with open(file_path, 'r') as file:
        for line in file:
            tables.append(json.loads(line))
    return tables

def is_title_in_question(title, question):
    title = re.sub(r'[^\w\s]', '', title)
    question = re.sub(r'[^\w\s]', '', question)
    title_words = set(title.lower().split())
    question_words = set(question.lower().split())
    common_words = title_words.intersection(question_words)
    common_percentage = (len(common_words) / len(title_words)) if len(title_words) > 0 else 0
    return common_percentage >= 0.6

def read_similar_tables(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def find_table_by_id(table_id, tables):
    for table in tables:
        if table['id'] == table_id:
            return table
    return None

def group_cells_by_row(cells):
    rows = {}
    for cell in cells:
        row_idx = cell['row_idx']
        if row_idx not in rows:
            rows[row_idx] = []
        rows[row_idx].append(cell['text'])
    return list(rows.values())

def compare_tables(primary_table, similar_tables, equality_log_file):
    """
    Compare the primary table and similar tables to generate unique_set and duplicate_set,
    and log identical table IDs.
    """
    primary_rows = group_cells_by_row(primary_table['cells'])
    primary_keys = set(tuple(row[:2]) for row in primary_rows if len(row) >= 2)

    similar_keys = set()
    identical_tables = []  # To record identical similar table IDs

    for similar_table in similar_tables:
        similar_rows = group_cells_by_row(similar_table['cells'])
        similar_keys_current = set(tuple(row[:2]) for row in similar_rows if len(row) >= 2)
        
        if primary_keys == similar_keys_current:
            identical_tables.append(similar_table['id'])
        similar_keys.update(similar_keys_current)
    
    # If identical tables exist, log them
    if identical_tables:
        log_data = {
            'primary_table_id': primary_table['id'],
            'identical_tables': identical_tables
        }
        with open(equality_log_file, 'a') as log_file:
            log_file.write(json.dumps(log_data) + '\n')
    
    unique_set = [row for row in primary_rows if tuple(row[:2]) not in similar_keys]
    duplicate_set = [row for row in primary_rows if tuple(row[:2]) in similar_keys]

    return unique_set, duplicate_set

def decontextualized(api, question, title):
    user_prompt = f"Table Title: {title}\nQuestion: {question}"
    messages = [
        {"role": "system", "content": template_title},
        {"role": "user", "content": user_prompt}
    ]
    response = api.generate_response(messages)
    return response

def generate_question(api, table, similar_tables, start_id, file_name, equality_log_file, error_log_file):
    """
    Generate a question, ensuring that the sampled rows from unique_set cannot be answered using duplicate_set.
    """
    print('*****************************Start Query Generation***********************')

    # Get unique_set and duplicate_set
    unique_set, duplicate_set = compare_tables(table, similar_tables, equality_log_file)
    print("****************************This is the unique set**********************")
    print(unique_set)
    
    # If unique_set is empty, skip this table
    if not unique_set:
        print("Skipping table as there is no valid unique set.")
        return None

    columns = [col['text'] for col in table['columns']]
    final_question = None
    synonyms = None
    last_error_response = None

    # Define the list of unique_set row counts to try
    unique_rows_options = [min(len(unique_set), random.randint(3,5)), 2, 1]

    for num_rows in unique_rows_options:
        attempt = 0
        max_retries = 3
        while attempt < max_retries:
            question_content = f"Table Title: {table['title']}\nColumns: " + ' | '.join(columns) + '\n'
            question_content += "Only source rows for the question generation are:\n"

            # Randomly sample num_rows rows from unique_set
            if len(unique_set) >= num_rows:
                selected_unique_rows = random.sample(unique_set, num_rows)
            else:
                selected_unique_rows = unique_set

            for row in selected_unique_rows:
                question_content += ' | '.join(row) + '\n'

            # Randomly sample 3-5 rows from duplicate_set
            if duplicate_set:
                selected_duplicate_rows = random.sample(duplicate_set, min(len(duplicate_set), random.randint(3, 5)))
            else:
                selected_duplicate_rows = []

            # Build the question based on SQL operator logic
            prob = 0.5
            condition_1 = ''
            if random.random() < prob:
                selected_operators = random.sample(SqlQuery.operators, min(len(SqlQuery.operators), 2))
                operator_str = ', '.join(selected_operators)
                condition_1 = f"(2) The question should implicitly contain operations: {operator_str}.\n"

            # Add conditions to the prompt to ensure the question cannot be answered using selected_duplicate_rows
            condition_2 = ""
            if selected_duplicate_rows:
                condition_2 = "(3) Please generate a question based on the provided table data that cannot be answered using the following rows:\n\n"
                for row in selected_duplicate_rows:
                    condition_2 += ' | '.join(row) + '\n'
                condition_2 += "\n"
                condition_2 += "(4) Please only output the generated question."
            else:
                condition_2 = "(3) Please only output the generated question."

            # Build prompt_sys
            prompt_sys = template_sql.format(condition_1=condition_1, condition_2=condition_2)
            messages = [
                {"role": "system", "content": prompt_sys},
                {"role": "user", "content": question_content}
            ]

            # Try calling the API to generate the original question
            try:
                original_question = api.generate_response(messages)
            except ContextLengthExceededError as e:
                print("Context length exceeded error occurred during question generation. Reducing number of sampled rows and retrying.")
                last_error_response = e.response
                break  # Break to reduce the number of sampled rows
            except Exception as e:
                print(f"Error: {e}. Retrying... ({attempt+1}/{max_retries})")
                attempt += 1
                time.sleep(5)
                continue

            print('*********************This is original_question*********************')
            print(original_question)

            # First check the quality of the question at the row level (before decontextualization)
            try:
                check_quality_result = check_quality(api, original_question, table)
            except ContextLengthExceededError as e:
                print("Context length exceeded during quality check. Reducing number of sampled rows and retrying.")
                last_error_response = e.response
                break  # Break to reduce the number of sampled rows
            except Exception as e:
                print(f"Error during quality check: {e}. Retrying... ({attempt+1}/{max_retries})")
                attempt += 1
                time.sleep(5)
                continue

            if not check_quality_result:
                attempt += 1
                print(f"Attempt {attempt}/{max_retries} failed at row-level quality check. Retrying...")
                continue  # Retry with the same number of rows

            # Proceed to decontextualization if the first quality check passes
            decontextualized_question = original_question
            if table['title'] and not is_title_in_question(table['title'], original_question):
                question_length = len(original_question.split(' '))
                decontextualize_prob = max(0, 1 - 0.02 * question_length)
                if random.random() < decontextualize_prob:
                    try:
                        decontextualized_question = decontextualized(api, original_question, table['title'])
                    except ContextLengthExceededError as e:
                        print("Context length exceeded during decontextualization. Skipping table.")
                        last_error_response = e.response
                        break  # Break to reduce the number of sampled rows
                    except Exception as e:
                        print(f"Error during decontextualization: {e}. Retrying... ({attempt+1}/{max_retries})")
                        attempt += 1
                        time.sleep(5)
                        continue

            # Now perform the table-level quality check
            try:
                table_quality_result = check_quality_table(api, decontextualized_question, table, similar_tables)
            except ContextLengthExceededError as e:
                print("Context length exceeded during table-level quality check. Reducing number of sampled rows and retrying.")
                last_error_response = e.response
                break  # Break to reduce the number of sampled rows
            except Exception as e:
                print(f"Error during table-level quality check: {e}. Retrying... ({attempt+1}/{max_retries})")
                attempt += 1
                time.sleep(5)
                continue

            # Proceed only if table-level quality check passes
            if table_quality_result:
                random_num = random.random()
                print(f"Random number for synonym and rewrite decision: {random_num}")
                if random_num < 0.4:
                    try:
                        synonyms = get_synonyms(api, decontextualized_question)
                        final_question = rewrite_question(api, decontextualized_question, synonyms)
                    except ContextLengthExceededError as e:
                        print("Context length exceeded during synonym generation or rewriting. Skipping table.")
                        last_error_response = e.response
                        break  # Break to reduce the number of sampled rows
                    except Exception as e:
                        print(f"Error during synonym generation or rewriting: {e}. Retrying... ({attempt+1}/{max_retries})")
                        attempt += 1
                        time.sleep(5)
                        continue
                else:
                    final_question = decontextualized_question
                break  # Exit retry loop if all checks pass
            else:
                attempt += 1
                print(f"Attempt {attempt}/{max_retries} failed at table-level quality check. Retrying...")
        else:
            continue  # Try the next number of rows

        if final_question:
            break  # Successfully generated a question, exit row count loop

    # If no question was generated, log the error and skip the table
    if not final_question:
        print(f"Skipping table {table['id']} due to context length exceeded or other errors.")
        if last_error_response:
            with open(error_log_file, 'a') as error_log:
                error_entry = {
                    'table_id': table['id'],
                    'table_name': table.get('title', ''),
                    'error_response': last_error_response
                }
                error_log.write(json.dumps(error_entry) + '\n')
        return None

    print('*********************This is final_question*********************')
    print(final_question)

    # Save the question to file
    with jsonlines.open(file_name, mode='a') as f:
        f.write({
            'qid': start_id,
            'table_id': table['id'],
            'original_question': original_question,
            'decontextualized_question': decontextualized_question if decontextualized_question else original_question,
            'synonyms': synonyms,
            'final_question': final_question,
            'table_text': question_content
        })

    return final_question

def check_quality(api, question, table):
    messages = [
        {"role": "system", "content": "Can you find the answer to the question in the given table rows? Please only output 'Yes' or 'No'."},
        {"role": "user", "content": f"Question: {question}\nTable Rows: {json.dumps(table['cells'])}"}
    ]
    try:
        response = api.generate_response(messages)
    except ContextLengthExceededError as e:
        raise e  # Reraise to be caught in generate_question
    return 'yes' in response.lower()

def check_quality_table(api, question, table, similar_tables):
    """
    Perform a table-level quality check to ensure the question cannot be answered by similar tables.
    """
    # Prepare the content for the prompt
    similar_tables_content = ''
    for similar_table in similar_tables:
        similar_tables_content += f"Table ID: {similar_table['id']}\nTitle: {similar_table['title']}\nColumns: {', '.join([col['text'] for col in similar_table['columns']])}\nCells: {json.dumps(similar_table['cells'])}\n\n"

    messages = [
        {"role": "system", "content": template_table_quality_check},
        {"role": "user", "content": f"Question: {question}\nTarget Table ID: {table['id']}\nTarget Table Title: {table['title']}\nTarget Table Cells: {json.dumps(table['cells'])}\n\nSimilar Tables:\n{similar_tables_content}"}
    ]
    try:
        response = api.generate_response(messages)
    except ContextLengthExceededError as e:
        raise e
    return 'yes' in response.lower()

def get_synonyms(api, question):
    messages = [
        {"role": "system", "content": template_synonyms},
        {"role": "user", "content": f"Question: {question}"}
    ]
    try:
        return api.generate_response(messages)
    except ContextLengthExceededError as e:
        raise e  # Reraise to be caught in generate_question

def rewrite_question(api, question, synonyms):
    messages = [
        {"role": "system", "content": template_rewrite},
        {"role": "user", "content": f"Question: {question}\nSynonyms: {synonyms}"}
    ]
    try:
        return api.generate_response(messages)
    except ContextLengthExceededError as e:
        raise e  # Reraise to be caught in generate_question

def main():
    api_key = "YOUR API KEY"  
    api = GPTInference(api_key)
    
    # File paths
    tables_file = 'YOUR TABLES FILE PATH'
    similar_tables_file = 'SIMILAR TABLES FILE PATH'
    output_file = 'OUTPUT FILE PATH'
    equality_log_file = 'EQUAL TABLES LOG'
    error_log_file = 'ERROR LOG FILE PATH'  # New error log file

    tables = read_tables_jsonl(tables_file)
    similar_tables = read_similar_tables(similar_tables_file)

    n = 5000  # Desired number of queries to generate
    qid = 0
    for main_table_id, similar_list in tqdm(similar_tables.items(), total=len(similar_tables), desc="Processing Tables"):
        if qid >= n:
            print(f"Reached desired number of queries ({n}), stopping.")
            break
        
        main_table = find_table_by_id(main_table_id, tables)
        if not main_table:
            continue
        
        # Collect similar tables for the main table
        similar_tables_group = [find_table_by_id(entry['id'], tables) for entry in similar_list]
        similar_tables_group = [table for table in similar_tables_group if table]  # Filter out missing tables
        
        question = generate_question(api, main_table, similar_tables_group, qid, output_file, equality_log_file, error_log_file)
        if question:
            qid += 1

if __name__ == "__main__":
    main()
