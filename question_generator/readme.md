Question-Table Pair Generation Script

This script generates training data for question-answering systems over tabular data. It leverages a Large Language Model (LLM) to create questions based on table data, ensuring that each question is uniquely answerable using a specific table and not by similar tables.

Overview

The script performs the following main steps:

	1.	Data Loading: Reads main tables and their similar tables from provided files.
	2.	Table Comparison: Identifies unique and duplicate rows between a main table and its similar tables.
	3.	Question Generation: Uses the LLM to generate questions based on sampled unique rows from the main table.
	4.	Quality Checks:
	•	Row-Level Check: Ensures the question can be answered using the main table’s rows.
	•	Table-Level Check: Ensures the question cannot be answered using similar tables.
	5.	Decontextualization: Optionally adds the table title to the question for additional context.
	6.	Synonym Replacement and Rewriting: Optionally rewrites the question using synonyms for diversity.
	7.	Output: Saves the generated questions and related data to an output file.

Requirements

	•	Python 3.x
	•	Libraries:
	•	json
	•	random
	•	http.client
	•	tqdm
	•	jsonlines
	•	time
	•	re

Setup

	1.	Install Dependencies:

pip install tqdm jsonlines


	2.	Set API Key:
Replace "YOUR_API_KEY" in the script with your actual API key for the LLM service.
	3.	Configure File Paths:
Update the file paths in the main() function to point to your data files and desired output locations.

Script Components

Classes

	•	SqlQuery: Contains SQL query elements and operators used in prompts.
	•	ContextLengthExceededError: Custom exception to handle cases where the LLM’s context length is exceeded.
	•	GPTInference: Handles API calls to the LLM, including retries and error handling.

Template Strings

Prompt templates used when interacting with the LLM:

	•	template_sql: For generating questions.
	•	template_synonyms: For extracting synonyms of entities in the question.
	•	template_rewrite: For rewriting the question using synonyms.
	•	template_title: For decontextualizing the question by adding the table title.
	•	template_table_quality_check: For checking if the question is uniquely answerable by the main table.

Functions

	•	Data Loading and Utilities:
	•	read_tables_jsonl(file_path): Reads tables from a JSONL file.
	•	read_similar_tables(file_path): Reads similar tables from a JSON file.
	•	find_table_by_id(table_id, tables): Finds a table by its ID.
	•	group_cells_by_row(cells): Groups table cells by their row indices.
	•	compare_tables(primary_table, similar_tables, equality_log_file): Identifies unique and duplicate rows.
	•	Question Generation:
	•	generate_question(api, table, similar_tables, start_id, file_name, equality_log_file, error_log_file): Main function that orchestrates question generation.
	•	Quality Checks:
	•	check_quality(api, question, table): Row-level quality check to ensure the question can be answered using the main table’s rows.
	•	check_quality_table(api, question, table, similar_tables): Table-level quality check to ensure the question cannot be answered by similar tables.
	•	Decontextualization and Rewriting:
	•	decontextualized(api, question, title): Adds the table title to the question if necessary.
	•	get_synonyms(api, question): Extracts synonyms for entities in the question.
	•	rewrite_question(api, question, synonyms): Rewrites the question using the extracted synonyms.

Main Execution

The main() function:

	•	Loads tables and similar tables.
	•	Sets the desired number of questions to generate (n = 5000).
	•	Iterates over tables to generate questions, performing quality checks and handling errors.
	•	Saves generated questions and related data to the specified output file.

Usage

Run the script using:

python generate_questions.py

Monitor progress via console output provided by tqdm.

Notes

	•	Data Files: Ensure your tables and similar tables are correctly formatted and accessible at the specified paths.
	•	API Usage: Be mindful of API usage limits and associated costs.
	•	Randomness: The script includes random elements in sampling and decision-making to enhance diversity in generated questions.
	•	Error Handling: Errors and identical tables are logged for analysis. The script retries operations when possible.

This readme provides a straightforward explanation of the script’s purpose and components, focusing on how it generates question-table pairs using an LLM while ensuring quality and uniqueness.