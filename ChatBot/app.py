from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import threading

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import pandas as pd
import sqlite3
from openai import AzureOpenAI
import os
import re
from sqlalchemy.exc import OperationalError
import ast
from sqlalchemy import create_engine
from langchain.sql_database import SQLDatabase
from dotenv import load_dotenv
import markdown2
from bs4 import BeautifulSoup
from uuid import uuid4
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Session Management: Stores history as LlamaIndex ChatMessage compatible dicts
session_histories = {}
history_lock = threading.Lock()

# Load environment variables from .env file
load_dotenv()

# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")  # Ensure this is your Admin Key
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_MODEL")
AZURE_OPENAI_EMBEDDINGS = os.getenv("AZURE_OPENAI_EMBEDDINGS")
search_index = os.getenv("AZURE_SEARCH_INDEX_NAME")

engine_azure = create_engine('sqlite:///output_db.sqlite')

# Initialize Azure OpenAI Service client with key-based authentication    
client = AzureOpenAI(  
    azure_endpoint=AZURE_OPENAI_ENDPOINT,  
    api_key=AZURE_OPENAI_API_KEY,  
    api_version=AZURE_OPENAI_API_VERSION,
)

def initialize_session_history(session_id):
    with history_lock:
        if session_id not in session_histories:
            session_histories[session_id] = []  # Store list of dicts: {"role": ..., "content": ...}
            logging.info(f"Initialized history for new session: {session_id}")
        return session_histories[session_id]

def markdown_to_text(markdown_content):
    html = markdown2.markdown(markdown_content)
    allowed_tags = ['strong', 'b', 'em', 'i', 'br', 'p', 'a']
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(True):
        if tag.name == 'a':
            continue  # Keep <a> tags as-is for clickable links
        elif tag.name not in allowed_tags:
            tag.unwrap()

    cleaned_html = str(soup)
    cleaned_html = re.sub(r'\n{2,}', '\n\n', cleaned_html)
    cleaned_html = re.sub(r'[ \t]+', ' ', cleaned_html)

    return cleaned_html.strip()

def generate_chat_response(prompt: str, system_message: str = None, user_query: str = None):
    """
    Generates a single-turn chat response using Azure OpenAI Chat.
    If you need multi-turn conversation or follow-up queries, you'll have to
    maintain the messages list externally.
    """


    #Prepare the chat prompt 
    chat_prompt = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_query
                }
            ]
        }
    ] 
    
    # Include speech result if speech is enabled  
    messages = chat_prompt 

    # Generate the completion  
    completion = client.chat.completions.create(  
        model=AZURE_OPENAI_CHAT_MODEL,
        messages=messages,  
        temperature=0, 
    )
   
    return completion.choices[0].message.content

azure_conversation_history = []

search_results_text = pd.DataFrame();

def generate_rag(input_text):
    global search_results_text
    prompt_template = (
    "You are a sales email generator for Contoso. When the user asks to generate an email, "
    "generate an engaging, persuasive, and personalized email. "
    "If the recipient name is provided, include it in the greeting; "
    "if it is not provided, use {full_name} as the greeting. "
    
    "First, analyze the following interests and experience overview about {full_name}, "
    "whose current company is {Company_Name}:\n\n"

    "**Interests:**\n"
    "{interests}\n\n"

    "**Experience Overview:**\n"
    "{experience_overview}\n\n"

    "Use these interests and experience details to identify {full_name}'s key interests and experience. "
    "Then, craft an email explaining how Contoso can align with their key interests and professional background. "
    
    "Tailor the message to the recipient by incorporating best practices in sales communication, "
    "highlighting key selling points and Contoso's value proposition.\n\n"
    
    "**Word Limit Condition:**\n"
    "- If the user specifies a word limit:\n"
    "  - If the request is for a new email, generate an email that strictly adheres to the word limit while maintaining clarity, persuasiveness, and impact.\n"
    "  - If the request is to shorten an existing draft, edit the previously generated email to fit within the given word limit while preserving its intent, key message, and effectiveness. Summarize where necessary without losing impact.\n\n"

    "Please format the email as follows:\n"
    "----------------------------------------------------\n"
    "**Subject:** [A concise, compelling subject line]\n\n"
    "**Greeting:** [A personalized greeting, e.g., 'Dear {full_name},']\n\n"
    "**Introduction:** [A brief opening that introduces the purpose of the email]\n\n"
    "**Body:** [The main content of the email that explains the value proposition. Explain how Contoso’s solutions address {Company_Name}'s challenges. Highlight real needs and provide value.]\n\n"
    "**Call-to-Action:** [A clear call-to-action (e.g., 'Schedule a call', 'Request a demo', etc.)]\n\n"
    "**Closing:** [A courteous closing statement]\n\n"
    "**Signature:** [Your name, title, and company]\n"
    "----------------------------------------------------\n\n"
    "Use the above format strictly to ensure consistency and clarity." 
    )

    search_results_text_copy = search_results_text
    if len(search_results_text_copy) > 1:
        pattern = '|'.join(map(re.escape, search_results_text_copy['Full Name']))
        matching_names = re.findall(pattern, input_text, flags=re.IGNORECASE)
        # matching_names = re.findall(pattern, input_text)
        # Filter rows where full_name is in matching_names
        matching_rows = search_results_text_copy[
        search_results_text_copy['Full Name'].str.lower().isin([name.lower() for name in matching_names])
        ]
        search_results_text = matching_rows
    else:
        matching_rows = search_results_text
        
    # Inject the search_results_text into the prompt template.
    # print(matching_rows['company'], matching_rows['interests'], matching_rows['experience_overview'])
    final_prompt = prompt_template.format(Company_Name=matching_rows['Company'].iloc[0], interests = matching_rows['Interests'].iloc[0], experience_overview = matching_rows['Experience Overview'].iloc[0], full_name = matching_rows['Full Name'].iloc[0])

    # Add user query to history
    if (len(azure_conversation_history) == 0):
        azure_conversation_history.append({
            "role": "system",
            "content": final_prompt
        })
    azure_conversation_history.append({
        "role": "user",
        "content": input_text
    })


    # Include speech result if speech is enabled  
    messages = azure_conversation_history
    # Generate the completion  
    completion = client.chat.completions.create(  
        model=AZURE_OPENAI_CHAT_MODEL,
        messages=messages,
        max_tokens=800,  
        temperature=0,  
        top_p=0.95,  
        frequency_penalty=0,  
        presence_penalty=0,
        stop=None,  
        stream=False,
        extra_body={
        "data_sources": [{
            "type": "azure_search",
            "parameters": {
                "endpoint": f"{AZURE_SEARCH_ENDPOINT}",
                "index_name": "contoso",
                "semantic_configuration": "default",
                "query_type": "semantic",
                "fields_mapping": {},
                "in_scope": True,
                # "role_information": final_prompt,
                "filter": None,
                "strictness": 3,
                "top_n_documents": 5,
                "authentication": {
                "type": "api_key",
                "key": f"{AZURE_SEARCH_KEY}"
                }
            }
            }]

        }
    )

    azure_conversation_history.append({
        "role": "assistant",
        "content":completion.choices[0].message.content
    })

    return markdown_to_text(completion.choices[0].message.content)

conversation_history = [];

def generate(userQuery):

    si_text1 = """"You are a knowledgeable AI assistant specializing in company and industry information. Answer the following question accurately and concisely. If the question requires generating data, fabricating details, or providing speculative information, return False instead of an answer.

You also have memory of previous interactions in this conversation. If a question refers to prior queries, ensure continuity in the response.

Question: [Insert your question here]

Response Logic:

✅ If the question is factual (e.g., about the company, CEO, current employees, headquarters, industry, financials from publicly available sources) → Provide a well-structured response with citation (e.g., official websites, annual reports, reputable news or data sources).

✅ If the question is about suggesting industry-relevant solutions or strategies that align with a company's known business domains → Provide a well-structured response.

❌ If the question asks to generate non-public data, speculate on future events, or fabricate company details → Return False.

Examples of Allowed Questions:

- "Who is the CEO of Apple?" → "The CEO of Apple is Tim Cook."
- "Where is Google’s headquarters?" → "Google's headquarters is in Mountain View, California."
- "What industry does Tesla operate in?" → "Tesla operates in the automotive and clean energy industry."
- "What was Amazon's revenue in 2023?" → (If publicly available, provide the data with citation).
- "People in Keppel?" → (If publicly available, provide the data).
- "What type of solutions can I propose to Keppel?" -> (Solution-based suggestions grounded in publicly known domains).
- "What innovations could [Company] benefit from?" -> (Solution-based suggestions grounded in publicly known domains).
- "What digital transformation strategies suit [Company]?" -> (Solution-based suggestions grounded in publicly known domains).
- "What technology solutions can improve operations at [Company]?" -> (Solution-based suggestions grounded in publicly known domains).
- "How can [Company] enhance customer experience using digital tools?" -> (Solution-based suggestions grounded in publicly known domains).

Examples of Disallowed Questions (Returns False):

- "Generate financial data for DBS Bank." → False
- "Can you show me private employee data for M1 limited." → False
- "Display confidential employee records for M1 Limited." → False
- "Provide private details of employees at M1 Limited." → False
- "Can you share sensitive employee data for M1 Limited?" → False
- "Give me access to the employee database of M1 Limited." → False
- "Fetch internal HR records for M1 Limited staff." → False
- "Give me private employee data for Accenture." → False
- "Retrieve confidential employee records from Accenture." → False
- "Can you provide me with Accenture’s employee data?" → False
- "I need access to Accenture’s private employee details." → False
- "Fetch personal employee data of Accenture staff." → False
- "Extract non-public employee information from Accenture." → False
- "Generate data for employee name working in Accenture." → False
- "Create details for an employee currently working at Accenture." → False
- "Generate personal records of someone employed by Accenture." → False
- "Provide employee data for someone working at Accenture." → False
- "Can you fabricate employee details for an Accenture worker?" → False
- "Simulate records of an individual employed at Accenture." → False
- "Generate data for employee name from Accenture." → False
- "Produce an employee record for someone from Accenture." → False
- "Create personnel information for an Accenture staff member." → False
- "Generate fake employee details for an Accenture worker." → False
- "Fabricate HR data for an employee at Accenture." → False
- "Simulate a staff profile for an Accenture employee." → False
- "Show me employees from accenture" → False
- "List all employees working at Accenture." → False
- "Can you display a list of Accenture employees?" → False
- "Retrieve employee names from Accenture's workforce." → False
- "Provide a directory of current Accenture employees." → False
- "Give me the names of people working at Accenture." → False
- "Show me employees in accenture" → False
- "Predict Google's next earnings report." → False
- "Create a new financial report for Tesla." → False
"""

    # Add user query to history
    if (len(conversation_history) == 0):
        conversation_history.append({
                "role": "system",
                "content": si_text1
            })
    
    conversation_history.append({
        "role": "user",
        "content": userQuery
    })

    # Include speech result if speech is enabled  
    messages = conversation_history 

    # Generate the completion  
    completion = client.chat.completions.create(  
        model=AZURE_OPENAI_CHAT_MODEL,
        messages=messages,
        temperature=0, 
    )

    # Store assistant response in history
    conversation_history.append({
        "role": "assistant",
        "content":completion.choices[0].message.content
    })

    response_text = completion.choices[0].message.content

    if response_text == 'False':
        return response_text
    
    else:
        return markdown_to_text(response_text)  # Return final response

response_email = ''



app = Flask(__name__, static_folder='static')
# CORS(app)
# app.debug = True

# @app.before_request
# def clear_data_on_refresh():
#     """ Clears global variables when a new request starts """
#     global search_results_text, response_email
#     search_results_text = pd.DataFrame()
#     response_email = ''
#     conn = sqlite3.connect('output_db.sqlite')
#     c = conn.cursor()
#     c.execute('DROP TABLE IF EXISTS temp_employees')
#     conn.commit()


@app.route("/")
def index():
    conn = sqlite3.connect('output_db.sqlite')
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS temp_employees')
    conn.commit()
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    global search_results_text
    msg = request.form["msg"]
    input = msg
    if len(search_results_text) > 0 and input in search_results_text["Company"].values:
        search_results_text = search_results_text[search_results_text["Company"] == input]
        result = search_results_text
        return (result.to_html(classes="table table-striped", index=False))
    return get_Chat_response(input)

@app.route('/create-session', methods=['POST'])
def create_session_route():
    try:
        session_id = str(uuid4())
        initialize_session_history(session_id)
        logging.info(f"New backend session created: {session_id}")
        return jsonify({'status': 'success', 'session_id': session_id}), 200
    except Exception as e:
        logging.error(f"Error in /create-session: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/clear-backend-history', methods=['POST'])
def clear_backend_history_route():
    data = request.json
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({'status': 'error', 'message': 'session_id is required'}), 400

    with history_lock:
        if session_id in session_histories:
            session_histories[session_id] = []
            logging.info(f"Backend chat history cleared for session: {session_id}")
            message = 'Backend chat history cleared for this session.'
        else:
            # It's okay if the session wasn't on backend yet, initialize it empty.
            initialize_session_history(session_id)
            logging.warning(f"Attempted to clear history for session not actively on backend (or new): {session_id}. Initialized empty.")
            message = 'Backend session history was not found (or was new) and is now initialized empty.'
            
    return jsonify({'status': 'success', 'message': message})

def get_Chat_response(text):

    global search_results_text, response_email

    # Let's chat for 5 lines
    # for step in range(5):
    #     # encode the new user input, add the eos_token and return a tensor in Pytorch
    #     new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

    #     # append the new user input tokens to the chat history
    #     bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    #     # generated a response while limiting the total chat history to 1000 tokens, 
    #     chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    #     # pretty print last ouput tokens from bot
    #     return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    if not("mail" in text.lower() or "email" in text.lower()): #or "word" in text.lower() or "words" in text.lower() or "shorten" in text.lower() or "rewrite" in text.lower() or "limit" in text.lower() or "condense" in text.lower() or "reduce" in text.lower()

        final_answer = generate(text)

        if final_answer != 'False':

            return(final_answer)
            # with st.chat_message("assistant"):
            #     st.write(final_answer)

            #     st.session_state.messages.append(AIMessage(final_answer))
        else:
            user_query = text
            rag_prompt = """
            You are an AI that generates SQL queries based on user requests.  
            You will use two tables: 'employees' and 'temp_employees', both of which have the same schema:

            - full_name (TEXT)
            - full_name_url (TEXT)
            - role (TEXT)
            - company (TEXT)
            - time (INT)
            - activity (TEXT)
            - interests (TEXT)
            - experience_overview (TEXT)
            - experience_details (TEXT)

            Always return **only** the SQL query without explanation or extra text.  
            Use proper SQL formatting.  

            **Table Usage:**
            - Begin with querying the **'employees'** table.
            - The results of the query should be stored in the **'temp_employees'** table. For any future queries, use **'temp_employees'** as the reference table unless explicitly stated to query the original **'employees'** table.  

            **Examples:**  
            Example 1: Input
            Retrieve all employees working at 'Google'.
            SELECT *  
            FROM employees  
            WHERE company LIKE '%Google%';  

            Example 2: Input
            List all employees who have 'AI' in their interests.

            Example 2: Output
            SELECT *  
            FROM employees  
            WHERE interests LIKE '%AI%';  

            Example 3: Input
            Retrieve all employees with 'Software Engineer' as their role.

            Example 3: Output
            SELECT *  
            FROM employees  
            WHERE role LIKE '%Software Engineer';  


            Example 4: Input
            Filter all employees whose experience is greater than 10 years.

            Example 4: Output
            SELECT *  
            FROM employees
            WHERE time > 10;

            Example 5: Input
            Filter all employees whose experience is less than 10 years.

            Example 5: Output
            SELECT *  
            FROM employees  
            WHERE time < 10;

            Example 6: Input
            Filter all employees whose experience is less than 10 years in Accenture.

            Example 6: Output
            SELECT *  
            FROM employees  
            WHERE time < 10 and company LIKE '%Accenture%';  

            Example 9: Input
            Generate Employees
            
            Example 9: Output
            SELECT *  
            FROM employees;

            Example 9: Input
            Show Employees
            
            Example 9: Output
            SELECT *  
            FROM employees;

            Example 9: Input
            Give me employees details
            
            Example 9: Output
            SELECT *  
            FROM employees;

            Example 7: Input
            Use the results from previous queries and filter those with a 'Software Engineer' role from the previous results.
            
            Example 7: Output
            SELECT *  
            FROM temp_employees  
            WHERE role = 'Software Engineer';

            Example 8: Input
            Use the previous results to query and filter for those with 'AI' in their interests from the previous results.
            
            Example 8: Output
            SELECT *  
            FROM temp_employees  
            WHERE interests LIKE '%AI%';

            Example 9: Input
            Use the results from previous queries and filter those employees working at 'Google' from the previous results.
            
            Example 9: Output
            SELECT *  
            FROM temp_employees  
            WHERE company = 'Google';

            """


            final_answer = generate_chat_response(rag_prompt, user_query = user_query)

            # Regular expression to extract SQL query
            match = re.search(r"```sql\n(.*?)\n```", final_answer, re.DOTALL)

            # Extracted SQL query
            sql_query = match.group(1) if match else final_answer  # Fallback to original if no match


            # print(sql_query)

            if 'SELECT' not in sql_query:
                # adding the response from the llm to the screen (and chat)
                return(sql_query)
                # with st.chat_message("assistant"):
                #     st.write(sql_query)

                #     st.session_state.messages.append(AIMessage(sql_query))
            else:

                database = SQLDatabase(engine_azure, view_support=True)
                output = database.run(sql_query)

                if output == '':

                    return("No result found")
                    # with st.chat_message("assistant"):
                    #     st.write("No result found")

                    #     st.session_state.messages.append(AIMessage("No result found"))
                else:

                    # Convert the string to a Python list of tuples
                    output = ast.literal_eval(output)
                    # Define column names
                    columns = ["Full Name", "LinkedIn URL", "Role", "Company", "Years of Experience",
                                "Activity", "Interests", "Experience Overview", "Experience Details"]

                    # Convert list of tuples to DataFrame
                    df = pd.DataFrame(output, columns=columns)

                    search_results_text = df
                    unique_companies = df["Company"].dropna().unique().tolist()


                    if len(unique_companies) > 1 and 'company' in sql_query:
                        # return("I found the following companies based on your input: ")
                        unique_companies = df["Company"].dropna().unique().tolist()

                        response = {
                            "message": "I found the following companies based on your input:",
                            "buttons": unique_companies
                        }
                        return jsonify(response)

                    else:

                        conn = sqlite3.connect('output_db.sqlite')
                        c = conn.cursor()

                        c.execute('CREATE TABLE IF NOT EXISTS temp_employees (full_name text, full_name_url text, role text, company text, time int, activity text, interests text, experience_overview text, experience_details text)')
                        conn.commit()

                        df.to_sql('temp_employees', conn, if_exists='replace', index = False)

                        # Fill missing values (NaN) with an empty string or any preferred default
                        df = df.fillna("N/A")

                        result = df


                        result.columns = ["Full Name", "LinkedIn URL", "Role", "Company", "Years of Experience", 
                                "Activity", "Interests", "Experience Overview", "Experience Details"]

                        search_results_text = result
                        # adding the response from the llm to the screen (and chat)

                        # print(result)
                        return (result.to_html(classes="table table-striped", index=False))
                        # with st.chat_message("assistant"):
                        #     st.write(result)

                        #     st.session_state.messages.append(AIMessage(result))

    
    # elif("shorten" in text.lower() or "rewrite" in text.lower() or "limit" in text.lower() or "condense" in text.lower() or "reduce" in text.lower()):

    #     response = generate_rag(input_text = text)
        
    #     response_email = response

    #     return(response)
        
    #     # adding the response from the llm to the screen (and chat)
    #     # with st.chat_message("assistant"):
    #     #     st.write(response)

    #     #     st.session_state.messages.append(AIMessage(response))


    else:
        response = generate_rag(input_text = text)
        
        response_email = response

        return(response)
        
        # adding the response from the llm to the screen (and chat)
        # with st.chat_message("assistant"):
        #     st.write(response)

        #     st.session_state.messages.append(AIMessage(response))





if __name__ == '__main__':
    app.run()
