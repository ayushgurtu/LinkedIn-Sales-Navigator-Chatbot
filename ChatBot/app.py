from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import threading

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import pandas as pd
# Remove sqlite3 import
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from openai import AzureOpenAI
import os
import re
# Update SQLAlchemy references
from sqlalchemy.exc import OperationalError
import ast
# Remove SQLite engine creation
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

# Add Cosmos DB configuration
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
COSMOS_DATABASE = os.getenv("COSMOS_DATABASE", "linkedin_data")
COSMOS_CONTAINER = os.getenv("COSMOS_CONTAINER", "employees")
COSMOS_TEMP_CONTAINER = os.getenv("COSMOS_TEMP_CONTAINER", "temp_employees")

# Initialize Cosmos DB client
cosmos_client = CosmosClient(COSMOS_ENDPOINT, credential=COSMOS_KEY)

# Create or get database and containers
database = cosmos_client.create_database_if_not_exists(id=COSMOS_DATABASE)
employees_container = database.create_container_if_not_exists(
    id=COSMOS_CONTAINER, 
    partition_key=PartitionKey(path="/company"),
    offer_throughput=400
)
temp_container = database.create_container_if_not_exists(
    id=COSMOS_TEMP_CONTAINER,
    partition_key=PartitionKey(path="/company"),
    offer_throughput=400
)

# Azure OpenAI configuration
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")  # Ensure this is your Admin Key
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_MODEL")
AZURE_OPENAI_EMBEDDINGS = os.getenv("AZURE_OPENAI_EMBEDDINGS")
search_index = os.getenv("AZURE_SEARCH_INDEX_NAME")

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
    # Clear temp container instead of dropping table
    try:
        # Delete all documents in temp container
        for item in temp_container.query_items(
            query="SELECT * FROM c",
            enable_cross_partition_query=True
        ):
            temp_container.delete_item(item, partition_key=item.get('company', ''))
        logging.info("Cleared temp_employees container")
    except Exception as e:
        logging.error(f"Error clearing temp container: {str(e)}")
    
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

# New route to clear history
@app.route('/clear-chat', methods=['POST'])
def clear_chat():
    try:
        # Clear temp container
        for item in temp_container.query_items(
            query="SELECT * FROM c",
            enable_cross_partition_query=True
        ):
            temp_container.delete_item(item, partition_key=item.get('company', ''))
        
        # Reset global variables
        global search_results_text, response_email
        search_results_text = pd.DataFrame()
        response_email = ''
        
        return jsonify({'status': 'success', 'message': 'Chat session cleared'}), 200
    except Exception as e:
        logging.error(f"Error in /clear-chat: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def get_Chat_response(text):

    global search_results_text, response_email

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
            You are an AI that converts natural language to Cosmos DB SQL queries based on user requests.  
            You will use two containers in Cosmos DB: 'employees' and 'temp_employees', both of which have the same schema:

            - full_name (TEXT)
            - full_name_url (TEXT)
            - role (TEXT)
            - company (TEXT)
            - time (NUMBER)
            - activity (TEXT)
            - interests (TEXT)
            - experience_overview (TEXT)
            - experience_details (TEXT)

            Always return **only** the SQL query without explanation or extra text.  
            Use proper SQL formatting for Cosmos DB.

            **Container Usage:**
            - Begin with querying the **'employees'** container.
            - For follow-up queries, use the **'temp_employees'** container as the reference.

            **Examples:**  
            Example 1: Input
            Retrieve all employees working at 'Google'.

            Example 1: Output
            SELECT * FROM employees WHERE CONTAINS(employees.company, 'Google')

            Example 2: Input
            List all employees who have 'AI' in their interests.

            Example 2: Output
            SELECT * FROM employees WHERE CONTAINS(employees.interests, 'AI')

            Example 3: Input
            Filter all employees whose experience is greater than 10 years.

            Example 3: Output
            SELECT * FROM employees WHERE employees.time > 10

            Example 4: Input
            Use the results from previous queries and filter those with a 'Software Engineer' role.
            
            Example 4: Output
            SELECT * FROM temp_employees WHERE CONTAINS(temp_employees.role, 'Software Engineer')
            """


            final_answer = generate_chat_response(rag_prompt, user_query = user_query)

            # Extract the SQL query - update regex for possible different formatting
            match = re.search(r"```sql\n(.*?)\n```|SELECT.*?;?$", final_answer, re.DOTALL)
            sql_query = match.group(1) if match and match.group(1) else final_answer

            if 'SELECT' not in sql_query:
                return sql_query
            else:
                try:
                    # Determine which container to query
                    target_container = temp_container if 'temp_employees' in sql_query else employees_container
                    
                    # Convert SQL query to Cosmos DB format if needed
                    cosmos_query = sql_query.replace('employees.', 'c.')
                    cosmos_query = cosmos_query.replace('temp_employees.', 'c.')
                    cosmos_query = cosmos_query.replace('FROM employees', 'FROM c')
                    cosmos_query = cosmos_query.replace('FROM temp_employees', 'FROM c')
                    
                    # Execute query
                    items = list(target_container.query_items(
                        query=cosmos_query,
                        enable_cross_partition_query=True
                    ))
                    
                    if not items:
                        return "No result found"
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(items)
                    
                    # Rename columns to match expected format
                    columns_mapping = {
                        'id': 'id',
                        'full_name': 'Full Name', 
                        'full_name_url': 'LinkedIn URL',
                        'role': 'Role', 
                        'company': 'Company', 
                        'time': 'Years of Experience',
                        'activity': 'Activity', 
                        'interests': 'Interests', 
                        'experience_overview': 'Experience Overview', 
                        'experience_details': 'Experience Details'
                    }
                    
                    df = df.rename(columns=columns_mapping)
                    
                    # Store results in temp container
                    if target_container == employees_container:
                        # Clear temp container first
                        for item in temp_container.query_items(
                            query="SELECT * FROM c",
                            enable_cross_partition_query=True
                        ):
                            try:
                                temp_container.delete_item(item, partition_key=item.get('company', ''))
                            except:
                                pass
                        
                        # Store results in temp container
                        for _, row in df.iterrows():
                            document = {
                                'id': str(uuid4()),
                                'full_name': row.get('Full Name', ''),
                                'full_name_url': row.get('LinkedIn URL', ''),
                                'role': row.get('Role', ''),
                                'company': row.get('Company', ''),
                                'time': row.get('Years of Experience', 0),
                                'activity': row.get('Activity', ''),
                                'interests': row.get('Interests', ''),
                                'experience_overview': row.get('Experience Overview', ''),
                                'experience_details': row.get('Experience Details', '')
                            }
                            temp_container.create_item(body=document)
                    
                    search_results_text = df
                    unique_companies = df["Company"].dropna().unique().tolist()
                    
                    if len(unique_companies) > 1 and 'company' in sql_query:
                        response = {
                            "message": "I found the following companies based on your input:",
                            "buttons": unique_companies
                        }
                        return jsonify(response)
                    else:
                        df = df.fillna("N/A")
                        return (df.to_html(classes="table table-striped", index=False))
                
                except Exception as e:
                    logging.error(f"Database error: {str(e)}")
                    return f"I encountered an error processing your request: {str(e)}"
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
