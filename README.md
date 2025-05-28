# LinkedIn Sales Navigator Chatbot

A conversational AI application that helps sales professionals find leads and generate sales emails using Azure OpenAI and Azure Cosmos DB.

## Setup

### Prerequisites
- Python 3.8+
- Azure subscription with:
  - Azure OpenAI service
  - Azure Cosmos DB account
  - Azure Cognitive Search service (optional, for RAG capabilities)

### Configuration
1. Clone this repository
2. Create a `.env` file based on the `.env.template` file
3. Fill in your Azure service credentials

### Azure Cosmos DB Setup
1. Create a Cosmos DB account in the Azure portal
2. Create a database named `linkedin_data` (or update the env var)
3. Create two containers:
   - `employees` - for permanent data storage
   - `temp_employees` - for temporary query results
4. Both containers should have the same schema with partition key `/company`
5. Import your LinkedIn profile data into the `employees` container

### Database Schema
Each document should have the following properties:
- `id` (string): Unique identifier
- `full_name` (string): LinkedIn user's name
- `full_name_url` (string): URL to LinkedIn profile
- `role` (string): Job title
- `company` (string): Current company (used as partition key)
- `time` (number): Years of experience
- `activity` (string): Recent activity
- `interests` (string): Professional interests
- `experience_overview` (string): Summary of experience
- `experience_details` (string): Detailed experience

### Running the Application
```bash
cd ChatBot
pip install -r requirements.txt
python app.py
```

The application will be available at http://localhost:5000
