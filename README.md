![2](https://github.com/user-attachments/assets/f53c22a2-eb52-414d-87cd-b5e4a2784fb9)

# TripleExtractor
TripleExtractor is a cutting-edge application designed to extract subject-predicate-object triples from input text, URLs, or text files. Powered by [TPLinker](https://github.com/131250208/TPlinker-joint-extraction), a state-of-the-art model for relation extraction, this project supports various output formats including JSON-LD, CSV, RDF, and XML.
This project integrates a FastAPI backend and a React frontend, providing an intuitive user interface and seamless server-client communication. For ease of deployment, the project is fully Dockerized.

## Features

- Multiple Input Options:
1. Enter plain text.
2. Provide a URL to fetch and extract text from web pages.
3. Upload text files for processing.

- Downloadable Output Formats:
1. JSON-LD
2. CSV
3. RDF (Turtle format)
4. XML

- Language Validation:
Ensures that the input text is in English, providing descriptive error messages for unsupported languages.

- Downloadable Results:
Allows users to download results in their preferred format.

- Advanced Error Handling:
Handles input errors, server errors, and network issues gracefully.

- Output:
The output is in the tabular format for better visualization.

- Latest Update:
An additional LLM is added to validate the generated triples which uses Llama 3 8B Instruct from Huggingface.

## Pre-Setup

1. You need to navigate to [HuggingFace-Llama 3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) page and request permission to access the model.
2. Generate a SSH key and set it in a .env variable in root of backend folder under like: HF_TOKEN_LLAMA = hf_***
3. You need at least 2GB of GPU access to run this code.

## Installation
Follow the steps below to set up the project locally.
- Download [BERT-BASE-CASED](https://huggingface.co/bert-base-cased) and put it under `backend/pretrained_models`.

- Download [Model-State](https://drive.google.com/file/d/1jgiPScLaWJoPd2BkeWe-jSapPLQMyKEx/view?usp=sharing) and put it under `backend/tplinker/default_log_dir/r584cHKZ`.

## Dockerized Setup

To run the project using Docker:

1. Clone the Repository:
```bash
git clone https://github.com/RezaeiAlireza/TripleExtractor.git
cd TripleExtractor
```
2. Build and Run Docker Containers:
```bash
docker-compose up --build
```
This command will:
- Build and start the backend container (FastAPI).
- Build and start the frontend container (React).
3. Access the Application:
- Frontend: Open your browser and go to http://localhost:3000.
- Backend API: Access the backend API at http://localhost:8000.

## Manual Setup
If you prefer to set up and run the project manually:

### Prerequisites
- Python 3.9, 3.10
- Node.js 16
- Conda (Recommended) for environment management
- Git for version control
  
### Backend Setup

1. Clone the Repository:
```bash
git clone https://github.com/RezaeiAlireza/TripleExtractor.git
cd TripleExtractor/backend
```
2. Set Up Conda Environment:
```bash
conda create -n tplinker python=3.9 -y
conda activate tplinker
```
3. Install Dependencies:
```bash
pip install -r requirements.txt
```
4. Run Backend:
```bash
uvicorn main:app --reload
```
The backend will be available at http://127.0.0.1:8000.

### Frontend Setup

1. Navigate to Frontend:
```bash
cd ../frontend
```
2. Install Dependencies:
```bash
npm install
```
3. Start Frontend:
```bash
npm start
```
The frontend will be available at http://127.0.0.1:3000.

## Usage
1. Launch the application in your browser: http://127.0.0.1:3000.
2. Choose your input type (Text, URL, or File).
3. Choose the model between NYT or WebNLG.
- Optional: You can select additional LLM verification to yes in order for a post-filtering on result to improve results.
4. Enter your text, paste the URL, or upload a text file.
5. Press extract and wait for the model to process and show the output.
5. Use the Download section to choose the desired format to download the generated output.

## Project Structure
```plaintext
TripleExtractor/
│
├── backend/
│   ├── data4bert/
│   ├── pretrained_models/ 
│   ├── common/
│   ├── tplinker/    
│   ├── main.py      
│   ├── Dockerfile
│   └── requirements.txt 
│
├── frontend/
│   ├── public/          
│   ├── Dockerfile
│   ├── src/             
│   │   ├── components/     
│   │   ├── pages/          
│   │   └── App.js        
│   └── package.json       
│
│   ├── docker-compose.yaml    
└── README.md              
```
### Example Input and Output
#### Input (Text)
Text:
"Vienna is the capital of Austria."

#### Output (JSON-LD)
```json
{
  "@context": {
    "subject": "http://schema.org/subject",
    "predicate": "http://schema.org/predicate",
    "object": "http://schema.org/object"
  },
  "@graph": [
    {
      "@type": "Triple",
      "subject": "Vienna",
      "predicate": "/location/location/contains",
      "object": "Austria"
    }
  ]
}
```
## Screenshots
![image](https://github.com/user-attachments/assets/23649844-24fc-48f7-8900-b4eef81ae159)
![1](https://github.com/user-attachments/assets/9b814077-2fa8-4bd6-a546-04548c38d80c)


## License
This project is licensed under the MIT License. See the LICENSE file for details.

### Acknowledgments
TPLinker: A joint extraction framework for subject-predicate-object triples.
FastAPI: For providing a robust backend framework.
React: For a user-friendly frontend interface.
For questions or support, feel free to contact [rezaei.alireza1290@gmail.com].


