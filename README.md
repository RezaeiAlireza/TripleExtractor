# TripleExtractor
TripleExtractor is a cutting-edge application designed to extract subject-predicate-object triples from input text, URLs, or text files. Powered by [TPLinker](https://github.com/131250208/TPlinker-joint-extraction), a state-of-the-art model for relation extraction, this project supports various output formats including JSON-LD, CSV, RDF, and XML.
This project integrates a FastAPI backend and a React frontend, providing an intuitive user interface and seamless server-client communication. For ease of deployment, the project is fully Dockerized.

## Features

- Multiple Input Options:
1. Enter plain text.
2. Provide a URL to fetch and extract text from web pages.
3. Upload text files for processing.

- Multiple Output Formats:
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
- Python 3.8+
- Node.js 16+
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
conda create -n tplinker python=3.8 -y
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
3. Enter your text, paste the URL, or upload a text file.
4. Select your desired output format: JSON-LD, CSV, RDF, or XML.
5. Click Submit to process the input and view the results.
6. Use the Download button to save the results in your selected format.

## Project Structure
```plaintext
TripleExtractor/
│
├── backend/
│   ├── app/
│   │   ├── data4bert/      # Data and relation mappings for TPLinker
│   │   ├── pretrained_models/  # Pretrained BERT models
│   │   ├── tplinker/       # TPLinker model
│   │   └── main.py         # FastAPI backend
│   └── requirements.txt    # Backend dependencies
│
├── frontend/
│   ├── public/             # Public assets
│   ├── src/                # React source code
│   │   ├── components/     # Reusable React components
│   │   ├── pages/          # Page components (e.g., HomePage)
│   │   └── App.js          # Main React app
│   └── package.json        # Frontend dependencies
│
└── README.md               # Project documentation
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

License
This project is licensed under the MIT License. See the LICENSE file for details.

### Acknowledgments
TPLinker: A joint extraction framework for subject-predicate-object triples.
FastAPI: For providing a robust backend framework.
React: For a user-friendly frontend interface.
For questions or support, feel free to contact [rezaei.alireza1290@gmail.com].


