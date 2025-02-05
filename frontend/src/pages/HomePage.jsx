import React, { useState } from 'react';
import axios from 'axios';
import '../styles/HomePage.css';
import InputForm from '../components/InputForm';
import ResultDisplay from '../components/ResultDisplay';

const HomePage = () => {
  const [results, setResults] = useState(null);
  const [format, setFormat] = useState('json-ld');
  const [input, setInput] = useState('');
  const [inputType, setInputType] = useState('text');
  const [model, setModel] = useState('NYT');
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [useAdditionalLLM, setUseAdditionalLLM] = useState(false);


  const handleLLMChange = (event) => {
    setUseAdditionalLLM(event.target.value === "true"); // Convert to boolean
  };


  const handleSubmit = async ({ text, url, model }) => {
    setIsLoading(true);
    setResults(null);
    setErrorMessage('');
    setInputType(url ? 'url' : 'text');
    setInput(url || text);
    setModel(model);

    try {
      const response = await axios.post('http://127.0.0.1:8000/extract-triples/', {
        text: url ? null : text,
        url: url || null,
        format: 'json-ld',
        model,
        use_llm: useAdditionalLLM
      });
      setResults(response.data);
    } catch (error) {
      console.error('Error:', error);
      if (error.response) {
        setErrorMessage(error.response.data.detail || 'An error occurred on the server.');
      } else if (error.request) {
        setErrorMessage('No response from the server. Please check your connection.');
      } else {
        setErrorMessage('An unexpected error occurred. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownload = async (format) => {
    try {
      const response = await fetch('http://127.0.0.1:8000/extract-triples/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputType === 'url' ? null : input,
          url: inputType === 'url' ? input : null,
          format: format,
          model: model,
          use_llm: useAdditionalLLM

        }),
      });
  
      if (!response.ok) {
        throw new Error('Failed to fetch the downloadable content.');
      }
        const blob = await response.blob();
        const extensionMap = {
        'json-ld': 'jsonld',
        csv: 'csv',
        rdf: 'ttl',
        xml: 'xml',
      };
  
      const fileExtension = extensionMap[format] || 'txt';
      const fileName = `output.${fileExtension}`;
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = fileName;
      link.click();
      window.URL.revokeObjectURL(url);
  
      console.log(`File downloaded: ${fileName}`);
    } catch (error) {
      console.error('Error downloading file:', error);
      alert('Error downloading the file. Please try again.');
    }
  };
  const handleDownloadNotExtracted = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/download-not-extracted/', {
        method: 'GET',
      });
  
      if (!response.ok) {
        throw new Error('Failed to fetch the not extracted inputs file.');
      }
  
      const blob = await response.blob();
      const fileName = "not_extracted_inputs.txt";
  
      // Create a URL for the blob and initiate the download
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = fileName;
      link.click();
  
      // Clean up the URL
      window.URL.revokeObjectURL(url);
  
      console.log(`File downloaded: ${fileName}`);
    } catch (error) {
      console.error('Error downloading not extracted inputs file:', error);
      alert('Error downloading the file. Please try again.');
    }
  };
  

  return (
    <div className="homepage-root">
      {/* LEFT SIDEBAR */}
      <div className="sidebar">
        <div>
          <h2 style={{ color: '#31a3ba' , marginBottom:'50px'}}>Relation Discovery in Unstructured Text</h2>
        </div>
        <h3>Input type:</h3>
        <div className="sidebar-section">
          <label>
            <input
              type="radio"
              name="inputType"
              value="text"
              checked={inputType === 'text'}
              onChange={(e) => setInputType(e.target.value)}
            />
            Raw text
          </label>
          <label>
            <input
              type="radio"
              name="inputType"
              value="file"
              checked={inputType === 'file'}
              onChange={(e) => setInputType(e.target.value)}
            />
            File (txt)
          </label>
          <label>
            <input
              type="radio"
              name="inputType"
              value="url"
              checked={inputType === 'url'}
              onChange={(e) => setInputType(e.target.value)}
            />
            Web URL
          </label>
        </div>

        <h3>Model:</h3>
        <div className="sidebar-section">
          <label>
            <input
              type="radio"
              name="model"
              value="NYT"
              checked={model === 'NYT'}
              onChange={(e) => setModel(e.target.value)}
            />
            NYT
          </label>
          <label>
            <input
              type="radio"
              name="model"
              value="WebNLG"
              checked={model === 'WebNLG'}
              onChange={(e) => setModel(e.target.value)}
            />
            WebNLG
          </label>
          <h3>Download Output:</h3>
        <div className="sidebar-section">
          <button onClick={() => handleDownload('json-ld')}>JSON-LD</button>
          <button onClick={() => handleDownload('csv')}>CSV</button>
          <button onClick={() => handleDownload('rdf')}>Turtle</button>
          <button onClick={() => handleDownload('xml')}>XML</button>
          <button onClick={handleDownloadNotExtracted}>Not Extracted Inputs</button>
        </div>
        <h3>Use Additional LLM for Validation:</h3>
        <div className="sidebar-section">
          <label>
            <input
              type="radio"
              name="additional-llm"
              value="true"
              checked={useAdditionalLLM === true}
              onChange={handleLLMChange}
            />
            Yes
          </label>
          <label>
            <input
              type="radio"
              name="additional-llm"
              value="false"
              checked={useAdditionalLLM === false}
              onChange={handleLLMChange}
            />
            No
          </label>
        </div>
        </div>
      </div>

      {/* RIGHT CONTENT AREA */}
      <div className="main-content">
        {/* INPUT BOX */}
        <div className="box-container">
          <h3>Input</h3>
          <InputForm
            inputType={inputType}
            format={format}
            model={model}
            onSubmit={handleSubmit}
          />
        </div>

        {/* OUTPUT BOX */}
        <div className="box-container">
          <h3>Output</h3>
          {isLoading ? (
            <p>Loading...</p>
          ) : errorMessage ? (
            <div style={{ color: 'red' }}>{errorMessage}</div>
          ) : (
            <ResultDisplay
              results={results}
              handleDownload={handleDownload}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default HomePage;
