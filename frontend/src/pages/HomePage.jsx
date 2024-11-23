import React, { useState } from 'react';
import InputForm from '../components/InputForm';
import ResultDisplay from '../components/ResultDisplay';
import axios from 'axios';

const HomePage = () => {
  const [results, setResults] = useState(null);
  const [format, setFormat] = useState('json-ld'); // Default format
  const [input, setInput] = useState(''); // Holds text or URL
  const [inputType, setInputType] = useState('text'); // Default input type
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(''); // For displaying detailed error messages

  const handleSubmit = async ({ text, url, format }) => {
    setIsLoading(true);
    setResults(null); // Clear previous results
    setErrorMessage(''); // Clear previous error message
    setInputType(url ? 'url' : 'text');
    setInput(url || text);
    setFormat(format);

    try {
      const response = await axios.post('http://127.0.0.1:8000/extract-triples/', {
        text: url ? null : text, // Only send text when inputType is 'text'
        url: url || null,       // Only send URL when inputType is 'url'
        format,
      });
      setResults(response.data);
    } catch (error) {
      console.error('Error:', error);

      if (error.response) {
        // Server responded with a specific error
        setErrorMessage(error.response.data.detail || 'An error occurred on the server.');
      } else if (error.request) {
        // No response received
        setErrorMessage('No response from the server. Please check your connection.');
      } else {
        // Something else happened
        setErrorMessage('An unexpected error occurred. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container">
      <InputForm onSubmit={handleSubmit} />
      {isLoading ? (
        <p>Loading...</p>
      ) : errorMessage ? (
        <div style={{ color: 'red', marginTop: '20px' }}>{errorMessage}</div>
      ) : (
        <ResultDisplay results={results} format={format} input={input} inputType={inputType} />
      )}
    </div>
  );
};

export default HomePage;
