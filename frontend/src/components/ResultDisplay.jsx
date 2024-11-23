import React from 'react';

const ResultDisplay = ({ results, format, input, inputType }) => {
  if (!results) return null;

  if (results.error) {
    return <div style={{ color: 'red' }}>{results.error}</div>;
  }

  const handleDownload = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/extract-triples/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputType === 'url' ? null : input,
          url: inputType === 'url' ? input : null,
          format,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch data from the server.');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `output.${format}`;
      link.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error downloading file:', error);
      alert('Error downloading the file. Please try again.');
    }
  };

  return (
    <div>
      <h3>Results:</h3>
      {typeof results === 'string' ? (
        <pre style={{ whiteSpace: 'pre-wrap', wordWrap: 'break-word' }}>{results}</pre>
      ) : (
        <pre>{JSON.stringify(results, null, 2)}</pre>
      )}
      <button onClick={handleDownload}>Download as {format.toUpperCase()}</button>
    </div>
  );
};

export default ResultDisplay;
