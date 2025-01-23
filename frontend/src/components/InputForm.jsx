import React, { useState, useEffect } from 'react';

const InputForm = ({ inputType, format, model, onSubmit }) => {
  const [text, setText] = useState('');
  const [file, setFile] = useState(null);
  const [url, setUrl] = useState('');

  useEffect(() => {
    // Reset input fields if the user toggles input type
    setText('');
    setFile(null);
    setUrl('');
  }, [inputType]);

  const handleFileChange = (event) => {
    const uploadedFile = event.target.files[0];
    if (uploadedFile) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setText(e.target.result);
      };
      reader.readAsText(uploadedFile);
      setFile(uploadedFile);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    // Submit based on whichever inputType is set
    if (inputType === 'text' && text.trim()) {
      onSubmit({ text, format, model });
    } else if (inputType === 'file' && file) {
      onSubmit({ text, format, model });
    } else if (inputType === 'url' && url.trim()) {
      onSubmit({ url, format, model });
    } else {
      alert('Please provide valid input.');
    }
  };

  return (
    <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', height: '100%' }}>
      {inputType === 'text' && (
        <div style={{ flexGrow: 1 }}>
          <textarea
            rows={5}
            style={{ width: '100%', height: '250px', borderRadius: '10px', background: '#eee', border: 'none', padding: '10px' }}
            placeholder="Enter your text here..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
        </div>
      )}

      {inputType === 'file' && (
        <div style={{ flexGrow: 1 }}>
          <label
            htmlFor="fileInput"
            style={{
              display: 'inline-block',
              padding: '10px 10px',
              cursor: 'pointer',
              backgroundColor: '#31a3ba',
              color: 'white',
              borderRadius: '4px',
              textAlign: 'center',
              width: '80vh',
            }}
          >
            Choose File
          </label>
          <input
            id="fileInput"
            type="file"
            accept=".txt"
            onChange={handleFileChange}
            style={{ display: 'none' }} // Hide the native input
          />
          {file && <p style={{ marginTop: '10px' }}>Loaded file: {file.name}</p>}
        </div>
      )}


      {inputType === 'url' && (
        <div style={{ flexGrow: 1 }}>
          <input
            style={{ width: '95%' , borderRadius: '10px', background: '#eee', border: 'none', padding: '10px' }}
            type="text"
            placeholder="Enter web URL"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
        </div>
      )}

      <button
        type="submit"
        style={{
          marginBottom: '30px',
          padding: '10px 20px',
          cursor: 'pointer',
          alignSelf: 'flex-end'
        }}
      >
        Extract
      </button>
    </form>
  );
};

export default InputForm;
