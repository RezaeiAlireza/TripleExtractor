import React, { useState } from 'react';
import { TextField, Button, Select, MenuItem, FormControl, InputLabel, Typography, RadioGroup, FormControlLabel, Radio } from '@mui/material';

const InputForm = ({ onSubmit }) => {
  const [inputType, setInputType] = useState('text'); // Default to text input
  const [text, setText] = useState('');
  const [file, setFile] = useState(null);
  const [url, setUrl] = useState('');
  const [format, setFormat] = useState('json-ld');

  const handleFileChange = (event) => {
    const uploadedFile = event.target.files[0];
    if (uploadedFile) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setText(e.target.result); // Set file content as input text
      };
      reader.readAsText(uploadedFile);
      setFile(uploadedFile);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputType === 'text' && text.trim()) {
      onSubmit({ text, format });
    } else if (inputType === 'file' && file) {
      onSubmit({ text, format });
    } else if (inputType === 'url' && url.trim()) {
      onSubmit({ url, format });
    } else {
      alert('Please provide valid input.');
    }
  };

  return (
    <form onSubmit={handleSubmit} style={{ marginTop: '20px' }}>
      <Typography variant="h6" gutterBottom>
        Choose Input Type:
      </Typography>
      <RadioGroup row value={inputType} onChange={(e) => setInputType(e.target.value)}>
        <FormControlLabel value="text" control={<Radio />} label="Text Input" />
        <FormControlLabel value="file" control={<Radio />} label="Upload .txt File" />
        <FormControlLabel value="url" control={<Radio />} label="URL" />
      </RadioGroup>

      {inputType === 'text' && (
        <TextField
          label="Input Text"
          multiline
          rows={4}
          fullWidth
          value={text}
          onChange={(e) => setText(e.target.value)}
          variant="outlined"
          style={{ marginBottom: '20px' }}
        />
      )}

      {inputType === 'file' && (
        <input
          type="file"
          accept=".txt"
          onChange={handleFileChange}
          style={{ marginBottom: '20px', display: 'block' }}
        />
      )}

      {inputType === 'url' && (
        <TextField
          label="Input URL"
          fullWidth
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          variant="outlined"
          style={{ marginBottom: '20px' }}
        />
      )}

      <FormControl fullWidth style={{ marginBottom: '20px' }}>
        <InputLabel>Output Format</InputLabel>
        <Select value={format} onChange={(e) => setFormat(e.target.value)}>
          <MenuItem value="json-ld">JSON-LD</MenuItem>
          <MenuItem value="csv">CSV</MenuItem>
          <MenuItem value="rdf">RDF</MenuItem>
          <MenuItem value="xml">XML</MenuItem>
        </Select>
      </FormControl>
      <Button variant="contained" color="primary" type="submit">
        Extract
      </Button>
    </form>
  );
};

export default InputForm;
