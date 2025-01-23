import React from 'react';

const ResultDisplay = ({ results }) => {
  if (!results) return null;
  if (results.error) {
    return <div style={{ color: 'red' }}>{results.error}</div>;
  }

  // Convert JSON-LD => array of triples
  let triples = [];
  if (results["@graph"]) {
    // e.g. results = { "@context": {...}, "@graph": [ ... ] }
    triples = results["@graph"];
  } else if (Array.isArray(results)) {
    // e.g. results = [ ... ]
    triples = results;
  }

  return (
    <div>
      <h3>Results:</h3>
      <div style={{ maxHeight: '250px', overflowY: 'auto', background: '#eee', padding: '10px', borderRadius: '10px' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: '#ddd', textAlign: 'left' }}>
              <th style={{ padding: '8px', border: '1px solid #ccc' }}>Subject</th>
              <th style={{ padding: '8px', border: '1px solid #ccc' }}>Predicate</th>
              <th style={{ padding: '8px', border: '1px solid #ccc' }}>Object</th>
            </tr>
          </thead>
          <tbody>
            {triples.length > 0 ? (
              triples.map((triple, index) => (
                <tr key={index}>
                  <td style={{ padding: '8px', border: '1px solid #ccc' }}>{triple.subject}</td>
                  <td style={{ padding: '8px', border: '1px solid #ccc' }}>{triple.predicate}</td>
                  <td style={{ padding: '8px', border: '1px solid #ccc' }}>{triple.object}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan="3" style={{ textAlign: 'center', padding: '8px' }}>
                  No results available.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ResultDisplay;
