import React from 'react';
import { AppBar, Toolbar, Typography, Container } from '@mui/material';

const Layout = ({ children }) => (
  <div>
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6">Relation Extraction Tool</Typography>
      </Toolbar>
    </AppBar>
    <Container sx={{ mt: 4 }}>{children}</Container>
  </div>
);

export default Layout;
