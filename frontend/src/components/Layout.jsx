import React from 'react';
import { AppBar, Toolbar, Typography, Container } from '@mui/material';

const Layout = ({ children }) => (
  <div>

    <Container sx={{ mt: 4 }}>{children}</Container>
  </div>
);

export default Layout;
