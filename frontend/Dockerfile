# Use Node.js as the base image
FROM node:alpine

# Set the working directory
WORKDIR /app

# Copy the dependency files first to leverage Docker's caching
COPY package.json package-lock.json /app/

# Install dependencies
RUN npm install

# Copy the rest of the application files
COPY . /app/

# Use npm start to serve the application
CMD ["npm", "start"]

# Expose port 3000
EXPOSE 3000
