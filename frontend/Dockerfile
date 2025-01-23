# Use Node.js 16 LTS as the base image
FROM node:16-alpine

# Set the working directory
WORKDIR /app

# Copy the dependency files first to leverage Docker's caching
COPY package.json package-lock.json /app/

# Install dependencies
RUN npm install

# Copy the rest of the application files
COPY . /app/

# Build the application
RUN npm run build

# Use npm start to serve the application
CMD ["npm", "start"]

# Expose port 3000
EXPOSE 3000
