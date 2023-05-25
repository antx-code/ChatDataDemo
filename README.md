# ChatDataDemo

A simple chat with data application backend api service using langchain and llama(gpt_index) with fastapi and qdrant.

## Usage
To integrate ChatDataDemo into your own documentation, follow these steps:
1. **Configuration:** Start by modifying the config.yaml file, replacing it with your own OpenAI API key and Qdrant API key.
2. **Data Upload:** Upload your unique dataset to Qdrant, and use the API to retrieve the response.
3. **Service Startup:** Execute the following commands to start the backend service using Docker-compose:
- **build the service**
    ```bash
    docker-compose -f docker-compose-build.yml build
    ```
- **start the service**
    ```bash
    docker-compose up -f docker-compose-deploy.yml up -d
    ```
This will build and launch your backend service, providing a robust API for your data-centric chat application.
