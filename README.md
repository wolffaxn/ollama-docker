# ollama-docker

[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub last commit (branch)](https://img.shields.io/github/last-commit/wolffaxn/ollama-docker/main.svg)](https://github.com/wolffaxn/ollama-docker)

## Overview

This repository contains a RAG pipeline using LlamaIndex framework, Ollama as local LLM server, Qdrant as vector database and Open WebUI as web frontend.

## 📦 Installation and Setup

Ensure Python 3.12 is installed.

  ```sh
  python --version
  Python 3.12.4
  ```

Clone the repository (or download it as a zip file):

  ```sh
  git clone https://github.com/wolffaxn/ollama-docker.git
  cd ollama-docker
  ```

Create the virtual environment named venv using Pip with Python version 3.12:

  ```sh
  python3 -m venv .venv && source .venv/bin/activate
  ```

Install the project dependencies included in the requirements.txt:

  ```sh
  pip install -r requirements.txt
  ```

 ## ⚡ Quick Start with Docker

Create the following new networks:

  ```sh
  docker network create backend
  docker network create frontend 
  ```
> [!NOTE]
> The step above only needs to be done once.

To start all containers, run the following command:

  ```sh
  docker compose up -d
  ```

> [!TIP] 
> To start a single container, run the follwing command.
>
> ```sh
> docker compose up -d [ollama|open-webui|pipelines|qdrant|redis|traefik]
> ```

Manage Open WebUI configurations:

  - Open [Open WebUI](https://open-webui.localhost).
  - Navigate to the **Settings > Connections > OpenAI API** section in Admin Panel.
    - Add a OpenAI API URL and set it to `http://pipelines:9099` and the API key to `0p3n-w3bu!`.
    - Add a Ollama API URL and set it to `http://host.docker.internal:7869`.
  - Navigate to the **Settings > Models** section in Admin Panel.
    - Pull the following models from Ollama.com:
      - `jina/jina-embeddings-v2-base-de:latest`
      - `llama3.1:8b`

> [!TIP]
> With `ollama.sh` you can run an ollama server with gpu support (metal) on macos. 
> In this case set the Ollama API URL to http://host.docker.internal:11434.

All services can be accessed via the following links.

- [Ollama](https://ollama.localhost)
- [Open WebUI](https://open-webui.localhost)
- [Pipelines](https://pipelines.localhost)
- [Promptfoo](https://promptfoo.localhost)
- [Qdrant](https://qdrant.localhost/dashboard)
- [Traefik](https://traefik.localhost/dashboard/) (login: admin/admin)

## Ingestion and Retrieval

Run the ingestion pipeline.

  ```sh
  ./ingest.sh
  ```

Run a query.

  ```sh
  ./queryengine.sh <query>
  ./chatengine.sh
  ```
