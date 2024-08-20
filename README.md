# ollama-docker

[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub last commit (branch)](https://img.shields.io/github/last-commit/wolffaxn/ollama-docker/main.svg)](https://github.com/wolffaxn/ollama-docker)

# Getting started

Clone Git Repository and change into directory.

```sh
git clone https://github.com/wolffaxn/ollama-docker.git
cd ollama-docker
```

Creating a new network in docker (this step only needs to be done once).

```sh
docker network create ollama-net 
```

To start all containers, run the following command.

```sh
docker compose up -d
```

To start a single container, run the follwing command.

```sh
docker compose up -d [traefik|ollama|ollama-web|qdrant]
```

To stop all containers, run the following command.

```sh
docker compose down
```

The UIs can be accessed at the following URLs.

- Ollama - http://ollama.localhost
- Open WebUI - http://ollama-web.localhost
- Qdrant - http://qdrant.localhost/dashboard 
- Traefik - http://localhost:8080/dashboard
