#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

CERT_PATH="./docker/traefik/certs"
DOMAIN="localhost"

openssl req \
  -x509 \
  -newkey rsa:4096 \
  -sha256 \
  -days 3650 \
  -nodes \
  -keyout "$CERT_PATH/$DOMAIN.key" \
  -out "$CERT_PATH/$DOMAIN.crt" \
  -subj "/CN=traefik-dev" \
  -addext "subjectAltName=DNS:$DOMAIN, DNS:*.$DOMAIN"

echo "New self-signed certificate created."
