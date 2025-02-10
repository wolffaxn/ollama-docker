#!/usr/bin/env bash
set -euo pipefail

DOMAIN="localhost"

openssl req \
  -x509 \
  -newkey rsa:4096 \
  -sha256 \
  -days 3650 \
  -nodes \
  -keyout "./config/certs/$DOMAIN.key" \
  -out "./config/certs/$DOMAIN.crt" \
  -subj "/CN=traefik" \
  -addext "subjectAltName=DNS:$DOMAIN, DNS:*.$DOMAIN"

echo "New self-signed certificate created."
