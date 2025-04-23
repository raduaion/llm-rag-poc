#!/bin/bash
source .env
ENVIRONMENT=$1

if [ -f "$ENVIRONMENT.env" ]; then
	source "$ENVIRONMENT.env"
fi

IMAGE_ID=$REPO:$TAG