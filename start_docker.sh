#!/usr/bin/env bash
nvidia-docker run -it -v "$(pwd)/competition-docker-files:/app/codalab" wahaha909/autonlp:gpu
