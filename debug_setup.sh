#!/bin/bash
apt update
apt upgrade -y
apt install -y software-properties-common build-essential libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev libffi-dev
apt install -y python3.12 python3.12-venv python3.12-dev
apt install git-lfs
python3.12 -m venv venv
source venv/bin/activate
pip install uv
uv pip install -r new_requirements.txt
