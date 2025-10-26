pip3 install uv
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
git clone git@github.com:rbalestr-lab/stable-pretraining.git
cd stable-pretraining
uv pip install -e ".[all]"