#!/usr/bin/env bash
#
# setup_rag_env.sh
# Sets up a Python virtual environment for the Multi-Agent RAG notebook
# and launches Jupyter.
#
# Usage:
#   chmod +x setup_rag_env.sh
#   ./setup_rag_env.sh
#
# After it runs, Jupyter opens in your browser. Open:
#   multi_agent_rag_flights.ipynb
#
set -e  # exit immediately if any command fails

ENV_NAME="rag-env"
PYTHON_BIN="python3"

echo "=============================================="
echo "  Multi-Agent RAG — Environment Setup"
echo "=============================================="

# 1. Check Python is available
if ! command -v $PYTHON_BIN &> /dev/null; then
    echo "ERROR: python3 not found. Install Python 3.9+ first."
    exit 1
fi
echo "Using: $($PYTHON_BIN --version)"

# 2. Create the virtual environment (only if it doesn't exist yet)
if [ ! -d "$ENV_NAME" ]; then
    echo ""
    echo "[1/4] Creating virtual environment '$ENV_NAME'..."
    $PYTHON_BIN -m venv $ENV_NAME
else
    echo ""
    echo "[1/4] Virtual environment '$ENV_NAME' already exists — reusing it."
fi

# 3. Activate it
echo "[2/4] Activating environment..."
# shellcheck disable=SC1090
source "$ENV_NAME/bin/activate"

# 4. Upgrade pip and install dependencies
echo "[3/4] Installing dependencies (this may take a minute)..."
pip install --quiet --upgrade pip
pip install --quiet \
    openai \
    faiss-cpu \
    numpy \
    tiktoken \
    jupyter \
    notebook \
    ipykernel

# 5. Register the environment as a Jupyter kernel
echo "[4/4] Registering Jupyter kernel..."
python -m ipykernel install --user --name=$ENV_NAME --display-name="Python (rag-env)"

echo ""
echo "=============================================="
echo "  Setup complete!"
echo "=============================================="
echo ""
echo "IMPORTANT: Set your OpenAI API key before running the notebook."
echo "Either export it now:"
echo "    export OPENAI_API_KEY='sk-your-key-here'"
echo "or paste it into cell 2 of the notebook."
echo ""
echo "Launching Jupyter Notebook..."
echo "When it opens, select the 'Python (rag-env)' kernel."
echo ""

# 6. Launch Jupyter
jupyter notebook multi_agent_rag_flights.ipynb