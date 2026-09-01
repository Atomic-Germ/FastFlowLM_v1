# FLM Test

A comprehensive testing framework for **[FastFlowLM (FLM)](https://fastflowlm.com)** that validates the functionality of various AI model categories including Language, Embedding, Audio, and Vision models.

## Overview

FLM_test is designed to thoroughly test FastFlowLM's API compatibility and model functionality across multiple modalities:

- **LLM Tests**: Language model inference with both streaming and non-streaming modes
- **Embedding Tests**: Text embedding model validation
- **Audio Tests**: Speech recognition and audio processing validation  
- **Vision Tests**: Vision-Language Model (VLM) tests with multi-image support

Each test suite automatically:
- Detects the FLM server version
- Fetches available models
- Runs standardized test prompts
- Saves results to CSV with timestamps
- Handles errors gracefully with detailed logging

## Prerequisites

- **Python 3.8 or higher**
- **FastFlowLM server** running locally or remotely
- **pip** (Python package manager)

## Quick Start

### 1. Clone the project

```bash
git clone https://github.com/ZaneNi/FLM_test.git
cd FLM_test
```

### 2. Setup (Linux/Mac or Windows)

**For Linux/Mac:**
```bash
source ./setup.sh
```

**For Windows (PowerShell):**
```powershell
.\setup.ps1
```


### 3. Configure the Server URL

Edit `backend.json` to point to your FLM server:

```json
{
    "base_url": "http://127.0.0.1:52625/v1"
}
```

Default settings:
- **Host:** 127.0.0.1 (localhost)
- **Port:** 52625

Update to match your FLM server deployment:

- **Local testing:** `http://127.0.0.1:port/v1`
- **Remote server:** `http://your-server-ip:port/v1`


### 4. Start FLM Server

Ensure your FastFlowLM server is running before running tests. Start the server with appropriate flags based on the tests you plan to run:

**Basic local server:**
```bash
flm serve
```

**For remote access (accessible from other machines):**
```bash
flm serve --host 0.0.0.0
```

**Load embedding models (required for embedding tests):**
```bash
flm serve -e 1
```

**Load audio models (required for audio tests):**
```bash
flm serve -a 1
```

**Combined flags (for running all tests):**
```bash
flm serve -e 1 -a 1
```


### 5. Run Tests

Activate the virtual environment:

**Linux/Mac:**
```bash
source venv/bin/activate
```

**Windows:**
```powershell
.\venv\Scripts\activate.ps1
```

Run tests with:

```bash
# Run all tests
python main.py --all

# Run specific tests
python main.py --llm                    # LLM tests only
python main.py --embedding              # Embedding tests only
python main.py --audio                  # Audio tests only
python main.py --vision                 # vision tests only

# Limit token generation
python main.py --llm --gen-lim 32      # Limit LLM output to 32 tokens
```

## Test Types

### LLM Tests
Tests language models with conversation capabilities.

**What it tests:**
- Non-streaming mode: Single API calls with standard responses
- Streaming mode: Continuous token-by-token responses
- Multi-turn conversations: Context preservation across exchanges
- Reasoning content extraction (if supported by model)

**Test Flow:**

**Non-stream** test:
  1. Initial prompt: "Teach me Maxwell's equations."
  2. Follow-up: "Summarize your answer."

**Stream** test:
  1. Initial prompt:"Tell me a joke and explain why it's funny." 
  2. Follow-up: "Summarize the joke and its explanation."

**Output:** `llm_results_v{version}_{timestamp}.csv`

### Embedding Tests

> TODO

**Output:** `embedding_results_v{version}_{timestamp}.csv`



### Audio Tests

> TODO

**Output:** `audio_results_v{version}_{timestamp}.csv`


### Vision Tests
Tests Vision-Language Models (VLM) with multi-image analysis.

**Tested Models:**
- `gemma3:4b`
- `medgemma:4b`
- `medgemma1.5:4b`
- `qwen2.5vl-it:3b`
- `qwen3vl-it:4b`
- `translategemma:4b`

**What it tests:**
- Multi-image understanding
- Detailed description generation
- Creative story generation connecting multiple images
- Streaming responses for image-to-text

**Test Images:**
- `test_files/image/test_image1.jpeg`
- `test_files/image/test_image2.jpg`

**Output:** `vison_results_v{version}_{timestamp}.csv`

## Understanding Results

Test results are saved as CSV files with the format:
```
{test_type}_results_v{version}_{timestamp}.csv
```

**Example filenames:**
- `llm_results_v0.9.35_20260308_202906.csv`
- `vision_results_vunknown_version_20260308_203124.csv`

### CSV Columns

**LLM Results:**
| Column | Description |
|--------|-------------|
| Model | Model ID/name |
| Mode | "Stream" or "Non-Stream" |
| Input | The prompt sent to the model |
| Reasoning Content | Internal reasoning (if available) |
| Output Content | Model's response |

**Vision Results:**
| Column | Description |
|--------|-------------|
| Model | VLM model ID |
| Input | The prompt sent to the model |
| Reasoning Content | Internal reasoning (if available) |
| Output Content | Model's response |

### Interpreting Results

- **N/A**: Feature not supported by the model
- **ERROR: {message}**: Test failed with specific error
- **Empty content**: Model timeout or connection issue


