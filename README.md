# The Modal Llama

The Modal Llama is a wrapper around llama.cpp, ik_llama.cpp and llama-swap to serve LLMs on modal.com and other providers like runpod and vast.

## How to run - Modal

You can either run the server directly, which will download the models or download them separately. The difference is that downloading them separately will give more feedback on the download progress, as huggingface_hub will show the progress bar properly.

**Download models separately:**

```bash
uv run modal run -i -m modal_llama.modal.serve::prep
```

**Serve the app:**

```bash
uv run modal serve -m modal_llama.modal.serve
# or
uv run src/modal_llama/modal/serve.py
```

Now you can use the URL printed in the terminal to access the app.

## How to run - Generic SSH Server

You can also deploy to a generic server with SSH access using the `modal_llama.baremetal.deploy` module:

```bash
# Deploy to a server using the current user
uv run -m modal_llama.baremetal.deploy my-server-hostname

# Deploy with custom username and custom SSH port
uv run -m modal_llama.baremetal.deploy my-server-hostname --username my-username --port 2222
```

This will:
1. Copy the project files to the server using rsync
2. Install system dependencies using apt and Python dependencies using uv
3. Build llama.cpp, ik_llama.cpp, and llama-swap on the server
4. Download the configured models
5. Start the llama-swap and nginx servers
6. Print the access URL

You can then access the app using the URL printed in the terminal.
