# The Modal Llama

The Modal Llama is a wrapper around llama.cpp, ik_llama.cpp and llama-swap to serve LLMs on modal.com and other providers like runpod and vast.

This will:
1. Copy the project files to the server using rsync
2. Install system dependencies using apt and Python dependencies using uv
3. Build llama.cpp, ik_llama.cpp, and llama-swap on the server
4. Download the configured models
5. Start the llama-swap and nginx servers
6. Print the access URL

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

You can then access the app using the URL printed in the terminal.

## How to run - RunPod

You can deploy to RunPod using the `modal_llama.runpod.deploy` module:

```bash
# Rent a pod with SSH access, enough space for models and GPU support (You might want a bigger "gun" i.e. GPU for larger models)
runpodctl create pod --containerDiskSize 60 --gpuCount 1 --gpuType "NVIDIA L40S" --imageName "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04" --mem 60 --vcpu 16 --startSSH --volumePath "/workspace" --volumeSize 300 --name "modal-llama" --ports '22/tcp'

# Get pod's SSH access information
runpodctl ssh connect modal-llama

# This will print something along the lines of:
# ssh root@<pod-ip> -p <ssh-port> # and somme comments about the pod

# Deploy to the pod via SSH
uv run -m modal_llama.baremetal.deploy ${pod_ip} --username root --port ${ssh_port}
```

### Tips

You can list all available GPU types with:

```bash
uv run -m modal_llama.runpod.list_gpu_types
```