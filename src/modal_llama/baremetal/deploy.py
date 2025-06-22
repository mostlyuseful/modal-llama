import getpass
import pathlib
from shlex import quote
import subprocess

import typer
from fabric import Connection


def get_current_user() -> str:
    """
    Get the current system user.
    """
    return getpass.getuser()


app = typer.Typer()


@app.command()
def serve(
    host: str = typer.Option(..., help="SSH server host to deploy to"),
    port: int = typer.Option(22, help="SSH server port, defaults to 22"),
    user: str = typer.Option(
        get_current_user(), help="SSH username, defaults to current user"
    ),
    remote_dir: str = typer.Option(
        "/tmp/modal-llama-deploy",
        help="Remote directory to deploy the modal-llama server to, defaults to /tmp/modal-llama-deploy",
    ),
    remote_models_dir: str = typer.Option(
        ...,
        help="Remote directory to store the models. This directory will be used by the llama-swap server to load the models.",
    ),
    remote_nginx_port: int = typer.Option(
        8000,
        help="Port for the nginx reverse proxy on the remote server, defaults to 8000. This port will have to be open in the firewall for external access.",
    ),
):
    """
    Deploy the modal-llama server to a remote SSH server.
    This command uses Fabric to connect to the remote server and run the necessary commands to set up the modal-llama server:

    1. Copy the project files to the server using rsync
    2. Install system dependencies using apt and Python dependencies using uv
    3. Build llama.cpp, ik_llama.cpp, and llama-swap on the server
    4. Download the configured models
    5. Start the llama-swap and nginx servers
    6. Print the access URL
    """

    c = Connection(host=host, user=user, port=port)
    c.run(
        "apt-get update && apt-get install -y rsync",
        pty=True,
        env={"DEBIAN_FRONTEND": "noninteractive"},
    )

    print(f"Copying project files to {user}@{host}:{remote_dir} ...")
    project_root = pathlib.Path(__file__).parent.parent.parent.parent.resolve()
    # fmt: off
    rsync_cmd = [
        "rsync",
        "-avzhP",
        "--exclude", ".venv",
        "--exclude", "__pycache__",
        "--exclude", "*.pyc",
        "--exclude", ".git",
        "--no-owner",
        "--no-group",
    ]
    # fmt: on
    if port != 22:
        rsync_cmd += ["-e", f"ssh -p {port} -o StrictHostKeyChecking=no"]
    else:
        rsync_cmd += ["-e", "ssh -o StrictHostKeyChecking=no"]
    rsync_cmd += [
        f"{project_root}/",
        f"{user}@{host}:{remote_dir}/",
    ]
    subprocess.run(rsync_cmd, check=True)

    print("Installing system and Python dependencies on remote host ...")
    c.run(
        "apt-get update && apt-get install -y build-essential ninja-build cmake gcc g++ git nginx python3.12 python3.12-venv tmux byobu",
        pty=True,
        env={"DEBIAN_FRONTEND": "noninteractive"},
    )
    c.run(f"python3.12 -m venv {remote_dir}/venv")
    c.run(f"{remote_dir}/venv/bin/pip install --upgrade pip")
    c.run(f"{remote_dir}/venv/bin/pip install --upgrade uv")
    c.run(
        f"cd {quote(remote_dir)} && {quote(remote_dir)}/venv/bin/uv sync --frozen",
        pty=True,
        env={"UV_PROJECT_ENVIRONMENT": f"{remote_dir}/venv"},
    )

    print("Building llama.cpp, ik_llama.cpp, and llama-swap on remote host ...")
    c.run(
        f"{quote(remote_dir)}/venv/bin/python3 -m modal_llama.baremetal.build --llama-cpp {quote(remote_dir)}/ext/llama-cpp --ik-llama-cpp {quote(remote_dir)}/ext/ik-llama-cpp --llama-swap {quote(remote_dir)}/ext/llama-swap",
        pty=True,
    )

    print("Starting llama-swap and nginx on remote host in a tmux session ...")
    # Start the server in a dedicated tmux session named 'modal-llama'
    cmd = f"tmux kill-session -t modal-llama; tmux new -d -s modal-llama 'cd {quote(remote_dir)} && {quote(remote_dir)}/venv/bin/python3 -m modal_llama.baremetal.serve --nginx-port {remote_nginx_port} --models-cache-path {quote(remote_models_dir)} --llama-cpp-backend-path {quote(remote_dir)}/ext/llama-cpp/build/bin/llama-server --ik-llama-cpp-backend-path {quote(remote_dir)}/ext/ik-llama-cpp/build/bin/llama-server --llama-swap-bin-dir {quote(remote_dir)}/ext/llama-swap/build'"
    print(cmd)
    c.run(
        cmd,
        pty=True,
    )

    print("Fetching public IP and printing access URL ...")
    result = c.run("curl -s ifconfig.me", hide=True)
    public_ip = result.stdout.strip()
    print("\nDeployment complete!")
    print(f"  Access the app at: http://{public_ip}:{remote_nginx_port}/\n")
    print(
        f"  Or via SSH tunnel: ssh -L 1234:{public_ip}:{remote_nginx_port} {user}@{host} (and then connect to localhost:1234)\n"
    )


if __name__ == "__main__":
    app()
