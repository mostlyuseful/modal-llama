import tempfile
import subprocess


def start_nginx_reverse_proxy(
    api_token: str | None, llama_swap_port: int, listen_port: int
):
    """
    Starts an Nginx reverse proxy that:
    - Listens on 0.0.0.0:{listen_port}
    - Proxies to llama-swap on localhost:{llama_swap_port}
    - Requires Bearer token authentication using {api_token}
    """
    # Create temporary nginx config file
    # Conditionally include bearer token check
    auth_block = (
        f"""
                if ($http_authorization != "Bearer {api_token}") {{
                    return 403;
                }}"""
        if api_token
        else ""
    )

    config_content = f"""events {{
    worker_connections 32;
}}

http {{
    # Increase timeouts for long-lived connections
    proxy_read_timeout 24h;
    proxy_send_timeout 24h;
    
    server {{
        listen {listen_port};
        server_name 0.0.0.0;
        
        location / {{
{auth_block}
            proxy_pass http://localhost:{llama_swap_port};
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }}
        
        # Special handling for SSE streams
        location ~* (/logs/streamSSE/|/api/modelsSSE) {{
            {auth_block}
            proxy_pass http://localhost:{llama_swap_port};
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            
            # Disable buffering for SSE
            proxy_buffering off;
            proxy_cache off;
            
            # Increase timeouts specifically for SSE
            proxy_read_timeout 24h;
            proxy_send_timeout 24h;
        }}
    }}
}}"""

    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".conf", delete=False) as f:
        f.write(config_content)
        config_path = f.name

    print(f"\n\n\n=== Generated Nginx config ===\n{config_content}\n\n\n")

    # Start nginx with the config
    nginx_process = subprocess.Popen(
        ["nginx", "-c", config_path, "-g", "daemon off;"],
        stdout=None,
        stderr=None,
    )

    # Return the process object and config path
    return nginx_process, config_path
