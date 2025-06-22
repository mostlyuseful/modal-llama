import re
from huggingface_hub import snapshot_download
from pathlib import Path
from modal_llama.llama_cpp import LlamaCppConfig
from more_itertools import one

from modal_llama.llama_swap import LlamaSwapConfig


def abbreviate_entrypoint_name(entrypoint_path: Path) -> str:
    """
    Abbreviate the entrypoint name by removing the file extension and replacing underscores with hyphens.
    """
    return entrypoint_path.stem.replace("_", "-")


def find_gguf_entrypoint(
    repo_dir: Path, include: list[str] | str | None = None
) -> Path:
    if include is None:
        include = ["*.gguf"]

    candidates = set()
    for pattern in include:
        candidates.update(repo_dir.rglob(pattern))

    # Search for the first entrypoint that matches one of these patterns:
    # 1. Single-file GGUF models: "qwen2.5-coder-32b-instruct-q6_k.gguf"
    # 2. Multi-file GGUF models: "qwen2.5-coder-32b-instruct-q8_0-00001-of-00005.gguf"
    #    Take lowest numbered file.

    if not candidates:
        raise ValueError(f"No entrypoint found in {repo_dir} matching {include}")

    # Check multi-file GGUF models
    multi_file_candidates = [
        p
        for p in candidates
        if p.name.endswith(".gguf") and re.search(r"\d+-of-\d+\.gguf$", p.name)
    ]

    if multi_file_candidates:
        # Sort by the number in the filename and return the first one
        multi_file_candidates.sort(
            key=lambda p: int(re.search(r"(\d+)-of-\d+", p.name).group(1))  # type: ignore
        )
        return multi_file_candidates[0]

    # If no multi-file candidates, return the first single-file GGUF model
    return one(candidates)


def gguf(
    huggingface_repo_name: str,
    backend: str | Path,
    include: list[str] | str | None = None,
    hub_dir: str | Path | None = None,
) -> LlamaCppConfig:
    repo_dir = Path(
        snapshot_download(
            huggingface_repo_name, allow_patterns=include, cache_dir=hub_dir
        )
    )
    print("Repo directory:", repo_dir)
    entrypoint_path = find_gguf_entrypoint(repo_dir, include)
    print("Entrypoint found:", entrypoint_path)
    abbrv_entrypoint_name = abbreviate_entrypoint_name(entrypoint_path)
    print("Abbreviated entrypoint name:", abbrv_entrypoint_name)
    return LlamaCppConfig(
        name=abbrv_entrypoint_name,
        binary=backend,
        model=entrypoint_path,
    )


def dots_llm1(backend: str | Path) -> LlamaCppConfig:
    """
    Dots LLM1 model configuration.
    """
    return gguf(
        "unsloth/dots.llm1.inst-GGUF",
        backend,
        include=["UD-Q6_K_XL/*.gguf"],
    ).with_params(
        ctx_size=32768,  # 32k context size
        jinja=True,
        port="${PORT}",
    )


def kimi_dev_72b(backend: str | Path, quant: str = "Q6_K") -> LlamaCppConfig:
    """
    Kimi Dev 72B model configuration.
    """
    return gguf(
        "bullerwins/Kimi-Dev-72B-GGUF",
        backend,
        include=[f"Kimi-Dev-72B-{quant}-*.gguf"],
    ).with_params(
        ctx_size=131072,  # 128k context size
        jinja=True,
        port="${PORT}",
    )


def devstral_small_2505(backend: str | Path) -> LlamaCppConfig:
    """
    Devstral Small 2505 model configuration.
    """
    return gguf(
        "Mungert/Devstral-Small-2505-GGUF",
        backend,
        include=["*q6_k_l.gguf"],
    ).with_params(
        ctx_size=131072,  # 128k context size
        jinja=True,
        port="${PORT}",
    )


def mistral_small_3v2_2506(
    backend: str | Path, quant: str = "UD-Q6_K_XL"
) -> LlamaCppConfig:
    """
    Mistral Small 3.2 2506 model configuration.
    """
    return gguf(
        "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
        backend,
        include=[f"*-{quant}.gguf"],
    ).with_params(
        ctx_size=131072,  # 128k context size
        jinja=True,
        temp=0.15,
        top_p=1.0,
        min_p=0.0,
        repeat_penalty=1.0,
        port="${PORT}",
    )


def prep_common_models(
    cfg: LlamaSwapConfig,
    llama_cpp_backend: str | Path,
    ik_llama_cpp_backend: str | Path,
) -> LlamaSwapConfig:
    """
    Populate the configuration with common models and download them.
    """

    cfg.add_model(
        gguf(
            "lmstudio-community/DeepSeek-R1-0528-Qwen3-8B-GGUF",
            llama_cpp_backend,
            include="*Q4_K_M*",
        )
        .with_params(
            jinja=True,
            n_gpu_layers=100,  # OFFLOAD EVERYTHING
            port="${PORT}",  # Replaced by llama-swap
            ctx_size=131072,
        )
        .build()
    )

    # cfg.add_model(dots_llm1(llama_cpp_backend).with_params(n_gpu_layers=100, cache_type_k="q8_0", cache_type_v="q8_0", flash_attn=True, ctx_size=4096).build())
    cfg.add_model(dots_llm1(llama_cpp_backend).with_params(n_gpu_layers=100).build())

    cfg.add_model(kimi_dev_72b(llama_cpp_backend).with_params(n_gpu_layers=100).build())

    cfg.add_model(
        mistral_small_3v2_2506(llama_cpp_backend).with_params(n_gpu_layers=100).build()
    )

    # cfg.add_model(ik_r1_0528(ik_llama_cpp_backend))
    # cfg.add_model(minimax_m1_40k(llama_cpp_backend))

    return cfg
