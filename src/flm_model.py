#!/usr/bin/env python3
"""flm model — unified CLI for model build, install, test, and list.

Standalone script (stdlib only) that dispatches to the flm-add, flm-test,
and q4nx-build Python utilities.  Resolves tools from the installed venv
($PREFIX/.venv) or falls back to the repo's utilities/ directory.

Usage:
    flm model build  -i <repo-or-file> -o <output> [-f family] [--deploy TAG]
    flm model install <repo-or-dir> [--tag TAG]
    flm model test   [--llm|--embedding|--audio|--vision|--tools] [--model TAG]
    flm model list
"""

import importlib
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Prefix / environment resolution
# ---------------------------------------------------------------------------

def _bin_dir() -> Path:
    """Directory containing this script (or the flm binary)."""
    return Path(__file__).resolve().parent


def _prefix_from_binary() -> Path:
    """Derive the install prefix from this script's location.

    Layout:  $PREFIX/bin/flm-model   (this script)
             $PREFIX/bin/flm         (the C++ binary)
             $PREFIX/share/flm/      (model_list.json, xclbins)
             $PREFIX/.venv/          (Python packages)
    """
    return _bin_dir().parent


def _prefix_from_flm() -> Path | None:
    """Locate the flm binary on PATH and derive the prefix."""
    import shutil
    flm = shutil.which("flm")
    if flm:
        return Path(flm).resolve().parent.parent
    return None


def resolve_prefix() -> Path:
    """Best-effort prefix resolution."""
    # 1. Already inside the prefix (flm-model installed next to flm)
    p = _prefix_from_binary()
    if (p / "share" / "flm" / "model_list.json").is_file():
        return p
    # 2. flm on PATH
    p2 = _prefix_from_flm()
    if p2 and (p2 / "share" / "flm" / "model_list.json").is_file():
        return p2
    # 3. Repo root (dev mode)
    repo = _bin_dir().parent.parent
    if (repo / "src" / "model_list.json").is_file():
        return repo
    return p  # best guess


def setup_environment(prefix: Path) -> None:
    """Set FLM_CONFIG_PATH, FLM_XCLBIN_PATH, FLM_MODEL_PATH if not already set.

    FLM_CONFIG_PATH is only set when a *user-level* registry exists at
    ~/.config/flm/model_list.json.  We intentionally do NOT point it at the
    system file (e.g. /opt/fastflowlm/share/flm/model_list.json) because
    flm-add uses FLM_CONFIG_PATH as the *write* destination — it discovers
    the system list on its own via find_system_model_list().
    """
    user_config = Path.home() / ".config" / "flm" / "model_list.json"
    if "FLM_CONFIG_PATH" not in os.environ and user_config.is_file():
        os.environ["FLM_CONFIG_PATH"] = str(user_config)

    share = prefix / "share" / "flm"
    if "FLM_XCLBIN_PATH" not in os.environ:
        # Prefer user-level xclbin dir if it exists (flm-add writes symlinks here)
        user_xclbin = Path.home() / ".config" / "flm"
        if (user_xclbin / "xclbins").is_dir():
            os.environ["FLM_XCLBIN_PATH"] = str(user_xclbin)
        elif share.is_dir():
            os.environ["FLM_XCLBIN_PATH"] = str(share)
        else:
            # Dev mode: repo/src
            os.environ["FLM_XCLBIN_PATH"] = str(prefix / "src")

    if "FLM_MODEL_PATH" not in os.environ:
        os.environ["FLM_MODEL_PATH"] = str(Path.home() / ".config" / "flm")


# ---------------------------------------------------------------------------
# Venv activation
# ---------------------------------------------------------------------------

def activate_venv(prefix: Path) -> None:
    """Activate $PREFIX/.venv if it exists."""
    venv_activate = prefix / ".venv" / "bin" / "activate"
    if venv_activate.is_file():
        # Source the activate script's env changes into this process
        import subprocess
        result = subprocess.run(
            ["bash", "-c", f"source {venv_activate} && env -0"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            for entry in result.stdout.split("\0"):
                if "=" in entry:
                    key, _, value = entry.partition("=")
                    if key:
                        os.environ[key] = value


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------

def _add_utilities_to_path(prefix: Path) -> None:
    """Add the repo's utilities/ subdirectories to sys.path for dev mode."""
    repo_root = prefix
    # Walk up to find a repo root (has src/ and utilities/)
    for candidate in [prefix, prefix.parent, prefix.parent.parent]:
        if (candidate / "utilities").is_dir() and (candidate / "src").is_dir():
            repo_root = candidate
            break
    utils = repo_root / "utilities"
    if utils.is_dir():
        for sub in ("flm-add", "flm-test", "q4nx-build"):
            p = utils / sub
            if p.is_dir() and str(p) not in sys.path:
                sys.path.insert(0, str(p))


def import_flm_add():
    """Import and return the flm_add module."""
    try:
        return importlib.import_module("flm_add")
    except ImportError:
        pass
    raise SystemExit(
        "flm-add is not installed. Install it with:\n"
        "  uv pip install --python <prefix>/.venv/bin/python utilities/flm-add\n"
        "  or: pip install utilities/flm-add"
    )


def import_flm_test():
    """Import and return the flm_test module."""
    try:
        return importlib.import_module("flm_test")
    except ImportError:
        pass
    raise SystemExit(
        "flm-test is not installed. Install it with:\n"
        "  uv pip install --python <prefix>/.venv/bin/python utilities/flm-test\n"
        "  or: pip install utilities/flm-test"
    )


def import_q4nx_cli():
    """Import and return the q4nx.cli module."""
    try:
        return importlib.import_module("q4nx.cli")
    except ImportError:
        pass
    raise SystemExit(
        "q4nx-build is not installed. Install it with:\n"
        "  uv pip install --python <prefix>/.venv/bin/python utilities/q4nx-build\n"
        "  or: pip install utilities/q4nx-build"
    )


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_build(prefix: Path, args: list[str]) -> None:
    """Run q4nx-build with --deploy auto-added."""
    # If no --deploy flag, derive one from -o or -i
    has_deploy = any(a.startswith("--deploy") for a in args)
    if not has_deploy:
        # Find -o value to derive a deploy tag
        output = None
        for i, a in enumerate(args):
            if a in ("-o", "--output") and i + 1 < len(args):
                output = args[i + 1]
                break
        if output:
            # Derive tag from output dir name
            out_name = Path(output).name
            # Strip common suffixes
            tag = out_name.lower().replace("-npu2", "")
            if ":" not in tag:
                # Try to extract size
                import re
                size_match = re.search(r"(\d+(?:\.\d+)?[bm])\b", tag)
                if size_match:
                    size = size_match.group(1)
                    name_part = tag[:size_match.start()].rstrip("-_")
                    tag = f"{name_part}:{size}"
            args = args + ["--deploy", tag]

    # Ensure output is absolute
    new_args = []
    skip_next = False
    for i, a in enumerate(args):
        if skip_next:
            skip_next = False
            if not Path(a).is_absolute():
                new_args.append(str(Path(a).resolve()))
            else:
                new_args.append(a)
            continue
        if a in ("-o", "--output"):
            skip_next = True
        new_args.append(a)

    cli = import_q4nx_cli()
    # q4nx.cli.main() reads sys.argv by default; set it
    sys.argv = ["q4nx-build"] + new_args
    cli.main(new_args)


def cmd_install(prefix: Path, args: list[str]) -> None:
    """Run flm-add."""
    mod = import_flm_add()
    sys.argv = ["flm-add"] + args
    mod.main()


def cmd_test(prefix: Path, args: list[str]) -> None:
    """Run flm-test."""
    mod = import_flm_test()
    # flm_test.main() calls argparse.parse_args() with no args (reads sys.argv)
    sys.argv = ["flm-test"] + args
    mod.main()


def cmd_list(prefix: Path, args: list[str]) -> None:
    """Print registered models from model_list.json."""
    # Find the system model list
    system_path = None
    for candidate in [
        prefix / "share" / "flm" / "model_list.json",
        prefix / "src" / "model_list.json",
        Path("/opt/fastflowlm/share/flm/model_list.json"),
    ]:
        if candidate.is_file():
            system_path = candidate
            break

    # Find the user model list
    user_path = Path.home() / ".config" / "flm" / "model_list.json"

    if not system_path and not user_path.is_file():
        raise SystemExit("Could not locate model_list.json. Set FLM_CONFIG_PATH.")

    system_models = {}
    if system_path and system_path.is_file():
        with open(system_path, encoding="utf-8") as f:
            system_models = json.load(f).get("models", {})

    user_models = {}
    if user_path.is_file():
        with open(user_path, encoding="utf-8") as f:
            user_models = json.load(f).get("models", {})

    if not system_models and not user_models:
        print("No models registered.")
        return

    # Merge: system models as base, user models override/supplement
    all_tags = set()
    for bucket, sizes in system_models.items():
        for sz in sizes:
            all_tags.add((bucket, sz))
    for bucket, sizes in user_models.items():
        for sz in sizes:
            all_tags.add((bucket, sz))

    if system_path:
        print(f"System: {system_path}")
    if user_path.is_file():
        print(f"User:   {user_path}")
    print()

    for bucket_name, size_token in sorted(all_tags):
        # User entry takes precedence for display
        info = None
        source = ""
        if bucket_name in user_models and size_token in user_models[bucket_name]:
            info = user_models[bucket_name][size_token]
            if (bucket_name in system_models and
                    size_token in system_models.get(bucket_name, {})):
                source = " (user override)"
            else:
                source = " *"
        elif bucket_name in system_models and size_token in system_models[bucket_name]:
            info = system_models[bucket_name][size_token]

        if not info:
            continue
        name = info.get("name", "?")
        family = (info.get("details") or {}).get("family", "?")
        tag = f"{bucket_name}:{size_token}" if size_token else bucket_name
        print(f"  {tag:<30s}  {name:<40s}  family={family}{source}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

HELP = """\
usage: flm model <command> [args...]

Commands:
  build   -i <repo-or-file> -o <output> [-f family] [--deploy TAG]
          Convert a GGUF or HF-safetensors model to Q4NX format.
          --deploy is auto-added from -o if not specified.

  install <repo-or-dir> [--tag TAG]
          Download/copy model files and register with FLM.
          Aliases: flm-add

  test    [--llm|--embedding|--audio|--vision|--tools] [--model TAG]
          Run model test suites against a running FLM server.
          Aliases: flm-test

  list    Show all registered models from model_list.json.

Environment (auto-configured, override if needed):
  FLM_CONFIG_PATH   Path to model_list.json
  FLM_XCLBIN_PATH   Path to xclbins parent directory
  FLM_MODEL_PATH    Models root directory (default: ~/.config/flm)
"""


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(HELP)
        sys.exit(0)

    command = sys.argv[1]
    rest = sys.argv[2:]

    # In dev mode (running from repo/src/), add utilities to path
    prefix = resolve_prefix()
    _add_utilities_to_path(prefix)
    setup_environment(prefix)
    activate_venv(prefix)

    # Re-check imports after venv activation
    if command == "build":
        cmd_build(prefix, rest)
    elif command == "install":
        cmd_install(prefix, rest)
    elif command in ("test", "tests"):
        cmd_test(prefix, rest)
    elif command == "list":
        cmd_list(prefix, rest)
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print(HELP, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
