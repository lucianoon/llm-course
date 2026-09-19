"""Diagnóstico do ambiente local do curso.

Exemplos:

    uv run python tools/doctor.py --profile cpu
    uv run python tools/doctor.py --profile gpu --require-cuda
    uv run python tools/doctor.py --profile serving --json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


PROFILES = {
    "cpu": ("torch", "transformers", "accelerate"),
    "gpu": ("torch", "transformers", "accelerate", "bitsandbytes"),
    "serving": ("torch", "transformers", "accelerate", "vllm"),
    "mlx": ("mlx", "mlx-lm"),
}


def package_check(package: str) -> Check:
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return Check(package, False, "não instalado")
    return Check(package, True, f"{version} instalado")


def import_check(package: str) -> Check:
    check = package_check(package)
    if not check.ok:
        return check
    module_name = {"mlx-lm": "mlx_lm", "bitsandbytes": "bitsandbytes"}.get(package, package)
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import importlib; importlib.import_module(__import__('sys').argv[1])", module_name],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Check(package, False, "importação excedeu 15s")
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "erro desconhecido"
        return Check(package, False, f"instalado, mas não importa: {detail}")
    return check


def torch_checks() -> list[Check]:
    probe = (
        "import json, torch; "
        "cuda=bool(torch.cuda.is_available()); "
        "print(json.dumps({'cuda': cuda, 'device': torch.cuda.get_device_name() if cuda else 'nenhuma GPU CUDA detectada', 'bf16': bool(cuda and torch.cuda.is_bf16_supported())}))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, timeout=15, check=False
        )
    except subprocess.TimeoutExpired:
        return [Check("torch runtime", False, "importação excedeu 15s")]
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "erro desconhecido"
        return [Check("torch runtime", False, detail)]
    try:
        values = json.loads(result.stdout)
    except json.JSONDecodeError:
        return [Check("torch runtime", False, "saída de diagnóstico inválida")]

    cuda = values["cuda"]
    device = values["device"]
    bf16 = values["bf16"]
    return [
        Check("CUDA", cuda, device, required=False),
        Check("bf16", bf16, "suportado" if bf16 else "não disponível", required=False),
    ]


def nvidia_check() -> Check:
    if shutil.which("nvidia-smi") is None:
        return Check("nvidia-smi", False, "comando não encontrado")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    detail = result.stdout.strip() or result.stderr.strip()
    return Check("nvidia-smi", result.returncode == 0, detail)


def run(profile: str, require_cuda: bool = False) -> list[Check]:
    checks = [
        Check("Python", (3, 11) <= sys.version_info[:2] < (3, 13), platform.python_version()),
        Check("plataforma", True, f"{platform.system()} {platform.machine()}", required=False),
    ]
    for package in PROFILES[profile]:
        checks.append(import_check(package))
    if profile in {"gpu", "serving"} or require_cuda:
        checks.extend(torch_checks())
        checks.append(nvidia_check())
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="cpu")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    checks = run(args.profile, args.require_cuda)

    if args.as_json:
        print(json.dumps([asdict(check) for check in checks], ensure_ascii=False, indent=2))
    else:
        for check in checks:
            icon = "OK" if check.ok else "FALHA"
            print(f"[{icon}] {check.name}: {check.detail}")

    return 0 if all(check.ok or not check.required for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
