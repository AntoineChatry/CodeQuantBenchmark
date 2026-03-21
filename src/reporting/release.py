"""Story 4.3 : Publication et Versioning — Git tag + GitHub Release."""

import subprocess
from pathlib import Path
from typing import Any

from src.utils.config import get_nested, load_config
from src.utils.logging import setup_logger

logger = setup_logger("release")


def run_git(args: list[str]) -> tuple[bool, str]:
    cmd = ["git"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout.strip() or result.stderr.strip()
    return result.returncode == 0, output


def create_tag(version: str) -> bool:
    success, output = run_git(["tag", "-a", version, "-m", f"Release {version}"])
    if not success:
        logger.error(f"Failed to create tag {version}: {output}")
        return False
    logger.info(f"Created tag: {version}")
    return True


def push_tag(version: str) -> bool:
    success, output = run_git(["push", "origin", version])
    if not success:
        logger.error(f"Failed to push tag {version}: {output}")
        return False
    logger.info(f"Pushed tag: {version}")
    return True


def create_github_release(
    version: str, title: str, notes: str, assets: list[str]
) -> bool:
    cmd = [
        "gh", "release", "create", version,
        "--title", title,
        "--notes", notes,
    ]
    for asset in assets:
        if Path(asset).exists():
            cmd.append(asset)
        else:
            logger.warning(f"Asset not found, skipping: {asset}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"GitHub release failed: {result.stderr.strip()}")
        return False

    logger.info(f"GitHub release created: {result.stdout.strip()}")
    return True


def collect_assets(config: dict[str, Any]) -> list[str]:
    results_dir = Path(get_nested(config, "reporting", "output_dir", default="./reports"))
    assets: list[str] = []

    for pattern in ["*.csv", "*.json"]:
        assets.extend(str(p) for p in results_dir.glob(pattern))

    assets_dir = results_dir / "assets"
    if assets_dir.exists():
        assets.extend(str(p) for p in assets_dir.glob("*.png"))

    data_dir = Path(get_nested(config, "data", "output_dir", default="./data"))
    test_file = data_dir / "test.jsonl"
    if test_file.exists():
        assets.append(str(test_file))

    return assets


def run(config_path: str = "config.yaml", version: str = "v1.0.0-benchmark") -> bool:
    config = load_config(config_path)
    project_name = get_nested(config, "project", "name", default="CodeQuantBenchmark")

    if not create_tag(version):
        return False

    assets = collect_assets(config)
    logger.info(f"Release assets: {len(assets)} files")

    title = f"{project_name} {version}"
    notes = (
        f"Benchmark results for {project_name}.\n\n"
        f"Quantization formats tested: {', '.join(get_nested(config, 'quantization', 'formats', default=[]))}\n"
        f"Base model: {get_nested(config, 'model', 'base_name', default='N/A')}"
    )

    return create_github_release(version, title, notes, assets)
