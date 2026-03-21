"""Clone de dépôts GitHub pour la collecte de données."""

import subprocess
from pathlib import Path
from typing import Any

import requests

from src.utils.config import get_nested, load_config
from src.utils.logging import setup_logger

logger = setup_logger("data_clone")

GITHUB_API = "https://api.github.com"


def search_repos(
    query: str, language: str = "Python", max_repos: int = 10
) -> list[dict[str, str]]:
    params = {
        "q": f"{query} language:{language}",
        "sort": "stars",
        "order": "desc",
        "per_page": max_repos,
    }
    resp = requests.get(f"{GITHUB_API}/search/repositories", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    repos = []
    for item in data.get("items", []):
        repos.append({
            "name": item["full_name"],
            "clone_url": item["clone_url"],
            "stars": str(item["stargazers_count"]),
            "description": item.get("description", ""),
        })
    logger.info(f"Found {len(repos)} repos for query '{query}'")
    return repos


def clone_repo(clone_url: str, target_dir: Path, shallow: bool = True) -> bool:
    repo_name = clone_url.rstrip("/").split("/")[-1].replace(".git", "")
    dest = target_dir / repo_name

    if dest.exists():
        logger.info(f"Already cloned: {dest}")
        return True

    cmd = ["git", "clone"]
    if shallow:
        cmd.extend(["--depth", "1"])
    cmd.extend([clone_url, str(dest)])

    logger.info(f"Cloning {clone_url} -> {dest}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Clone failed: {result.stderr.strip()}")
        return False

    return True


def run(config_path: str = "config.yaml") -> int:
    config = load_config(config_path)
    source_dir = Path(get_nested(config, "data", "source_dir", default="./repos"))
    source_dir.mkdir(parents=True, exist_ok=True)

    github_cfg = get_nested(config, "data", "github", default={})
    repos_list: list[dict[str, str]] = github_cfg.get("repos", [])
    search_query: str = github_cfg.get("search_query", "")
    search_language: str = github_cfg.get("search_language", "Python")
    max_repos: int = github_cfg.get("max_repos", 10)
    shallow: bool = github_cfg.get("shallow_clone", True)

    if search_query and not repos_list:
        repos_list = search_repos(search_query, search_language, max_repos)

    cloned = 0
    for repo in repos_list:
        url = repo if isinstance(repo, str) else repo.get("clone_url", "")
        if url and clone_repo(url, source_dir, shallow=shallow):
            cloned += 1

    logger.info(f"Cloned {cloned}/{len(repos_list)} repos into {source_dir}")
    return cloned
