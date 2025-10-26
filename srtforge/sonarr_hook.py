"""Sonarr custom script entry point."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

from .logging import get_console
from .pipeline import PipelineConfig, run_pipeline

TRIGGER_EVENTS = {"onimport", "onupgrade"}
EPISODE_PATH_KEYS = [
    "episodefile_path",
    "episode_file_path",
    "sonarr_episodefile_path",
    "sonarr_episode_file_path",
]


def _read_env(keys: Iterable[str]) -> dict[str, str]:
    env = {}
    for key in keys:
        value = os.environ.get(key) or os.environ.get(key.upper()) or os.environ.get(key.lower())
        if value:
            env[key.lower()] = value
    return env


def _resolve_episode_path() -> Optional[Path]:
    env = _read_env(EPISODE_PATH_KEYS)
    for key in EPISODE_PATH_KEYS:
        normalized = key.lower()
        if normalized in env:
            return Path(env[normalized]).expanduser()
    return None


def _resolve_event_type() -> str:
    event = os.environ.get("sonarr_eventtype") or os.environ.get("SONARR_EVENTTYPE") or ""
    return event.lower()


def main() -> None:
    console = get_console()
    event = _resolve_event_type()
    if event not in TRIGGER_EVENTS:
        console.log(f"[yellow]Ignoring Sonarr event[/yellow] {event or 'unknown'}")
        return

    episode_path = _resolve_episode_path()
    if not episode_path:
        console.log("[bold red]EpisodeFile.Path missing in environment[/bold red]")
        return

    console.log(f"Processing Sonarr episode file: {episode_path}")
    config = PipelineConfig(media_path=episode_path)
    run_pipeline(config)


if __name__ == "__main__":  # pragma: no cover
    main()
