"""A dataclass with a shared mutable default (dataclass bug)."""

from dataclasses import dataclass


@dataclass
class Task:
    tags: list = []
