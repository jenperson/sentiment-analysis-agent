from dataclasses import dataclass


@dataclass
class PostRecord:
    id: str
    title: str
    body: str
    score: int
    created_utc: float
    permalink: str


@dataclass
class CommentRecord:
    id: str
    body: str
    score: int
    created_utc: float
    permalink: str