"""Face identity helpers for ROI-filtered fighter candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class FaceMatcher:
    """Compare candidate faces against reference face embeddings."""

    def __init__(
        self,
        reference_paths: list[Path],
        *,
        backend: str = "insightface",
        detection_size: tuple[int, int] = (640, 640),
        detector_backend: str = "opencv",
    ) -> None:
        self.backend = backend
        self.detector_backend = detector_backend
        if backend == "deepface-arcface":
            self._init_deepface()
        elif backend == "insightface":
            self._init_insightface(detection_size)
        else:
            raise ValueError(f"Unsupported face match backend: {backend}")
        self._references = self._load_references(reference_paths)

    def _init_insightface(self, detection_size: tuple[int, int]) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "Face matching requires insightface. Install requirements-boxmot.txt on the pod."
            ) from exc

        self._app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self._app.prepare(ctx_id=0, det_size=detection_size)
        self._deepface = None

    def _init_deepface(self) -> None:
        try:
            from deepface import DeepFace
        except ImportError as exc:
            raise RuntimeError(
                "DeepFace ArcFace matching requires deepface. Install requirements-boxmot.txt on the pod."
            ) from exc

        self._app = None
        self._deepface = DeepFace

    @property
    def reference_count(self) -> int:
        return len(self._references)

    def match_candidate(self, frame: Any, box: tuple[int, int, int, int]) -> tuple[float, bool]:
        crop = _crop(frame, box)
        if crop is None:
            return 0.0, False
        if self.backend == "deepface-arcface":
            return self._match_deepface(crop)
        faces = self._app.get(crop)
        if not faces:
            return 0.0, False
        if not self._references:
            return 0.0, True
        scores = [
            _cosine_similarity(_normalized_embedding(face), reference)
            for face in faces
            for reference in self._references
        ]
        return max(scores, default=0.0), True

    def _match_deepface(self, crop: Any) -> tuple[float, bool]:
        embeddings = self._deepface_embeddings(crop)
        if not embeddings:
            return 0.0, False
        if not self._references:
            return 0.0, True
        scores = [
            _cosine_similarity(embedding, reference)
            for embedding in embeddings
            for reference in self._references
        ]
        return max(scores, default=0.0), True

    def _load_references(self, paths: list[Path]) -> list[np.ndarray]:
        references = []
        for path in reference_image_paths(paths):
            image = cv2.imread(str(path))
            if image is None:
                continue
            if self.backend == "deepface-arcface":
                embeddings = self._deepface_embeddings(image)
                if embeddings:
                    references.append(embeddings[0])
            else:
                faces = self._app.get(image)
                if not faces:
                    continue
                face = max(faces, key=lambda item: _face_area(item))
                references.append(_normalized_embedding(face))
        return references

    def _deepface_embeddings(self, image: Any) -> list[np.ndarray]:
        try:
            representations = self._deepface.represent(
                img_path=image,
                model_name="ArcFace",
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True,
            )
        except Exception:
            return []
        if isinstance(representations, dict):
            representations = [representations]
        embeddings = []
        for representation in representations or []:
            confidence = float(representation.get("face_confidence", 0.0) or 0.0)
            if confidence <= 0.0:
                continue
            embedding = np.asarray(representation.get("embedding", []), dtype=np.float32)
            if embedding.size:
                embeddings.append(_normalize_vector(embedding))
        return embeddings


def reference_image_paths(paths: list[Path]) -> list[Path]:
    image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    expanded = []
    for path in paths:
        if path.is_dir():
            expanded.extend(
                sorted(child for child in path.rglob("*") if child.suffix.lower() in image_suffixes)
            )
        elif path.suffix.lower() in image_suffixes:
            expanded.append(path)
    return expanded


def _crop(frame: Any, box: tuple[int, int, int, int]) -> Any | None:
    height, width = frame.shape[:2]
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(width, int(box[2])), min(height, int(box[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _normalized_embedding(face: Any) -> np.ndarray:
    embedding = np.asarray(face.normed_embedding if hasattr(face, "normed_embedding") else face.embedding)
    return _normalize_vector(embedding)


def _normalize_vector(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    if not norm:
        return embedding
    return embedding / norm


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    if first.size == 0 or second.size == 0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(first, second))))


def _face_area(face: Any) -> float:
    x1, y1, x2, y2 = face.bbox
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
