from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from ..data_loader import DatasetConfig, encode_neutrosophic, load_dataset
from ..encoders import EncoderConfig
from ..simulate import generate_synthetic_spectrum
from ..study_config import ModelSettings, StudyConfig
from .utils import get_project_root, load_real_dataset


@dataclass
class FigureContext:
    """Shared context for manuscript/analysis figures."""

    X: np.ndarray
    y: np.ndarray
    x_tif: np.ndarray
    y_tif: np.ndarray
    feature_names: List[str]
    wavelengths: np.ndarray
    wavelength_label: str
    dataset_name: str
    encoder_config: EncoderConfig
    model_settings: ModelSettings
    model_name: str
    random_state: int = 42
    metadata: Optional[Dict[str, Any]] = None
    snv: bool = False
    task: str = "regression"
    mode: str = "paper"

    @staticmethod
    def _infer_wavelengths(feature_names: Sequence[str]) -> Tuple[np.ndarray, str]:
        numeric_values: List[float] = []
        numeric_count = 0
        for name in feature_names:
            try:
                numeric_values.append(float(name))
                numeric_count += 1
            except Exception:
                numeric_values.append(float("nan"))
        if feature_names and numeric_count / len(feature_names) >= 0.8:
            return np.array(numeric_values, dtype=float), "Wavelength (nm)"
        return np.arange(len(feature_names)), "Feature Index"

    @classmethod
    def from_session(cls, state: "SessionState", random_state: int = 42) -> "FigureContext":
        if state.dataframe is None:
            raise ValueError("Session state is missing a loaded dataframe.")
        if not state.feature_columns or state.target_column is None:
            raise ValueError("Session state is missing feature/target selections.")

        df = state.dataframe
        X = df[state.feature_columns].to_numpy(dtype=float)
        y = df[state.target_column].to_numpy()

        encoder_config = EncoderConfig.from_value(state.encoder_name or "probabilistic")
        x_tif = state.x_tif
        y_tif = state.y_tif
        if x_tif is None or y_tif is None:
            x_tif, y_tif = encode_neutrosophic(
                X,
                y,
                task=state.task,
                snv=state.snv,
                encoding=encoder_config,
            )

        feature_names = list(state.feature_columns)
        wavelengths, wavelength_label = cls._infer_wavelengths(feature_names)
        dataset_name = state.data_path.stem if state.data_path else "interactive"
        model_name = state.model_name or "NPLS"
        model_settings = ModelSettings(
            method=model_name,
            max_components=state.n_components,
            channel_weights=state.channel_weights,
            lambda_indeterminacy=state.lambda_indeterminacy,
            lambda_falsity=state.lambda_falsity,
            alpha=state.alpha,
        )

        return cls(
            X=X,
            y=y,
            x_tif=x_tif,
            y_tif=y_tif,
            feature_names=feature_names,
            wavelengths=wavelengths,
            wavelength_label=wavelength_label,
            dataset_name=dataset_name,
            encoder_config=encoder_config,
            model_settings=model_settings,
            model_name=model_name,
            random_state=random_state,
            metadata=state.encoding_metadata,
            snv=state.snv,
            task=state.task,
            mode="analysis",
        )

    @classmethod
    def from_config(cls, config: StudyConfig, random_state: int = 42) -> "FigureContext":
        dataset_cfg = DatasetConfig(
            path=config.dataset.path,
            target=config.dataset.target,
            task=config.dataset.task,
            features=config.dataset.features,
            exclude_columns=config.dataset.exclude_columns,
            snv=config.dataset.snv,
            encoding=config.dataset.encoding,
            spectral_noise_db=config.dataset.spectral_noise_db,
            format=config.dataset.format,
            name=config.dataset.name,
        )
        data = load_dataset(dataset_cfg)
        df = data["dataframe"]
        metadata = data["metadata"]
        feature_names = metadata.get("feature_names", [])
        X = df[feature_names].to_numpy(dtype=float)
        y = df[metadata["target_name"]].to_numpy()

        wavelengths, wavelength_label = cls._infer_wavelengths(feature_names)
        encoder_config = EncoderConfig.from_value(config.dataset.encoding)
        model_name = config.model.method if config.model.method != "all" else "NPLS"

        return cls(
            X=X,
            y=y,
            x_tif=data["x_tif"],
            y_tif=data["y_tif"],
            feature_names=list(feature_names),
            wavelengths=wavelengths,
            wavelength_label=wavelength_label,
            dataset_name=metadata.get("name", Path(config.dataset.path).stem),
            encoder_config=encoder_config,
            model_settings=config.model,
            model_name=model_name,
            random_state=random_state,
            metadata=metadata,
            snv=config.dataset.snv,
            task=config.dataset.task,
            mode="paper",
        )

    @classmethod
    def paper_default(cls, dataset_name: str = "MA_A2", random_state: int = 42) -> "FigureContext":
        data_path = get_project_root() / "data" / f"{dataset_name}.csv"
        if data_path.exists():
            x_tif, y_tif, metadata, df, _ = load_real_dataset(data_path)
            feature_names = metadata.get("feature_cols", metadata.get("feature_names", []))
            target_name = metadata.get("target", metadata.get("target_name", df.columns[0]))
            X = df[feature_names].to_numpy(dtype=float)
            y = df[target_name].to_numpy()

            wavelengths, wavelength_label = cls._infer_wavelengths(feature_names)
            encoder_name = metadata.get("encoder", {}).get("name", "probabilistic")
            encoder_config = EncoderConfig.from_value(encoder_name)

            return cls(
                X=X,
                y=y,
                x_tif=x_tif,
                y_tif=y_tif,
                feature_names=list(feature_names),
                wavelengths=wavelengths,
                wavelength_label=wavelength_label,
                dataset_name=metadata.get("name", dataset_name),
                encoder_config=encoder_config,
                model_settings=ModelSettings(),
                model_name="NPLS",
                random_state=random_state,
                metadata=metadata,
                snv=bool(metadata.get("snv", metadata.get("snv_applied", False))),
                task=metadata.get("task", "regression"),
                mode="paper",
            )

        X, y, _ = generate_synthetic_spectrum(n_samples=80, n_features=120, seed=random_state)
        x_tif, y_tif = encode_neutrosophic(X, y.reshape(-1, 1))
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        wavelengths, wavelength_label = cls._infer_wavelengths(feature_names)
        metadata = {"name": "Synthetic", "task": "regression"}

        return cls(
            X=X,
            y=y,
            x_tif=x_tif,
            y_tif=y_tif,
            feature_names=feature_names,
            wavelengths=wavelengths,
            wavelength_label=wavelength_label,
            dataset_name="Synthetic",
            encoder_config=EncoderConfig(),
            model_settings=ModelSettings(),
            model_name="NPLS",
            random_state=random_state,
            metadata=metadata,
            snv=False,
            task="regression",
            mode="paper",
        )


if TYPE_CHECKING:
    from ..interactive import SessionState
