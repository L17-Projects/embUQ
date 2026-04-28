from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EmbArchitecture:
    width: int
    depth: int
    name: str


@dataclass(frozen=True)
class EmbDatasetSpec:
    name: str
    modality: str
    diameter_um: str
    data_relpath: str
    dnn_artifact_relpath: str
    bnn_artifact_relpath: str
    dnn_train_script_relpath: str
    dnn_multi_arch_script_relpath: str
    bnn_train_script_relpath: str
    group_holdout_script_relpath: str

    def resolve(self, repo_root: str | Path) -> dict[str, str]:
        root = Path(repo_root).resolve()
        return {
            "name": self.name,
            "modality": self.modality,
            "diameter_um": self.diameter_um,
            "data": str((root / self.data_relpath).resolve()),
            "dnn_artifact": str((root / self.dnn_artifact_relpath).resolve()),
            "bnn_artifact": str((root / self.bnn_artifact_relpath).resolve()),
            "dnn_train_script": str((root / self.dnn_train_script_relpath).resolve()),
            "dnn_multi_arch_script": str((root / self.dnn_multi_arch_script_relpath).resolve()),
            "bnn_train_script": str((root / self.bnn_train_script_relpath).resolve()),
            "group_holdout_script": str((root / self.group_holdout_script_relpath).resolve()),
        }


DEFAULT_DNN_ARCHITECTURES: tuple[EmbArchitecture, ...] = (
    EmbArchitecture(32, 2, "w32_d2"),
    EmbArchitecture(32, 3, "w32_d3"),
    EmbArchitecture(32, 4, "w32_d4"),
    EmbArchitecture(64, 2, "w64_d2"),
    EmbArchitecture(64, 3, "w64_d3"),
    EmbArchitecture(64, 4, "w64_d4"),
    EmbArchitecture(64, 5, "w64_d5"),
    EmbArchitecture(128, 2, "w128_d2"),
    EmbArchitecture(128, 3, "w128_d3"),
    EmbArchitecture(128, 4, "w128_d4"),
    EmbArchitecture(256, 2, "w256_d2"),
    EmbArchitecture(256, 3, "w256_d3"),
)


DEFAULT_BNN_ARCHITECTURES: tuple[EmbArchitecture, ...] = DEFAULT_DNN_ARCHITECTURES


EMB_DATASET_SPECS: tuple[EmbDatasetSpec, ...] = (
    EmbDatasetSpec(
        name="compression_2.1um",
        modality="compression",
        diameter_um="2.1",
        data_relpath="compression/surrogate/diameters/2.1um/data/F_Delta.dat",
        dnn_artifact_relpath="compression/surrogate/diameters/2.1um/trained/microbubble_force_BEST.pkl",
        bnn_artifact_relpath="compression/surrogate/diameters/2.1um/trained/microbubble_force_BNN.pt",
        dnn_train_script_relpath="compression/surrogate/scripts/emb_train.py",
        dnn_multi_arch_script_relpath="compression/surrogate/scripts/train_multi_arch.py",
        bnn_train_script_relpath="compression/surrogate/scripts/emb_train_bnn.py",
        group_holdout_script_relpath="compression/surrogate/scripts/run_group_holdout.py",
    ),
    EmbDatasetSpec(
        name="compression_2.9um",
        modality="compression",
        diameter_um="2.9",
        data_relpath="compression/surrogate/diameters/2.9um/data/F_Delta.dat",
        dnn_artifact_relpath="compression/surrogate/diameters/2.9um/trained/microbubble_force_BEST.pkl",
        bnn_artifact_relpath="compression/surrogate/diameters/2.9um/trained/microbubble_force_BNN.pt",
        dnn_train_script_relpath="compression/surrogate/scripts/emb_train.py",
        dnn_multi_arch_script_relpath="compression/surrogate/scripts/train_multi_arch.py",
        bnn_train_script_relpath="compression/surrogate/scripts/emb_train_bnn.py",
        group_holdout_script_relpath="compression/surrogate/scripts/run_group_holdout.py",
    ),
    EmbDatasetSpec(
        name="compression_3.0um",
        modality="compression",
        diameter_um="3.0",
        data_relpath="compression/surrogate/diameters/3.0um/data/F_Delta.dat",
        dnn_artifact_relpath="compression/surrogate/diameters/3.0um/trained/microbubble_force_BEST.pkl",
        bnn_artifact_relpath="compression/surrogate/diameters/3.0um/trained/microbubble_force_BNN.pt",
        dnn_train_script_relpath="compression/surrogate/scripts/emb_train.py",
        dnn_multi_arch_script_relpath="compression/surrogate/scripts/train_multi_arch.py",
        bnn_train_script_relpath="compression/surrogate/scripts/emb_train_bnn.py",
        group_holdout_script_relpath="compression/surrogate/scripts/run_group_holdout.py",
    ),
    EmbDatasetSpec(
        name="indentation_3.2um",
        modality="indentation",
        diameter_um="3.2",
        data_relpath="indentation/surrogate/diameters/3.2um/data/samples_all.dat",
        dnn_artifact_relpath="indentation/surrogate/diameters/3.2um/trained/microbubble_displacement_BEST.pkl",
        bnn_artifact_relpath="indentation/surrogate/diameters/3.2um/trained/microbubble_displacement_BNN.pt",
        dnn_train_script_relpath="indentation/surrogate/scripts/emb_train.py",
        dnn_multi_arch_script_relpath="indentation/surrogate/scripts/train_multi_arch.py",
        bnn_train_script_relpath="indentation/surrogate/scripts/emb_train_bnn.py",
        group_holdout_script_relpath="indentation/surrogate/scripts/run_group_holdout.py",
    ),
    EmbDatasetSpec(
        name="indentation_3.4um",
        modality="indentation",
        diameter_um="3.4",
        data_relpath="indentation/surrogate/diameters/3.4um/data/samples_all.dat",
        dnn_artifact_relpath="indentation/surrogate/diameters/3.4um/trained/microbubble_displacement_BEST.pkl",
        bnn_artifact_relpath="indentation/surrogate/diameters/3.4um/trained/microbubble_displacement_BNN.pt",
        dnn_train_script_relpath="indentation/surrogate/scripts/emb_train.py",
        dnn_multi_arch_script_relpath="indentation/surrogate/scripts/train_multi_arch.py",
        bnn_train_script_relpath="indentation/surrogate/scripts/emb_train_bnn.py",
        group_holdout_script_relpath="indentation/surrogate/scripts/run_group_holdout.py",
    ),
    EmbDatasetSpec(
        name="indentation_5.8um",
        modality="indentation",
        diameter_um="5.8",
        data_relpath="indentation/surrogate/diameters/5.8um/data/samples_all.dat",
        dnn_artifact_relpath="indentation/surrogate/diameters/5.8um/trained/microbubble_displacement_BEST.pkl",
        bnn_artifact_relpath="indentation/surrogate/diameters/5.8um/trained/microbubble_displacement_BNN.pt",
        dnn_train_script_relpath="indentation/surrogate/scripts/emb_train.py",
        dnn_multi_arch_script_relpath="indentation/surrogate/scripts/train_multi_arch.py",
        bnn_train_script_relpath="indentation/surrogate/scripts/emb_train_bnn.py",
        group_holdout_script_relpath="indentation/surrogate/scripts/run_group_holdout.py",
    ),
)


def resolve_emb_dataset_specs(repo_root: str | Path) -> list[dict[str, str]]:
    return [spec.resolve(repo_root) for spec in EMB_DATASET_SPECS]
