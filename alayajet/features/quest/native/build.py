from __future__ import annotations

import hashlib
import logging
import subprocess
import sysconfig
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_SRC = _THIS_DIR / "resident_write.cpp"
_EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
_CACHE_DIR = Path(__file__).resolve().parents[4] / ".cache" / "alayajet-quest-native"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_OUT = _CACHE_DIR / f"_quest_native{_EXT_SUFFIX}"
_HASH = _OUT.with_suffix(_OUT.suffix + ".sha256")


def _find_package_path(name: str) -> Path:
    import importlib

    mod = importlib.import_module(name)
    paths = getattr(mod, "__path__", None)
    if paths:
        return Path(list(paths)[0])
    f = getattr(mod, "__file__", None)
    if f:
        return Path(f).parent
    raise RuntimeError(f"Cannot locate package {name!r}")


def _package_version(name: str) -> str:
    import importlib

    mod = importlib.import_module(name)
    return str(getattr(mod, "__version__", ""))


@dataclass(frozen=True)
class _BuildSpec:
    cmd: list[str]
    nb_src: Path
    mlx_lib: Path
    mlx_include: Path
    py_include: str
    mlx_version: str
    nb_version: str


def _build_spec() -> _BuildSpec:
    py_include = sysconfig.get_paths()["include"]
    nb_path = _find_package_path("nanobind")
    mlx_path = _find_package_path("mlx")
    mlx_include = mlx_path / "include"
    mlx_lib = mlx_path / "lib"
    metal_cpp = mlx_include / "metal_cpp"
    nb_src = nb_path / "src" / "nb_combined.cpp"

    cmd = [
        "clang++",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-O2",
        "-fvisibility=default",
        f"-I{py_include}",
        f"-I{nb_path / 'include'}",
        f"-I{nb_path / 'src'}",
        f"-I{nb_path / 'ext' / 'robin_map' / 'include'}",
        f"-I{mlx_include}",
        f"-I{metal_cpp}",
        f"-L{mlx_lib}",
        "-lmlx",
        "-framework",
        "Metal",
        "-framework",
        "Foundation",
        f"-Wl,-rpath,{mlx_lib}",
        "-D_METAL_",
        "-DACCELERATE_NEW_LAPACK",
        "-undefined",
        "dynamic_lookup",
        str(nb_src),
        str(_SRC),
        "-o",
        str(_OUT),
    ]
    return _BuildSpec(
        cmd=cmd,
        nb_src=nb_src,
        mlx_lib=mlx_lib,
        mlx_include=mlx_include,
        py_include=py_include,
        mlx_version=_package_version("mlx.core"),
        nb_version=_package_version("nanobind"),
    )


def _input_hash(spec: _BuildSpec) -> str:
    h = hashlib.sha256()
    h.update("\0".join(spec.cmd).encode())
    h.update(b"\0")
    h.update(f"mlx={spec.mlx_version}\0nb={spec.nb_version}\0".encode())
    h.update(_SRC.read_bytes())
    h.update(b"\0")
    h.update(spec.nb_src.read_bytes())
    return h.hexdigest()


def needs_rebuild() -> bool:
    if not _OUT.exists() or not _HASH.exists():
        return True
    try:
        spec = _build_spec()
        return _HASH.read_text().strip() != _input_hash(spec)
    except Exception:
        return True


def build() -> Path:
    spec = _build_spec()
    expected_hash = _input_hash(spec)
    if _OUT.exists() and _HASH.exists():
        try:
            if _HASH.read_text().strip() == expected_hash:
                return _OUT
        except OSError:
            pass

    for p, label in [
        (spec.py_include, "Python include"),
        (spec.mlx_include, "MLX include"),
        (spec.mlx_lib / "libmlx.dylib", "MLX lib"),
        (spec.nb_src, "nanobind source"),
    ]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    logger.info("Building quest native extension ...")
    result = subprocess.run(spec.cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to build quest native extension:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    _HASH.write_text(expected_hash)
    return _OUT
