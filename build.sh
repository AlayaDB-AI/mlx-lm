#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

APP_NAME="AlayaJetServer"
ENTRY="$ROOT_DIR/app_entry.py"
MLX_LIB_DIR="$ROOT_DIR/3rdparty/mlx/python/mlx/lib"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYINSTALLER_BIN="${PYINSTALLER_BIN:-pyinstaller}"

if ! command -v "$PYINSTALLER_BIN" >/dev/null 2>&1; then
  echo "pyinstaller not found. Install with: $PYTHON_BIN -m pip install -U pyinstaller"
  exit 1
fi

if [ ! -f "$ENTRY" ]; then
  echo "Missing entry: $ENTRY"
  exit 1
fi

if [ ! -f "$MLX_LIB_DIR/libmlx.dylib" ] || [ ! -f "$MLX_LIB_DIR/mlx.metallib" ]; then
  echo "Missing MLX binaries in $MLX_LIB_DIR"
  exit 1
fi

"$PYINSTALLER_BIN" --clean --noconfirm --name "$APP_NAME" --console --onedir \
  --paths "$ROOT_DIR/3rdparty/mlx/python" \
  --collect-all mlx \
  --collect-all mlx_lm \
  --collect-submodules mlx_lm \
  --add-binary "$MLX_LIB_DIR/libmlx.dylib:." \
  --add-binary "$MLX_LIB_DIR/mlx.metallib:." \
  "$ENTRY"

if command -v ditto >/dev/null 2>&1; then
  ditto -c -k --sequesterRsrc --keepParent \
    "$ROOT_DIR/dist/$APP_NAME" \
    "$ROOT_DIR/dist/$APP_NAME.zip"
elif command -v zip >/dev/null 2>&1; then
  (cd "$ROOT_DIR/dist" && zip -r "$APP_NAME.zip" "$APP_NAME" >/dev/null)
else
  echo "No zip tool found; skipping zip creation."
  exit 0
fi

echo "Build complete: $ROOT_DIR/dist/$APP_NAME"
echo "Zip: $ROOT_DIR/dist/$APP_NAME.zip"
