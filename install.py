"""Install AudioSR into the same Python environment that runs ComfyUI."""

import importlib.metadata
import os
import subprocess
import sys


def update_compatibility_metadata():
    """Replace the three obsolete upstream pins with the tested modern ranges."""
    distribution = importlib.metadata.distribution("audiosr")
    metadata_file = next(file for file in distribution.files if str(file).endswith(".dist-info/METADATA"))
    filename = os.fspath(distribution.locate_file(metadata_file))
    with open(filename, "r", encoding="utf-8") as stream:
        metadata = stream.read()
    replacements = {
        "numpy": ("Requires-Dist: numpy (<=1.23.5)", "Requires-Dist: numpy (>=2.2.6)"),
        "librosa": ("Requires-Dist: librosa (==0.9.2)", "Requires-Dist: librosa (>=0.11.0)"),
        "transformers": ("Requires-Dist: transformers (==4.30.2)", "Requires-Dist: transformers (>=5.13.0)"),
    }
    for package, obsolete_lines in replacements.items():
        for obsolete in obsolete_lines:
            metadata = metadata.replace(obsolete, "Requires-Dist: " + package)
    marker = "X-HR-Endless-Tested-Compatibility: numpy-2.2.6_librosa-0.11.0_transformers-5.13.0"
    if marker not in metadata:
        metadata = metadata.replace("Metadata-Version: 2.1", "Metadata-Version: 2.1\n" + marker, 1)
    with open(filename, "w", encoding="utf-8") as stream:
        stream.write(metadata)


def main():
    """Install upstream AudioSR code without applying its obsolete dependencies."""
    installed = {
        distribution.metadata["Name"].lower(): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    compatible = []
    if installed.get("torchlibrosa") != "0.1.0":
        compatible.append("torchlibrosa==0.1.0")
    if installed.get("progressbar") != "2.5":
        compatible.append("progressbar==2.5")
    if compatible:
        subprocess.run([sys.executable, "-m", "pip", "install", *compatible], check=True)
    if installed.get("audiosr") != "0.0.7":
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "audiosr==0.0.7"], check=True)
    update_compatibility_metadata()
    subprocess.run([sys.executable, "python/audio_sr.py", "--check"], check=True)


if __name__ == "__main__":
    main()
