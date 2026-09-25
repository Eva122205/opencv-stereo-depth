"""Download the OpenCV aloe stereo pair into the chosen data directory."""
import argparse
from pathlib import Path
from urllib.request import urlopen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    options = parser.parse_args()
    options.data_dir.mkdir(parents=True, exist_ok=True)
    base_url = "https://raw.githubusercontent.com/opencv/opencv/4.12.0/samples/data"
    for name in ("aloeL.jpg", "aloeR.jpg"):
        destination = options.data_dir / name
        if destination.exists():
            print(f"Already exists: {destination}")
            continue
        with urlopen(f"{base_url}/{name}", timeout=60) as response:
            content = response.read()
        if not content.startswith(b"\xff\xd8\xff"):
            raise ValueError(f"The downloaded file is not a JPEG: {name}")
        destination.write_bytes(content)
        print(f"Downloaded: {destination} ({len(content)} bytes)")


if __name__ == "__main__":
    main()
