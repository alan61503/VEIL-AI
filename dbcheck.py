
from pathlib import Path
for path in Path("data/images").iterdir():
    print(path, path.is_file())
