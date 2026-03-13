from pathlib import Path

DATA_DIR = Path("data")


def convert_file(file_path: Path) -> None:
    # Lê em latin1 para conseguir regravar arquivos legados em UTF-8.
    text = file_path.read_text(encoding="latin1")
    file_path.write_text(text, encoding="utf-8")
    print(f"Converted: {file_path}")


def main() -> None:
    for file_path in DATA_DIR.rglob("*.csv"):
        convert_file(file_path)


if __name__ == "__main__":
    main()
