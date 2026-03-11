from pathlib import Path

DATA_DIR = Path("data")

def convert_file(file_path: Path):
    # lê usando latin1 (funciona para quase todos os arquivos brasileiros)
    text = file_path.read_text(encoding="latin1")

    # salva novamente em utf-8
    file_path.write_text(text, encoding="utf-8")

    print(f"Converted: {file_path}")


def main():
    for file in DATA_DIR.rglob("*.csv"):
        convert_file(file)


if __name__ == "__main__":
    main()