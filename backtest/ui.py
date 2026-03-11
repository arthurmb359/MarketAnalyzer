from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText


class BacktestWindow:
    def __init__(self, registry: BacktestRegistry) -> None:
        self.registry = registry

        self.root = tk.Tk()
        self.root.title("MarketAnalyzer - Backtest")
        self.root.geometry("1000x700")
        self.root.minsize(800, 500)

        self.selected_algorithm = tk.StringVar()

        self._build_ui()

    def _build_ui(self) -> None:
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill="x")

        ttk.Label(top_frame, text="Algoritmo:").pack(side="left", padx=(0, 8))

        self.combo = ttk.Combobox(
            top_frame,
            textvariable=self.selected_algorithm,
            values=self.registry.names(),
            state="readonly",
            width=40,
        )
        self.combo.pack(side="left", padx=(0, 8))

        if self.registry.names():
            self.combo.current(0)

        self.run_button = ttk.Button(
            top_frame,
            text="Iniciar Backtest",
            command=self._run_selected_algorithm,
        )
        self.run_button.pack(side="left")

        output_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        output_frame.pack(fill="both", expand=True)

        self.output = ScrolledText(
            output_frame,
            wrap="word",
            font=("Consolas", 11),
        )
        self.output.pack(fill="both", expand=True)

        self._write_line("Backtest UI inicializada.")
        self._write_line("Escolha um algoritmo e clique em 'Iniciar Backtest'.")
        self._write_line("")

    def _write(self, text: str) -> None:
        self.output.insert("end", text)
        self.output.see("end")
        self.root.update_idletasks()

    def _write_line(self, text: str) -> None:
        self._write(text + "\n")

    def _run_selected_algorithm(self) -> None:
        name = self.selected_algorithm.get().strip()
        if not name:
            self._write_line("Nenhum algoritmo selecionado.")
            return

        self.output.delete("1.0", "end")
        self._write_line(f"Executando: {name}")
        self._write_line("")

        try:
            fn = self.registry.get(name)
            result = fn()
            self._write_line(result)
        except Exception as exc:
            self._write_line(f"Erro ao executar backtest: {exc}")

    def run(self) -> None:
        self.root.mainloop()