from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import TYPE_CHECKING

import pandas as pd
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from backtest.registry import BacktestRegistry

class BacktestWindow:
    def __init__(self, registry: "BacktestRegistry") -> None:
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


class MarketAnalyzerWindow:
    def __init__(self, daily: pd.DataFrame, registry: "BacktestRegistry") -> None:
        self.daily = daily.copy()
        self.registry = registry

        self.root = tk.Tk()
        self.root.title("MarketAnalyzer")
        self.root.geometry("1200x800")
        self.root.minsize(960, 600)

        self.series_views = ["taxa", "zscore"]
        self.series_view_index = 0
        self.selected_algorithm = tk.StringVar()

        self._build_ui()

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.series_tab = ttk.Frame(notebook, padding=10)
        self.backtest_tab = ttk.Frame(notebook, padding=10)

        notebook.add(self.series_tab, text="Series")
        notebook.add(self.backtest_tab, text="Backtests")

        self._build_series_tab(self.series_tab)
        self._build_backtest_tab(self.backtest_tab)

    def _build_series_tab(self, parent: ttk.Frame) -> None:
        controls = ttk.Frame(parent)
        controls.pack(fill="x", pady=(0, 8))

        self.series_view_label = ttk.Label(controls, text="View: Taxa")
        self.series_view_label.pack(side="left", padx=(0, 8))

        ttk.Button(controls, text="Prev", command=self._prev_series_view).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(controls, text="Next", command=self._next_series_view).pack(
            side="left", padx=(0, 8)
        )

        ttk.Label(
            controls,
            text="Use Left/Right to switch views and toolbar for zoom/pan.",
        ).pack(side="left")

        chart_frame = ttk.Frame(parent)
        chart_frame.pack(fill="both", expand=True)

        self.series_figure = Figure(figsize=(10, 6), dpi=100)
        self.series_ax = self.series_figure.add_subplot(111)
        self.series_canvas = FigureCanvasTkAgg(self.series_figure, master=chart_frame)
        self.series_canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill="x")
        self.series_toolbar = NavigationToolbar2Tk(
            self.series_canvas,
            toolbar_frame,
            pack_toolbar=False,
        )
        self.series_toolbar.update()
        self.series_toolbar.pack(side="left", fill="x")

        canvas_widget = self.series_canvas.get_tk_widget()
        canvas_widget.bind("<Enter>", lambda _e: canvas_widget.focus_set())
        self.series_canvas.mpl_connect("key_press_event", self._on_series_key)

        self._draw_series()
        canvas_widget.focus_set()

    def _draw_series(self) -> None:
        self.series_ax.clear()

        if self.daily.empty:
            self.series_ax.set_title("Series")
            self.series_ax.text(0.5, 0.5, "No data available.", ha="center", va="center")
            self.series_canvas.draw_idle()
            return

        if "Data Base" not in self.daily.columns:
            self.series_ax.set_title("Series")
            self.series_ax.text(
                0.5,
                0.5,
                "Column 'Data Base' not found in dataframe.",
                ha="center",
                va="center",
            )
            self.series_canvas.draw_idle()
            return

        x = self.daily["Data Base"]
        view = self.series_views[self.series_view_index]

        if view == "taxa":
            self._plot_series_column("taxa_media", x, "Daily average rate", linewidth=1.5)
            self._plot_series_column("media_historica", x, "Historical mean", linestyle="--", linewidth=1.2)
            self._plot_series_column("media_rolling_5a", x, "Rolling mean 5y", linewidth=1.2)
            self._plot_series_column("mm_252", x, "Moving avg 252d", linewidth=1.0)
            self._plot_series_column("mm_756", x, "Moving avg 756d", linewidth=1.0)
            self._plot_series_column("banda_1dp_sup", x, "+1 std", linestyle=":", linewidth=1.0)
            self._plot_series_column("banda_1dp_inf", x, "-1 std", linestyle=":", linewidth=1.0)
            self._plot_series_column("banda_2dp_sup", x, "+2 std", linestyle=":", linewidth=1.0)
            self._plot_series_column("banda_2dp_inf", x, "-2 std", linestyle=":", linewidth=1.0)
            self.series_ax.set_title("IPCA+ Long - Historical Rate Series")
            self.series_ax.set_ylabel("Real Rate (%)")
            self.series_view_label.configure(text="View: Taxa")
        else:
            self._plot_series_column("zscore", x, "Historical z-score", linewidth=1.2)
            self._plot_series_column("zscore_rolling_5a", x, "Rolling z-score 5y", linewidth=1.2)
            self.series_ax.axhline(0, linestyle="--", linewidth=1.0)
            self.series_ax.axhline(1, linestyle=":", linewidth=1.0)
            self.series_ax.axhline(2, linestyle=":", linewidth=1.0)
            self.series_ax.axhline(-1, linestyle=":", linewidth=1.0)
            self.series_ax.axhline(-2, linestyle=":", linewidth=1.0)
            self.series_ax.set_title("IPCA+ Long - Z-score")
            self.series_ax.set_ylabel("Z-score")
            self.series_view_label.configure(text="View: Z-score")

        self.series_ax.set_xlabel("Date")
        self.series_ax.grid(True, alpha=0.3)
        self.series_ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
        self.series_ax.format_xdata = mdates.DateFormatter("%d/%m/%Y")
        self.series_ax.format_coord = (
            lambda x, y: f"Data={mdates.num2date(x).strftime('%d/%m/%Y')}  Valor={y:.4f}"
        )

        if self.series_ax.lines:
            self.series_ax.legend()

        self.series_figure.tight_layout()
        self.series_canvas.draw_idle()

    def _next_series_view(self) -> None:
        self.series_view_index = (self.series_view_index + 1) % len(self.series_views)
        self._draw_series()

    def _prev_series_view(self) -> None:
        self.series_view_index = (self.series_view_index - 1) % len(self.series_views)
        self._draw_series()

    def _on_series_key(self, event) -> None:
        if event.key == "right":
            self._next_series_view()
        elif event.key == "left":
            self._prev_series_view()

    def _plot_series_column(self, column: str, x: pd.Series, label: str, **kwargs) -> None:
        if column not in self.daily.columns:
            return
        self.series_ax.plot(x, self.daily[column], label=label, **kwargs)

    def _build_backtest_tab(self, parent: ttk.Frame) -> None:
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill="x")

        ttk.Label(top_frame, text="Algorithm:").pack(side="left", padx=(0, 8))

        self.combo = ttk.Combobox(
            top_frame,
            textvariable=self.selected_algorithm,
            values=self.registry.names(),
            state="readonly",
            width=46,
        )
        self.combo.pack(side="left", padx=(0, 8))

        if self.registry.names():
            self.combo.current(0)

        self.run_button = ttk.Button(
            top_frame,
            text="Run Backtest",
            command=self._run_selected_algorithm,
        )
        self.run_button.pack(side="left")

        output_frame = ttk.Frame(parent, padding=(0, 10, 0, 0))
        output_frame.pack(fill="both", expand=True)

        self.output = ScrolledText(
            output_frame,
            wrap="word",
            font=("Consolas", 11),
        )
        self.output.pack(fill="both", expand=True)

        self._write_line("MarketAnalyzer UI ready.")
        self._write_line("Use the tabs to switch between charts and backtests.")
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
            self._write_line("No algorithm selected.")
            return

        self.output.delete("1.0", "end")
        self._write_line(f"Running: {name}")
        self._write_line("")

        try:
            fn = self.registry.get(name)
            result = fn()
            self._write_line(result)
        except Exception as exc:
            self._write_line(f"Error while running backtest: {exc}")

    def run(self) -> None:
        self.root.mainloop()
