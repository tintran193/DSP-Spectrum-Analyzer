import tkinter as tk

from tkinter import ttk, filedialog, messagebox

import numpy as np

from matplotlib.figure import Figure

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg
)

from src.input.signal_generator import (
    generate_sine,
    generate_multi_sine,
    generate_square
)

from src.input.wav_loader import load_wav

from src.analysis.analyzer import SpectrumAnalyzer

from src.visualization.plot import (
    plot_time_domain,
    plot_spectrum,
    plot_db_spectrum
)


class SpectrumAnalyzerGUI:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Digital Spectrum Analyzer"
        )

        # Keep the window within a typical desktop screen height.
        self.root.geometry("1350x780")
        self.root.minsize(1100, 650)

        self.analyzer = SpectrumAnalyzer()

        self.wav_path = None

        # Input settings.
        self.source_var = tk.StringVar(
            value="Generated"
        )

        self.signal_type_var = tk.StringVar(
            value="Sine"
        )

        self.frequency_var = tk.StringVar(
            value="1000"
        )

        self.amplitude_var = tk.StringVar(
            value="1.0"
        )

        self.phase_var = tk.StringVar(
            value="0"
        )

        self.fs_var = tk.StringVar(
            value="8000"
        )

        self.duration_var = tk.StringVar(
            value="0.128"
        )

        # FFT settings.
        self.fft_size_var = tk.StringVar(
            value="1024"
        )

        self.window_var = tk.StringVar(
            value="Rectangular"
        )

        # Signal information.
        self.info_var = tk.StringVar(
            value="No signal analyzed."
        )

        self._create_gui()

    # =========================================
    # GUI layout
    # =========================================

    def _create_gui(self):

        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Use grid for the three main vertical regions so the peak table
        # always reserves its own space below the plots.
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=0)  # Header
        main_frame.rowconfigure(1, weight=1)  # Controls + plots
        main_frame.rowconfigure(2, weight=0)  # Peak table

        title = ttk.Label(
            main_frame,
            text="DIGITAL SPECTRUM ANALYZER",
            font=("Arial", 18, "bold")
        )
        title.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        # Main content.
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=1, column=0, sticky="nsew")

        # Left side: controls.
        control_frame = ttk.LabelFrame(
            content_frame,
            text="Input and FFT Settings",
            padding=10
        )
        control_frame.pack(side="left", fill="y", padx=5)
        self._create_controls(control_frame)

        # Right side: plots.
        display_frame = ttk.Frame(content_frame)
        display_frame.pack(
            side="right", fill="both", expand=True, padx=5
        )
        self._create_plots(display_frame)

        # Bottom: peak table. It has a reserved row and cannot be pushed
        # below the visible window by the expanding plot area.
        self._create_peak_table(main_frame)

    # =========================================
    # Input controls
    # =========================================

    def _create_controls(self, parent):

        ttk.Label(
            parent,
            text="Signal Source"
        ).pack(anchor="w")

        source_box = ttk.Combobox(
            parent,
            textvariable=self.source_var,
            values=["Generated", "WAV"],
            state="readonly"
        )

        source_box.pack(
            fill="x",
            pady=5
        )

        ttk.Label(
            parent,
            text="Signal Type"
        ).pack(anchor="w")

        type_box = ttk.Combobox(
            parent,
            textvariable=self.signal_type_var,
            values=["Sine", "Multi-sine", "Square"],
            state="readonly"
        )

        type_box.pack(
            fill="x",
            pady=5
        )

        self._add_entry(
            parent,
            "Frequency(s) Hz (comma separated for multi-sine)",
            self.frequency_var
        )

        self._add_entry(
            parent,
            "Amplitude(s) (comma separated for multi-sine)",
            self.amplitude_var
        )

        self._add_entry(
            parent,
            "Phase(s) rad (comma separated for multi-sine)",
            self.phase_var
        )

        self._add_entry(
            parent,
            "Sampling Frequency (Hz)",
            self.fs_var
        )

        self._add_entry(
            parent,
            "Duration (s)",
            self.duration_var
        )

        ttk.Separator(
            parent,
            orient="horizontal"
        ).pack(
            fill="x",
            pady=10
        )

        ttk.Label(
            parent,
            text="FFT Size"
        ).pack(anchor="w")

        fft_box = ttk.Combobox(
            parent,
            textvariable=self.fft_size_var,
            values=[
                "256",
                "512",
                "1024",
                "2048",
                "4096"
            ],
            state="readonly"
        )

        fft_box.pack(
            fill="x",
            pady=5
        )

        ttk.Label(
            parent,
            text="Window"
        ).pack(anchor="w")

        window_box = ttk.Combobox(
            parent,
            textvariable=self.window_var,
            values=[
                "Rectangular",
                "Hamming",
                "Hann"
            ],
            state="readonly"
        )

        window_box.pack(
            fill="x",
            pady=5
        )

        ttk.Button(
            parent,
            text="Load WAV",
            command=self.load_wav_file
        ).pack(
            fill="x",
            pady=5
        )

        ttk.Button(
            parent,
            text="ANALYZE",
            command=self.analyze_signal
        ).pack(
            fill="x",
            pady=5
        )

        ttk.Button(
            parent,
            text="RESET",
            command=self.reset
        ).pack(
            fill="x",
            pady=5
        )

        ttk.Separator(
            parent,
            orient="horizontal"
        ).pack(
            fill="x",
            pady=10
        )

        ttk.Label(
            parent,
            text="Signal Information"
        ).pack(anchor="w")

        ttk.Label(
            parent,
            textvariable=self.info_var,
            wraplength=250,
            justify="left"
        ).pack(
            anchor="w",
            pady=5
        )

    def _add_entry(
        self,
        parent,
        label,
        variable
    ):

        ttk.Label(
            parent,
            text=label,
            wraplength=250
        ).pack(anchor="w")

        ttk.Entry(
            parent,
            textvariable=variable
        ).pack(
            fill="x",
            pady=5
        )

    # =========================================
    # Plot creation
    # =========================================

    def _create_plots(self, parent):

        # A less tall initial figure leaves enough room for Peak Information.
        # The Tk canvas will still expand/shrink with its display frame.
        self.figure = Figure(
            figsize=(9, 6),
            dpi=100
        )

        self.ax_time = self.figure.add_subplot(311)

        self.ax_magnitude = self.figure.add_subplot(312)

        self.ax_db = self.figure.add_subplot(313)

        self.figure.tight_layout(pad=3)

        self.canvas = FigureCanvasTkAgg(
            self.figure,
            master=parent
        )

        self.canvas.get_tk_widget().pack(
            fill="both",
            expand=True
        )

    # =========================================
    # Peak table
    # =========================================

    def _create_peak_table(self, parent):

        table_frame = ttk.LabelFrame(
            parent,
            text="Peak Information",
            padding=5
        )
        table_frame.grid(
            row=2, column=0,
            sticky="ew",
            pady=(6, 0)
        )
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        columns = ("frequency", "magnitude", "db")

        self.peak_table = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=4
        )

        self.peak_table.heading("frequency", text="Frequency (Hz)")
        self.peak_table.heading("magnitude", text="Magnitude")
        self.peak_table.heading("db", text="Magnitude (dB)")

        self.peak_table.column(
            "frequency", width=180, anchor="center", stretch=True
        )
        self.peak_table.column(
            "magnitude", width=180, anchor="center", stretch=True
        )
        self.peak_table.column(
            "db", width=180, anchor="center", stretch=True
        )

        scrollbar = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.peak_table.yview
        )
        self.peak_table.configure(yscrollcommand=scrollbar.set)

        self.peak_table.grid(row=0, column=0, sticky="ew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    # =========================================
    # Load WAV
    # =========================================

    def load_wav_file(self):

        path = filedialog.askopenfilename(
            filetypes=[
                ("WAV files", "*.wav"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        self.wav_path = path

        self.source_var.set("WAV")

        messagebox.showinfo(
            "WAV Loaded",
            "WAV file selected successfully."
        )

    # =========================================
    # Parse comma-separated values
    # =========================================

    @staticmethod
    def parse_float_list(text):

        try:
            return [
                float(value.strip())
                for value in text.split(",")
            ]

        except ValueError:
            raise ValueError(
                "Invalid number format."
            )

    # =========================================
    # Get input signal
    # =========================================

    def get_input_signal(self):

        source = self.source_var.get()

        # -------------------------------------
        # WAV input
        # -------------------------------------

        if source == "WAV":

            if self.wav_path is None:
                raise ValueError(
                    "Please load a WAV file first."
                )

            x, fs = load_wav(
                self.wav_path
            )

            return x, fs

        # -------------------------------------
        # Generated input
        # -------------------------------------

        fs = float(self.fs_var.get())

        duration = float(
            self.duration_var.get()
        )

        if not np.isfinite(fs) or fs <= 0:
            raise ValueError(
                "Sampling frequency must be positive."
            )

        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(
                "Duration must be positive."
            )

        signal_type = self.signal_type_var.get()

        # Sine.
        if signal_type == "Sine":

            frequency = float(
                self.frequency_var.get()
            )

            amplitude = float(
                self.amplitude_var.get()
            )

            phase = float(
                self.phase_var.get()
            )

            if not 0 < frequency < fs / 2:
                raise ValueError(
                    "Frequency must be between 0 and Nyquist."
                )

            if amplitude < 0:
                raise ValueError(
                    "Amplitude must be non-negative."
                )

            return generate_sine(
                frequency=frequency,
                amplitude=amplitude,
                phase=phase,
                fs=fs,
                duration=duration
            )

        # Multi-sine.
        if signal_type == "Multi-sine":

            frequencies = self.parse_float_list(
                self.frequency_var.get()
            )

            amplitudes = self.parse_float_list(
                self.amplitude_var.get()
            )

            phases = self.parse_float_list(
                self.phase_var.get()
            )

            if not (
                len(frequencies)
                == len(amplitudes)
                == len(phases)
            ):
                raise ValueError(
                    "Frequency, amplitude and phase "
                    "must have the same number of elements."
                )

            if any(
                f <= 0 or f >= fs / 2
                for f in frequencies
            ):
                raise ValueError(
                    "All frequencies must be below Nyquist."
                )

            if any(a < 0 for a in amplitudes):
                raise ValueError(
                    "Amplitudes must be non-negative."
                )

            return generate_multi_sine(
                frequencies=frequencies,
                amplitudes=amplitudes,
                phases=phases,
                fs=fs,
                duration=duration
            )

        # Square.
        if signal_type == "Square":

            frequency = float(
                self.frequency_var.get()
            )

            amplitude = float(
                self.amplitude_var.get()
            )

            if not 0 < frequency < fs / 2:
                raise ValueError(
                    "Frequency must be between 0 and Nyquist."
                )

            if amplitude < 0:
                raise ValueError(
                    "Amplitude must be non-negative."
                )

            return generate_square(
                frequency=frequency,
                amplitude=amplitude,
                fs=fs,
                duration=duration
            )

        raise ValueError(
            "Unsupported signal type."
        )

    # =========================================
    # Analyze signal
    # =========================================

    def analyze_signal(self):

        try:

            # Step 1: Get signal.
            x, fs = self.get_input_signal()

            # Step 2: Get FFT settings.
            fft_size = int(
                self.fft_size_var.get()
            )

            window_name = (
                self.window_var.get().lower()
            )

            # Step 3: Run analysis pipeline.
            result = self.analyzer.analyze(
                x=x,
                fs=fs,
                fft_size=fft_size,
                window_name=window_name
            )

            # Step 4: Plot waveform.
            plot_time_domain(
                self.ax_time,
                result["frame"],
                fs
            )

            # Step 5: Plot magnitude spectrum.
            plot_spectrum(
                self.ax_magnitude,
                result["frequency"],
                result["magnitude"],
                result["peaks"]
            )

            # Step 6: Plot dB spectrum.
            plot_db_spectrum(
                self.ax_db,
                result["frequency"],
                result["db"],
                result["peaks"]
            )

            self.figure.tight_layout(pad=3)

            self.canvas.draw_idle()

            # Step 7: Update signal information.
            self.info_var.set(
                f"Sampling frequency: {fs:.2f} Hz\n"
                f"FFT size: {fft_size}\n"
                f"Window: {window_name}\n"
                f"Frequency resolution: "
                f"{result['frequency_resolution']:.4f} Hz\n"
                f"Number of samples: {len(x)}"
            )

            # Step 8: Update peak table.
            self.update_peak_table(
                result["peaks"]
            )

        except (ValueError, TypeError) as error:

            messagebox.showerror(
                "Input Error",
                str(error)
            )

        except Exception as error:

            messagebox.showerror(
                "Analysis Error",
                str(error)
            )

    # =========================================
    # Update peak table
    # =========================================

    def update_peak_table(self, peaks):

        # Clear old rows.
        for item in self.peak_table.get_children():

            self.peak_table.delete(item)

        # Insert new rows.
        for peak in peaks:

            self.peak_table.insert(
                "",
                "end",
                values=(
                    f"{peak['frequency']:.2f}",
                    f"{peak['magnitude']:.4f}",
                    f"{peak['db']:.2f}"
                )
            )

    # =========================================
    # Reset GUI
    # =========================================

    def reset(self):

        self.source_var.set("Generated")

        self.signal_type_var.set("Sine")

        self.frequency_var.set("1000")

        self.amplitude_var.set("1.0")

        self.phase_var.set("0")

        self.fs_var.set("8000")

        self.duration_var.set("0.128")

        self.fft_size_var.set("1024")

        self.window_var.set("Rectangular")

        self.wav_path = None

        self.info_var.set(
            "No signal analyzed."
        )

        # Clear table.
        for item in self.peak_table.get_children():

            self.peak_table.delete(item)

        # Clear plots.
        self.ax_time.clear()

        self.ax_magnitude.clear()

        self.ax_db.clear()

        self.canvas.draw_idle()


def run_app():

    root = tk.Tk()

    app = SpectrumAnalyzerGUI(root)

    root.mainloop()


if __name__ == "__main__":
    run_app()