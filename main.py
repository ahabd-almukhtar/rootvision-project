import tkinter as tk
from tkinter import ttk, messagebox
import math
import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Dict, Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


SAFE_NAMES = {
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'exp': np.exp,
    'log': np.log,
    'sqrt': np.sqrt,
    'abs': np.abs,
    'pi': np.pi,
    'e': np.e,
}


@dataclass
class IterationRow:
    n: int
    x: float
    fx: float
    error: Optional[float]
    note: str = ""


class RootVisionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("RootVision Studio - Numerical Methods Explorer")
        self.root.geometry("1400x850")
        self.root.minsize(1200, 760)

        self.examples = self._build_examples()
        self.rows: List[IterationRow] = []
        self.current_points = []
        self.play_index = 0
        self.play_job = None

        self._build_style()
        self._build_ui()
        self._load_example("Parachutist Drag Equation")

    def _build_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 15, "bold"))
        style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Info.TLabel", font=("Segoe UI", 9))
        style.configure("Run.TButton", font=("Segoe UI", 10, "bold"))

    def _build_examples(self) -> Dict[str, Dict]:
        return {
            "Parachutist Drag Equation": {
                "f": "668.06/x*(1-exp(-0.146843*x)) - 40",
                "g": "668.06/40*(1-exp(-0.146843*x))",
                "method": "Bisection",
                "x_low": "12",
                "x_high": "16",
                "x0": "14",
                "x1": "16",
                "tol": "0.0001",
                "max_iter": "25",
                "description": (
                    "Engineering example from root-finding lectures. The root represents the drag coefficient c "
                    "for a parachutist reaching 40 m/s after 10 s."
                ),
            },
            "cos(x) - x = 0": {
                "f": "cos(x) - x",
                "g": "cos(x)",
                "method": "Newton-Raphson",
                "x_low": "0",
                "x_high": "1",
                "x0": "0.5",
                "x1": "1.0",
                "tol": "1e-5",
                "max_iter": "30",
                "description": "Classic nonlinear equation that works well for comparing convergence speed.",
            },
            "x^3 - x - 1 = 0": {
                "f": "x**3 - x - 1",
                "g": "(x + 1)**(1/3)",
                "method": "Secant",
                "x_low": "1",
                "x_high": "2",
                "x0": "1.0",
                "x1": "2.0",
                "tol": "1e-5",
                "max_iter": "30",
                "description": "Useful for Bisection, Newton, Secant, and Fixed-Point comparisons.",
            },
        }

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        controls = ttk.Frame(outer, padding=(6, 6, 10, 6))
        controls.grid(row=0, column=0, sticky="nsw")

        main = ttk.Frame(outer)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=3)
        main.rowconfigure(1, weight=2)

        ttk.Label(controls, text="RootVision Studio", style="Title.TLabel").pack(anchor="w", pady=(0, 10))
        ttk.Label(
            controls,
            text="Educational GUI for visualizing root-finding iterations and convergence.",
            wraplength=300,
            style="Info.TLabel",
            justify="left",
        ).pack(anchor="w", pady=(0, 12))

        example_box = ttk.LabelFrame(controls, text="Example Problem", padding=8)
        example_box.pack(fill="x", pady=(0, 8))
        self.example_var = tk.StringVar()
        example_combo = ttk.Combobox(
            example_box,
            textvariable=self.example_var,
            values=list(self.examples.keys()),
            state="readonly",
            width=34,
        )
        example_combo.pack(fill="x")
        example_combo.bind("<<ComboboxSelected>>", lambda e: self._load_example(self.example_var.get()))

        input_box = ttk.LabelFrame(controls, text="User Inputs", padding=8)
        input_box.pack(fill="x", pady=(0, 8))

        self.method_var = tk.StringVar(value="Bisection")
        self.function_var = tk.StringVar()
        self.g_var = tk.StringVar()
        self.x_low_var = tk.StringVar()
        self.x_high_var = tk.StringVar()
        self.x0_var = tk.StringVar()
        self.x1_var = tk.StringVar()
        self.tol_var = tk.StringVar(value="1e-5")
        self.max_iter_var = tk.StringVar(value="30")

        self._labeled_combo(input_box, "Method", self.method_var, ["Bisection", "Newton-Raphson", "Secant", "Fixed-Point"])
        self._labeled_entry(input_box, "f(x)", self.function_var, width=34)
        self._labeled_entry(input_box, "g(x) for Fixed-Point", self.g_var, width=34)
        self._labeled_entry(input_box, "Lower bound / xL", self.x_low_var)
        self._labeled_entry(input_box, "Upper bound / xU", self.x_high_var)
        self._labeled_entry(input_box, "Initial guess x0", self.x0_var)
        self._labeled_entry(input_box, "Second guess x1", self.x1_var)
        self._labeled_entry(input_box, "Tolerance", self.tol_var)
        self._labeled_entry(input_box, "Max iterations", self.max_iter_var)

        button_row = ttk.Frame(controls)
        button_row.pack(fill="x", pady=(6, 8))
        ttk.Button(button_row, text="Run", style="Run.TButton", command=self.run_solver).pack(side="left", padx=(0, 6))
        ttk.Button(button_row, text="Animate", command=self.animate_iterations).pack(side="left", padx=6)
        ttk.Button(button_row, text="Clear", command=self.clear_results).pack(side="left", padx=6)

        theory_box = ttk.LabelFrame(controls, text="What this teaches", padding=8)
        theory_box.pack(fill="x", pady=(0, 8))
        self.theory_text = tk.Text(theory_box, width=40, height=12, wrap="word", font=("Segoe UI", 9))
        self.theory_text.pack(fill="both", expand=True)
        self.theory_text.configure(state="disabled")

        # Plot panel
        plot_frame = ttk.LabelFrame(main, text="Visualization", padding=6)
        plot_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(10, 5.8), dpi=100)
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Bottom section: table + summary
        bottom = ttk.Frame(main)
        bottom.grid(row=1, column=0, sticky="nsew")
        bottom.columnconfigure(0, weight=3)
        bottom.columnconfigure(1, weight=2)
        bottom.rowconfigure(0, weight=1)

        table_frame = ttk.LabelFrame(bottom, text="Iteration Table", padding=6)
        table_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        columns = ("n", "x", "f(x)", "error", "note")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)
        for col, w in [("n", 60), ("x", 160), ("f(x)", 160), ("error", 120), ("note", 260)]:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="center")
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=vsb.set)

        summary_frame = ttk.LabelFrame(bottom, text="Summary & Interpretation", padding=8)
        summary_frame.grid(row=0, column=1, sticky="nsew")
        summary_frame.rowconfigure(0, weight=1)
        summary_frame.columnconfigure(0, weight=1)
        self.summary_text = tk.Text(summary_frame, wrap="word", font=("Segoe UI", 9))
        self.summary_text.grid(row=0, column=0, sticky="nsew")
        self.summary_text.configure(state="disabled")

        self.method_var.trace_add("write", lambda *_: self._update_theory_panel())

    def _labeled_entry(self, parent, label, variable, width=18):
        ttk.Label(parent, text=label, style="Section.TLabel").pack(anchor="w", pady=(4, 0))
        ttk.Entry(parent, textvariable=variable, width=width).pack(fill="x", pady=(0, 2))

    def _labeled_combo(self, parent, label, variable, values):
        ttk.Label(parent, text=label, style="Section.TLabel").pack(anchor="w", pady=(4, 0))
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        combo.pack(fill="x", pady=(0, 2))

    def _load_example(self, name: str):
        ex = self.examples[name]
        self.example_var.set(name)
        self.method_var.set(ex["method"])
        self.function_var.set(ex["f"])
        self.g_var.set(ex["g"])
        self.x_low_var.set(ex["x_low"])
        self.x_high_var.set(ex["x_high"])
        self.x0_var.set(ex["x0"])
        self.x1_var.set(ex["x1"])
        self.tol_var.set(ex["tol"])
        self.max_iter_var.set(ex["max_iter"])
        self._set_text(self.summary_text, ex["description"])
        self._update_theory_panel()
        self.clear_results(reset_summary=False)

    def _update_theory_panel(self):
        method = self.method_var.get()
        texts = {
            "Bisection": (
                "Bisection is a bracketing method. It needs two initial values that lie on opposite sides of the root. "
                "At every step, the interval is halved, so the error shrinks in a predictable way. This makes it highly stable and easy to explain."
            ),
            "Newton-Raphson": (
                "Newton-Raphson uses the tangent line at the current guess. It is usually very fast near the root, but a poor starting value or a nearly zero derivative can cause problems."
            ),
            "Secant": (
                "Secant replaces the derivative with a slope through the last two guesses. It is often faster than Bisection and avoids computing derivatives explicitly."
            ),
            "Fixed-Point": (
                "Fixed-Point iteration rewrites f(x)=0 into x=g(x). Convergence depends strongly on the chosen form g(x). This makes it excellent for teaching why some iteration formulas converge while others diverge."
            ),
        }
        self._set_text(self.theory_text, texts.get(method, ""))

    def _set_text(self, widget: tk.Text, content: str):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    def _safe_eval(self, expr: str, x):
        local_names = dict(SAFE_NAMES)
        local_names['x'] = x
        return eval(expr, {"__builtins__": {}}, local_names)

    def _build_function(self, expr: str) -> Callable:
        def func(x):
            return self._safe_eval(expr, x)
        return func

    def _num_derivative(self, f: Callable[[float], float], x: float, h: float = 1e-6) -> float:
        return (f(x + h) - f(x - h)) / (2 * h)

    def clear_results(self, reset_summary=True):
        if self.play_job:
            self.root.after_cancel(self.play_job)
            self.play_job = None
        self.rows = []
        self.current_points = []
        self.play_index = 0
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.set_title("Function plot")
        self.ax2.set_title("Convergence")
        self.canvas.draw_idle()
        if reset_summary:
            self._set_text(self.summary_text, "Results will appear here after running the solver.")

    def run_solver(self):
        try:
            method = self.method_var.get()
            f = self._build_function(self.function_var.get().strip())
            g_expr = self.g_var.get().strip()
            g = self._build_function(g_expr) if g_expr else None
            xl = float(self.x_low_var.get())
            xu = float(self.x_high_var.get())
            x0 = float(self.x0_var.get())
            x1 = float(self.x1_var.get())
            tol = float(self.tol_var.get())
            max_iter = int(self.max_iter_var.get())

            if method == "Bisection":
                self.rows, self.current_points = self.solve_bisection(f, xl, xu, tol, max_iter)
            elif method == "Newton-Raphson":
                self.rows, self.current_points = self.solve_newton(f, x0, tol, max_iter)
            elif method == "Secant":
                self.rows, self.current_points = self.solve_secant(f, x0, x1, tol, max_iter)
            elif method == "Fixed-Point":
                if g is None:
                    raise ValueError("Please enter g(x) for Fixed-Point iteration.")
                self.rows, self.current_points = self.solve_fixed_point(f, g, x0, tol, max_iter)
            else:
                raise ValueError("Unsupported method.")

            self.populate_table()
            self.plot_results(f)
            self.write_summary(method)
        except Exception as exc:
            messagebox.showerror("Run error", str(exc))

    def solve_bisection(self, f, xl, xu, tol, max_iter):
        fl = f(xl)
        fu = f(xu)
        if fl * fu > 0:
            raise ValueError("For Bisection, f(xL) and f(xU) must have opposite signs.")
        rows, pts = [], []
        xr_old = None
        for i in range(1, max_iter + 1):
            xr = (xl + xu) / 2.0
            fr = f(xr)
            err = abs(xr - xr_old) if xr_old is not None else None
            note = "left interval" if fl * fr < 0 else "right interval"
            rows.append(IterationRow(i, xr, fr, err, note))
            pts.append((xr, fr))
            if abs(fr) < tol or (err is not None and err < tol):
                break
            if fl * fr < 0:
                xu = xr
                fu = fr
            else:
                xl = xr
                fl = fr
            xr_old = xr
        return rows, pts

    def solve_newton(self, f, x0, tol, max_iter):
        rows, pts = [], []
        x = x0
        for i in range(1, max_iter + 1):
            fx = f(x)
            dfx = self._num_derivative(f, x)
            if abs(dfx) < 1e-12:
                raise ValueError("Derivative became too small. Try another initial guess.")
            x_new = x - fx / dfx
            err = abs(x_new - x)
            rows.append(IterationRow(i, x_new, f(x_new), err, f"slope={dfx:.4g}"))
            pts.append((x_new, f(x_new)))
            x = x_new
            if abs(fx) < tol or err < tol:
                break
        return rows, pts

    def solve_secant(self, f, x0, x1, tol, max_iter):
        rows, pts = [], []
        prev, curr = x0, x1
        for i in range(1, max_iter + 1):
            f_prev, f_curr = f(prev), f(curr)
            denom = (f_curr - f_prev)
            if abs(denom) < 1e-12:
                raise ValueError("Secant denominator became too small. Change starting values.")
            x_new = curr - f_curr * (curr - prev) / denom
            err = abs(x_new - curr)
            rows.append(IterationRow(i, x_new, f(x_new), err, "secant intersection"))
            pts.append((x_new, f(x_new)))
            prev, curr = curr, x_new
            if abs(f_curr) < tol or err < tol:
                break
        return rows, pts

    def solve_fixed_point(self, f, g, x0, tol, max_iter):
        rows, pts = [], []
        x = x0
        for i in range(1, max_iter + 1):
            x_new = g(x)
            err = abs(x_new - x)
            rows.append(IterationRow(i, x_new, f(x_new), err, "x_{n+1}=g(x_n)"))
            pts.append((x_new, f(x_new)))
            x = x_new
            if err < tol or abs(f(x)) < tol:
                break
        return rows, pts

    def populate_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for row in self.rows:
            self.tree.insert(
                "",
                "end",
                values=(
                    row.n,
                    f"{row.x:.8f}",
                    f"{row.fx:.8e}",
                    "-" if row.error is None else f"{row.error:.3e}",
                    row.note,
                ),
            )

    def plot_results(self, f):
        self.ax1.clear()
        self.ax2.clear()

        xs = [p[0] for p in self.current_points] if self.current_points else [0, 1]
        lo = min(xs + [float(self.x_low_var.get()), float(self.x0_var.get())]) - 1
        hi = max(xs + [float(self.x_high_var.get()), float(self.x1_var.get())]) + 1
        x_plot = np.linspace(lo, hi, 500)
        y_plot = f(x_plot)
        y_plot = np.clip(y_plot, -1e3, 1e3)

        self.ax1.axhline(0, linewidth=1)
        self.ax1.plot(x_plot, y_plot, linewidth=2)
        if self.current_points:
            px = [p[0] for p in self.current_points]
            py = [p[1] for p in self.current_points]
            self.ax1.plot(px, py, 'o--', markersize=5)
        self.ax1.set_title("f(x) and iteration points")
        self.ax1.set_xlabel("x")
        self.ax1.set_ylabel("f(x)")
        self.ax1.grid(True, alpha=0.3)

        errors = [row.error for row in self.rows if row.error is not None]
        iters = list(range(2, len(errors) + 2)) if errors else []
        if errors:
            self.ax2.semilogy(iters, errors, 'o-')
        else:
            self.ax2.text(0.5, 0.5, "Need at least 2 points\nfor an error curve.", ha="center", va="center")
        self.ax2.set_title("Convergence history")
        self.ax2.set_xlabel("iteration")
        self.ax2.set_ylabel("absolute change")
        self.ax2.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def animate_iterations(self):
        if not self.rows:
            messagebox.showinfo("Nothing to animate", "Run the solver first.")
            return
        self.play_index = 1
        self._animate_step()

    def _animate_step(self):
        if self.play_index > len(self.current_points):
            self.play_job = None
            return
        shown = self.current_points[: self.play_index]
        self.ax1.clear()
        self.ax2.clear()
        f = self._build_function(self.function_var.get().strip())
        xs = [p[0] for p in self.current_points]
        lo = min(xs + [float(self.x_low_var.get()), float(self.x0_var.get())]) - 1
        hi = max(xs + [float(self.x_high_var.get()), float(self.x1_var.get())]) + 1
        x_plot = np.linspace(lo, hi, 500)
        y_plot = np.clip(f(x_plot), -1e3, 1e3)
        self.ax1.axhline(0, linewidth=1)
        self.ax1.plot(x_plot, y_plot, linewidth=2)
        px = [p[0] for p in shown]
        py = [p[1] for p in shown]
        self.ax1.plot(px, py, 'o--', markersize=6)
        self.ax1.set_title(f"Animation step {self.play_index}/{len(self.current_points)}")
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_xlabel("x")
        self.ax1.set_ylabel("f(x)")

        errors = [row.error for row in self.rows[: self.play_index] if row.error is not None]
        iters = list(range(2, len(errors) + 2)) if errors else []
        if errors:
            self.ax2.semilogy(iters, errors, 'o-')
        self.ax2.set_title("Convergence during animation")
        self.ax2.set_xlabel("iteration")
        self.ax2.set_ylabel("absolute change")
        self.ax2.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()

        self.play_index += 1
        self.play_job = self.root.after(650, self._animate_step)

    def write_summary(self, method: str):
        last = self.rows[-1]
        converged = last.error is None or (last.error is not None and last.error < float(self.tol_var.get())) or abs(last.fx) < float(self.tol_var.get())
        summary = [
            f"Method used: {method}",
            f"Estimated root: {last.x:.8f}",
            f"Function value at estimate: {last.fx:.3e}",
            f"Iterations performed: {len(self.rows)}",
            f"Final step error: {'-' if last.error is None else f'{last.error:.3e}'}",
            "",
        ]
        if method == "Bisection":
            summary.append(
                "Interpretation: Bisection moved safely inside a valid bracket. You can see the midpoint approaching the x-axis while the interval effectively shrinks by half each step."
            )
        elif method == "Newton-Raphson":
            summary.append(
                "Interpretation: Newton-Raphson used tangent-line updates. Near the solution, the error usually dropped quickly, which is why the convergence plot becomes steep."
            )
        elif method == "Secant":
            summary.append(
                "Interpretation: Secant approximated the tangent by using the last two points. This often gives fast convergence without evaluating a derivative explicitly."
            )
        else:
            summary.append(
                "Interpretation: Fixed-Point iteration depends on the chosen form g(x). The plot and iteration table help you judge whether the chosen formula is converging or not."
            )
        summary.append("")
        summary.append(f"Convergence status: {'Likely converged successfully.' if converged else 'Stopped before clear convergence.'}")
        self._set_text(self.summary_text, "\n".join(summary))


def main():
    root = tk.Tk()
    app = RootVisionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
