"""Offline plot generation runtime for logger_node."""

from __future__ import annotations


class PlotRunner:

    def _generate_offline_plots(self) -> None:
        """Generate offline plots from logged data."""
        total = 0
        self._safe_log_info("Generating offline plots...")

        try:
            from ..plotting.offline_plotter import OfflinePlotter
            plotter = OfflinePlotter(self._run_session.session_path, output_format='pdf')
            generated = plotter.generate_plots()
            total += len(generated)
        except ImportError as e:
            self._safe_log_warn(f"Could not import offline plotter: {e}")
        except Exception as e:
            self._safe_log_warn(f"Failed to generate vehicle offline plots: {e}")

        try:
            from ..plotting.offline_cone_plotter import OfflineConePlotter
            cone_plotter = OfflineConePlotter(self._run_session.session_path, output_format='pdf')
            generated_paths = cone_plotter.generate_all_range_rmse_plots(delete_legacy_png=True)
            total += len(generated_paths)
            for output_path in generated_paths:
                self._safe_log_info(f"Generated cone range RMSE offline plot: {output_path}")
        except ImportError as e:
            self._safe_log_warn(f"Could not import cone offline plotter: {e}")
            msg = str(e).lower()
            if 'numpy' in msg or 'multiarray' in msg or '_array_api' in msg:
                self._safe_log_warn(
                    "Detected NumPy/Matplotlib binary mismatch. "
                    "Reinstall compatible versions to enable cone range RMSE plot generation."
                )
        except Exception as e:
            self._safe_log_warn(f"Failed to generate cone offline plots: {e}")

        self._safe_log_info(f"Generated {total} plots in {self._run_session.plots_path}")
