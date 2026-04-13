"""Shared matplotlib font sizing for offline session plots."""

PLOT_FONT_SCALE = 2.5

DEFAULT_TITLE_FONTSIZE = 12.0 * PLOT_FONT_SCALE
COMPACT_TITLE_FONTSIZE = 10.0 * PLOT_FONT_SCALE
SUPTITLE_FONTSIZE = 14.0 * PLOT_FONT_SCALE
AXIS_LABEL_FONTSIZE = 10.0 * PLOT_FONT_SCALE
TICK_LABEL_FONTSIZE = 10.0 * PLOT_FONT_SCALE
LEGEND_FONTSIZE = 10.0 * PLOT_FONT_SCALE * 0.7
CONE_STATS_FONTSIZE = 9.0 * PLOT_FONT_SCALE


def apply_axis_label_fontsize(ax) -> None:
    """Scale the x/y axis label text."""
    ax.xaxis.label.set_size(AXIS_LABEL_FONTSIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_FONTSIZE)


def apply_tick_label_fontsize(ax) -> None:
    """Scale the numeric tick labels on both axes."""
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='minor', labelsize=TICK_LABEL_FONTSIZE)
