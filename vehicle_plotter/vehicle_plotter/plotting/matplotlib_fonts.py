"""Shared matplotlib font sizing for offline session plots."""

SERIF_FONT_FAMILY = [
    "Georgia",
    "DejaVu Serif",
    "Times New Roman",
    "Times",
    "serif",
]

DEFAULT_TITLE_FONTSIZE = 11.0
COMPACT_TITLE_FONTSIZE = 10.0
SUPTITLE_FONTSIZE = 12.0
AXIS_LABEL_FONTSIZE = 10.0
TICK_LABEL_FONTSIZE = 10.0
LEGEND_FONTSIZE = 10.0
CONE_STATS_FONTSIZE = 10.0


def apply_serif_font_preferences() -> None:
    """Configure matplotlib to prefer a Georgia-like serif stack."""
    import matplotlib

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": SERIF_FONT_FAMILY,
            "mathtext.fontset": "dejavuserif",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def apply_axis_label_fontsize(ax) -> None:
    """Scale the x/y axis label text."""
    ax.xaxis.label.set_size(AXIS_LABEL_FONTSIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_FONTSIZE)


def apply_tick_label_fontsize(ax) -> None:
    """Scale the numeric tick labels on both axes."""
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='minor', labelsize=TICK_LABEL_FONTSIZE)
