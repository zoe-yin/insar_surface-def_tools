"""Tiny fallback for when the ESA acquisition-plan scrape isn't available: guess each
track's next pass as (last acquisition + repeat_days). No network, no schedule file.
"""
import pandas as pd


def guess_next_acquisitions(df, repeat_days=12, n_future=1, track_col="Path Number",
                             time_col="Start Time"):
    """
    For each track in `df`, guess its next `n_future` acquisition date(s) as
    (most recent Start Time) + repeat_days, 2*repeat_days, ...

    df          : your ASF search results (must have `track_col` and `time_col`).
    repeat_days : assumed repeat interval. Pass a dict {track: days} if it
                  differs by track (e.g. during constellation changes).
    n_future    : how many future dates to guess per track.

    Returns one row per (track, guess) with columns:
        [track_col, "last_start", "guess_n", "guessed_start"]
    """
    rows = []
    for track, g in df.groupby(track_col):
        last = g[time_col].max()
        days = repeat_days[track] if isinstance(repeat_days, dict) else repeat_days
        for n in range(1, n_future + 1):
            rows.append({
                track_col: track,
                "last_start": last,
                "guess_n": n,
                "guessed_start": last + pd.Timedelta(days=days * n),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "search_results_s1.csv")
    df["Start Time"] = pd.to_datetime(df["Start Time"])
    print(guess_next_acquisitions(df, repeat_days=12).to_string(index=False))