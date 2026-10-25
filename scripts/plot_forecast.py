#!/usr/bin/env python3
"""Plot an HTTP forecast's median and p10-p90 band alongside its history."""
import argparse
import json
import math
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def plot_forecast(request, response, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = [math.nan if value is None else value for value in request["history"]]
    horizon = request["horizon"]
    quantiles = response["quantiles"]
    lower, median, upper = (quantiles[level] for level in ("0.1", "0.5", "0.9"))
    if not history or any(len(values) != horizon for values in (lower, median, upper)):
        raise ValueError("Expected nonempty history and horizon values for each quantile")
    future_days = range(len(history) + 1, len(history) + horizon + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(history) + 1), history, "o-", color="0.3",
            markersize=3, label="Observed history")
    ax.fill_between(future_days, lower, upper, color="tab:blue", alpha=0.2,
                    label="p10–p90 forecast band (80%)")
    ax.plot(future_days, median, "o-", color="tab:blue", markersize=4,
            label="Median forecast")
    ax.axvline(len(history) + 0.5, color="0.5", linestyle="--", linewidth=1)
    ax.set(title=f"{response.get('model', 'Timestral')} forecast", xlabel="Time step (day in sales example)",
           ylabel="Value (units sold in sales example)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--request", type=Path,
                        default=Path(__file__).resolve().parents[1] / "examples/sales_request.json")
    parser.add_argument("--response", type=Path,
                        help="Plot a saved response instead of calling the server")
    parser.add_argument("--output", type=Path, default=Path("output/sales-forecast.png"))
    args = parser.parse_args()
    try:
        request = json.loads(args.request.read_text())
        # The chart always needs these three levels, even for a custom request.
        request["quantiles"] = [0.1, 0.5, 0.9]
        if args.response:
            response = json.loads(args.response.read_text())
        else:
            http_request = Request(
                args.url.rstrip("/") + "/v1/forecast",
                data=json.dumps(request, allow_nan=False).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(http_request, timeout=60) as result:
                response = json.load(result)
        plot_forecast(request, response, args.output)
    except HTTPError as error:
        with error:
            parser.exit(1, f"Server returned HTTP {error.code}: {error.read().decode()}\n")
    except (OSError, URLError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Could not plot forecast: {error}\n")
    except ModuleNotFoundError as error:
        parser.exit(1, f"Missing plotting dependency: {error}. Install matplotlib in your Python environment.\n")
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
