import argparse
import logging

from pipeline.config import load_pipeline_config
from pipeline.pipeline import run_pipeline


logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the QuicKB pipeline.")
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help=(
            "Path to the pipeline configuration file. Defaults to 'config.yaml' in the "
            "current working directory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    config = load_pipeline_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI feedback
        logger.error("Fatal error: %s", exc)
        raise