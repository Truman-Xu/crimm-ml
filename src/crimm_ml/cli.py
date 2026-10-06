"""`crimm-ml` command line: manage the model and schedule files.

    crimm-ml assets fetch [--from DIR_OR_URL] [--force]
    crimm-ml assets verify
    crimm-ml assets path
    crimm-ml assets warm [--T 15 50]
"""
import argparse
import logging
import sys

from . import __version__, assets
from .errors import CrimmMLError


def main(argv=None):
    ap = argparse.ArgumentParser(prog="crimm-ml", description="crimm-ml model and cache management")
    ap.add_argument("--version", action="version", version=f"crimm-ml {__version__}")
    sub = ap.add_subparsers(dest="group", required=True)
    a = sub.add_parser("assets", help="the model file and the schedule tables").add_subparsers(
        dest="cmd", required=True)
    f = a.add_parser("fetch", help="download the model into the cache (or copy it with --from)")
    f.add_argument("--from", dest="source", default=None,
                   help="a directory holding the model files, or a base URL")
    f.add_argument("--force", action="store_true", help="replace files that are already cached")
    a.add_parser("verify", help="check sizes and SHA-256 checksums of the cached model")
    a.add_parser("path", help="print the cache locations")
    w = a.add_parser("warm", help="build the IGSO(3) schedule tables now (about a minute per T)")
    w.add_argument("--T", type=int, nargs="+", default=[15, 50], help="step counts (default: 15 50)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        if args.cmd == "fetch":
            print(assets.fetch(source=args.source, force=args.force))
        elif args.cmd == "verify":
            problems = assets.verify()
            for p in problems:
                print(p, file=sys.stderr)
            print("ok" if not problems else "FAILED")
            return 1 if problems else 0
        elif args.cmd == "path":
            print(f"cache:     {assets.cache_dir()}")
            print(f"model:     {assets.model_dir()}")
            print(f"schedules: {assets.schedule_dir()}")
        elif args.cmd == "warm":
            from .minimal_loop_sampler import warm_schedules
            from .models import load_model
            model = load_model(assets.model_path(), device="cpu", threads=1)
            print(warm_schedules(model, args.T))
    except CrimmMLError as e:
        print(f"crimm-ml: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
