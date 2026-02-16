import os
import argparse
from pathlib import Path

from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_args():
    p = argparse.ArgumentParser(
        description="Query Materials Project and download structures as CIF files into ./cif"
    )
    p.add_argument(
        "--api-key",
        default=os.getenv("MP_API_KEY", ""),
        help="Materials Project API key (or set env MP_API_KEY)",
    )

    # --- Query controls ---
    p.add_argument(
        "--elements",
        default="Cu",
        help='Elements that MUST be present (comma-separated). e.g. "Cu,O" means contains at least Cu and O.',
    )
    p.add_argument(
        "--chemsys",
        default="",
        help='Chemical system string, e.g. "Cu-O" or "Cu-O-Zn". If provided, it overrides elements matching.',
    )
    p.add_argument(
        "--exclude-elements",
        default="",
        help='Elements that must NOT be present (comma-separated). e.g. "Hg,Cd".',
    )
    p.add_argument(
        "--is-stable",
        action="store_true",
        help="Only stable materials on the (GGA/GGA+U) hull (summary.is_stable==True).",
    )
    p.add_argument(
        "--energy-above-hull-max",
        type=float,
        default=None,
        help="Keep only entries with energy_above_hull <= this value (eV/atom). Example: 0.05",
    )
    p.add_argument(
        "--band-gap-min",
        type=float,
        default=None,
        help="band_gap >= this value (eV)",
    )
    p.add_argument(
        "--band-gap-max",
        type=float,
        default=None,
        help="band_gap <= this value (eV)",
    )
    p.add_argument(
        "--num-elements-max",
        type=int,
        default=None,
        help="Max number of elements (e.g. 3 for ternaries).",
    )

    # --- Download controls ---
    p.add_argument(
        "--outdir",
        default="cif",
        help='Output directory name. Default: "cif".',
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of materials to download (0 means no limit).",
    )
    p.add_argument(
        "--conventional",
        action="store_true",
        help="Use conventional standard cell when exporting CIF (often easier to read).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip downloading if CIF file already exists.",
    )
    return p.parse_args()


def to_list(csv: str):
    csv = csv.strip()
    if not csv:
        return []
    return [x.strip() for x in csv.split(",") if x.strip()]


def main():
    args = parse_args()

    if not args.api_key:
        raise SystemExit(
            "Missing API key. Provide --api-key or set environment variable MP_API_KEY."
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    elements = to_list(args.elements)
    exclude_elements = to_list(args.exclude_elements)

    # Build search kwargs for mp-api summary endpoint
    search_kwargs = {
        "fields": [
            "material_id",
            "formula_pretty",
            "elements",
            "is_stable",
            "energy_above_hull",
            "band_gap",
        ]
    }

    if args.chemsys.strip():
        search_kwargs["chemsys"] = args.chemsys.strip()
    else:
        # "elements" here means "containing at least these elements"
        search_kwargs["elements"] = elements

    if exclude_elements:
        # mp-api summary.search supports "exclude_elements"
        search_kwargs["exclude_elements"] = exclude_elements

    if args.is_stable:
        search_kwargs["is_stable"] = True

    if args.energy_above_hull_max is not None:
        search_kwargs["energy_above_hull"] = (0, args.energy_above_hull_max)

    if args.band_gap_min is not None or args.band_gap_max is not None:
        bg_min = args.band_gap_min if args.band_gap_min is not None else 0
        bg_max = args.band_gap_max
        search_kwargs["band_gap"] = (bg_min, bg_max)

    if args.num_elements_max is not None:
        search_kwargs["num_elements"] = (1, args.num_elements_max)

    with MPRester(args.api_key) as mpr:
        docs = list(mpr.materials.summary.search(**search_kwargs))

        if args.limit and args.limit > 0:
            docs = docs[: args.limit]

        iterator = docs
        if tqdm is not None:
            iterator = tqdm(docs, desc="Downloading CIFs", unit="mat")

        downloaded = 0
        skipped = 0
        failed = 0

        for d in iterator:
            mpid = d.material_id
            formula = getattr(d, "formula_pretty", mpid)

            cif_path = outdir / f"{mpid}.cif"

            if args.skip_existing and cif_path.exists():
                skipped += 1
                continue

            try:
                structure = mpr.get_structure_by_material_id(
                    mpid, conventional_unit_cell=args.conventional
                )
                w = CifWriter(structure)
                w.write_file(str(cif_path))
                downloaded += 1
            except Exception as e:
                failed += 1
                with open(outdir / "download_errors.txt", "a", encoding="utf-8") as f:
                    f.write(f"{mpid}\t{formula}\t{repr(e)}\n")

        print("\nDone.")
        print(f"Downloaded: {downloaded}")
        print(f"Skipped:    {skipped}")
        print(f"Failed:     {failed}")
        print(f"Output dir: {outdir.resolve()}")


if __name__ == "__main__":
    main()
