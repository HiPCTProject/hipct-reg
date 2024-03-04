from pathlib import Path

import click

from hipct_reg.ITK_registration import registration_pipeline


class PointType(click.ParamType):
    name = "point"

    def convert(
        self, value: str, param: click.Parameter | None, ctx: click.Context | None
    ) -> tuple[int, int, int]:
        vals = value.split()
        if len(value) != 3:
            self.fail(f"Did not find three values: {value}", param, ctx)

        return (int(value[0]), int(value[1]), int(value[2]))


POINT = PointType()


@click.command()
@click.option(
    "--path_roi",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to ROI dataset",
)
@click.option(
    "--path_full",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to ROI dataset",
)
@click.option(
    "--pt_roi",
    required=True,
    type=POINT,
    help="Common point in ROI image",
)
@click.option(
    "--pt_full",
    required=True,
    type=POINT,
    help="Common point in full image",
)
def registration_pipeline_cli(
    path_roi: Path,
    path_full: Path,
    pt_roi: tuple[int, int, int],
    pt_full: tuple[int, int, int],
) -> None:
    transform = registration_pipeline(
        path_roi=path_roi, path_full=path_full, pt_roi=pt_roi, pt_full=pt_full
    )


if __name__ == "__main__":
    registration_pipeline_cli()
