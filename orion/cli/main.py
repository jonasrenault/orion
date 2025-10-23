import logging

import typer
from rich.logging import RichHandler

from orion.datasets.prepare import app as prepare_app
from orion.yolo.yolo import app as yolo_app

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler(markup=True)]
)

app = typer.Typer(no_args_is_help=True)

app.add_typer(prepare_app)
app.add_typer(yolo_app)

if __name__ == "__main__":
    app()
