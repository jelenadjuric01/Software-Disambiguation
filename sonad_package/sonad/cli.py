import click
from .core import process_files
from .config import configure_token, get_token
@click.group(help="""
SONAD: Software Name Disambiguation Tool

Available commands:

  process    Process input CSV file and generate disambiguation results
             Options:
               -i, --input TEXT     Input CSV file path (required)
               -o, --output TEXT    Output CSV file path (required)
               -t, --temp-dir TEXT  Optional temporary directory path

  configure  Configure GitHub token for API access
""")
def cli():
    pass

@cli.command()
@click.option('-i', '--input', required=True, help='Input CSV file path')
@click.option('-o', '--output', required=True, help='Output CSV file path')
@click.option('-t', '--temp-dir', default=None, help='Optional temporary directory path')
def process(input, output, temp_dir):
    """Process input file and generate disambiguation results"""
    token = get_token()
    process_files(input_path=input, output_path=output, folder_path=temp_dir, github_token=token)

@cli.command()
def configure():
    """Configure GitHub token for API access"""
    configure_token()

if __name__ == '__main__':
    cli()