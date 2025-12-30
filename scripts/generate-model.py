"""
Generate Pydantic models from Jupyter Notebook JSON schema.
Cross-platform version of gen.sh
"""
import re
import subprocess
import sys
from pathlib import Path


def find_schema_file():
    """Find the nbformat schema file in the virtual environment."""
    # Try common virtual environment locations
    venv_paths = [
        Path('.venv'),
        Path('venv'),
    ]

    for venv_path in venv_paths:
        if not venv_path.exists():
            continue

        # Check for both Windows and POSIX paths
        site_packages_patterns = [
            venv_path / 'Lib' / 'site-packages',  # Windows
            venv_path / 'lib',  # POSIX (will need to glob for python version)
        ]

        for sp_pattern in site_packages_patterns:
            if sp_pattern.name == 'lib' and sp_pattern.exists():
                # POSIX: need to find python* directory
                for python_dir in sp_pattern.glob('python*'):
                    site_packages = python_dir / 'site-packages'
                    schema_file = site_packages / 'nbformat' / 'v4' / 'nbformat.v4.schema.json'
                    if schema_file.exists():
                        return schema_file
            elif sp_pattern.exists():
                # Windows: direct path to site-packages
                schema_file = sp_pattern / 'nbformat' / 'v4' / 'nbformat.v4.schema.json'
                if schema_file.exists():
                    return schema_file

    return None


def run_datamodel_codegen(schema_file: Path, output_file: Path):
    """Run datamodel-codegen to generate the model."""
    cmd: list[str] = [
        'datamodel-codegen',
        '--input', str(schema_file),
        '--input-file-type', 'jsonschema',
        '--output', str(output_file),
        '--output-model-type', 'pydantic_v2.BaseModel',
        '--formatters', 'ruff-check', 'ruff-format',
        '--target-python-version', '3.11',
        '--enum-field-as-literal', 'all',
        '--disable-future-imports',
        '--use-annotated',
        '--use-schema-description',
        '--use-unique-items-as-set',
        '--use-field-description',
        '--class-name', 'JupyterNotebook',
        '--collapse-root-models',
        '--reuse-model',
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error running datamodel-codegen:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    print("Model generated successfully")


def post_process_model(output_file: Path):
    """Apply post-processing replacements to the generated model."""
    content = output_file.read_text(encoding='utf-8')

    # Define replacements
    replacements = [
        # Rename Metadata classes
        (r'^class Metadata1\(BaseModel\):', 'class RawCellMetadata(BaseModel):'),
        (r'^class Metadata2\(BaseModel\):', 'class MarkdownCellMetadata(BaseModel):'),
        (r'^class Metadata3\(BaseModel\):', 'class CodeCellMetadata(BaseModel):'),
        (r'^class Metadata4\(BaseModel\):', 'class UnrecognizedCellMetadata(BaseModel):'),
        # Update references to Metadata classes
        (r'metadata: Metadata1$', 'metadata: RawCellMetadata'),
        (r'metadata: Metadata2$', 'metadata: MarkdownCellMetadata'),
        (r'metadata: Metadata3$', 'metadata: CodeCellMetadata'),
        (r'metadata: Metadata4$', 'metadata: UnrecognizedCellMetadata'),
    ]

    # Apply replacements
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    # Remove Misc type alias and following two lines
    # Match both TypeAliasType and type syntax
    content = re.sub(
        r'^(?:Misc = TypeAliasType\("Misc", Any\)|type Misc = Any)\n.*\n.*\n',
        '',
        content,
        flags=re.MULTILINE
    )

    output_file.write_text(content, encoding='utf-8')
    print("Post-processing completed")


def main():
    """Main entry point."""
    # Find schema file
    schema_file = find_schema_file()
    if schema_file is None:
        print("Error: Could not find nbformat schema file", file=sys.stderr)
        print("Please ensure nbformat is installed in a virtual environment", file=sys.stderr)
        sys.exit(1)

    print(f"Found schema file: {schema_file}")

    # Define output file
    output_file = Path('src/jupymcp/model.py')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Generate model
    run_datamodel_codegen(schema_file, output_file)

    # Post-process
    post_process_model(output_file)

    print(f"Successfully generated {output_file}")


if __name__ == '__main__':
    main()
