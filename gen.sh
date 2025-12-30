datamodel-codegen \
  --input .venv/lib/python3.11/site-packages/nbformat/v4/nbformat.v4.schema.json \
  --input-file-type jsonschema \
  --output src/jupymcp/model.py \
  --output-model-type pydantic_v2.BaseModel \
  --formatters ruff-check ruff-format\
  --target-python-version 3.11 \
  --enum-field-as-literal all \
  --disable-future-imports \
  --use-annotated \
  --use-schema-description \
  --use-unique-items-as-set \
  --use-field-description \
  --class-name JupyterNotebook \
  --collapse-root-models \
  --reuse-model

sed -i '' -E \
  -e 's/^class Metadata1\(BaseModel\):/class RawCellMetadata(BaseModel):/' \
  -e 's/^class Metadata2\(BaseModel\):/class MarkdownCellMetadata(BaseModel):/' \
  -e 's/^class Metadata3\(BaseModel\):/class CodeCellMetadata(BaseModel):/' \
  -e 's/^class Metadata4\(BaseModel\):/class UnrecognizedCellMetadata(BaseModel):/' \
  -e 's/metadata: Metadata1$/metadata: RawCellMetadata/' \
  -e 's/metadata: Metadata2$/metadata: MarkdownCellMetadata/' \
  -e 's/metadata: Metadata3$/metadata: CodeCellMetadata/' \
  -e 's/metadata: Metadata4$/metadata: UnrecognizedCellMetadata/' \
  -e '/^(Misc = TypeAliasType\("Misc", Any\)|type Misc = Any)$/{' \
  -e 'N' \
  -e 'N' \
  -e 'd' \
  -e '}' \
  src/jupymcp/model.py
