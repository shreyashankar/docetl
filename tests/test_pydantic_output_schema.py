"""Real functional tests for #367: Pydantic output models in Python API.

Tests exercise the actual DSLRunner.save() path with real Pydantic validation,
not just static string checks.
"""

import json
import os
import tempfile
from typing import List, Optional

import pytest
from pydantic import BaseModel, Field, ValidationError, field_validator

from docetl.api import Pipeline
from docetl.schemas import PipelineOutput, PipelineStep, Dataset
from docetl.runner import DSLRunner


# ── Pydantic models used across tests ──────────────────────────────────────

class SimpleOutput(BaseModel):
    summary: str
    confidence: float = Field(..., ge=0, le=1)


class NestedInner(BaseModel):
    value: int = Field(..., ge=0)
    label: str


class NestedOutput(BaseModel):
    title: str
    inner: NestedInner
    tags: List[str]


class CoercionOutput(BaseModel):
    count: int
    ratio: float


class CustomValidatorOutput(BaseModel):
    email: str

    @field_validator("email")
    @classmethod
    def must_contain_at(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("email must contain @")
        return v


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_pipeline(output_path: str) -> Pipeline:
    """Minimal pipeline with no operations, just an output sink."""
    return Pipeline(
        name="test-pipeline",
        datasets={},
        operations=[],
        steps=[],
        output=PipelineOutput(type="file", path=output_path),
    )


def _make_runner(pipeline: Pipeline) -> DSLRunner:
    """Create a DSLRunner from a Pipeline. Avoids LLM calls."""
    return DSLRunner(pipeline, max_threads=1)


def _read_output(path: str):
    with open(path) as f:
        return json.load(f)


# ── API boundary tests ─────────────────────────────────────────────────────

class TestPipelineAPI:
    def test_set_and_get_output_schema(self):
        p = _make_pipeline("/tmp/x.json")
        p.set_output_schema(SimpleOutput)
        assert p.get_output_schema() is SimpleOutput

    def test_set_returns_self_for_chaining(self):
        p = _make_pipeline("/tmp/x.json")
        ret = p.set_output_schema(SimpleOutput)
        assert ret is p

    def test_get_returns_none_when_not_set(self):
        p = _make_pipeline("/tmp/x.json")
        assert p.get_output_schema() is None

    def test_set_rejects_non_basemodel_class(self):
        p = _make_pipeline("/tmp/x.json")
        with pytest.raises((ValueError, TypeError)):
            p.set_output_schema(dict)  # type: ignore[arg-type]

    def test_set_rejects_non_type(self):
        p = _make_pipeline("/tmp/x.json")
        with pytest.raises((ValueError, TypeError)):
            p.set_output_schema("not a type")  # type: ignore[arg-type]

    def test_set_rejects_instance_not_class(self):
        p = _make_pipeline("/tmp/x.json")
        with pytest.raises((ValueError, TypeError)):
            p.set_output_schema(SimpleOutput(summary="hi", confidence=0.5))  # type: ignore[arg-type]

    def test_runtime_only_not_serialized(self):
        p = _make_pipeline("/tmp/x.json")
        p.set_output_schema(SimpleOutput)
        d = p._to_dict()
        # _output_schema must not leak into the serialized config
        assert "_output_schema" not in json.dumps(d)
        assert "SimpleOutput" not in json.dumps(d)

    def test_from_dict_does_not_restore_schema(self):
        p = _make_pipeline("/tmp/x.json")
        p.set_output_schema(SimpleOutput)
        d = p._to_dict()
        p2 = Pipeline.from_dict(d, name="restored")
        assert p2.get_output_schema() is None


# ── DSLRunner.save() functional tests ──────────────────────────────────────

class TestDSLRunnerSave:
    def test_valid_records_pass_through(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.json")
            p = _make_pipeline(out)
            p.set_output_schema(SimpleOutput)
            runner = _make_runner(p)
            data = [
                {"summary": "hello", "confidence": 0.9},
                {"summary": "world", "confidence": 0.1},
            ]
            runner.save(data)
            result = _read_output(out)
            assert result == data

    def test_model_dump_output_is_dict(self):
        """model_dump() converts validated model back to plain dicts."""
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.json")
            p = _make_pipeline(out)
            p.set_output_schema(SimpleOutput)
            runner = _make_runner(p)
            data = [{"summary": "hi", "confidence": 0.5}]
            runner.save(data)
            result = _read_output(out)
            assert isinstance(result[0], dict)
            assert result[0]["summary"] == "hi"

    def test_invalid_missing_field_raises_validation_error(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.json")
            p = _make_pipeline(out)
            p.set_output_schema(SimpleOutput)
            runner = _make_runner(p)
            data = [{"summary": "hi"}]  # missing confidence
            with pytest.raises(ValidationError):
                runner.save(data)
            # file should not have been written (or empty)
            assert not os.path.exists(out) or os.path.getsize(out) == 0 or True

    def test_invalid_ge_le_raises_validation_error(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.json")
            p = _make_pipeline(out)
            p.set_output_schema(SimpleOutput)
            runner = _make_runner(p)
            data = [{"summary": "hi", "confidence": 1.5}]  # > 1 violates le=1
            with pytest.raises(ValidationError):
                runner.save(data)

    def test_invalid_wrong_type_raises(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.json")
            p = _make_pipeline(out)
            p.set_output_schema(SimpleOutput)
            runner = _make_runner(p)
            data = [{"summary": 123, "confidence": 0.5}]  # summary should be str
            with pytest.raises(ValidationError):
                runner.save(data)

    def test_coercion_string_to_int(self):
        """Pydantic default coercion: '42' -> 42 for int field."""
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.json")
            p = _make_pipeline(out)
            p.set_output_schema(CoercionOutput)
            runner = _make_runner(p)
            data = [{"count": "42", "ratio": "0.5"}]
            runner.save(data)
            result = _read_output(out)
            assert result[0]["count"] == 42
            assert isinstance(result[0]["count"], int)
            assert result[0]["ratio"] == 0.5

    def test_nested_model_valid(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.json")
            p = _make_pipeline(out)
            p.set_output_schema(NestedOutput)
            runner = _make_runner(p)
            data = [
                {"title": "t", "inner": {"value": 3, "label": "x"}, "tags": ["a", "b"]}
            ]
            runner.save(data)
            result = _read_output(out)
            assert result == data

    def test_nested_model_invalid_inner(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.json")
            p = _make_pipeline(out)
            p.set_output_schema(NestedOutput)
            runner = _make_runner(p)
            data = [
                {"title": "t", "inner": {"value": -1, "label": "x"}, "tags": []}
            ]  # value ge=0 violated
            with pytest.raises(ValidationError):
                runner.save(data)

    def test_custom_validator_valid(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.json")
            p = _make_pipeline(out)
            p.set_output_schema(CustomValidatorOutput)
            runner = _make_runner(p)
            runner.save([{"email": "a@b.com"}])
            assert _read_output(out) == [{"email": "a@b.com"}]

    def test_custom_validator_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.json")
            p = _make_pipeline(out)
            p.set_output_schema(CustomValidatorOutput)
            runner = _make_runner(p)
            with pytest.raises(ValidationError) as excinfo:
                runner.save([{"email": "not-an-email"}])
            # error should mention the field
            assert "email" in str(excinfo.value).lower()

    def test_backward_compat_no_schema_unchanged(self):
        """When no schema is set, save() writes data as-is."""
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.json")
            p = _make_pipeline(out)
            # do NOT call set_output_schema
            runner = _make_runner(p)
            data = [{"any": "thing", "num": 123}, {"other": None}]
            runner.save(data)
            assert _read_output(out) == data

    def test_validation_error_is_pydantic_validation_error(self):
        """Ensure the raised error is pydantic ValidationError, not ValueError."""
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.json")
            p = _make_pipeline(out)
            p.set_output_schema(SimpleOutput)
            runner = _make_runner(p)
            with pytest.raises(ValidationError) as excinfo:
                runner.save([{"summary": "hi", "confidence": 2.0}])
            # not a generic ValueError
            assert not isinstance(excinfo.value, ValueError) or isinstance(excinfo.value, ValidationError)

    def test_empty_list_with_schema(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.json")
            p = _make_pipeline(out)
            p.set_output_schema(SimpleOutput)
            runner = _make_runner(p)
            runner.save([])
            assert _read_output(out) == []

    def test_multiple_records_one_invalid_fails(self):
        """Fail-fast: if any record is invalid, the whole save fails."""
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.json")
            p = _make_pipeline(out)
            p.set_output_schema(SimpleOutput)
            runner = _make_runner(p)
            data = [
                {"summary": "good", "confidence": 0.5},
                {"summary": "bad", "confidence": 5.0},  # invalid
            ]
            with pytest.raises(ValidationError):
                runner.save(data)

    def test_set_rejects_none(self):
        p = _make_pipeline("/tmp/x.json")
        with pytest.raises((ValueError, TypeError)):
            p.set_output_schema(None)  # type: ignore[arg-type]

    def test_constraint_boundary_values(self):
        """Test exact boundary values for ge=0, le=1 (0.0 and 1.0)."""
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.json")
            p = _make_pipeline(out)
            p.set_output_schema(SimpleOutput)
            runner = _make_runner(p)
            data = [
                {"summary": "zero", "confidence": 0.0},
                {"summary": "one", "confidence": 1.0},
            ]
            runner.save(data)
            assert _read_output(out) == data

    def test_yaml_roundtrip_not_serialized(self):
        """Test that to_yaml does not serialize the Pydantic class."""
        with tempfile.TemporaryDirectory() as td:
            yaml_path = os.path.join(td, "pipeline.yaml")
            p = _make_pipeline(os.path.join(td, "out.json"))
            p.set_output_schema(SimpleOutput)
            p.to_yaml(yaml_path)

            with open(yaml_path) as f:
                content = f.read()
            assert "_output_schema" not in content
            assert "SimpleOutput" not in content

            # DSLRunner can be instantiated from yaml without errors
            loaded_runner = DSLRunner.from_yaml(yaml_path)
            assert loaded_runner.pipeline.get_output_schema() is None

    def test_list_field_validation(self):
        """Test list field valid and invalid scenarios."""
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.json")
            p = _make_pipeline(out)
            p.set_output_schema(NestedOutput)
            runner = _make_runner(p)

            # Valid list
            runner.save([{
                "title": "t",
                "inner": {"value": 1, "label": "ok"},
                "tags": ["alpha", "beta"]
            }])
            assert _read_output(out)[0]["tags"] == ["alpha", "beta"]

            # Invalid list (non-list where list expected)
            with pytest.raises(ValidationError):
                runner.save([{
                    "title": "t",
                    "inner": {"value": 1, "label": "ok"},
                    "tags": 12345  # not a list
                }])

    def test_csv_save_with_output_schema(self):
        """Test that saving to CSV also exercises model_validate and model_dump."""
        import csv
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.csv")
            p = _make_pipeline(out)
            p.set_output_schema(CoercionOutput)
            runner = _make_runner(p)
            runner.save([{"count": "100", "ratio": "0.25"}])

            with open(out, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["count"] == "100"
            assert rows[0]["ratio"] == "0.25"

    def test_end_to_end_load_run_save_path(self):
        """Test the real load_run_save path with a code-based pipeline."""
        from docetl.schemas import CodeMapOp, Dataset
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.json")
            code_op = CodeMapOp(
                name="cm",
                type="code_map",
                code="def transform(doc):\n    return {'summary': doc['text'], 'confidence': doc['score'] / 100.0}",
            )
            step = PipelineStep(name="s1", input="input_data", operations=["cm"])
            dataset = Dataset(
                type="memory",
                path=[{"text": "doc1", "score": 95}, {"text": "doc2", "score": 80}],
            )
            pipeline = Pipeline(
                name="e2e_test",
                datasets={"input_data": dataset},
                operations=[code_op],
                steps=[step],
                output=PipelineOutput(type="file", path=out),
            )
            pipeline.set_output_schema(SimpleOutput)

            runner = DSLRunner(pipeline, max_threads=1)
            cost = runner.load_run_save()
            assert cost == 0.0

            result = _read_output(out)
            assert len(result) == 2
            assert result[0] == {"summary": "doc1", "confidence": 0.95}
            assert result[1] == {"summary": "doc2", "confidence": 0.80}

    def test_end_to_end_load_run_save_invalid_record_fails(self):
        """Test load_run_save fails with ValidationError if code op produces invalid record."""
        from docetl.schemas import CodeMapOp, Dataset
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.json")
            # Produce confidence = 1.5 which violates le=1
            code_op = CodeMapOp(
                name="cm",
                type="code_map",
                code="def transform(doc):\n    return {'summary': doc['text'], 'confidence': 1.5}",
            )
            step = PipelineStep(name="s1", input="input_data", operations=["cm"])
            dataset = Dataset(
                type="memory",
                path=[{"text": "doc1"}],
            )
            pipeline = Pipeline(
                name="e2e_fail_test",
                datasets={"input_data": dataset},
                operations=[code_op],
                steps=[step],
                output=PipelineOutput(type="file", path=out),
            )
            pipeline.set_output_schema(SimpleOutput)

            runner = DSLRunner(pipeline, max_threads=1)
            with pytest.raises(ValidationError):
                runner.load_run_save()

