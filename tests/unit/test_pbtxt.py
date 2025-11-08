from tsbk.utils.pbtxt import (
    find_matching_brace,
    parse_pbtxt,
    parse_section,
    parse_section_body,
)


def test_find_matching_brace():
    assert find_matching_brace("{}", 0) == 1
    assert find_matching_brace("{{}}", 0) == 3


def test_parse_section_body():
    assert parse_section_body("  text  ") == "text"
    assert parse_section_body("\n  text  \n") == "text"

    assert parse_section_body(" {name: value} ") == {"name": "value"}
    assert parse_section_body(" {name: value  name2: value2} ") == {
        "name": "value",
        "name2": "value2",
    }
    assert parse_section_body("\n  {\nname: value\n  name2: value2}  \n") == {
        "name": "value",
        "name2": "value2",
    }

    assert parse_section_body(' [ {\n    name: "MODEL_NAME"\n   dims: [ 1, 2]\n  }\n]') == [
        {"name": "MODEL_NAME", "dims": [1, 2]}
    ]
    assert parse_section_body(
        ' [{ name: "MODEL_NAME"  \n  data_type: TYPE_STRING  }\n , '
        '{ name: "MODEL_NAME" \n   data_type: TYPE_STRING  }]'
    ) == [
        {"name": "MODEL_NAME", "data_type": "TYPE_STRING"},
        {"name": "MODEL_NAME", "data_type": "TYPE_STRING"},
    ]
    assert parse_section_body("[]") == []
    assert parse_section_body("[ ]") == []

    assert parse_section_body('  {name: "INPUT0"\n dims: [ 16 ]\n data_type: TYPE_STRING\n}\n') == {
        "name": "INPUT0",
        "dims": [16],
        "data_type": "TYPE_STRING",
    }

    assert parse_section_body(
        '{\n      model_name: "add1"\n      model_version: -1\n      input_map {\n        key: "a"\n\t    value: "a"\n      },\n      output_map {\n        key: "b"\n\t    value: "b"\n      }\n    }'
    ) == {
        "model_name": "add1",
        "model_version": -1,
        "input_map": {"key": "a", "value": "a"},
        "output_map": {"key": "b", "value": "b"},
    }


def test_parse_section():
    assert parse_section("###Some comment") is None

    assert parse_section("  text: value") == ("text", "value")
    assert parse_section(" \n text: value\n") == ("text", "value")

    assert parse_section("property: {name: value} ") == ("property", {"name": "value"})
    assert parse_section("\nproperty:  \n{\n  name: value  \n  name2: value2}  \n  ") == (
        "property",
        {"name": "value", "name2": "value2"},
    )

    assert parse_section('property [ {\n    name: "MODEL_NAME"\n   dims: [ 1, 2]\n  }\n]') == (
        "property",
        [{"name": "MODEL_NAME", "dims": [1, 2]}],
    )
    assert parse_section('property \n[ {    name: "MODEL_NAME"   dims: [ 1, 2]  }]') == (
        "property",
        [{"name": "MODEL_NAME", "dims": [1, 2]}],
    )
    assert parse_section(
        'parameters: {key: "EXECUTION_ENV_PATH" value: {string_value: "$$TRITON_MODEL_DIRECTORY/some-hash.tar.gz"}}\n'
    ) == (
        "parameters",
        {"key": "EXECUTION_ENV_PATH", "value": {"string_value": "$$TRITON_MODEL_DIRECTORY/some-hash.tar.gz"}},
    )

    assert parse_section(
        'output [\n\t{\n\t\tname: "label"\n\t\tdata_type: TYPE_STRING\n\t\tdims: [1]\n\t},\n\t{\n\t\tname: "score"\n\t\tdata_type: TYPE_FP32\n\t\tdims: [1]\n\t}\n]'
    ) == (
        "output",
        [
            {"name": "label", "data_type": "TYPE_STRING", "dims": [1]},
            {"name": "score", "data_type": "TYPE_FP32", "dims": [1]},
        ],
    )


def test_parse_pbtxt_file1(assets_dir):
    path = assets_dir.joinpath("pbtxts/bls.pbtxt")
    config = parse_pbtxt(file_path=path)
    assert config == parse_pbtxt(content=path.read_text())

    assert config["name"] == "bls"
    assert config["backend"] == "python"
    assert len(config["input"]) == 3
    assert config["input"][0]["name"] == "MODEL_NAME"
    assert config["input"][1]["name"] == "INPUT0"
    assert config["input"][2]["name"] == "INPUT1"
    assert len(config["output"]) == 2
    assert config["output"][0]["name"] == "OUTPUT0"
    assert config["output"][1]["name"] == "OUTPUT1"
    assert config["instance_group"] == [{"kind": "KIND_CPU"}]


def test_parse_pbtxt_file2(assets_dir):
    path = assets_dir.joinpath("pbtxts/simple.pbtxt")
    config = parse_pbtxt(file_path=path)
    assert config == parse_pbtxt(content=path.read_text())

    assert config["platform"] == "tensorflow_graphdef"
    assert len(config["input"]) == 2
    assert config["input"][0]["name"] == "INPUT0"
    assert config["input"][1]["name"] == "INPUT1"
    assert len(config["output"]) == 2
    assert config["output"][0]["name"] == "OUTPUT0"
    assert config["output"][1]["name"] == "OUTPUT1"


def test_parse_pbtxt_file3(assets_dir):
    path = assets_dir.joinpath("pbtxts/repeat.pbtxt")
    config = parse_pbtxt(file_path=path)
    assert config == parse_pbtxt(content=path.read_text())

    assert config["name"] == "repeat"
    assert config["backend"] == "python"
    assert "model_transaction_policy" in config
    assert isinstance(config["model_transaction_policy"], dict)
    assert config["model_transaction_policy"]["decoupled"] == "True"


def test_parse_ensemble_pbtxt(assets_dir):
    path = assets_dir.joinpath("ensemble/ensemble.pbtxt")
    config = parse_pbtxt(file_path=path)
    assert config == parse_pbtxt(content=path.read_text())

    assert config["platform"] == "ensemble"
    assert config["ensemble_scheduling"] == {
        "step": [
            {
                "input_map": {"key": "a", "value": "a"},
                "model_name": "add1",
                "model_version": -1,
                "output_map": {"key": "b", "value": "b"},
            },
            {
                "input_map": {"key": "b", "value": "b"},
                "model_name": "sub1",
                "model_version": -1,
                "output_map": {"key": "c", "value": "c"},
            },
        ]
    }
