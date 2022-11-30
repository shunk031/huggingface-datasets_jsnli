import datasets as ds
import pytest


@pytest.fixture
def dataset_path() -> str:
    return "jsnli.py"


@pytest.mark.parametrize(
    "dataset_name, expected_num_train,",
    (
        ("wo_filtering", 548014),
        ("w_filtering", 533005),
    ),
)
def test_load_dataset(
    dataset_path: str,
    dataset_name: str,
    expected_num_train: int,
    expected_num_valid: int = 3916,
):
    dataset = ds.load_dataset(path=dataset_path, name=dataset_name)

    assert dataset["train"].num_rows == expected_num_train  # type: ignore
    assert dataset["validation"].num_rows == expected_num_valid  # type: ignore
