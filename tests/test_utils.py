import pytest

import utils


def test_find_raptor_checkpoint_single_match(tmp_path):
    models_dir = tmp_path
    expected = models_dir / "final_raptor3_any_seed_42_step_123.pt"
    expected.write_text("ok")

    result = utils.find_raptor_checkpoint("raptor3", 42, str(models_dir))

    assert result == str(expected)


def test_find_raptor_checkpoint_multiple_matches(tmp_path):
    models_dir = tmp_path
    (models_dir / "final_raptor2_a_seed_7_step_1.pt").write_text("ok")
    (models_dir / "final_raptor2_b_seed_7_step_2.pt").write_text("ok")

    with pytest.raises(ValueError):
        utils.find_raptor_checkpoint("raptor2", 7, str(models_dir))


def test_find_raptor_checkpoint_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        utils.find_raptor_checkpoint("raptor4", 999, str(tmp_path))
