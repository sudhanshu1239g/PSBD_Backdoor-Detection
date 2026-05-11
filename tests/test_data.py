from pathlib import Path

from psbd_nlp.data import load_samples_csv


def test_load_samples_csv_parses_boolean_strings(tmp_path: Path):
    csv_path = tmp_path / "samples.csv"
    csv_path.write_text("text,label,is_poisoned\nhello,1,false\ntrigger,0,true\n", encoding="utf-8")

    samples = load_samples_csv(csv_path)

    assert samples[0].is_poisoned is False
    assert samples[1].is_poisoned is True
