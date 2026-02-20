from election_sim.phase3.voting import _extract_profiles_loose


def test_extract_profiles_loose_handles_valid_json():
    raw = """```json
{
  "profiles": [
    {
      "china": "c1",
      "healthcare": "h1",
      "guns": "g1"
    },
    {
      "china": "c2",
      "healthcare": "h2",
      "guns": "g2"
    }
  ]
}
```"""
    rows = _extract_profiles_loose(raw)
    assert len(rows) == 2
    assert rows[0]["china"] == "c1"
    assert rows[1]["guns"] == "g2"


def test_extract_profiles_loose_salvages_truncated_json():
    raw = """```json
{
  "profiles": [
    {
      "china": "c1",
      "healthcare": "h1",
      "guns": "g1"
    },
    {
      "china": "c2",
      "healthcare": "h2",
      "guns": "g2"
    },
    {
      "china": "broken",
      "healthcare": "broken"
```"""
    rows = _extract_profiles_loose(raw)
    assert len(rows) == 2
    assert rows[0]["healthcare"] == "h1"
    assert rows[1]["china"] == "c2"
