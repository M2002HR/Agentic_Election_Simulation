from election_sim.phase2.debate import _parse_question_list


def test_parse_question_list_handles_fenced_json():
    raw = """```json
{
  "questions": [
    "What tradeoff will your policy make in year one?",
    "Give a costed timeline with measurable outcomes for healthcare reform."
  ]
}
```"""
    out = _parse_question_list(raw, n=2, title="Healthcare")
    assert len(out) == 2
    assert "tradeoff" in out[0].lower() or "timeline" in out[1].lower()


def test_parse_question_list_filters_invalid_lines():
    raw = """```json
{
"questions": [
```"""
    out = _parse_question_list(raw, n=3, title="China")
    assert len(out) == 3
    assert all(len(x) > 10 for x in out)
