from election_sim.phase3.voters import generate_voters


def test_generate_voters_with_distributions_and_seeded_assignment():
    value_pool = [
        {"profile_id": "vp_a", "china": "A", "healthcare": "B", "guns": "C"},
        {"profile_id": "vp_b", "china": "D", "healthcare": "E", "guns": "F"},
    ]
    voters = generate_voters(
        count=50,
        trait_names=["wisdom", "fear", "anger", "adaptability", "distrust"],
        seed=123,
        trait_distributions={
            "wisdom": {"low": 10, "medium": 20, "high": 70},
            "fear": {"low": 30, "medium": 40, "high": 30},
        },
        value_pool=value_pool,
        assignment_mode="seeded_random",
    )
    assert len(voters) == 50
    assert all(v.value_profile_id in {"vp_a", "vp_b"} for v in voters)
    high_wisdom = sum(1 for v in voters if v.traits["wisdom"] >= 7)
    assert high_wisdom >= 25  # loose check for 70% high with stochastic sampling


def test_generate_voters_respects_distribution_counts_exactly():
    voters = generate_voters(
        count=200,
        trait_names=["wisdom", "fear", "anger", "adaptability", "distrust"],
        seed=42,
        trait_distributions={
            "wisdom": {"low": 60, "medium": 30, "high": 10},
            "fear": {"low": 30, "medium": 40, "high": 30},
        },
        value_pool=[{"profile_id": "vp_0", "china": "c", "healthcare": "h", "guns": "g"}],
        assignment_mode="seeded_random",
    )

    wisdom_low = sum(1 for v in voters if v.traits["wisdom"] <= 3)
    wisdom_medium = sum(1 for v in voters if 4 <= v.traits["wisdom"] <= 6)
    wisdom_high = sum(1 for v in voters if v.traits["wisdom"] >= 7)
    assert (wisdom_low, wisdom_medium, wisdom_high) == (120, 60, 20)

    fear_low = sum(1 for v in voters if v.traits["fear"] <= 3)
    fear_medium = sum(1 for v in voters if 4 <= v.traits["fear"] <= 6)
    fear_high = sum(1 for v in voters if v.traits["fear"] >= 7)
    assert (fear_low, fear_medium, fear_high) == (60, 80, 60)


def test_generate_voters_seeded_random_uses_unique_profiles_when_possible():
    value_pool = [
        {"profile_id": f"vp_{i:03d}", "china": f"c{i}", "healthcare": f"h{i}", "guns": f"g{i}"}
        for i in range(20)
    ]
    voters = generate_voters(
        count=20,
        trait_names=["wisdom", "fear", "anger", "adaptability", "distrust"],
        seed=7,
        trait_distributions={},
        value_pool=value_pool,
        assignment_mode="seeded_random",
    )
    profile_ids = [v.value_profile_id for v in voters]
    assert len(set(profile_ids)) == 20
