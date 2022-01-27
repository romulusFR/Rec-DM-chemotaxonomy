import pytest
import downloader_v2 as dl


COMPOUNDS = ["acridine", "anthraquinone", "tetraterpene/carotenoid", "triterpene"]
ACTIVITIES = ["germination", "herbicidal", "cytotoxicity", "sedative"]
REF_COMPOUNDS_2_OR = '((KEY("acridine")) OR (KEY("anthraquinone")))'
REF_ACTIVITIES_2_OR = '((KEY("germination")) OR (KEY("herbicidal")))'
REF_COMPOUNDS_2_AND= '((KEY("acridine")) AND (KEY("anthraquinone")))'
REF_COMPOUND_1 = '((KEY("acridine")))'


class TestClausalQuery:
    def test_compound_one(self):
        res = dl.clausal_query(COMPOUNDS[:1], [], [], [])
        assert res == REF_COMPOUND_1

    def test_compound_split(self):
        res = dl.clausal_query(COMPOUNDS[2:3], [], [], [])
        assert res == '((KEY("tetraterpene") OR KEY("carotenoid")))'

    def test_compound_two(self):
        res = dl.clausal_query(COMPOUNDS[:2], [], [], [])
        assert res == REF_COMPOUNDS_2_OR

    def test_activities(self):
        res = dl.clausal_query([], ACTIVITIES[:2], [], [])
        assert res == REF_ACTIVITIES_2_OR

    def test_compounds_and_activities(self):
        res = dl.clausal_query(COMPOUNDS[:2], ACTIVITIES[:2], [], [])
        assert res == f"{REF_COMPOUNDS_2_OR} AND {REF_ACTIVITIES_2_OR}"

    def test_pos_kw_empty(self):
        res = dl.clausal_query([], [], COMPOUNDS[:1], [])
        assert res == REF_COMPOUND_1

    def test_pos_kw_empty_2(self):
        res = dl.clausal_query([], [], COMPOUNDS[:2], [])
        assert res == REF_COMPOUNDS_2_AND

    def test_pos_kw(self):
        res = dl.clausal_query(COMPOUNDS[:2], ACTIVITIES[:2], COMPOUNDS[0:1], [])
        assert res == f"{REF_COMPOUNDS_2_OR} AND {REF_ACTIVITIES_2_OR} AND {REF_COMPOUND_1}"

    def test_neg_kw_only(self):
        with pytest.raises(IndexError):
            res = dl.clausal_query([], [], [], COMPOUNDS[:1])

    def test_neg_kw_nopos(self):
        res = dl.clausal_query(COMPOUNDS[:2], ACTIVITIES[:2], [], COMPOUNDS[0:1])
        assert res == f"{REF_COMPOUNDS_2_OR} AND {REF_ACTIVITIES_2_OR} AND NOT {REF_COMPOUND_1}"

    def test_neg_kw_nopos_2(self):
        res = dl.clausal_query([], [], COMPOUNDS[:2], ACTIVITIES[:2])
        assert res == f"{REF_COMPOUNDS_2_AND} AND NOT {REF_ACTIVITIES_2_OR}"