from pathlib import Path
from unittest.mock import patch

import pytest

import benchmark as b
from research_eval import LANGUAGES, assess, load_cases

CASES = Path(__file__).parents[1] / 'research_cases.json'


def test_answer_key_never_reaches_model_prompt():
    case = load_cases(CASES, '2026-11-01', 'hu', 'crowdbricks-batching-typo')[0]
    assert case['expected']['target'] not in case['query']
    assert 'https://' not in case['query']
    assert 'crowdbriks rehan 10k 4090 batcing' in case['query']
    assert 'Hungarian' in case['query']


def test_each_supported_language_has_separate_case():
    cases = load_cases(CASES, '2026-11-01', 'all')
    assert len(cases) == 2 * len(LANGUAGES)
    assert len({case['id'] for case in cases}) == len(cases)
    hungarian = next(c for c in cases if c['id'] == 'motionbricks-release-typo/hu')
    assert 'új sulyok' in hungarian['query']


@pytest.mark.parametrize('as_of,age', [('2026-11-01', 0), ('2026-11-05', 7), ('2026-10-26', 7)])
def test_same_day_stale_and_future_sources_rejected(as_of, age):
    with pytest.raises(ValueError, match='source age'):
        load_cases(CASES, as_of, 'en-us', max_age_days=age)


def test_original_typo_is_preserved_as_historical_not_fresh():
    cases = load_cases(CASES, '2026-11-01', 'en-us', 'legacy-rek-typo-2025', 0)
    assert len(cases) == 1
    assert cases[0]['case_kind'] == 'historical'
    assert 'Latest REK fight in San fransico VR Unitree bots' in cases[0]['query']
    assert '2025' in cases[0]['query']


def test_article_suite_rejects_supplied_evidence_before_network(tmp_path):
    with patch.object(b, 'request_json') as network, pytest.raises(SystemExit) as error:
        b.main(['--env-file', str(tmp_path/'absent'), '--search-mode', 'shared', '--run'])
    network.assert_not_called()
    assert error.value.code == 2


def target_result(mode='native', text='CrowdBricks MotionBricks RTX 4090'):
    return {'mode': mode, 'status': 'complete', 'response': text, 'usage': {'search_count': 3},
            'citations': [{'url': 'https://github.com/patch-gg/crowdbricks'}]}


def test_matching_topic_and_source_still_requires_human_review():
    case = load_cases(CASES, '2026-11-01', 'hu', 'crowdbricks-batching-typo')[0]
    result = assess(case, target_result())
    assert result['research_gate'] == 'candidate_requires_human_review'
    assert result['review_required']
    assert result['review']['publishable'] is None


def test_fluent_wrong_topic_is_not_qualified():
    case = load_cases(CASES, '2026-11-01', 'hu', 'crowdbricks-batching-typo')[0]
    assert assess(case, target_result(text='NVIDIA GR00T N1.7 is impressive'))['research_gate'] == 'unverified_target_or_primary_source'


def test_supplied_evidence_cannot_qualify_an_article_model():
    case = load_cases(CASES, '2026-11-01', 'hu', 'crowdbricks-batching-typo')[0]
    assert assess(case, target_result(mode='shared'))['research_gate'] == 'ineligible_no_native_research'


def test_search_must_be_observed_not_merely_claimed():
    case = load_cases(CASES, '2026-11-01', 'hu', 'crowdbricks-batching-typo')[0]
    response = target_result()
    response['usage']['search_count'] = None
    assert assess(case, response)['research_gate'] == 'unverified_search_not_observed'


def test_lookalike_domains_do_not_match_primary_source():
    case = load_cases(CASES, '2026-11-01', 'hu', 'crowdbricks-batching-typo')[0]
    response = target_result()
    response['citations'] = [{'url': 'https://github.com.evil.example/patch-gg/crowdbricks'}]
    assert not assess(case, response)['matching_primary_citations']


def test_web_retrieval_can_trigger_long_input_price_tier():
    registry = b.load_models(b.ROOT/'models.json')
    meta = registry['models']['anthropic-claude-haiku-5-5']
    raw = {'usage': {'input_tokens': 100001, 'output_tokens': 1000}}
    cost = b.usage_and_cost(meta, raw, 'none')
    assert cost['rates_used']['input'] == 0.5
    assert cost['token_usd'] == pytest.approx((100001*0.5+1000*2.5)/1e6)


def test_returned_grounding_redirects_keep_primary_links_as_hints_only():
    case = load_cases(CASES, '2026-11-01', 'hu', 'crowdbricks-batching-typo')[0]
    response = target_result(text='CrowdBricks MotionBricks RTX 4090: https://github.com/patch-gg/crowdbricks')
    response['citations'] = [{'url': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/example'}]
    result = assess(case, response)
    assert result['primary_links_in_article']
    assert not result['matching_primary_citations']
    assert result['research_gate'] == 'candidate_requires_human_review'
    assert result['review']['fresh_source_verified'] is None
