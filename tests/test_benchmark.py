import json
import io
from pathlib import Path
from unittest.mock import patch

import pytest
from urllib.error import HTTPError

import benchmark as b


def model(provider='google', **extra):
    return {'provider': provider, 'model': 'test', 'input': 1, 'output': 2,
            'cached_input': 0.1, 'search_price': 0.01, 'enabled': True, **extra}


def test_import_and_dry_run_need_no_keys(tmp_path, capsys):
    with patch.dict(b.os.environ, {}, clear=True), patch.object(b, 'request_json') as network:
        assert b.main(['--env-file', str(tmp_path/'absent'), '--list-models']) == 0
        assert b.main(['--env-file', str(tmp_path/'absent')]) == 0
    network.assert_not_called()
    assert 'Dry run' in capsys.readouterr().out


def test_free_selection_does_not_include_free_quota_or_paid_models():
    reg = b.load_models(b.ROOT/'models.json')
    chosen = b.select_models(reg, mode='shared', free_only=True)
    assert set(chosen) == {'openrouter-free','gemma-4-31b-it','gemma-4-26b-a4b-it'}
    assert not b.select_models(reg, mode='native', free_only=True)


def test_invalid_free_label_is_rejected(tmp_path):
    path = tmp_path/'models.json'
    path.write_text(json.dumps({'models': {'bad': model('openrouter', free=True, input=0, output=0)}}))
    with pytest.raises(ValueError, match='Cannot certify'):
        b.load_models(path)


def test_unknown_model_and_explicit_unsupported_search_are_errors():
    reg = b.load_models(b.ROOT/'models.json')
    with pytest.raises(ValueError, match='Unknown model'):
        b.select_models(reg, names='typo')
    with pytest.raises(ValueError, match='no native search'):
        b.select_models(reg, names='groq-gpt-oss-120b')


def test_gemini_counts_thinking_cached_input_and_queries():
    raw = {'usageMetadata': {'promptTokenCount': 1000, 'cachedContentTokenCount': 200,
           'candidatesTokenCount': 100, 'thoughtsTokenCount': 300},
           'candidates': [{'groundingMetadata': {'webSearchQueries': ['one', 'two']}}]}
    cost = b.usage_and_cost(model(), raw, 'native')
    assert cost['output_tokens_including_thinking'] == 400
    assert cost['token_usd'] == pytest.approx((800+20+800)/1e6)
    assert cost['search_usd'] == 0.02
    assert cost['total_usd'] == pytest.approx(0.02162)


def test_gemini_legacy_bills_one_grounded_prompt_not_each_query():
    raw = {'usageMetadata': {'promptTokenCount': 100, 'candidatesTokenCount': 100},
           'candidates': [{'groundingMetadata': {'webSearchQueries': ['one', 'two']}}]}
    assert b.usage_and_cost(model(search_unit='grounded_prompt'), raw, 'native')['search_count'] == 1


def test_responses_reasoning_is_already_in_output():
    raw = {'usage': {'input_tokens': 100, 'output_tokens': 500,
                    'output_tokens_details': {'reasoning_tokens': 400}}}
    cost = b.usage_and_cost(model('openai'), raw, 'none')
    assert cost['thinking_tokens'] == 400
    assert cost['token_usd'] == pytest.approx(0.0011)


def test_anthropic_input_cache_counts_are_additive():
    raw = {'usage': {'input_tokens': 100, 'output_tokens': 50,
                    'cache_read_input_tokens': 200, 'cache_creation_input_tokens': 300}}
    cost = b.usage_and_cost(model('anthropic'), raw, 'none')
    assert cost['input_tokens'] == 600
    assert cost['token_usd'] == pytest.approx((100+20+375+100)/1e6)


def test_missing_usage_or_prices_are_unknown_not_zero():
    assert b.usage_and_cost(model(), {}, 'none')['total_usd'] is None
    raw = {'usageMetadata': {'promptTokenCount': 100}}
    assert b.usage_and_cost(model(), raw, 'none')['total_usd'] is None
    raw = {'usage': {'prompt_tokens': 100, 'completion_tokens': 100}}
    assert b.usage_and_cost(model('cerebras', input=None), raw, 'none')['total_usd'] is None


def test_provider_reported_cost_takes_precedence():
    raw = {'usage': {'input_tokens': 100, 'output_tokens': 100, 'cost': {'total_cost': 0.123}}}
    cost = b.usage_and_cost(model('perplexity'), raw, 'native')
    assert cost['total_usd'] == 0.123
    assert cost['cost_kind'] == 'provider_reported'


def test_promotion_expiry_changes_rates():
    meta = model(prices_after={'date': '2027-01-01', 'input': 2, 'output': 4})
    raw = {'usageMetadata': {'promptTokenCount': 1000, 'candidatesTokenCount': 1000}}
    with patch.object(b, 'today', return_value='2026-12-31'):
        assert b.usage_and_cost(meta, raw, 'none')['total_usd'] == 0.003
    with patch.object(b, 'today', return_value='2027-01-01'):
        assert b.usage_and_cost(meta, raw, 'none')['total_usd'] == 0.006


@pytest.mark.parametrize('provider', ['openai', 'perplexity', 'xai'])
def test_responses_adapters_use_current_endpoint_and_tools(provider):
    raw = {'status': 'completed', 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'Answer'}]}]}
    with patch.object(b, 'api_key', return_value='test-key'), patch.object(b, 'request_json', return_value=raw) as call:
        _, text, status = b.invoke(model(provider), 'prompt', 'native', 2048, 10)
    assert text == 'Answer' and status == 'complete'
    assert call.call_args.args[0].endswith('/responses')
    assert call.call_args.args[2]['tools'] == [{'type': 'web_search'}]


def test_gemini_shared_mode_has_no_search_tool():
    raw = {'candidates': [{'content': {'parts': [{'text': 'hidden', 'thought': True}, {'text': 'Answer'}]}, 'finishReason': 'STOP'}]}
    with patch.object(b, 'api_key', return_value='test'), patch.object(b, 'request_json', return_value=raw) as call:
        _, text, status = b.invoke(model(), 'prompt', 'shared', 2048, 10)
    assert 'tools' not in call.call_args.args[2]
    assert text == 'Answer' and status == 'complete'


def test_free_router_disables_paid_fallbacks_and_plugins():
    raw = {'choices': [{'message': {'content': 'Answer'}, 'finish_reason': 'stop'}]}
    with patch.object(b, 'api_key', return_value='test'), patch.object(b, 'request_json', return_value=raw) as call:
        b.invoke(model('openrouter', free=True, model='openrouter/free'), 'prompt', 'shared', 100, 10)
    payload = call.call_args.args[2]
    assert payload['plugins'] == []
    assert payload['provider'] == {'allow_fallbacks': False, 'max_price': {'prompt': 0, 'completion': 0}}
    assert 'models' not in payload and 'tools' not in payload


def test_incomplete_and_empty_results_are_not_success():
    raw = {'stop_reason': 'pause_turn', 'content': [{'type': 'text', 'text': 'Still researching'}]}
    with patch.object(b, 'api_key', return_value='test'), patch.object(b, 'request_json', return_value=raw):
        assert b.invoke(model('anthropic'), 'q', 'native', 100, 10)[2] == 'incomplete'
    raw = {'candidates': [{'finishReason': 'STOP', 'content': {'parts': []}}]}
    with patch.object(b, 'api_key', return_value='test'), patch.object(b, 'request_json', return_value=raw):
        assert b.invoke(model(), 'q', 'none', 100, 10)[2] == 'empty'


def test_errors_are_retained_with_unknown_cost():
    with patch.object(b, 'invoke', side_effect=RuntimeError('HTTP 401; no retry')):
        result = b.run_one('test', model(), {'id': 'q', 'query': 'q'}, 'none', 'prompt', 100, 10)
    assert result['status'] == 'error'
    assert result['usage']['total_usd'] is None


def test_excel_preserves_errors_full_text_and_literal_formulas(tmp_path):
    from openpyxl import load_workbook
    path = tmp_path/'results.xlsx'
    text = '=SUM(A1:A2)' + 'a'*40000
    rows = [{'model': 'test', 'query_id': 'q', 'mode': 'shared', 'status': 'complete', 'response': text,
             'usage': {'total_usd': 0}},
            {'model': 'error', 'query_id': 'q', 'mode': 'shared', 'status': 'error', 'error': 'HTTP 401'}]
    b.excel_export(path, rows)
    wb = load_workbook(path)
    assert wb['Summary'].max_row == 3
    assert wb['Summary']['G3'].value is None
    assert wb['Responses']['E2'].data_type == 's'
    assert ''.join(wb['Responses'].cell(r, 5).value for r in (2, 3)) == text


def test_shared_sources_are_identical_and_checkpointed(tmp_path):
    evidence = [{'title': 'Primary source', 'url': 'https://example.org', 'text': 'Source facts'}]
    reg = {'verified_at': '2026-11-01', 'models': {'a': model(), 'b': model()}}
    registry_path = tmp_path/'registry.json'
    registry_path.write_text(json.dumps(reg))
    seen = []
    def fake_invoke(meta, prompt, *args):
        seen.append(prompt)
        return {'usageMetadata': {'promptTokenCount': 10, 'candidatesTokenCount': 10}}, 'Answer', 'complete'
    with patch.dict(b.os.environ, {'GEMINI_API_KEY': 'test'}, clear=True), patch.object(b, 'shared_search', return_value=evidence) as search, patch.object(b, 'invoke', side_effect=fake_invoke):
        code = b.main(['--env-file', str(tmp_path/'absent'), '--models-file', str(registry_path),
                       '--search-mode', 'shared', '--query', 'Question', '--max-calls', '2', '--run',
                       '--output-dir', str(tmp_path/'out'), '--no-excel'])
    assert code == 0
    search.assert_called_once()
    assert seen[0] == seen[1]
    bundle = json.loads(next((tmp_path/'out').glob('*/results.json')).read_text())
    assert len(bundle['results']) == 2
    assert bundle['evidence']['custom'] == evidence
    assert bundle['results'][0]['prompt_sha256'] == bundle['results'][1]['prompt_sha256']


def test_no_live_web_search_claim_in_none_mode():
    prompt = b.build_prompt('Translate this text', 'none')
    assert 'do not claim to have searched the web' in prompt


def test_free_native_cli_is_rejected():
    with pytest.raises(SystemExit) as exc:
        b.main(['--free-only'])
    assert exc.value.code == 2


def test_error_diagnostics_classify_billing_without_leaking_body():
    raw = json.dumps({'error': {'message': 'Credit balance too low; echoed key SECRET-DO-NOT-PRINT'}}).encode()
    error = HTTPError('https://api.example.test', 400, 'bad request', {}, io.BytesIO(raw))
    with patch.object(b.urllib.request, 'urlopen', side_effect=error), pytest.raises(RuntimeError) as exc:
        b.request_json('https://api.example.test', 'SECRET-DO-NOT-PRINT', {'model': 'test'})
    assert 'credit/billing issue' in str(exc.value)
    assert 'SECRET-DO-NOT-PRINT' not in str(exc.value)


def test_paid_lookup_cannot_be_misrepresented_as_free():
    with pytest.raises(SystemExit) as exc:
        b.main(['--free-only','--search-mode','agent-google'])
    assert exc.value.code==2


def test_google_lookup_uses_observed_grounding_not_model_invented_links():
    raw={'usageMetadata':{'promptTokenCount':10,'candidatesTokenCount':20},
         'candidates':[{'groundingMetadata':{'webSearchQueries':['q'],
          'groundingChunks':[{'web':{'uri':'https://example.org/source','title':'Source'}}],
          'groundingSupports':[{'groundingChunkIndices':[0],'segment':{'text':'Verified source statement'}}]}}]}
    with patch.object(b,'invoke',return_value=(raw,'unverified invented URL: https://fake.example','complete')) as invoke:
        result=b.google_lookup('noisy keywords',5)
    assert result['results']==[{'url':'https://example.org/source','title':'Source','text':'Verified source statement'}]
    assert 'Do not return JSON' in invoke.call_args.args[1]
    assert result['usage']['search_count']==1


def test_excel_keeps_diagnostic_phases_apart_and_free_zero_distinct_from_unknown(tmp_path):
    from openpyxl import load_workbook
    rows=[{'model':'same','query_id':'q','mode':'agent','phase':phase,'status':status,'usage':usage}
          for phase,status,usage in [('screening','complete',{'total_usd':0}),
                                    ('diagnostic','error',{'total_usd':None,'known_cost_usd':0.02}),
                                    ('unavailable','not_run',{'total_usd':None})]]
    path=tmp_path/'phases.xlsx'
    b.excel_export(path,rows)
    wb=load_workbook(path)
    comparison=list(wb['Model Comparison'].values)
    assert len(comparison)==4
    groups={r[-1]:dict(zip(comparison[0],r)) for r in comparison[1:]}
    assert groups['screening']['mean_fully_priced_USD']==0
    assert groups['diagnostic']['unknown_cost_attempts']==1
    assert groups['diagnostic']['known_lower_bound_USD']==0.02
    assert groups['unavailable']['known_lower_bound_USD'] is None
