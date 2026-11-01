import json
from unittest.mock import patch

import pytest

from agent_search import public_url, research


def test_agent_chooses_its_query_fetches_source_and_counts_all_model_calls():
    actions=iter([{'action':'search','query':'corrected project query'},
                  {'action':'fetch','url':'https://example.org/source'},
                  {'action':'final','article':'Verified article'}])
    calls=[]
    def invoke(meta,prompt,mode,max_tokens,timeout):
        calls.append(prompt)
        return {'model':'actual-model','usage':{}},json.dumps(next(actions)),'complete'
    def price(*args):return {'total_usd':0.001,'token_usd':0.001,'input_tokens':100,'output_tokens_including_thinking':50,'thinking_tokens':10}
    def search(query,limit):
        assert query=='corrected project query'
        return [{'url':'https://example.org/source','title':'Primary source','text':'Snippet'}]
    with patch('agent_search.fetch_source',return_value={'url':'https://example.org/source','text':'Actual source text'}):
        raw,text,status=research({},'Noisy keywords',100,10,invoke,price,search)
    assert text=='Verified article' and status=='complete'
    assert raw['agent_usage']['total_usd']==pytest.approx(0.003)
    assert raw['agent_usage']['search_count']==1
    assert raw['agent_usage']['fetch_count']==1
    assert raw['agent_usage']['inference_calls']==3
    assert 'Actual source text' in calls[-1]


def test_failed_later_step_preserves_known_spend_without_claiming_total():
    count=0
    def invoke(*args):
        nonlocal count
        count+=1
        if count==2:raise RuntimeError('HTTP 429')
        return {},'{"action":"search","query":"q"}','complete'
    raw,text,status=research({},'keywords',100,10,invoke,lambda *a:{'total_usd':0.01},lambda *a:[])
    assert status=='error'
    assert raw['agent_usage']['known_cost_usd']==0.01
    assert raw['agent_usage']['total_usd'] is None


def test_agent_never_fetches_urls_not_returned_by_search():
    actions=iter([{'action':'fetch','url':'http://localhost/secret'}, {'action':'final','article':'Uncertain'}])
    with patch('agent_search.fetch_source') as fetch:
        research({},'q',100,10,lambda *a:({},json.dumps(next(actions)),'complete'),lambda *a:{'total_usd':0},lambda *a:[],steps=2)
    fetch.assert_not_called()


def test_fetch_blocks_private_and_credential_bearing_urls():
    with pytest.raises(ValueError):public_url('http://user:password@example.com')
    with patch('agent_search.socket.getaddrinfo',return_value=[(None,None,None,None,('127.0.0.1',80))]),pytest.raises(ValueError):
        public_url('http://example.com')


def test_paid_lookup_cost_is_added_to_all_writer_turns():
    actions=iter([{'action':'search','query':'q'},{'action':'fetch','url':'https://example.org/a'},
                  {'action':'final','article':'Article'}])
    def lookup(*args):return {'results':[{'url':'https://example.org/a','title':'Source'}],
                             'usage':{'total_usd':0.02,'token_usd':0.006,'search_usd':0.014,'search_count':1},'raw':{'model':'lookup-model'}}
    with patch('agent_search.fetch_source',return_value={'text':'Source text'}):
        raw,_,_=research({},'q',100,10,lambda *a:({},json.dumps(next(actions)),'complete'),lambda *a:{'total_usd':0.001,'token_usd':0.001},lookup)
    assert raw['agent_usage']['total_usd']==pytest.approx(0.023)
    assert raw['agent_usage']['search_usd']==0.014
    assert raw['agent_usage']['lookup_inference_calls']==1
    assert raw['lookup_responses'][0]['model']=='lookup-model'


def test_plain_final_article_is_accepted_after_actual_research():
    article='A'*110+'\n\n'+'B'*110+'\n\n'+'C'*110
    actions=iter(['{"action":"search","query":"q"}',
                  '{"action":"fetch","url":"https://example.org/a"}',article])
    with patch('agent_search.fetch_source',return_value={'text':'Source text'}):
        raw,text,status=research({},'q',100,10,lambda *a:({},next(actions),'complete'),lambda *a:{'total_usd':0},
                                 lambda *a:[{'url':'https://example.org/a','title':'Source'}])
    assert text==article and status=='complete'


def test_technical_search_failure_is_not_a_successful_article():
    actions=iter(['{"action":"search","query":"q"}','{"action":"final","article":"Could not identify the story"}'])
    def failed(*args):raise ValueError('Search backend unavailable')
    raw,_,status=research({},'q',100,10,lambda *a:({},next(actions),'complete'),lambda *a:{'total_usd':0.001},failed)
    assert status=='retrieval_failed'
    assert raw['agent_usage']['search_attempts']==1
    assert raw['agent_usage']['retrieval_failures']==1
