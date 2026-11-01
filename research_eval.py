"""Dated, blinded research cases and transparent review hints (not truth scores)."""
from datetime import date, timedelta
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

LANGUAGES = {
    'en-us': 'American English', 'hu': 'Hungarian', 'en-gb': 'British English',
    'de': 'German', 'fr': 'French', 'it': 'Italian', 'zh': 'Simplified Chinese',
    'pl': 'Polish', 'es': 'Spanish', 'pt': 'Portuguese', 'ja': 'Japanese', 'ko': 'Korean',
    'zh-hant': 'Traditional Chinese',
}


def load_cases(path, as_of, languages, selected=None, max_age_days=7):
    manifest = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    requested = selected.split(',') if selected else None
    if requested and any(name not in {c['id'] for c in manifest['cases']} for name in requested):
        raise ValueError('Unknown research case')
    codes = list(LANGUAGES) if languages == 'all' else languages.split(',')
    if any(code not in LANGUAGES for code in codes):
        raise ValueError('Unknown output language; use ' + ','.join(LANGUAGES))
    scenarios = []
    for case in manifest['cases']:
        if requested is not None and case['id'] not in requested:
            continue
        if requested is None and not case.get('enabled', False):
            continue
        if case['kind'] == 'fresh':
            published = date.fromisoformat(case['published_at'])
            age = (date.fromisoformat(as_of)-published).days
            if age < 0 or age > max_age_days:
                raise ValueError(f"{case['id']}: source age {age} days is outside 0..{max_age_days}; refresh the verified case rather than treating old news as fresh")
        for code in codes:
            # Only the noisy fragment, test instructions, date window and desired
            # language reach the model. The correct name and URLs stay blinded.
            window = (date.fromisoformat(as_of)-timedelta(days=max_age_days)).isoformat()
            fragment = case.get('input_variants', {}).get(code, case['input'])
            query = (
                f"Identify the exact robotics story intended by this short, possibly misspelled or ambiguous submission: {fragment}\n"
                + (f"Find the relevant recent update between {window} and {as_of}; do not substitute a similarly named older project.\n" if case['kind']=='fresh' else case.get('context', '')+'\n')
                + f"Research it on the live web, then write a 3-paragraph RoboHorizon article in {LANGUAGES[code]}. "
                "Use concrete verified facts and restrained dry humor. Check the original announcement and distinguish the event/release date from a paper or article publication date. "
                "Include relevant verified official source, GitHub, arXiv and Hugging Face links naturally. Do not invent URLs, dates or results. "
                "Only return the article, without greetings, numbered citations or a bibliography. If the clues do not support a unique identification, state the uncertainty rather than confidently writing about another topic."
            )
            scenarios.append({'id': case['id'] + '/' + code, 'case_id': case['id'], 'language': code,
                              'query': query, 'requires_live_research': True, 'case_kind': case['kind'],
                              'published_at': case.get('published_at'), 'expected': case['expected']})
    return scenarios


def normalize_url(url):
    parts = urlsplit(url)
    return (parts.netloc.lower().removeprefix('www.'), parts.path.rstrip('/').lower())


def assess(query, result):
    expected = query.get('expected', {})
    text = result.get('response', '').casefold()
    groups = expected.get('keyword_groups', [])
    keyword_hints = [any(word.casefold() in text for word in group) for group in groups]
    returned = result.get('citations', [])
    known = [normalize_url(url) for url in expected.get('primary_urls', [])]
    matches = []
    for item in returned:
        url = item.get('url', '')
        host, path = normalize_url(url)
        if any(host == h and (path == p or path.startswith(p+'/')) for h, p in known):
            matches.append(url)
    article_links = []
    for url in re.findall(r'https?://[^\s)<>]+', result.get('response', '')):
        host, path = normalize_url(url)
        if any(host == h and (path == p or path.startswith(p+'/')) for h, p in known):
            article_links.append(url)
    usage = result.get('usage', {})
    searched = (usage.get('search_count') or 0) > 0
    if result.get('mode') != 'native':
        gate = 'ineligible_no_native_research'
    elif result.get('status') != 'complete':
        gate = 'ineligible_incomplete_or_failed'
    elif not searched:
        gate = 'unverified_search_not_observed'
    elif not (matches or (article_links and returned)) or (keyword_hints and not all(keyword_hints)):
        gate = 'unverified_target_or_primary_source'
    else:
        gate = 'candidate_requires_human_review'
    return {'expected_target': expected.get('target'), 'language': query.get('language'),
            'case_kind': query.get('case_kind'), 'keyword_hints': keyword_hints,
            'matching_primary_citations': matches, 'research_gate': gate,
            'primary_links_in_article': list(dict.fromkeys(article_links)),
            'review_required': True,
            'review': {'correct_topic_and_event': None, 'fresh_source_verified': None,
                       'dates_and_units_correct': None, 'no_unsupported_claims': None,
                       'natural_target_language_1_to_5': None, 'publishable': None, 'notes': ''},
            'review_traps': expected.get('traps', []), 'date_review_note': expected.get('event_date_note')}
