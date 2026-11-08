"""Fresh state per Streamlit session; invalidate on replacement AND deletion."""
from __future__ import annotations
import hashlib

RESULT_KEYS = ('report', 'excel_bytes', 'cached_question', 'question_pdf_bytes', 'last_student', 'last_latency')


def reset_analysis(state):
    for key in RESULT_KEYS:
        state[key] = None
    state['messages'] = []
    state['chat_context'] = {}


def init_session(state):
    if 'uploads_hash' not in state:
        reset_analysis(state)
        state['uploads_hash'] = None
    state.setdefault('chat_context', {})


def sync_uploads(state, files):
    digest = hashlib.sha256()
    for index, file in enumerate(files):
        digest.update(bytes([index]))
        if file is None:
            digest.update(b'missing')
        else:
            name, data = file.name.encode('utf-8'), file.getvalue()
            digest.update(len(name).to_bytes(8, 'big'))
            digest.update(name)
            digest.update(len(data).to_bytes(8, 'big'))
            digest.update(data)
    signature = digest.hexdigest()
    if state.get('uploads_hash') != signature:
        reset_analysis(state)
        state['uploads_hash'] = signature
        return True
    return False
