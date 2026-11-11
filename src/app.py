import os, html
from pathlib import Path
import gradio as gr
from inference import Predictor

# One source of truth for entity names and colors.
# The colors are the only "loud" thing in the UI, so they carry the meaning.
LABELS = {
    'PER': ('Person', '#6366f1', '#4338ca'),
    'ORG': ('Organization', '#f59e0b', '#b45309'),
    'LOC': ('Place', '#10b981', '#047857'),
    'MISC': ('Other', '#f43f5e', '#be123c'),
}
FALLBACK = ('Other', '#64748b', '#334155')

EMPTY_TEXT = 'Highlighted text will appear here.'
HINT_EMPTY = 'Choose an example or enter some English text.'


def _entity_css():
    rules = []
    for code, (_, color, dark_color) in LABELS.items():
        rules.append(f'.ent-{code}{{background:{color}26;border-bottom-color:{color};}}')
        rules.append(f'.ent-{code} small{{color:{dark_color} !important;}}')
        rules.append(f'.dot-{code}{{background:{color};}}')
    return '\n'.join(rules)


CUSTOM_CSS = f"""
html {{
    scrollbar-gutter: stable;
}}
.gradio-container {{
    width: 920px !important;
    max-width: 95vw !important;
    margin-left: auto !important;
    margin-right: auto !important;
    --ink: #0f172a;
    --on-ink: #f8fafc;
}}
.dark .gradio-container {{
    --ink: #f1f5f9;
    --on-ink: #0f172a;
}}

/* Neutral primary button: no accent color competing with the entity colors */
.gradio-container button.primary {{
    background: var(--ink) !important;
    color: var(--on-ink) !important;
    border: 1px solid var(--ink) !important;
}}
.gradio-container button.primary:hover {{ opacity: 0.88; }}

/* Results card with clean white surface, removing muddy grey background */
.results-card, .results-card > .styler, .results-card .styler {{
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
}}
.results-card .status p {{
    margin: 0;
    color: #334155 !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
}}

/* Highlighted text: clean document card */
.ner-view {{
    white-space: pre-wrap;
    line-height: 2.2;
    font-size: 1.05rem;
    color: #0f172a;
    padding: 12px 14px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    margin: 6px 0 12px;
}}
.ner-view.empty {{
    color: #64748b;
    border-style: dashed;
    font-size: 0.95rem;
}}
.ent {{
    padding: 2px 6px;
    border-radius: 4px;
    border-bottom: 2px solid transparent;
    color: #0f172a !important;
}}
.ent small {{
    margin-left: 5px;
    font-size: 0.75em;
    font-weight: 700;
    letter-spacing: 0.02em;
}}
{_entity_css()}

/* Legend */
.legend {{ display: flex; flex-wrap: wrap; gap: 16px; margin: 0 0 10px; }}
.legend span {{
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 0.88rem; color: #334155; font-weight: 500;
}}
.legend i {{
    width: 12px; height: 12px; border-radius: 3px;
    display: inline-block;
}}
"""

EMPTY_VIEW = f'<div class="ner-view empty">{EMPTY_TEXT}</div>'


def legend_html():
    items = ''.join(
        f'<span><i class="dot-{code}"></i>{name}</span>'
        for code, (name, _, _) in LABELS.items()
    )
    return f'<div class="legend">{items}</div>'


def colored_html(text, spans):
    out, last = [], 0
    for s in spans:
        name = LABELS.get(s['label'], FALLBACK)[0]
        cls = f"ent-{s['label']}" if s['label'] in LABELS else ''
        out.append(html.escape(text[last:s['start']]))
        out.append(
            f'<mark class="ent {cls}" title="{name}">'
            + html.escape(text[s['start']:s['end']])
            + f'<small>{name}</small></mark>'
        )
        last = s['end']
    out.append(html.escape(text[last:]))
    return f'<div class="ner-view">{"".join(out)}</div>'


EXAMPLES = [
    'Sara works at Google in Cairo.',
    'Ahmed visited Dubai with Omar.',
    'Apple opened an office in London.',
    'An Egyptian student moved to Paris.',
]


def build_app(model_dir=None):
    predictor = Predictor(model_dir or Path(__file__).parent / 'selected_model')

    def analyze(text):
        if not text or not text.strip():
            return EMPTY_VIEW, gr.update(value=[], visible=False), HINT_EMPTY
        try:
            result = predictor.analyze(text)
        except ValueError as exc:
            raise gr.Error(str(exc))
        rows = [[x['text'], LABELS.get(x['label'], FALLBACK)[0]] for x in result['entities']]
        n = len(rows)
        if n:
            msg = f'Found {n} ' + ('entity.' if n == 1 else 'entities.')
        else:
            msg = 'No entities found. Try another sentence.'
        return colored_html(text, result['entities']), gr.update(value=rows, visible=bool(n)), msg

    def clear():
        return '', EMPTY_VIEW, gr.update(value=[], visible=False), HINT_EMPTY

    with gr.Blocks(title='Find entities') as demo:
        gr.HTML(f'<style>{CUSTOM_CSS}</style>')
        gr.Markdown(
            '# Find entities in your text\n'
            'Paste English text to find people, organizations, places and other named entities.'
        )

        box = gr.Textbox(
            label='Your text', lines=4, max_lines=12,
            placeholder='Sara works at Google in Cairo.',
        )
        for pair in (EXAMPLES[:2], EXAMPLES[2:]):
            with gr.Row():
                for ex in pair:
                    gr.Button(ex, size='sm', variant='secondary').click(
                        lambda ex=ex: ex, outputs=box
                    )

        with gr.Row():
            button = gr.Button('Find entities', variant='primary', scale=3)
            clear_btn = gr.Button('Clear', variant='secondary', scale=1)

        with gr.Group(elem_classes=['results-card']):
            gr.Markdown('### Results')
            status = gr.Markdown(HINT_EMPTY, elem_classes=['status'])
            gr.HTML(legend_html())
            view = gr.HTML(value=EMPTY_VIEW)
            table = gr.Dataframe(
                headers=['Entity', 'Category'], datatype=['str', 'str'],
                interactive=False, visible=False, label='Detected entities',
            )

        with gr.Accordion('About this demo', open=False):
            gr.Markdown(
                'This English-language demo finds **people**, **organizations**, **places** '
                'and **other named entities**. Other includes nationalities, such as Egyptian.\n\n'
                'The underlying categories are PER, ORG, LOC and MISC. The model was trained on '
                'English news articles and can make mistakes. Dates are not a separate supported category.\n\n'
                'You can enter up to **20,000 characters**. Your original spacing and punctuation '
                'are preserved. Text is processed when you select Find entities.'
            )

        button.click(analyze, inputs=box, outputs=[view, table, status],
                     concurrency_limit=1, concurrency_id='inference')
        clear_btn.click(clear, outputs=[box, view, table, status])
    return demo.queue(max_size=8, default_concurrency_limit=1)


if __name__ == '__main__':
    build_app().launch(server_name='127.0.0.1', server_port=int(os.environ.get('PORT', '7860')))
