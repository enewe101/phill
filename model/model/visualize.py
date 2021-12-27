import os
import pdb
import xml.etree.ElementTree as et
import shutil
import model as m

initial_html = "<html><head></head><body></body></html>"
html_dir = '../../data/html'
DEFAULT_OUT_PATH = os.path.join(m.const.HTML_DIR, "test.html")


def make_sentences(tokens_batch, head_ptrs_up, head_ptrs_down, dictionary):
    div = et.Element("div")
    zipped = zip(tokens_batch, head_ptrs_up, head_ptrs_down)
    for tokens, heads_up, heads_down in zipped:
        p = et.Element("p")
        div.append(p)
        for token_id, head_up, head_down in zip(tokens, heads_up, heads_down):
            word = et.Element("word")
            word.text = dictionary.get_token(token_id)
            word.set("head", str(head_up))
            word.set("alt-head", str(head_down))
            p.append(word)
    return div


def print_trees(
        tokens_batch,
        head_ptrs_up,
        head_ptrs_down,
        dictionary,
        out_path=m.const.HTML_DIR
    ):

    if not os.path.exists(html_dir):
        os.makedirs(html_dir)
    out_path_html = os.path.join(out_path, "test.html")
    with open(out_path_html, 'wb') as f:
        doc = build_trees(
            tokens_batch, head_ptrs_up, head_ptrs_down, dictionary)
        f.write(et.tostring(doc, method="html"))
    in_path_js = os.path.join(m.const.SCRIPT_DIR, "template.js")
    out_path_js = os.path.join(out_path, "template.js")
    shutil.copy(in_path_js, out_path_js)

    in_path_svg = os.path.join(m.const.SCRIPT_DIR, "arc.svg")
    out_path_svg= os.path.join(out_path, "arc.svg")
    shutil.copy(in_path_svg, out_path_svg)
    return doc


def build_trees(tokens_batch, head_ptrs_up, head_ptrs_down, dictionary):

    template_path = os.path.join(m.const.SCRIPT_DIR, 'template.html')
    with open(template_path) as template_file:
        template = template_file.read()

    doc = et.fromstring(template)
    sentences_elm = make_sentences(
        tokens_batch, head_ptrs_up, head_ptrs_down, dictionary)
    body = doc.find("body")
    body.append(sentences_elm)
    return doc








