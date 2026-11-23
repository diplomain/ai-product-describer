import json
from pathlib import Path
import pytest
from app.core import BPETokenizer, CausalSelfAttention, MiniGPT, collect_corpus, load_corpus, train_model, load_model, generate_description

def test_corpus_written_and_loaded(tmp_path):
    p=collect_corpus(tmp_path/'products.jsonl')
    rows=load_corpus(p)
    assert len(rows)>=10 and 'Description:' in rows[0]

def test_bpe_round_trip_unicode():
    tok=BPETokenizer(300).train(['coffee coffee morning','coffee beans aroma','☕ coffee'])
    text='☕ coffee morning'
    assert tok.decode(tok.encode(text))==text

def test_attention_is_causal_shape():
    import torch
    layer=CausalSelfAttention(16,4,12)
    assert layer(torch.randn(2,7,16)).shape==(2,7,16)

def test_gpt_forward_and_generation():
    import torch
    model=MiniGPT(260,block_size=12,n_embd=16,n_head=4,n_layer=1)
    logits,loss=model(torch.randint(0,260,(2,6)),torch.randint(0,260,(2,6)))
    assert logits.shape==(2,6,260) and loss.item()>0
    assert model.generate(torch.randint(0,260,(1,4)),max_new_tokens=2).shape[1]==6

def test_train_save_load_generate(tmp_path):
    save=tmp_path/'tiny.pt'
    train_model(['hello product','nice item','product descriptions are fun'],steps=2,vocab_size=258,block_size=16,save_path=save)
    model,tok=load_model(save)
    assert generate_description('Item: mug',model,tok,max_new_tokens=2)

