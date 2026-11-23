"""Tiny GPT trained from scratch on product description examples (educational project)."""
from __future__ import annotations
import json, re
from pathlib import Path
from collections import Counter
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = ROOT / "data" / "products.jsonl"
MODEL_PATH = ROOT / "models" / "product_gpt.pt"

# Milestone 1: starter corpus (small synthetic examples; expand with original/licensed data).
STARTER = [
 {"category":"coffee","name":"Highland Arabica","description":"A smooth medium roast coffee with notes of cocoa and toasted almond. Freshly roasted beans bring a balanced aroma to your morning cup."},
 {"category":"coffee","name":"Coastal Blend","description":"Bright and lively with citrus notes and a gentle finish. This carefully blended coffee makes an inviting cup for slow mornings."},
 {"category":"coffee","name":"Midnight Espresso","description":"Bold espresso beans deliver a rich crema, dark chocolate flavor, and a full-bodied finish. Ideal for espresso and moka pot brewing."},
 {"category":"coffee","name":"Golden Decaf","description":"Enjoy a mellow, comforting cup with caramel sweetness and a soft finish. Decaffeinated beans preserve the flavor for any time of day."},
 {"category":"coffee","name":"Mountain Peaberry","description":"Distinctive peaberry beans offer a fragrant cup with berry-like brightness and a clean finish. A thoughtful choice for coffee enthusiasts."},
 {"category":"coffee","name":"Morning House Roast","description":"A dependable everyday roast with warm nutty notes and balanced body. It pairs easily with breakfast and your favorite brewing routine."},
 {"category":"coffee","name":"Velvet Mocha Blend","description":"Rich coffee flavor meets smooth cocoa tones in this comforting blend. Its rounded taste is delicious with milk or served black."},
 {"category":"coffee","name":"Estate Dark Roast","description":"Deep roasted beans create smoky aromas and a lingering finish. Brew a strong, satisfying cup for an afternoon break."},
 {"category":"coffee","name":"Floral Light Roast","description":"A light roast with delicate floral fragrance and gentle fruit notes. Its crisp character shines with pour-over brewing."},
 {"category":"coffee","name":"Weekend Blend","description":"A balanced blend with hints of brown sugar and spice. Smooth, easy-drinking flavor makes it a welcome part of relaxed weekends."},
 {"category":"coffee","name":"Rainforest Reserve","description":"Carefully selected beans create a silky body and subtle sweetness. The aroma opens with roasted nuts and finishes pleasantly clean."},
 {"category":"coffee","name":"Spiced Winter Roast","description":"A cozy roast with cinnamon-like warmth, rich aroma, and a rounded finish. Enjoy it freshly brewed on cool evenings."},
 {"category":"coffee","name":"Classic Filter Coffee","description":"A traditional, full-flavored roast designed for filter brewing. It delivers comforting strength, fragrant aroma, and a smooth finish."},
 {"category":"coffee","name":"Cocoa Trail Beans","description":"Chocolate-forward coffee with a soft nutty character and low-key acidity. A versatile roast for espresso, filter, or French press."},
 {"category":"coffee","name":"Sunrise Single Origin","description":"Bright single-origin beans reveal gentle fruit tones and a clean, refreshing finish. Freshly ground coffee brings the aroma to life."},
]

def collect_corpus(path: str | Path = DEFAULT_CORPUS, examples=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    records = examples if examples is not None else STARTER
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in records) + "\n", encoding="utf-8")
    return path

def load_corpus(path: str | Path = DEFAULT_CORPUS):
    path = Path(path)
    if not path.exists(): collect_corpus(path)
    rows=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row=json.loads(line); rows.append(f"Product: {row.get('name','')} | Category: {row.get('category','')} | Description: {row.get('description','')}")
    return rows

# Milestone 2: byte-level BPE. UTF-8 byte alphabet ensures all text is encodable.
class BPETokenizer:
    def __init__(self, vocab_size=320):
        self.vocab_size=max(256,int(vocab_size)); self.merges=[]; self.token_to_id={bytes([i]):i for i in range(256)}; self.id_to_token={i:bytes([i]) for i in range(256)}
    def train(self, texts):
        sequences=[list(t.encode("utf-8")) for t in texts]
        for _ in range(self.vocab_size-256):
            counts=Counter(pair for seq in sequences for pair in zip(seq,seq[1:]))
            if not counts: break
            pair, freq=counts.most_common(1)[0]
            if freq < 2: break
            new_id=len(self.id_to_token); merged=bytes([pair[0]]) if False else self.id_to_token[pair[0]]+self.id_to_token[pair[1]]
            self.merges.append(pair); self.id_to_token[new_id]=merged; self.token_to_id[merged]=new_id
            updated=[]
            for seq in sequences:
                out=[]; i=0
                while i<len(seq):
                    if i<len(seq)-1 and (seq[i],seq[i+1])==pair: out.append(new_id); i+=2
                    else: out.append(seq[i]); i+=1
                updated.append(out)
            sequences=updated
        return self
    def encode(self, text):
        seq=list(text.encode("utf-8"))
        for new_id,pair in enumerate(self.merges, start=256):
            out=[]; i=0
            while i<len(seq):
                if i<len(seq)-1 and (seq[i],seq[i+1])==pair: out.append(new_id); i+=2
                else: out.append(seq[i]); i+=1
            seq=out
        return seq
    def decode(self, ids):
        return b"".join(self.id_to_token[int(i)] for i in ids).decode("utf-8",errors="replace")
    def state_dict(self): return {"vocab_size":self.vocab_size,"merges":self.merges}
    def load_state_dict(self, state):
        self.vocab_size=state["vocab_size"]; self.merges=[tuple(x) for x in state["merges"]]
        self.id_to_token={i:bytes([i]) for i in range(256)}; self.token_to_id={v:k for k,v in self.id_to_token.items()}
        for pair in self.merges:
            idx=len(self.id_to_token); tok=self.id_to_token[pair[0]]+self.id_to_token[pair[1]]
            self.id_to_token[idx]=tok; self.token_to_id[tok]=idx
        return self

# Milestone 3: decoder-only Transformer with causal self-attention.
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__(); assert n_embd%n_head==0
        self.n_head=n_head; self.head_dim=n_embd//n_head
        self.qkv=nn.Linear(n_embd,3*n_embd); self.proj=nn.Linear(n_embd,n_embd)
        self.register_buffer("mask",torch.tril(torch.ones(block_size,block_size)).view(1,1,block_size,block_size))
    def forward(self,x):
        b,t,c=x.shape; q,k,v=self.qkv(x).chunk(3,dim=-1)
        q=q.view(b,t,self.n_head,self.head_dim).transpose(1,2); k=k.view(b,t,self.n_head,self.head_dim).transpose(1,2); v=v.view(b,t,self.n_head,self.head_dim).transpose(1,2)
        scores=(q@k.transpose(-2,-1))/self.head_dim**0.5; scores=scores.masked_fill(self.mask[:,:,:t,:t]==0,float("-inf"))
        y=F.softmax(scores,dim=-1)@v; y=y.transpose(1,2).contiguous().view(b,t,c)
        return self.proj(y)
class TransformerBlock(nn.Module):
    def __init__(self,n_embd,n_head,block_size):
        super().__init__(); self.ln1=nn.LayerNorm(n_embd); self.attn=CausalSelfAttention(n_embd,n_head,block_size); self.ln2=nn.LayerNorm(n_embd)
        self.mlp=nn.Sequential(nn.Linear(n_embd,4*n_embd),nn.GELU(),nn.Linear(4*n_embd,n_embd))
    def forward(self,x): return x+self.attn(self.ln1(x))+self.mlp(self.ln2(x))
class MiniGPT(nn.Module):
    def __init__(self,vocab_size,block_size=128,n_embd=96,n_head=4,n_layer=3):
        super().__init__(); self.block_size=block_size; self.token_emb=nn.Embedding(vocab_size,n_embd); self.pos_emb=nn.Embedding(block_size,n_embd)
        self.blocks=nn.Sequential(*[TransformerBlock(n_embd,n_head,block_size) for _ in range(n_layer)]); self.ln=nn.LayerNorm(n_embd); self.head=nn.Linear(n_embd,vocab_size)
    def forward(self,idx,targets=None):
        b,t=idx.shape; pos=torch.arange(t,device=idx.device); x=self.token_emb(idx)+self.pos_emb(pos); logits=self.head(self.ln(self.blocks(x)))
        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1)) if targets is not None else None
        return logits,loss
    @torch.no_grad()
    def generate(self,idx,max_new_tokens=120,temperature=0.8,top_k=40):
        for _ in range(max_new_tokens):
            idx_cond=idx[:,-self.block_size:]; logits,_=self(idx_cond); logits=logits[:,-1,:]/max(temperature,1e-4)
            if top_k:
                values,_=torch.topk(logits,min(top_k,logits.size(-1))); logits[logits<values[:,-1,None]]=float("-inf")
            probs=F.softmax(logits,dim=-1); next_id=torch.multinomial(probs,1); idx=torch.cat((idx,next_id),dim=1)
        return idx

# Milestone 4: next-token training.
def train_model(texts=None,steps=1200,vocab_size=320,block_size=128,save_path=MODEL_PATH,seed=42):
    torch.manual_seed(seed); texts=texts or load_corpus(); tok=BPETokenizer(vocab_size).train(texts); ids=torch.tensor([tok.encode("\n".join(texts))],dtype=torch.long)
    model=MiniGPT(len(tok.id_to_token),block_size=block_size); optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    if ids.size(1)<3: raise ValueError("Training text is too short.")
    for step in range(steps):
        start=torch.randint(0,max(1,ids.size(1)-block_size), (1,)).item(); x=ids[:,start:start+block_size]; y=ids[:,start+1:start+block_size+1]; x=x[:,:y.size(1)]
        if x.size(1)<2: continue
        logits,loss=model(x,y); optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
    save_path=Path(save_path); save_path.parent.mkdir(parents=True,exist_ok=True); torch.save({"model":model.state_dict(),"tokenizer":tok.state_dict(),"config":{"block_size":block_size,"n_embd":96,"n_head":4,"n_layer":3}},save_path)
    return save_path

def load_model(path=MODEL_PATH):
    ckpt=torch.load(path,map_location="cpu",weights_only=False); tok=BPETokenizer().load_state_dict(ckpt["tokenizer"]); cfg=ckpt["config"]
    model=MiniGPT(len(tok.id_to_token),**cfg); model.load_state_dict(ckpt["model"]); model.eval(); return model,tok

def generate_description(prompt,model=None,tokenizer=None,model_path=MODEL_PATH,max_new_tokens=120,temperature=.8,top_k=40):
    if model is None or tokenizer is None: model,tokenizer=load_model(model_path)
    encoded=tokenizer.encode(prompt)
    if not encoded: encoded=[ord(" ")]
    out=model.generate(torch.tensor([encoded],dtype=torch.long),max_new_tokens=max_new_tokens,temperature=temperature,top_k=top_k)[0].tolist()
    return tokenizer.decode(out[len(encoded):]).strip()
