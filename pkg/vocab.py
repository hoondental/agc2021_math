import re
import sentencepiece as spm

'''
1. special token {token_name:token} 형태로 지정, token = <token_name> 추천
2. pad, bos, eos, unk 는 id 지정 가능
3. 이후에는 special token 부터 차례로 번호 지정

'''

class Vocab:
    def __init__(self):
        self._pad_id = None
        self._bos_id = None
        self._eos_id = None
        self._space_id = None
        self.id2token = []
        self.token2id = {}
        
    def pad_id(self):
        return self._pad_id
        
    def bos_id(self):
        return self._bos_id
        
    def eos_id(self):
        return self._eos_id
        
    def unk_id(self):
        return self._unk_id
        
    def space_id(self):
        return self._space_id
    
    def pad_token(self):
        return self.id2token[self._pad_id] if self._pad_id else None
    
    def bos_token(self):
        return self.id2token[self._bos_id] if self._bos_id else None
    
    def eos_token(self):
        return self.id2token[self._eos_id] if self._eos_id else None
    
    def unk_token(self):
        return self.id2token[self._unk_id] if self._unk_id else None
    
    def space_token(self):
        return self.id2token[self._space_id] if self._space_id else None
    
    def vocab_size(self):
        return len(self.tokens)
        
    def encode_as_tokens(self, text):
        return list(text)
        
    def encode_as_ids(self, text):
        tokens = self.encode_as_tokens(text)
        ids = [self.token2id[t] if t in self.id2token else self.unk_id() for t in tokens]
        return ids
        
    def decode_tokens(self, tokens):
        raise NotImplementedError
        
    def decode_ids(self, ids):
        tokens = [self.id2token[i] for i in ids]
        text = self.decode_tokens(tokens)
        return text
        
    @property
    def tokens(self):
        return self.id2token
    
    def ids_to_tokens(self, ids):
        return [self.id2token[i] for i in ids]
    
    def tokens_to_ids(self, tokens):
        return [self.token2id[t] if t in self.id2token else self.unk_id() for t in tokens]
        
        
        
        
        

class CharVocab(Vocab):
    def __init__(self, pad_id=0, bos_id=1, eos_id=2, unk_id=3, 
                 pad_token='<pad>', bos_token='<bos>', eos_token='<eos>', unk_token='<unk>', 
                 chars=list("abcdefghijklmnopqrstuvwxyz '"), specials={}, contract=False):      
        super().__init__()
        id2tokens = {}
        used_ids = []
        def _next_id(idx):
            while(idx in used_ids):
                idx += 1
            return idx
                
        self._pad_id = pad_id
        self._bos_id = bos_id
        self._eos_id = eos_id 
        self._unk_id = unk_id
        self._pad_token = pad_token
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._unk_token = unk_token
        if pad_id is not None:
            used_ids.append(pad_id)
            id2tokens[pad_id] = pad_token
        if bos_id is not None:
            used_ids.append(bos_id)
            id2tokens[bos_id] = bos_token
        if eos_id is not None:
            used_ids.append(eos_id)
            id2tokens[eos_id] = eos_token
        if unk_id is not None:
            used_ids.append(unk_id)
            id2tokens[unk_id] = unk_token
            
        idx = _next_id(0)
        for k, v in specials.items():
            setattr(self, k + '_id', idx)
            setattr(self, k + '_token', v)
            used_ids.append(idx)
            id2tokens[idx] = v
            idx = _next_id(idx)
            
        for c in chars:
            used_ids.append(idx)
            id2tokens[idx] = c
            idx = _next_id(idx)

        self.id2token = []
        self.token2id = {}
        for i in range(len(id2tokens)):
            token = id2tokens[i]
            self.id2token.append(token)
            self.token2id[token] = i               
            
        if ' ' in self.token2id.keys():
            self._space_id = self.token2id[' ']
            self._space_token = ' '
        else:
            self._space_id = None
            self._space_token = None            
        self.contract = contract
            
    def encode_as_tokens(self, text):
        if self.contract:
            text = re.sub(r'(.)\1+', r'\1', text)
        text = list(text)
        tokens = list(map(lambda c: c if c in self.tokens else self.unk_token(), text))
        return tokens
        
    def encode_as_ids(self, text):
        tokens = self.encode_as_tokens(text)
        ids = [self.token2id[t] for t in tokens]
        return ids
        
    def decode_tokens(self, tokens):
        if self._pad_token:
            tokens = list(filter(self._pad_token.__ne__, tokens))
        if self._bos_token:
            tokens = list(filter(self._bos_token.__ne__, tokens))
        if self._eos_token:
            tokens = list(filter(self._eos_token.__ne__, tokens))
        text = ''.join(tokens)
        return(text)
        
    def decode_ids(self, ids):
        tokens = [self.id2token[i] for i in ids]
        text = self.decode_tokens(tokens)
        return text
    
        
   
    
    
class SPVocab(Vocab):
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        
        id2token = []
        token2id = {}
        for i in range(self.sp.vocab_size()):
            p = self.sp.id_to_piece(i)
            id2token.append(p)
            token2id[p] = i
            
        self.id2token = id2token
        self.token2id = token2id           
        
        self._space_id = self.token2id['▁'] if '▁' in self.token2id.keys() else None

        self.pad_id = self.sp.pad_id
        self.bos_id = self.sp.bos_id
        self.eos_id = self.sp.eos_id
        self.unk_id = self.sp.unk_id
        
        self.enocode_as_ids = self.sp.encode_as_ids
        self.encode_as_tokens = self.sp.encode_as_pieces
        self.decode_ids = self.sp.decode_ids
        self.decode_tokens = self.sp.decode_pieces


        