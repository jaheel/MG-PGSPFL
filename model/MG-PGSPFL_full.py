"""MG-PGSPFL — federated event sequence prediction model."""

import json, os, time, gc
from datetime import datetime
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from metrics import evaluate

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # release package root
DS_NAME = 'eicu'      # dataset: mimic_iv / eicu / instacart (folder under data/)
DATA_DIR = os.path.join(PKG_DIR, 'data', DS_NAME, 'preprocessed')
RANDOM_SEED = 1024
LR=0.001; BATCH_SIZE=64; LOCAL_EPOCHS=5; EMB_DIM=160; HIDDEN_DIM=160
RUN_NAME = f'MG-PGSPFL_lr{LR}_B{BATCH_SIZE}_E{LOCAL_EPOCHS}_s{EMB_DIM}_seed{RANDOM_SEED}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
OUT_DIR = os.path.join(PKG_DIR, 'outputs', DS_NAME, RUN_NAME); os.makedirs(OUT_DIR, exist_ok=True)

NUM_LSTM_LAYERS=1; DROPOUT=0.1
COMM_ROUNDS=20; GAMMA_MIN=0.1; GAMMA_MAX=1.0
MU=0.0001; NU=1e-06; MU_REPR=1e-5
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)

LOG_FILE=os.path.join(OUT_DIR,'training_log.txt')
def log(msg):
    with open(LOG_FILE,'a') as f: f.write(msg+'\n')
    print(msg, flush=True)

log(f'=== MG-PGSPFL Full ({DS_NAME}) ===')
log(f'  seed={RANDOM_SEED}, lr={LR}, B={BATCH_SIZE}, E={LOCAL_EPOCHS}, s={EMB_DIM}, dp={DROPOUT}')
log(f'  K={COMM_ROUNDS}, γ_min={GAMMA_MIN}, γ_max={GAMMA_MAX}')
log(f'  μ={MU}, ν={NU}, μ_repr={MU_REPR}')
log(f'  R_prox: ||W_c-W_global||²+||θ-θ_global||²')
log(f'  R_cons: ||W_f-W_c^(prev)||^2 (fixed target, no FREQ_W)')

log('Loading data...')
sd=torch.load(os.path.join(DATA_DIR,'sequences.pt'),weights_only=False)
SEQ_MATRIX=sd['matrix']; PATIENT_INDEX=sd['index']
train_patients=torch.load(os.path.join(DATA_DIR,'train_patients.pt'),weights_only=False)
test_patients=torch.load(os.path.join(DATA_DIR,'test_patients.pt'),weights_only=False)
with open(os.path.join(DATA_DIR,'config.json')) as f: cfg=json.load(f)
N_ICD=cfg['n_icd']; centers=cfg['centers']
log(f'  ICD: {N_ICD}, Centers: {len(centers)}')

log('Computing client frequency vectors...')
client_freq={}
for center in centers:
    pids=train_patients[center]
    if len(pids)==0: client_freq[center]=np.ones(N_ICD)/N_ICD; continue
    freq=np.zeros(N_ICD)
    for pid in pids:
        s,e,_=PATIENT_INDEX[pid]; steps=SEQ_MATRIX[s:e].numpy()
        freq+=steps.sum(axis=0)
    freq=freq/max(freq.sum(),1); client_freq[center]=freq

GLOBAL_FREQ=np.zeros(N_ICD)
for center in centers: GLOBAL_FREQ+=client_freq[center]*len(train_patients[center])
GLOBAL_FREQ/=max(GLOBAL_FREQ.sum(),1e-8)
_FW=GLOBAL_FREQ/max(GLOBAL_FREQ.max(),1e-8)
FREQ_W=torch.FloatTensor(np.where(_FW>=0.01,_FW,0.0)).to(DEVICE)

prev_local_Wc={}

class SeqDataset(Dataset):
    def __init__(self,pids):
        self.samples=[(SEQ_MATRIX[s:e][:-1],SEQ_MATRIX[s:e][-1]) for pid in pids for s,e,_ in [PATIENT_INDEX[pid]]]
    def __len__(self): return len(self.samples)
    def __getitem__(self,i): return self.samples[i][0].clone(),self.samples[i][1].clone()
def collate(batch):
    seqs,labels=zip(*batch); lengths=[len(s) for s in seqs]; mx=max(lengths)
    padded=torch.zeros(len(seqs),mx,seqs[0].size(-1))
    for i,s in enumerate(seqs): padded[i,:len(s)]=s
    return padded,torch.tensor(lengths),torch.stack(labels)

class Classifier(nn.Module):
    def __init__(self,in_dim,out_dim,dropout=0.,activation=None):
        super().__init__()
        self.linear=nn.Linear(in_dim,out_dim); self.dropout=nn.Dropout(dropout); self.activation=activation
    def forward(self,x):
        return self.activation(self.linear(self.dropout(x))) if self.activation else self.linear(self.dropout(x))

class OursModel(nn.Module):
    def __init__(self,n_icd,emb_dim,hidden_dim,n_layers,dropout):
        super().__init__()
        self.embed=nn.Linear(n_icd,emb_dim,bias=False)
        self.lstm=nn.LSTM(emb_dim,hidden_dim,n_layers,batch_first=True,bidirectional=True)
        self.coarse_head=Classifier(2*hidden_dim,n_icd,dropout,activation=None)
        self.fine_head=Classifier(2*hidden_dim,n_icd,dropout,activation=None)
        self.fine_head.linear.weight.data.copy_(self.coarse_head.linear.weight.data)
        self.fine_head.linear.bias.data.copy_(self.coarse_head.linear.bias.data)
    def forward(self,x,lengths):
        x=self.embed(x)
        packed=nn.utils.rnn.pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,(hn,_)=self.lstm(packed); h=torch.cat([hn[-2],hn[-1]],dim=-1)
        return torch.sigmoid(self.coarse_head(h)+self.fine_head(h))

def get_params(m): return {k:v.cpu().detach().numpy() for k,v in m.state_dict().items()}
def set_params(m,p): m.load_state_dict({k:torch.FloatTensor(v).to(DEVICE) for k,v in p.items()})
def global_tensors(params):
    return {k:torch.FloatTensor(v).to(DEVICE) for k,v in params.items() if 'fine_head' not in k}

log('=== MG-PGSPFL Training ===')
global_model=OursModel(N_ICD,EMB_DIM,HIDDEN_DIM,NUM_LSTM_LAYERS,DROPOUT).to(DEVICE)
global_params=get_params(global_model); GLOBAL_TENSORS=global_tensors(global_params)
criterion=nn.BCELoss()
cnts={c:len(train_patients[c]) for c in centers}; total_tr=sum(cnts.values())

history={'round':[],'train_loss':[],
    'train_recall_at_10':[],'train_recall_at_20':[],
    'train_recall_head_at_10':[],'train_recall_head_at_20':[],
    'train_recall_tail_at_10':[],'train_recall_tail_at_20':[],
    'train_client_recall_at_10':[],'train_client_recall_std_at_10':[],
    'train_client_recall_at_20':[],'train_client_recall_std_at_20':[],
    'test_recall_at_10':[],'test_recall_at_20':[],
    'test_recall_head_at_10':[],'test_recall_head_at_20':[],
    'test_recall_tail_at_10':[],'test_recall_tail_at_20':[],
    'test_client_recall_at_10':[],'test_client_recall_std_at_10':[],
    'test_client_recall_at_20':[],'test_client_recall_std_at_20':[],
    'round_time_s':[],'total_time_min':[]}

t_start=time.time()
try:
    for rnd in range(1,COMM_ROUNDS+1):
        t0=time.time(); client_params=[]; weights=[]; round_losses=[]
        for center in centers:
            if cnts[center]==0: continue
            ds=SeqDataset(train_patients[center]); dl=DataLoader(ds,BATCH_SIZE,shuffle=True,collate_fn=collate)
            model=OursModel(N_ICD,EMB_DIM,HIDDEN_DIM,NUM_LSTM_LAYERS,DROPOUT).to(DEVICE)
            set_params(model,global_params); opt=optim.Adam(model.parameters(),lr=LR); model.train()

            has_target=(center in prev_local_Wc)
            if has_target:
                Wc_target={k:v.to(DEVICE) for k,v in prev_local_Wc[center].items()}

            cl,cs=0.0,0
            for _ in range(LOCAL_EPOCHS):
                for x,lens,y in dl:
                    x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad()
                    pred=model(x,lens); loss_ce=criterion(pred,y)

                    r_prox_head=torch.zeros((),device=DEVICE)
                    r_prox_repr=torch.zeros((),device=DEVICE)
                    for n,p in model.named_parameters():
                        if n not in GLOBAL_TENSORS: continue
                        if n.startswith('embed.') or n.startswith('lstm.'):
                            r_prox_repr=r_prox_repr+((p-GLOBAL_TENSORS[n])**2).sum(); continue
                        d2=(p-GLOBAL_TENSORS[n])**2
                        if n=='coarse_head.linear.weight': r_prox_head=r_prox_head+(d2.sum(dim=1)*FREQ_W).sum()
                        else: r_prox_head=r_prox_head+(d2*FREQ_W).sum()

                    r_cons=torch.zeros((),device=DEVICE)
                    if has_target:
                        for n,p in model.named_parameters():
                            if not n.startswith('fine_head.'): continue
                            kc=n.replace('fine_head','coarse_head')
                            d2=(p-Wc_target[kc])**2
                            if n.endswith('.weight'): r_cons=r_cons+d2.sum()
                            else: r_cons=r_cons+d2.sum()

                    loss=loss_ce+(MU/2)*r_prox_head+(MU_REPR/2)*r_prox_repr+(NU/2)*r_cons
                    loss.backward(); opt.step()
                    cl+=loss_ce.item()*x.size(0); cs+=x.size(0)

            prev_local_Wc[center]={
                'coarse_head.linear.weight':model.coarse_head.linear.weight.detach().cpu().clone(),
                'coarse_head.linear.bias':model.coarse_head.linear.bias.detach().cpu().clone()
            }

            lp=get_params(model)
            f_i=client_freq[center]
            gamma=GAMMA_MIN+(GAMMA_MAX-GAMMA_MIN)*(f_i/max(f_i.max(),1e-8))
            w_fine=lp['fine_head.linear.weight']; b_fine=lp['fine_head.linear.bias']
            lp_scaled=dict(lp)
            lp_scaled['fine_head.linear.weight']=w_fine*gamma[:,np.newaxis]
            lp_scaled['fine_head.linear.bias']=b_fine*gamma
            upload_params={}
            for k,v in lp_scaled.items():
                if 'fine_head' in k:
                    k_coarse=k.replace('fine_head','coarse_head')
                    upload_params[k_coarse]=lp[k_coarse]+v
                elif 'coarse_head' not in k: upload_params[k]=v
            client_params.append(upload_params); weights.append(cnts[center]/total_tr)
            round_losses.append(cl/max(cs,1))
            del model,ds,dl

        tw=sum(weights)
        for k in global_params:
            if 'fine_head' in k or 'coarse_head' in k: continue
            if k in client_params[0]: global_params[k]=sum(cp[k]*(w/tw) for cp,w in zip(client_params,weights))
        if 'coarse_head.linear.weight' in client_params[0]:
            for subk in ['coarse_head.linear.weight','coarse_head.linear.bias']:
                global_params[subk]=sum(cp[subk]*(w/tw) for cp,w in zip(client_params,weights))
        GLOBAL_TENSORS=global_tensors(global_params); set_params(global_model,global_params)

        m_train=evaluate(global_model,train_patients,centers,N_ICD,BATCH_SIZE,DEVICE,SeqDataset,collate)
        m_test=evaluate(global_model,test_patients,centers,N_ICD,BATCH_SIZE,DEVICE,SeqDataset,collate)
        m={'round':rnd}; m['train_loss']=round(np.mean(round_losses),6)
        m['round_time_s']=round(time.time()-t0,1); m['total_time_min']=round((time.time()-t_start)/60,1)
        for k,v in m_train.items(): m[f'train_{k}']=v
        for k,v in m_test.items(): m[f'test_{k}']=v
        for k,v in m.items(): history.setdefault(k,[]).append(v)
        log(f'  Round {rnd:>3}: TrLoss={m["train_loss"]:.4f} | '
            f'Train R@10/20={m["train_recall_at_10"]:.3f}/{m["train_recall_at_20"]:.3f} | '
            f'Test R@10/20={m["test_recall_at_10"]:.3f}/{m["test_recall_at_20"]:.3f} | '
            f'RH@10={m["test_recall_head_at_10"]:.4f} RT@10={m["test_recall_tail_at_10"]:.4f} | '
            f'CR@10={m["test_client_recall_at_10"]:.4f}±{m["test_client_recall_std_at_10"]:.4f} | '
            f'{m["round_time_s"]:.0f}s')
        gc.collect(); torch.cuda.empty_cache()
except Exception:
    import traceback; log('\n=== EXCEPTION ===')
    for line in traceback.format_exc().splitlines(): log(line); raise

pd.DataFrame(history).to_csv(os.path.join(OUT_DIR,'training_history.csv'),index=False)
final={k:v[-1] for k,v in history.items() if k not in ('round','round_time_s','total_time_min')}
total_time=round((time.time()-t_start)/60,1)
log(f'\n=== FINAL after {total_time:.0f} min ===')
for k,v in final.items(): log(f'  {k}: {v:.4f}')
json.dump(final,open(os.path.join(OUT_DIR,'final_metrics.json'),'w'),indent=2)
torch.save(global_params,os.path.join(OUT_DIR,'final_model.pth'))
run_cfg={'run_name':RUN_NAME,'model':'MG-PGSPFL','variant':'Full','emb_dim':EMB_DIM,'hidden_dim':HIDDEN_DIM,
    'dropout':DROPOUT,'batch_size':BATCH_SIZE,'lr':LR,'local_epochs':LOCAL_EPOCHS,'comm_rounds':COMM_ROUNDS,
    'gamma_min':GAMMA_MIN,'gamma_max':GAMMA_MAX,'mu':MU,'nu':NU,'mu_repr':MU_REPR,
    'seed':RANDOM_SEED,'device':str(DEVICE),'total_time_min':total_time}
json.dump(run_cfg,open(os.path.join(OUT_DIR,'run_config.json'),'w'),indent=2)
log(f'\nSaved to {OUT_DIR}')
