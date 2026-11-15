"""Compare lexical and latent features on grouped synthetic intent examples."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

LABELS=['billing','technical','account','delivery']

def load(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def evaluate(rows,seed=42):
    texts=np.array([r['text'] for r in rows]); y=np.array([r['label'] for r in rows])
    groups=np.array([r['family'] for r in rows])
    train,test=next(GroupShuffleSplit(n_splits=1,test_size=.3,random_state=seed).split(texts,y,groups))
    if set(y[train])!=set(LABELS) or set(y[test])!=set(LABELS):
        raise ValueError('Split needs every class in train and test')
    vectorizer=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)
    a=vectorizer.fit_transform(texts[train]);b=vectorizer.transform(texts[test])
    svd=TruncatedSVD(n_components=min(16,a.shape[0]-1,a.shape[1]-1),random_state=seed)
    latent_a=svd.fit_transform(a);latent_b=svd.transform(b)
    report={'seed':seed,'train_size':len(train),'test_size':len(test),
            'train_families':sorted(set(groups[train])),'test_families':sorted(set(groups[test])),
            'labels':LABELS,'results':{}}
    models={}
    for name,x,z in [('tfidf',a,b),('latent_svd',latent_a,latent_b)]:
        clf=LogisticRegression(max_iter=2000,random_state=seed).fit(x,y[train])
        pred=clf.predict(z)
        report['results'][name]={'accuracy':float(accuracy_score(y[test],pred)),
            'macro_f1':float(f1_score(y[test],pred,average='macro')),
            'confusion_matrix':confusion_matrix(y[test],pred,labels=LABELS).tolist()}
        models[name]=clf
    return report,(vectorizer,svd,models)

def predict(bundle,texts):
    vectorizer,svd,models=bundle
    x=vectorizer.transform(texts)
    return {name:clf.predict(x if name=='tfidf' else svd.transform(x)).tolist() for name,clf in models.items()}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default=str(Path(__file__).with_name('data')/'intents.jsonl'))
    p.add_argument('--output',default='results/metrics.json')
    p.add_argument('--text',nargs='+');p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();report,bundle=evaluate(load(a.data),a.seed)
    if a.text: report['predictions']=predict(bundle,a.text)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__': main()
