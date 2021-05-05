import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

import data_utils

def get_raw_Xy_chords(chords_translated, CHORDS_WINDOW, add_start_end=True):
    X_raw, y_raw = [], []
    row_to_song_idx_map = []
    
    for s_idx, chords in enumerate(chords_translated):
        if add_start_end:
            chords_with_end = ["START_0.0", *chords, "END_0.0"]
        else:
            chords_with_end = chords
        for i in range(len(chords_with_end)-CHORDS_WINDOW):
            x = chords_with_end[i:i+CHORDS_WINDOW]
            y = chords_with_end[i+CHORDS_WINDOW]
            X_raw.append(x)
            y_raw.append(y)
            row_to_song_idx_map.append(s_idx)
            
    assert len(X_raw) == len(y_raw) == len(row_to_song_idx_map)
    return X_raw, y_raw, row_to_song_idx_map

def get_chords_CountVectorizer(chords_translated, add_start_end=True, count_vectorizer_kwargs={}):
    if add_start_end:
        chords_flat = [
            ch.split('_')[0] 
            for chords in chords_translated 
            for ch in ["START", *chords, "END"]
        ]
    else:
        chords_flat = [
            ch.split('_')[0] 
            for chords in chords_translated 
            for ch in chords
        ]

    vectorizer = CountVectorizer(
        **count_vectorizer_kwargs
#         lowercase=False, 
#         preprocessor=lambda x : x, 
#         tokenizer=lambda x : x.split()
    )
    vectorizer.fit(chords_flat)
    
    print('Created vocabulary: ')
    print(vectorizer.vocabulary_)
    
    return vectorizer

def _neutral_embedding(embeddings):
    return np.mean(embeddings.wv.vectors, axis=0) / np.std(embeddings.wv.vectors, axis=0)

def _encode_chords_embeddings(rawx, duration_to_id, CHORDS_WINDOW, EMBEDDINGS, ONEHOT_DURATION_COUNT):
    CHORDS_EMB_SIZE = EMBEDDINGS["CHORDS"].vector_size
    chords_neutral_emb = _neutral_embedding(EMBEDDINGS["CHORDS"])
    chords_start_emb = np.zeros(shape=CHORDS_EMB_SIZE) + 10.0
    chords_end_emb = np.zeros(shape=CHORDS_EMB_SIZE) - 10.0
    
    vec = np.zeros(CHORDS_WINDOW*CHORDS_EMB_SIZE + CHORDS_WINDOW*(ONEHOT_DURATION_COUNT + 2))
    
    chs = [chord.split('_')[0] for chord in rawx]
    ds = [chord.split('_')[-1] for chord in rawx]

    of = 0
    for ch in chs:
        try:
            emb = EMBEDDINGS["CHORDS"][ch]
        except KeyError:
            if ch == "START":
                emb = chords_start_emb
            elif ch == "END":
                emb = chords_end_emb
            else:
                emb = chords_neutral_emb
        vec[of:of+CHORDS_EMB_SIZE] = emb
        of += CHORDS_EMB_SIZE

    for d in ds:
        onehot = data_utils._encode_duration(d, duration_to_id)
        vec[of:of+len(onehot)] = onehot
        of += len(onehot)
        vec[of] = float(d)
        of += 1
        
    return vec

def _encode_chords_onehot(rawx, duration_to_id, vectorizer, CHORDS_WINDOW, ONEHOT_DURATION_COUNT):
    VECTORIZER_SIZE = len(vectorizer.vocabulary_)
    vec = np.zeros(CHORDS_WINDOW*VECTORIZER_SIZE + CHORDS_WINDOW*(ONEHOT_DURATION_COUNT + 2))
    chs = [chord.split('_')[0] for chord in rawx]
    ds = [chord.split('_')[-1] for chord in rawx]

    of = 0
    for ch in chs:
        emb = vectorizer.transform([ch]).toarray().reshape(-1)
        vec[of:of+VECTORIZER_SIZE] = emb
        of += VECTORIZER_SIZE

    for d in ds:
        onehot = data_utils._encode_duration(d, duration_to_id)
        vec[of:of+len(onehot)] = onehot
        of += len(onehot)
        vec[of] = float(d)
        of += 1
        
    return vec

def get_X_chords_embeddings(X_raw, duration_to_id, row_to_song_idx_map, CHORDS_WINDOW, ONEHOT_DURATION_COUNT, EMBEDDINGS):
    CHORDS_EMB_SIZE = EMBEDDINGS["CHORDS"].vector_size
    # onehot_duration_count + anything outside this + actual value
    X = np.zeros(
            (
                len(X_raw), 
                CHORDS_WINDOW*CHORDS_EMB_SIZE + CHORDS_WINDOW*(ONEHOT_DURATION_COUNT + 2)
            ), 
            dtype=np.float32
        )
    
    indices_cache = {}
    for idx, x in enumerate(X_raw):
        X[idx] =  _encode_chords_embeddings(x, duration_to_id, CHORDS_WINDOW, EMBEDDINGS, ONEHOT_DURATION_COUNT)
        try:
            indices_cache[row_to_song_idx_map[idx]].append(idx)
        except KeyError:
            indices_cache[row_to_song_idx_map[idx]] = [idx]
        
    return X, indices_cache

def get_X_chords_onehot(X_raw, duration_to_id, row_to_song_idx_map, vectorizer, CHORDS_WINDOW, ONEHOT_DURATION_COUNT):
    X = np.zeros(
            (
                len(X_raw), 
                CHORDS_WINDOW*len(vectorizer.vocabulary_) + CHORDS_WINDOW*(ONEHOT_DURATION_COUNT + 2)
            ), 
            dtype=np.float32
        )
    
    indices_cache = {}
    for idx, x in enumerate(X_raw):
        X[idx] =  _encode_chords_onehot(x, duration_to_id, vectorizer, CHORDS_WINDOW, ONEHOT_DURATION_COUNT)
        try:
            indices_cache[row_to_song_idx_map[idx]].append(idx)
        except KeyError:
            indices_cache[row_to_song_idx_map[idx]] = [idx]
        
    return X, indices_cache

def batch_pad(sample_song, sample_y_p, sample_y_d, batch_size=32):
    ss = sample_song.copy()
    syp = sample_y_p.copy()
    syd = sample_y_d.copy()
    
    pad = batch_size - (ss.shape[0] % batch_size)
    print(pad)
    ss = np.concatenate(
        (
            ss, 
            np.zeros(shape=(pad, ss.shape[-1]))
        )
    )

    pad_onehot = np.zeros(shape=(pad, syp.shape[-1]))
    pad_onehot[-1] = 1.0
    syp = np.concatenate((syp, pad_onehot))

    pad_onehot = np.zeros(shape=(pad, syd.shape[-1]))
    pad_onehot[-1] = 1.0
    syd = np.concatenate((syd, pad_onehot))
    
    return ss, syp, syd