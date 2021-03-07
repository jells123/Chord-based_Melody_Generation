import numpy as np
import music21 as mc
import data_utils

def newsong_to_midistream(new_notes, song_chords, tr_chords_dict=None):
    newsong_stream = []
    off = 0.0

    for note in new_notes:
        p, d = note.split('_')
        d = round(float(d), 2)
        if p != "REST":
            new_note = mc.note.Note(p)
        else:
            new_note = mc.note.Rest()
        new_note.quarterLength = d
        newsong_stream.append(new_note)

        new_note.offset = off
        off += d

    off = 0.0
    for ch in song_chords:
        if ch.split('_')[0] in ["START", "END"] or ch.split('_')[-1] in ["0.0"]:
            continue
            
        if tr_chords_dict:
            pitch_names = tr_chords_dict[ch.split("_")[0]].split("+")
            pitches = [mc.note.Note(ptch).pitch.ps for ptch in pitch_names]
        else:
            pitches = ch.split('_')[0].split('+')
                
        d = float(ch.split('_')[1])
                
        if pitches[0] == "-1.0":
            new_chord = mc.note.Rest()
        else:
            new_chord = mc.chord.Chord([mc.note.Note(float(p)-12.0).nameWithOctave for p in pitches])
        new_chord.quarterLength = d
        newsong_stream.append(new_chord)

        new_chord.offset = off
        off += d
        
    return mc.stream.Stream(newsong_stream)

def sample_OLD(a, temperature=1.0):
    a = np.log(a) / temperature
    a = np.exp(a)
    a = a.astype(float)
    a /= a.sum()
    return np.argmax(np.random.multinomial(1, a.reshape(-1), 1))

def sample(a, temperature=1.0):
    result = np.log(a + 1e-12) / temperature
    result = np.exp(result)
    result = result.astype(float)
    result /= result.sum()
    try:
        sample = np.argmax(np.random.multinomial(1, result.reshape(-1), 1))
        return sample
    except Exception as ex:
        print(result)
        print(a)
        # ...
        raise(ex)

def predict_melody(mode, seed_notes, chords, model, id_to_token_pitches, id_to_token_durations, embeddings=None, vectorizer=None, scaler=None, notes_window=8, chords_window=3, chord_size=4, chords_mapping={}, temperature=None, duration_to_id_notes=None, duration_to_id_chords=None):
    # prediction starts here
    model.reset_states()

    note_off, chord_off, chidx = 0.0, 0.0, 0
    while len(seed_notes) < notes_window:
        seed_notes.insert(0, "NONE_0.0")
    current_notes = seed_notes.copy()
    ds = [float(note.split('_')[-1]) for note in current_notes]
    note_off = sum(ds)
    melody = seed_notes.copy()
    
    sample_note = current_notes[-1].split('_')[0]
    translate_notes = False
    if sample_note.replace('#', '').replace('-', '').isalnum():
        translate_notes = True

    while chidx < len(chords):

        while chord_off <= note_off and chidx < len(chords):
            chord_off += float(chords[chidx].split('_')[-1])
            chidx += 1

        # ____ CURRENT CHORD
        current_chord = chords[chidx-1]
        current_chord_rem = chord_off-note_off

        # ____ NEXT_CHORD
        if chidx < len(chords):
            next_chord = chords[chidx]
        else:
            next_chord = 0

        # ____ PREVIOUS CHORD
        prev_chords = chords[max(0, chidx-1-(chords_window-1)):max(0, chidx-1)]
        while len(prev_chords) < chords_window-1:
            prev_chords.insert(0, 0)

        prepared_chords = [*prev_chords, (current_chord, current_chord_rem), next_chord]
        
        if mode == "pitches":
            seed = scaler.transform(
                    _seed_melody_pitches(
                        current_notes, 
                        prepared_chords, 
                        translate_notes=translate_notes, 
                        notes_window=notes_window, 
                        chords_window=chords_window, 
                        chord_size=chord_size, 
                        chords_mapping=chords_mapping
                    )
                )
        elif mode == "embeddings":
            seed = _seed_melody_embeddings(
                        current_notes, 
                        prepared_chords, 
                        embeddings,
                        translate_notes=translate_notes, 
                        notes_window=notes_window, 
                        chords_window=chords_window, 
                    )
        elif mode == "custom":
            seed = _seed_melody_custom(
                current_notes, 
                prepared_chords, 
                embeddings, 
                vectorizer, 
                duration_to_id_notes, 
                duration_to_id_chords, 
                is_start=None, 
                translate_notes=translate_notes, 
                notes_window=8, 
                chords_window=3
            )
        
        pred_p_proba, pred_d_proba = model.predict(seed.reshape(1, 1, -1))
        pred_p_proba = pred_p_proba.reshape(-1)
        pred_d_proba = pred_d_proba.reshape(-1)
        
        if not temperature:
            pred_p, pred_d = id_to_token_pitches[np.argmax(pred_p_proba)], id_to_token_durations[np.argmax(pred_d_proba)]
            if pred_p == "UNK":
                pred_p_proba[np.argmax(pred_p_proba)] = 0.0
                pred_p = id_to_token_pitches[np.argmax(pred_p_proba)]
            if pred_d == "UNK":
                pred_d_proba[np.argmax(pred_d_proba)] = 0.0
                pred_d = id_to_token_pitches[np.argmax(pred_d_proba)]
        else:
            sample_p = sample(pred_p_proba, temperature=temperature)
            while id_to_token_pitches[sample_p] in ["UNK", "NOTHING"]:
                pred_p_proba[sample_p] = 0.0
                pred_p_proba = pred_p_proba / pred_p_proba.sum()
                sample_p = sample(pred_p_proba, temperature=temperature)
            pred_p = id_to_token_pitches[sample_p]
            
            sample_d = sample(pred_d_proba, temperature=temperature)
            while id_to_token_durations[sample_d] in ["UNK", "0.0", "NOTHING"]:
                pred_d_proba[sample_d] = 0.0
                pred_d_proba = pred_d_proba / pred_d_proba.sum()
                sample_d = sample(pred_d_proba, temperature=temperature)
            pred_d = id_to_token_durations[sample_d]
            
#             pred_p = id_to_token_pitches[sample(pred_p_proba, temperature=temperature)]
#             pred_d = id_to_token_durations[sample(pred_d_proba, temperature=temperature)]
            
#         while pred_p == "UNK":
#             prev_best = np.argmax(pred_p_proba)
#             pred_p_proba[prev_best] = 0.0
#             pred_p = id_to_token_pitches[np.argmax(pred_p_proba)]
#         while pred_d == "UNK":
#             prev_best = np.argmax(pred_d_proba)
#             pred_d_proba[prev_best] = 0.0
#             pred_d = id_to_token_durations[np.argmax(pred_d_proba)]

        new_note = f"{pred_p}_{pred_d}"

        current_notes[:notes_window-1] = current_notes[1:]
        current_notes[-1] = new_note
        note_off += float(pred_d)
        melody.append(new_note)
        
    return melody
        
def _seed_melody_pitches(current_notes, prepared_chords, is_start=0, translate_notes=True, notes_window=8, chords_window=3, chord_size=4, chords_mapping={}):
    return data_utils._encode_pitches(
                current_notes, 
                prepared_chords, 
                is_start=is_start,
                translate_notes=translate_notes, 
                notes_window=notes_window, 
                chords_window=chords_window, 
                chord_size=chord_size,
                chords_mapping=chords_mapping
            ).reshape(1, -1)
        

def _seed_melody_embeddings(seed_notes, prepared_chords, embeddings, is_start=0, translate_notes=True, notes_window=8, chords_window=3):
    return data_utils._encode_embeddings(
        seed_notes, 
        prepared_chords, 
        embeddings, 
        is_start=is_start, 
        translate_notes=translate_notes, 
        notes_window=notes_window, 
        chords_window=chords_window
    ).reshape(1, -1)

def _seed_melody_custom(seed_notes, prepared_chords, embeddings, vectorizer, duration_to_id_notes, duration_to_id_chords, is_start=None, translate_notes=True, notes_window=8, chords_window=3):
    return data_utils._encode_custom(
        seed_notes, 
        prepared_chords, 
        embeddings, 
        vectorizer,
        duration_to_id_notes,
        duration_to_id_chords,
        translate_notes=translate_notes, 
        notes_window=notes_window, 
        chords_window=chords_window
    ).reshape(1, -1)
