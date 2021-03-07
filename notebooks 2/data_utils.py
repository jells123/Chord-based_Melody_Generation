import os
import json
import music21 as mc
import numpy as np
from collections import Counter

import seaborn as sns
sns.set()

import random
random.seed(20201308)

from gensim.models import Word2Vec


def get_duration_to_id(X_raw, ONEHOT_DURATION_COUNT):
    x_durations = []
    for x in X_raw:
        for xx in x:
            x_durations.append(xx.split('_')[-1])
    print("Found", len(set(x_durations)), "unique durations")
    most_common_d = sorted([pair[0] for pair in Counter(x_durations).most_common(ONEHOT_DURATION_COUNT)])
    d_to_id = {d : idx for idx, d in enumerate(most_common_d)}
    return d_to_id

def _encode_duration(d, duration_to_id):
    onehot = np.zeros(len(duration_to_id)+1)
    d_str = str(d)
    try:
        d_idx = duration_to_id[d_str]
    except KeyError: #outside
        d_idx = len(duration_to_id)
    onehot[d_idx] = 1.0
    return onehot

def get_parsed_data(data_path):
    with open(os.path.join(data_path, "parse_4.json"), "r") as handle:
        return json.load(handle)

def _fix_with_most_common(chords_mapping_dict):
    chords_mapping_dict['F major seventh chord'] = 'C+E+F+A'
    chords_mapping_dict['C major second major tetrachord'] = 'C+D+E+G'
    chords_mapping_dict['enharmonic equivalent to major triad above C#'] = 'C#+F+G#'
    chords_mapping_dict['C major seventh chord'] = 'C+E+G+B'
    chords_mapping_dict['Bb major seventh chord'] = 'D+F+A+B-'
    chords_mapping_dict['F# half diminished seventh chord'] = 'C+E+F#+A'
    chords_mapping_dict['F phrygian tetrachord'] = 'E+F+G+A'
    chords_mapping_dict['C# quartal tetramirror'] = 'C#+E+F#+B'
    chords_mapping_dict['B tritone fourth'] = 'C+F+B'
    chords_mapping_dict['C minor seventh chord'] = 'C+E-+F+G#'
    chords_mapping_dict['Eb quartal trichord'] = 'E-+F+B-'
    chords_mapping_dict['enharmonic equivalent to minor triad above Eb'] = 'E-+G#+B'
    chords_mapping_dict['Eb major seventh chord'] = 'D+E-+G+B-'
    chords_mapping_dict['C# minor seventh chord'] = 'C#+E+G#+B'
    chords_mapping_dict['Eb phrygian tetrachord'] = 'D+E-+F+G'
    chords_mapping_dict['C# major second major tetrachord'] = 'C#+E-+F+G#'
    chords_mapping_dict['E tritone fourth'] = 'E+F+B-'
    chords_mapping_dict['enharmonic to dominant seventh chord above C#'] = 'C#+E-+G+B-'
    chords_mapping_dict['A tritone fourth'] = 'E-+E+A'
    chords_mapping_dict['C# half diminished seventh chord'] = 'C#+E+G+B'
    chords_mapping_dict['E all interval tetrachord'] = 'E+F+G+B'
    chords_mapping_dict['C phrygian tetrachord'] = 'C+D+E+B'
    chords_mapping_dict['E whole tone tetramirror'] = 'D+E+F#+G#'
    chords_mapping_dict['A all interval tetrachord'] = 'C#+E-+E+A'
    chords_mapping_dict['G# minor second diminished tetrachord'] = 'D+G#+A+B'
    chords_mapping_dict['C perfect fourth major tetrachord'] = 'C+E+F+G'
    chords_mapping_dict['G# whole tone tetramirror'] = 'C+D+G#+B-'
    chords_mapping_dict['enharmonic equivalent to diminished triad above C#'] = 'C#+E+B-'
    chords_mapping_dict['C all interval tetrachord'] = 'C+D+E-+G#'
    chords_mapping_dict['C# minor tetramirror'] = 'C#+G#+B-+B'
    chords_mapping_dict['A double fourth tetramirror'] = 'C#+D+G#+A'
    chords_mapping_dict['D all interval tetrachord'] = 'D+E-+F+A'
    chords_mapping_dict['F augmented major tetrachord'] = 'C#+F+G#+A'
    chords_mapping_dict['C minor second quartal tetrachord'] = 'C+D+G+G#'
    chords_mapping_dict['F# minor second diminished tetrachord'] = 'C+F#+G+A'
    chords_mapping_dict['F# tritone fourth'] = 'C+F#+G'
    chords_mapping_dict['C augmented seventh chord'] = 'C+D+E+G#'
    chords_mapping_dict['Eb minor second quartal tetrachord'] = 'D+E-+G+A'
    chords_mapping_dict['Eb perfect fourth major tetrachord'] = 'E-+E+F#+B'
    chords_mapping_dict['Eb tritone fourth'] = 'D+E-+G#'
    chords_mapping_dict['A tritone quartal tetrachord'] = 'C#+D+G+A'
    chords_mapping_dict['F# minor tetramirror'] = 'E+F#+G+A'
    chords_mapping_dict['Eb double fourth tetramirror'] = 'D+E-+G+G#'
    chords_mapping_dict['C# tritone fourth'] = 'C#+D+G'
    chords_mapping_dict['C# perfect fourth minor tetrachord'] = 'C#+E-+F+B-'
    chords_mapping_dict['F augmented seventh chord'] = 'C#+E-+F+A'
    chords_mapping_dict['Bb all interval tetrachord'] = 'D+E+F+B-'
    chords_mapping_dict['Bb augmented major tetrachord'] = 'D+F#+A+B-'
    chords_mapping_dict['C# minor second diminished tetrachord'] = 'C#+E+B-+B'
    return chords_mapping_dict
    
def get_chords_mapping(data_path):
    with open(os.path.join(data_path, "chord_names.json"), "r") as handle:
        content = json.load(handle)
    new_dict = {content[key] : key for key in content}
    new_dict = _fix_with_most_common(new_dict)
    return new_dict
    
def name_to_ps(name):
    return mc.pitch.Pitch(name).ps if name != "REST" else -1.0

def unify_note(note):
    name, duration = note.split('_')
    pitch = name_to_ps(name)
    return f"{pitch}_{duration}"

chord_names_dict = {}

def fix_equivalent_name(name):
    part2, part1 = name.split('enharmonic equivalent to ')[-1].split(' above ')
    return ' '.join([part1, part2])

def unify_chord(chord, truncate_chord=4):
    names, duration = chord.split('_')
    new_names = []

    for name in names.split('.'):
        if name not in new_names:
            new_names.append(name)
    new_names = new_names[:truncate_chord]

    new_name = '+'.join(new_names)
    global chord_names_dict    
    if new_name not in chord_names_dict:
        pitched_common_name = mc.chord.Chord(new_names).pitchedCommonName.replace('-', ' ')
#         if 'equivalent' in pitched_common_name:
#             pitched_common_name = fix_equivalent_name(pitched_common_name)
        chord_names_dict[new_name] = pitched_common_name
        
    return f"{chord_names_dict[new_name]}_{duration}"

def select_test(parsed_data, data_path, chorus_verse_only=True):
    if os.path.isfile(os.path.join(data_path, "test_idx.json")):
        with open(os.path.join(data_path, "test_idx.json"), "r") as handle:
            TEST_SELECT = json.load(handle)
            print("Loaded TEST indices")
    else:
        TEST_SIZE = int(len(parsed_data['parse_success']) * 0.1)
        TEST_SELECT = random.sample(range(0, len(parsed_data['parse_success'])), TEST_SIZE)

        with open(os.path.join(PATH, "test_idx.json"), "w") as handle:
            json.dump(TEST_SELECT, handle)
            
    result = []
    for idx in TEST_SELECT:
        path = parsed_data["parse_success"][idx]
        if (
            chorus_verse_only and ("chorus" in path.lower() or "verse" in path.lower())
        ) or not chorus_verse_only:
            result.append(idx)
            
    return result

def select_train(parsed_data, artist="*", test_select=None, chorus_verse_only=True):
    TRAIN_SELECT = []
    for idx, path in enumerate(parsed_data["parse_success"]):
        if (
            (isinstance(artist, str) and (artist == "*" or f"/{artist}/" in path))
            or
            (isinstance(artist, list) and any(f"/{ar}/" in path for ar in artist))
        ) and (idx not in test_select or not test_select):
            if (
                chorus_verse_only and ("chorus" in path.lower() or "verse" in path.lower())
            ) or not chorus_verse_only:
                TRAIN_SELECT.append(idx)
    return TRAIN_SELECT

def select_songs(parsed_data, SELECT, REPEAT=1):
    SONGS = []
    notes_translated, chords_translated = [], []
    for idx, (notes, chords) in enumerate(zip(parsed_data['notes'], parsed_data['chords'])):
        if idx in SELECT:
            notes_translated.append([unify_note(n) for n in notes] * REPEAT)
            chords_translated.append([unify_chord(n) for n in chords] * REPEAT)
            SONGS.append(parsed_data["parse_success"][idx])
    return SONGS, notes_translated, chords_translated

def get_train_songs(parsed_data, TRAIN_SELECT, REPEAT=1):
    TRAIN_SONGS = []
    notes_translated, chords_translated = [], []
    for idx, (notes, chords) in enumerate(zip(parsed_data['notes'], parsed_data['chords'])):
        if idx in TRAIN_SELECT:
            notes_translated.append([unify_note(n) for n in notes] * REPEAT)
            chords_translated.append([unify_chord(n) for n in chords] * REPEAT)
            TRAIN_SONGS.append(parsed_data["parse_success"][idx])
    return TRAIN_SONGS, notes_translated, chords_translated

def get_test_songs(parsed_data, TEST_SELECT, REPEAT=1):
    TEST_SONGS = []
    notes_translated_test, chords_translated_test = [], []
    for idx, (notes, chords) in enumerate(zip(parsed_data['notes'], parsed_data['chords'])):
        if idx in TEST_SELECT:
            notes_translated_test.append([unify_note(n) for n in notes])
            chords_translated_test.append([unify_chord(n) for n in chords])
            TEST_SONGS.append(parsed_data["parse_success"][idx])
    return TEST_SONGS, notes_translated_test, chords_translated_test

def load_embeddings(data_path, embeddings_dir="embeddings", emb_size=100, emb_window=8):
    models = {}
    notes_emb_filename = "embeddings_NOTES_size75_window8_CBOW_H-SFMAX.bin"
    chords_emb_filename = "embeddings_CHORDS-SHORT_size100_window2_CBOW_NEG-S.bin"
    print("> Load notes embeddings:", notes_emb_filename)
    print("> Load chords embeddings:", chords_emb_filename)
    models = {
        "NOTES": Word2Vec.load(os.path.join(data_path, embeddings_dir, "NOTES", notes_emb_filename)),
        "CHORDS": Word2Vec.load(os.path.join(data_path, embeddings_dir, "CHORDS", chords_emb_filename))
    }    
    return models

def load_embeddings_OLD(data_path, embeddings_dir="embeddings", emb_size=100, emb_window=8):
    models = {}
    assert emb_size == 100
    assert emb_window == 8

    for emb_type in ["NOTES",]: #, "CHORDS"]:
        emb_filename = f"embeddings_{emb_type}_size{emb_size}_window{emb_window}.bin"
        model = Word2Vec.load(os.path.join(data_path, embeddings_dir, emb_type, emb_filename))
        models[emb_type] = model
    
    custom_embeddings = "embeddings_CHORDS-SHORT_size100_window4_CBOW_H-SFMAX.bin"
    models["CHORDS"] = Word2Vec.load(os.path.join(data_path, embeddings_dir, "CHORDS", custom_embeddings))
    
    return models

def decode_note(note):
    pitch, duration = note.split('_')
    return pitch, float(duration)

def align_notes_to_chords(notes, chords):
    note_to_chord_align = []

    chord_idx = 0
    _, current_chord_duration = decode_note(chords[chord_idx])

    offset = 0.0
    for idx, note in enumerate(notes):
        current_align = []
        _, duration = decode_note(note)
        current_align.append(chord_idx)
        offset += duration

        while current_chord_duration < offset:
            if chord_idx == len(chords)-1:
                break
            chord_idx += 1
            _, nd = decode_note(chords[chord_idx])
            current_chord_duration += nd
            current_align.append(chord_idx)

        note_to_chord_align.append(current_align)

    return note_to_chord_align

def extract_notes_pitches_durations(notes_slice):
    input_pitches = []
    input_durations = []
    for n in notes_slice:
        pitch, duration = n.split('_')
        input_pitches.append(float(pitch))
        input_durations.append(float(duration))
    return input_pitches, input_durations

def filter_out_chords(chords, window_size):
    if window_size == 1:
        return [chords[-1]] # only last chord - list for compatibility? 
    elif window_size == 2:
        return [chords[0], chords[-1]] # first and last only
    else:
        result = []
        durations = np.array([decode_note(chord)[1] for chord in chords[1:-1]])
        top_durations_arg = np.argpartition(durations, -(window_size-2))[-(window_size-2):]
        top_durations = durations[top_durations_arg]
        for idx, d in enumerate(durations):
            if d in top_durations and len(result) < window_size-2:
                result.append(chords[idx])
            if len(result) >= window_size:
                break

        # always include the first and the last chord    
        result = [chords[0], *result, chords[-1]]
        assert len(result) == window_size
        return result
    
def find_aligned_chords(aligns, target_idx, chords_window=4):
    # this operates on INDICES of chords! this is what 'aligns' store
    target_aligned_chords = aligns[target_idx]
    j = 1
    while len(target_aligned_chords) < chords_window and target_idx-j > 0:
        more_chords = aligns[target_idx-j]
        for ch in more_chords[::-1]:
            if ch not in target_aligned_chords:
                target_aligned_chords.insert(0, ch)
            if len(target_aligned_chords) == chords_window:
                break
        j += 1
    
    j = 1
    next_chord = None
    while not next_chord and target_idx+j < len(aligns):
        next_note_aligned_chords = aligns[target_idx+j]
        for ch in next_note_aligned_chords:
            if ch != target_aligned_chords[-1]:
                next_chord = ch
                break
        j += 1

    return target_aligned_chords, next_chord

def chord_pitches_to_fixed(pitches, pitches_per_chord=4):
    pitches = pitches.split('+')
    result = [0] * pitches_per_chord
    for i in range(min(pitches_per_chord, len(pitches))):
        result[i] = float(pitches[i])
    return result

def _normalize_octave(note, min_octave=3, max_octave=6):
    if note.octave < min_octave:
        note.octave = min_octave
    if note.octave > max_octave:
        note.octave = max_octave
    return note

def _note_pitched_to_named(note):
    pitched, duration = note.split('_')
    if pitched.replace('#', '').replace('-', '').isalnum():
        # is 'C3'
        name = pitched
    else:
        # probably can convert to float
        pitched = float(pitched)
        if pitched == -1.0:
            name = "REST"
        else:
            mc_note = mc.note.Note(pitched)
            mc_note = _normalize_octave(mc_note)
            name = mc_note.nameWithOctave
    return '_'.join([name, duration])

def _split_notes(notes):
    splitted = [nt.split('_') for nt in notes]
    pitches = [s[0] for s in splitted]
    durations = [s[1] for s in splitted]
    return pitches, durations

def prepare_raw_X_y(input_notes, input_chords, notes_window=8, chords_window=4, use_next_chord=True, rep=1):
    X_notes_raw, X_chords_raw, y_raw = [], [], []
    is_start, row_to_song_idx_map = [], []
    
    translate_notes = _should_translate_notes(input_notes[-1][-1])

    for s_dx, (ns, chs) in enumerate(zip(input_notes, input_chords)):
        notes = (ns.copy()) * rep
        chords = (chs.copy()) * rep
        
        note_off = 0.0
        chord_off = 0.0
        chidx = 0
        for i in range(notes_window, len(notes)):
            # ____ NOTES
            notes_slice = notes[i-notes_window:i]
            notes_slice = [_note_pitched_to_named(nt) for nt in notes_slice]
            # ____ TARGET
            target = _note_pitched_to_named(notes[i])

            ds = [float(note.split('_')[-1]) for note in notes_slice]
            if note_off == 0.0:
                note_off += sum(ds)

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

            X_notes_raw.append(notes_slice)
            X_chords_raw.append([*prev_chords, (current_chord, current_chord_rem), next_chord])
            y_raw.append(target)

            target_d = float(target.split('_')[-1])
            note_off += target_d

            is_start.append(1 if i == 0 else 0)
            row_to_song_idx_map.append(s_dx)

    assert len(X_notes_raw) == len(X_chords_raw) == len(y_raw) == len(row_to_song_idx_map)
    return X_notes_raw, X_chords_raw, y_raw, is_start, row_to_song_idx_map

def trim_rare_labels(X_notes_raw, X_chords_raw, y_raw, is_start, row_to_song_idx_map, min_freq=3, unk="UNK"):
    y_counts = Counter(y_raw)
    X_notes_new, X_chords_new, y_new, is_start_new, row_to_song_idx_new = [], [], [], [], []

    for xn, xch, y, is_st, s_idx in zip(X_notes_raw, X_chords_raw, y_raw, is_start, row_to_song_idx_map):
        X_notes_new.append(xn)
        X_chords_new.append(xch)
    
        if y_counts[y] >= min_freq:
            y_new.append(y)
        else:
            y_new.append(unk)

        is_start_new.append(is_st)
        row_to_song_idx_new.append(s_idx)

    return X_notes_new, X_chords_new, y_new, is_start_new, row_to_song_idx_new

def trim_rare_labels_separately(y_pitches_raw, y_durations_raw, min_freq=3, unk="UNK", keep=["START", "END", "0.0"]):
    y_p_counts = Counter(y_pitches_raw)
    y_d_counts = Counter(y_durations_raw)
    y_pitches_new, y_durations_new = [], []

    for y_p, y_d in zip(y_pitches_raw, y_durations_raw):
        if y_p_counts[y_p] >= min_freq or y_p in keep:
            y_pitches_new.append(y_p)
        else:
            y_pitches_new.append(unk)
            
        if y_d_counts[y_d] >= min_freq or y_d in keep:
            y_durations_new.append(y_d)
        else:
            y_durations_new.append(unk)

    return y_pitches_new, y_durations_new

_note_name_to_pitch_cache = {}
def _note_name_to_pitch(note_name):
    global _note_name_to_pitch_cache
    if note_name == 'REST':
        return -1.0
    else:
        if note_name not in _note_name_to_pitch_cache:
            _note_name_to_pitch_cache[note_name] = mc.note.Note(note_name).pitch.ps
        return _note_name_to_pitch_cache[note_name]

def prepare_input_X(X_notes_raw, X_chords_raw, is_start, row_to_song_idx, mode="", notes_window=8, chords_window=4, chord_size=4, indices_cache={}, embeddings=None, vectorizer=None, chords_mapping={},
duration_to_id_notes=None, duration_to_id_chords=None,):
    if mode.lower() == "embeddings":
        return _prepare_input_X_embeddings(
            X_notes_raw, 
            X_chords_raw, 
            is_start, 
            row_to_song_idx, 
            embeddings, 
            notes_window=notes_window, 
            chords_window=chords_window, 
            indices_cache=indices_cache
        )
    elif mode.lower() == "pitches":
        return _prepare_input_X_pitches(
            X_notes_raw, 
            X_chords_raw, 
            is_start, 
            row_to_song_idx, 
            notes_window=notes_window, 
            chords_window=chords_window, 
            indices_cache=indices_cache,
            chord_size=chord_size,
            chords_mapping=chords_mapping
        )
    elif mode.lower() == "custom":
        return _prepare_input_X_custom(
            X_notes_raw, 
            X_chords_raw, 
            row_to_song_idx, 
            embeddings,
            vectorizer,
            duration_to_id_notes,
            duration_to_id_chords,
            notes_window=notes_window, 
            chords_window=chords_window, 
            indices_cache=indices_cache,
        )
    
def _neutral_embedding(embeddings):
    return np.mean(embeddings.wv.vectors, axis=0) / np.std(embeddings.wv.vectors, axis=0)

def _chord_emb(chords, ch_embeddings):
    ch_emb = []
    for ch in chords:
        if ch != 0:
            if isinstance(ch, str):
                ch_name = ch.split('_')[0]
            elif isinstance(ch, tuple):
                ch_name = ch[0].split('_')[0]
            try:
                ch_emb.append(ch_embeddings[ch_name])
            except KeyError:
                ch_emb.append(_neutral_embedding(ch_embeddings))
        elif ch == 0:
            ch_emb.append([0] * ch_embeddings.vector_size)
        else:
            raise Exception(f"CHORD is {ch}!")
    return ch_emb

def _encode_embeddings(notes, chords, embeddings, is_start=False, translate_notes=False, notes_window=8, chords_window=3):
    notes_emb_size = embeddings["NOTES"].vector_size
    chords_emb_size = embeddings["CHORDS"].vector_size
    
    vector = np.zeros(
        notes_window*notes_emb_size + notes_window + (chords_window+1)*chords_emb_size + (chords_window+1) + 1 + 1, 
        dtype=np.float32
    )
    of = 0
    
    # NOTE PITCHES
    n_pitches = [note.split('_')[0] for note in notes]
    if translate_notes:
        n_pitches = [str(_note_name_to_pitch(n)) for n in n_pitches]
        
    n_emb = []
    for n in n_pitches:
        try:
            n_emb.append(embeddings["NOTES"][n])
        except KeyError:
            n_emb.append(_neutral_embedding(embeddings["NOTES"]))
            
    vector[of:of+(notes_window*notes_emb_size)] = np.array(n_emb).reshape(-1)
    of += (notes_window*notes_emb_size)
    
    # NOTE DURATIONS
    vector[of:of+notes_window] = [float(note.split('_')[1]) for note in notes]
    of += notes_window

    # CHORDS PITCHES
    assert len(chords) == chords_window+1
    ch_emb = _chord_emb(chords, embeddings["CHORDS"])
    vector[of:of + (chords_window+1)*chords_emb_size] = np.array(ch_emb).reshape(-1)
    of += (chords_window+1)*chords_emb_size

    # CHORDS DURATIONS
    ch_durations = []
    for ch in chords:
        if ch != 0:
            if isinstance(ch, str):
                d = [float(ch.split('_')[1])]
            elif isinstance(ch, tuple):
                d1 = float(ch[0].split('_')[1]) # full duration
                d2 = ch[1] # currnt duration
                d = [d1, d2]
            ch_durations.extend(d)
        elif ch == 0:
            ch_durations.append(0)
        else:
            raise Exception(f"CHORD is {ch}!")

    vector[of:of + (chords_window+1+1)] = np.array(ch_durations)
    of += (chords_window+1+1)
    
    # IS START
    vector[-1] = 1 if is_start else 0
    
    return vector

def _encode_custom(notes, chords, embeddings, vectorizer, duration_to_id_notes, duration_to_id_chords, translate_notes=False, notes_window=8, chords_window=3):
    notes_emb_size = embeddings["NOTES"].vector_size
    vectorizer_size = len(vectorizer.vocabulary_)
    
    vector = np.zeros(
        notes_window*notes_emb_size + notes_window*(len(duration_to_id_notes)+2) + (chords_window+1)*vectorizer_size + (chords_window+1+1)*(len(duration_to_id_chords)+2), 
        dtype=np.float32
    )
    of = 0
    
    # NOTE PITCHES
    n_pitches = [note.split('_')[0] for note in notes]
    if translate_notes:
        n_pitches = [str(_note_name_to_pitch(n)) for n in n_pitches]
        
    n_emb = []
    for n in n_pitches:
        try:
            n_emb.append(embeddings["NOTES"][n])
        except KeyError:
            n_emb.append(_neutral_embedding(embeddings["NOTES"]))
            
    vector[of:of+(notes_window*notes_emb_size)] = np.array(n_emb).reshape(-1)
    of += (notes_window*notes_emb_size)
    
    # NOTE DURATIONS
    for d in [note.split('_')[1] for note in notes]:
        onehot = _encode_duration(d, duration_to_id_notes)
        vector[of:of+len(onehot)] = onehot
        of += len(onehot)
        vector[of] = float(d)
        of += 1

    # CHORDS PITCHES
    assert len(chords) == chords_window+1
    for ch in chords:
        if isinstance(ch, str):
            emb = vectorizer.transform([ch]).toarray().reshape(-1)
        else:
            emb = [0] * vectorizer_size
        vector[of:of+vectorizer_size] = emb
        of += vectorizer_size

    # CHORDS DURATIONS
    ch_durations = []
    for ch in chords:
        if ch != 0:
            if isinstance(ch, str):
                d = ch.split('_')[1]
                d_vec = [*_encode_duration(d, duration_to_id_chords), float(d)]
            elif isinstance(ch, tuple):
                d1 = ch[0].split('_')[1] # full duration
                d2 = ch[1] # currnt duration
                d_vec = [*_encode_duration(d1, duration_to_id_chords), float(d1), *_encode_duration(d2, duration_to_id_chords), float(d2)]
            ch_durations.extend(d_vec)
        elif ch == 0:
            ch_durations.extend([0] * (len(duration_to_id_chords) + 1 + 1))
        else:
            raise Exception(f"CHORD is {ch}!")
            
    assert len(ch_durations) == (chords_window+1+1) * (len(duration_to_id_chords)+1+1)
    vector[of:of + len(ch_durations)] = np.array(ch_durations)
    of += len(ch_durations)
    
    assert of == len(vector)
    return vector
    
def _prepare_input_X_embeddings(X_notes_raw, X_chords_raw, is_start, row_to_song_idx, embeddings, notes_window, chords_window, indices_cache):
    notes_emb_size = embeddings["NOTES"].vector_size
    chords_emb_size = embeddings["CHORDS"].vector_size
    
    chords_neutral_emb = _neutral_embedding(embeddings["CHORDS"])
    chord_miss_count = 0
    
    X = np.zeros(
        (
                len(X_notes_raw), 
                                                        #embed each chord   #prev, current & next duration: current x2!
                notes_window*notes_emb_size + notes_window + (chords_window+1)*chords_emb_size + (chords_window+1) + 1 + 1 # + is_start
        ), 
        dtype=np.float32
    )

    # NOTE PITCH, NOTE DURATION, CHORD PITCHES, CHORD DURATION, IS_START

    iterator = list(enumerate(zip(X_notes_raw, X_chords_raw, is_start, row_to_song_idx)))
        
    sample_note = X_notes_raw[-1][-1].split('_')[0]
    translate_notes = _should_translate_notes(X_notes_raw[-1][-1])
    
    for idx, (notes, chords, st, song_idx) in iterator:
        X[idx, :] = _encode_embeddings(
            notes, 
            chords, 
            embeddings, 
            is_start=False, 
            translate_notes=translate_notes, 
            notes_window=notes_window, 
            chords_window=chords_window, 
        )

        # CACHE HELPER
        try:
            indices_cache[song_idx].append(idx)
        except KeyError:
            indices_cache[song_idx] = [idx]

    # shrink cache helper
    for idx in indices_cache:
        indices_cache[idx] = [indices_cache[idx][0], indices_cache[idx][-1]]

    print(f"No. UNK chords: {chord_miss_count}")
    return X, indices_cache

def _should_translate_notes(sample):
    sample_note = sample.split('_')[0].replace('-', '').replace('#', '')
    if sample_note.isalnum() or sample_note == "REST":
        return True
    return False

def _prepare_input_X_custom(
    X_notes_raw, 
    X_chords_raw, 
    row_to_song_idx, 
    embeddings, 
    vectorizer, 
    duration_to_id_notes,
    duration_to_id_chords,
    notes_window, 
    chords_window, 
    indices_cache
):
    notes_emb_size = embeddings["NOTES"].vector_size
    chords_neutral_emb = _neutral_embedding(embeddings["CHORDS"])
    vectorizer_size = len(vectorizer.vocabulary_)
    
    X = np.zeros(
        (
                len(X_notes_raw), 
                notes_window*notes_emb_size + notes_window*(len(duration_to_id_notes)+2) + (chords_window+1)*vectorizer_size + (chords_window+1+1)*(len(duration_to_id_chords)+2)
        ), 
        dtype=np.float32
    )

    # NOTE PITCH, NOTE DURATION, CHORD PITCHES, CHORD DURATION, IS_START

    iterator = list(enumerate(zip(X_notes_raw, X_chords_raw, row_to_song_idx)))
    translate_notes = _should_translate_notes(X_notes_raw[-1][-1])
    
    for idx, (notes, chords, song_idx) in iterator:       
        X[idx, :] = _encode_custom(
            notes, 
            chords, 
            embeddings, 
            vectorizer,
            duration_to_id_notes,
            duration_to_id_chords,
            translate_notes=translate_notes, 
            notes_window=notes_window, 
            chords_window=chords_window, 
        )

        # CACHE HELPER
        try:
            indices_cache[song_idx].append(idx)
        except KeyError:
            indices_cache[song_idx] = [idx]

    # shrink cache helper
    for idx in indices_cache:
        indices_cache[idx] = [indices_cache[idx][0], indices_cache[idx][-1]]

    return X, indices_cache

def _encode_pitches(notes, chords, is_start=False, translate_notes=False, notes_window=8, chords_window=3, chord_size=4, chords_mapping={}):
    vector = np.zeros(
        (notes_window*1 + notes_window + (chords_window+1)*chord_size + (chords_window+1) + 1 + 1), 
        dtype=np.float32
    )
    of = 0

    # NOTE PITCHES
    n_pitches = [note.split('_')[0] for note in notes]
    if translate_notes:
        n_pitches = [_note_name_to_pitch(n) for n in n_pitches]
    else:
        n_pitches = [float(n) for n in n_pitches]
            
    vector[of:of+(notes_window*1)] = np.array(n_pitches).reshape(-1)
    of += (notes_window*1)

    # NOTE DURATIONS
    n_durations = [float(note.split('_')[1]) for note in notes]
    vector[of:of+notes_window] = n_durations
    of += notes_window

    # CHORDS PITCHES
    assert len(chords) == chords_window+1
    ch_p = []
    for ch in chords:
        if ch != 0:
            if isinstance(ch, str):
                ch_name = ch.split('_')[0]
            elif isinstance(ch, tuple):
                ch_name = ch[0].split('_')[0]
            ch_notes = chords_mapping[ch_name]
            ch_pitches = [_note_name_to_pitch(n) for n in ch_notes.split('+')][:chord_size]
            ch_pitches += [0] * max(0, chord_size-len(ch_pitches))
            assert len(ch_pitches) == chord_size
            ch_p.extend(ch_pitches)
        elif ch == 0:
            ch_p.extend([0] * chord_size)
        else:
            raise Exception(f"CHORD is {ch}!")
        
    vector[of:of + (chords_window+1)*chord_size] = np.array(ch_p).reshape(-1)
    of += (chords_window+1)*chord_size

    # CHORDS DURATIONS
    ch_durations = []
    for ch in chords:
        if ch != 0:
            if isinstance(ch, str):
                d = [float(ch.split('_')[1])]
            elif isinstance(ch, tuple):
                d1 = float(ch[0].split('_')[1]) # full duration
                d2 = ch[1] # currnt duration
                d = [d1, d2]
            ch_durations.extend(d)
        elif ch == 0:
            ch_durations.append(0)
        else:
            raise Exception(f"CHORD is {ch}!")

    vector[of:of + (chords_window+1+1)] = np.array(ch_durations)
    of += (chords_window+1+1)

    # IS START
    vector[-1] = 1 if is_start else 0
    return vector

def _prepare_input_X_pitches(X_notes_raw, X_chords_raw, is_start, row_to_song_idx, notes_window, chords_window, chord_size, indices_cache, chords_mapping={}):
    X = np.zeros(
        (
                len(X_notes_raw), 
                                                        #embed each chord   #prev, current & next duration: current x2!
                notes_window*1 + notes_window + (chords_window+1)*chord_size + (chords_window+1) + 1 + 1 # + is_start
        ), 
        dtype=np.float32
    )

    # NOTE PITCH, NOTE DURATION, CHORD PITCHES, CHORD DURATION, IS_START

    iterator = list(enumerate(zip(X_notes_raw, X_chords_raw, is_start, row_to_song_idx)))
    translate_notes = _should_translate_notes(X_notes_raw[-1][-1])
    
    for idx, (notes, chords, st, song_idx) in iterator:
        vector = _encode_pitches(
            notes, 
            chords, 
            is_start=st,
            translate_notes=translate_notes, 
            notes_window=notes_window, 
            chords_window=chords_window, 
            chord_size=chord_size,
            chords_mapping=chords_mapping
        )
        X[idx, :] = vector

        # CACHE HELPER
        try:
            indices_cache[song_idx].append(idx)
        except KeyError:
            indices_cache[song_idx] = [idx]

    # shrink cache helper
    for idx in indices_cache:
        indices_cache[idx] = [indices_cache[idx][0], indices_cache[idx][-1]]

    return X, indices_cache

from keras.utils import np_utils

def prepare_input_y(y_raw, limit=None, noneclass=True):
    all_tokens = list(sorted(set(y_raw)))
    if noneclass:
        all_tokens.append('NOTHING')
    token_to_id = {token : idx for idx, token in enumerate(all_tokens)}

    if not limit:
        y = np_utils.to_categorical([token_to_id[token] for token in y_raw], num_classes=len(all_tokens))
    else:
        y = np_utils.to_categorical([token_to_id[token] for token in y_raw[:limit]], num_classes=len(all_tokens))
    return y, token_to_id

def get_duration(piece):
    _, duration = piece.split("_")
    return float(duration)

def chord_to_pitches_duration_fixed(chord):
    pitches, duration = chord.split("_")
    pitches = chord_pitches_to_fixed(pitches)
    return pitches, float(duration)

def extract_X_by_song_idx(X, sample_song_idx, row_to_song_idx_map):
    indices = np.where(np.array(row_to_song_idx_map) == sample_song_idx)[0]
    first = indices[0]
    last = indices[-1]
    return X[first:last+1, :, :]

def extract_Xy_by_song_idx(X, y, sample_song_idx, row_to_song_idx_map, indices_cache={}):
    try:
        indices = indices_cache[sample_song_idx]
    except KeyError:
        indices = np.where(np.array(row_to_song_idx_map) == sample_song_idx)[0]
        indices_cache[sample_song_idx] = indices
    first = indices[0]
    last = indices[-1]
    return X[first:last+1, :, :], y[first:last+1], indices_cache

def extract_first_last_by_song_idx(sample_song_idx, row_to_song_idx_map, indices_cache={}):
    try:
        indices = indices_cache[sample_song_idx]
    except KeyError:
        indices = np.where(np.array(row_to_song_idx_map) == sample_song_idx)[0]
        indices_cache[sample_song_idx] = indices
    first = indices[0]
    last = indices[-1]
    return first, last, indices_cache

def get_bigram_transitions_chord_wise(notes_translated, chords_translated, translate_pitch_to_name=True, flat=False):
    transitions = {}

    for notes, chords in zip(notes_translated, chords_translated):

        note_t, chord_t = 0.0, 0.0
        ch_idx = 0
        prev_note = None
        for note in notes:
            note_tr = _note_pitched_to_named(note) if translate_pitch_to_name else note
            current_chord = chords[ch_idx]
            label = current_chord if not flat else current_chord.split('_')[0]
            if not prev_note:
                transitions.setdefault(label, {}).setdefault("START_0.0", []).append(note_tr)
            else:
                transitions.setdefault(label, {}).setdefault(prev_note, []).append(note_tr)
            note_t += float(note_tr.split('_')[-1])
            while chord_t < note_t:
                chord_t += float(chords[ch_idx].split('_')[-1])
                ch_idx += 1
                if ch_idx == len(chords):
                    break
                else: 
                    current_chord = chords[ch_idx]
                    # transitions.setdefault(chords[ch_idx], {}).setdefault(prev_note, []).append(note) # (TO CONSIDER...)

            if ch_idx == len(chords):
                break

        transitions.setdefault(label, {}).setdefault(note_tr, []).append("END_0.0")

    for ch in transitions:
        for prev_note in transitions[ch]:
            data = transitions[ch][prev_note]
            sum_events = len(data)
            counts = Counter(data).most_common()
            new_data = {}
            for pair in counts:
                new_data[pair[0]] = pair[1] / sum_events
            transitions[ch][prev_note] = new_data
            
    return transitions