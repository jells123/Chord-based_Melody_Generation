from collections import Counter
import music21 as mc

MODE_TO_OFFSET = {"1": 0, "2": 2, "3": 4, "4": 5, "5": 7, "6": 9, "7": 11}


def translate_to_c(sample, mode, debug=False):
    offset = MODE_TO_OFFSET.get(mode, -1)
    if offset > 0:
        if debug:
            print("Mode: ", mode)
        for part in sample.parts:
            part.transpose(offset, inPlace=True)
    return sample


def estimate_part_type(part):
    types = [type(elem) for elem in part.notesAndRests]
    if mc.note.Note not in types:
        return "chords"
    else:
        counts = Counter(types)
    return (
        "notes"
        if counts.get(mc.note.Note, 0) > counts.get(mc.chord.Chord, 0)
        else "chords"
    )


def _estimate_single_part_type(part):
    counts = Counter([type(elem) for elem in part.flat])
    if counts.get(mc.note.Note, 0) > counts.get(mc.chord.Chord, 0):
        return "notes"
    elif counts.get(mc.note.Note, 0) < counts.get(mc.chord.Chord, 0):
        return "chords"
    return "?"


def estimate_parts_types(part1, part2):
    t1 = _estimate_single_part_type(part1)
    t2 = _estimate_single_part_type(part2)
    if t1 == t2:
        print(f"Same types: {t1} and {t2}")
        return None, None
    else:
        return t1, t2


def extract_parts_types(sample):
    if not sample.parts or len(sample.parts) > 2:
        return (None, None)
    elif len(sample.parts) == 1:
        return (_estimate_single_part_type(sample.parts[0]),)
    else:
        t1 = _estimate_single_part_type(sample.parts[0])
        t2 = _estimate_single_part_type(sample.parts[1])
        if t1 == t2:
            print(f"Same types: {t1} and {t2}")
            return (None, None)
        else:
            return (t1, t2)


def estimate_parts_types_OLD(part1, part2):
    """
    LOGIC:
    1. For each part separately, if it has more notes than chords, then it's notes. If it has more chords than notes, then it's chords.
    If the number of notes and chords is equal, keep it as undefined.
    2. Check for duplicate part types, 2x. notes or 2x. chords.
    Raise exception if both were undefined (same count of notes and chords in both parts)
    Otherwise, compare counts of chords between parts. Assign 'chords' type to the one which has more chords and 'notes' to the other one.
    If counts of chords were equal, then compare lengths of parts. The shorter one will be assigned to 'chords' and longer one to 'notes'.
    If there was no match - lenghts are also equal, then raise exception. (? will this line of code execute ?)
    3. If there were no duplicate part types, and there is 'undefined' mark left, then fill it with the remaining type.
    4. At the end, two types should be different and defined. (assert)
    """
    t1, t2 = "?", "?"

    t1 = _estimate_single_part_type(part1)
    t2 = _estimate_single_part_type(part2)

    if t1 == t2:
        counts1 = Counter([type(elem) for elem in part1.notes])
        counts2 = Counter([type(elem) for elem in part2.notes])
        if t1 == "?":
            raise Exception("Counts of Notes and Chords are equal in both parts!")
        else:
            if counts1.get(mc.chord.Chord, 0) > counts2.get(mc.chord.Chord, 0):
                t1, t2 = "chords", "notes"
            elif counts1.get(mc.chord.Chord, 0) < counts2.get(mc.chord.Chord, 0):
                t1, t2 = "notes", "chords"
            else:
                if len(part1.notes) < len(part2.notes):
                    t1, t2 = "chords", "notes"
                elif len(part1.notes) < len(part2.notes):
                    t1, t2 = "notes", "chords"
                else:
                    raise Exception("NO WAY I CAN'T IDENTIFY THOSE PARTS")
    elif t1 == "?":
        t1 = "notes" if t2 == "chords" else "chords"
    elif t2 == "?":
        t2 = "notes" if t1 == "chords" else "chords"

    assert t1 != "?" != t2
    return t1, t2
