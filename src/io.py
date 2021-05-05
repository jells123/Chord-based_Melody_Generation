import music21 as mc
import json


def read_midi_from_archive(archive, path):
    mf = mc.midi.MidiFile()
    mf.readstr(archive.read(path))
    return mc.midi.translate.midiFileToStream(mf)


def read_mode(archive, midi_path):
    symbol_nokey_path = midi_path.replace("/pianoroll/", "/event/").replace(
        "nokey.mid", "symbol_nokey.json"
    )
    read_data = archive.read(symbol_nokey_path)
    json_data = json.loads(read_data)
    return json_data.get("metadata", {}).get("mode", "?")
