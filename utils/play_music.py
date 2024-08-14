from pretty_midi import pretty_midi
import sounddevice as sd


def play_midi_file(midi_file: str, _SAMPLING_RATE):
    """Plays the MIDI file using the sounddevice library."""
    # Load the MIDI file using pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_file)

    # Synthesize the MIDI to a waveform using fluidsynth
    waveform = pm.fluidsynth(fs=_SAMPLING_RATE)

    # Play the waveform
    sd.play(waveform, samplerate=_SAMPLING_RATE)
    sd.wait()  # Wait until the sound has finished playing
