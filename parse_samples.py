import os

flute_sample_dir = './res/flute'
violin_sample_dir = './res/violin'

class Sample:
    def __init__(filename, instr, note, octave, vol, condition):
        self.filename = filename
        self.instr = instr
        self.note = note
        self.octave = octave
        self.vol = vol
        self.condition

def parse_filename(filename):
    """Audio samples stored in repository are written in the following format:
    <instr>_<note><octave>_<duration>_<volume>_<condition>.mp3

    Returns a Sample encapsulating the information."""
    file_info = filename.split('_')
    print(file_info)

for sample_file in os.listdir(flute_sample_dir):
    # print(sample_file)
    pass

for sample_file in os.listdir(violin_sample_dir):
    # print(sample_file)
    parse_filename(sample_file)
