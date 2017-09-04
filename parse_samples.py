import os

flute_sample_dir = './res/flute'
violin_sample_dir = './res/violin'

class Sample:
    def __init__(self, filename, instr, note, octave, duration, vol, condition):
        self.filename = filename
        self.instr = instr
        self.note = note
        self.octave = octave
        self.duration = duration
        self.vol = vol
        self.condition = condition

    def __repr__(self):
        return self.instr + " " + self.note + " " + str(self.octave) + " " + self.duration + " " + self.vol + " " + self.condition

def parse_filename(filename):
    """Audio samples stored in repository are written in the following format:
    <instr>_<note><octave>_<duration>_<volume>_<condition>.mp3

    Returns a Sample encapsulating the information."""
    file_info = filename.split('_')
    file_sample = Sample(filename, file_info[0], file_info[1][:-1], int(file_info[1][-1:]), 
                         file_info[2], file_info[3], file_info[4])
    # print(file_sample)
    return file_sample

def get_flute_samples():
    for sample_file in os.listdir(flute_sample_dir):
        parse_filename(sample_file)

def get_violin_samples():
    for sample_file in os.listdir(violin_sample_dir):
        parse_filename(sample_file)

get_flute_samples()
