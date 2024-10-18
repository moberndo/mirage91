
#%%
from pylsl import resolve_byprop, StreamInlet
import numpy as np

streams=resolve_byprop('name', 'ClassifierOutput')
inlet = StreamInlet(streams[0])
while True:
    chunk, tmps = inlet.pull_chunk()
    if chunk:
        print(chunk)
        