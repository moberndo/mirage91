# import matplotlib.pyplot as plt
# import numpy as np
# import pyxdf

# path = "session1_run1.xdf"
# data, header = pyxdf.load_xdf(path)

# for stream in data:
#     y = stream["time_series"]

#     if isinstance(y, list):
#         # list of strings, draw one vertical line for each marker
#         for timestamp, marker in zip(stream["time_stamps"], y):
#             plt.axvline(x=timestamp)
#             print(f'Marker "{marker[0]}" @ {timestamp:.2f}s')
#     elif isinstance(y, np.ndarray):
#         # numeric data, draw as lines
#         plt.plot(stream["time_stamps"], y)
#     else:
#         raise RuntimeError("Unknown stream format")  # noqa: TRY004

# plt.show()
import pyxdf
import numpy as np

fname = "session1_run1.xdf"

streams, header = pyxdf.load_xdf(fname)

print("Number of streams:", len(streams))

for i, stream in enumerate(streams):

    info = stream["info"]

    print("\n" + "=" * 60)
    print("STREAM", i)
    print("=" * 60)

    print("Name:",
          info.get("name", [""])[0])

    print("Type:",
          info.get("type", [""])[0])

    print("Sampling rate:",
          info.get("nominal_srate", [""])[0])

    print("Channel count:",
          info.get("channel_count", [""])[0])

    print("Number of samples:",
          len(stream["time_stamps"]))

    print(
        "Data shape:",
        np.asarray(stream["time_series"]).shape
    )

    # Try to print channel labels
    try:
        channels = info["desc"][0]["channels"][0]["channel"]

        print("Channels:")

        for ch in channels:
            print(
                "  ",
                ch.get("label", ["?"])[0]
            )

    except Exception:
        print("No channel labels found.")