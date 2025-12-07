#!python3
from sys import argv


def print_metadata(blob):
    time_per_line = float(blob[23].split("=")[1][:-3].replace(",", "."))
    size = float(blob[21].split("=")[1][:-3].replace(",", "."))
    if blob[23].strip().endswith("ms"):
        time_per_line *= 1e-3
    speed = size / time_per_line
    overscan = 1 if blob[32].startswith("Overscan") else 0
    print(f"scanspeed: {speed} nm / s")
    print(f"image size: {size} nm")
    setpoint = blob[36 + overscan].split("=")[1]
    p = int(blob[37 + overscan].split("=")[1])
    i = int(blob[38 + overscan].split("=")[1])
    print(f"coeffs: P = {p}; I = {i}")
    print(f"setpoint = {setpoint}")
    print()
    print(blob[23])
    print(blob[21])

    print("".join([
                  "$v_R = \\SI{",
                  str(speed),
                  "}{\\nano\\meter\\per\\second}$, $I_\\text{Setpoint} = \\SI{",
                  str(setpoint[:-3]),
                  "}{\\nano\\ampere}$, $P=", str(p), "$, $I=", str(i), "$"]))


def main():
    handle = open(argv[1], "r")
    metadata = [handle.readline() for _ in range(113)]
    handle.close()
    print_metadata(metadata)


if __name__ == "__main__":
    main()
