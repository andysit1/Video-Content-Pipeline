from phase_one_extract import *
import matplotlib.pyplot as plt

def main():
  silence_chunk = read_file_silence("../output-text/vol.txt")
  freeze_chunk = read_file_freeze("../output-text/mine_freeze.txt")
  # volume_chunk = read_file_volumedetect("output-text")
  print(silence_chunk, len(silence_chunk))
  print("--------------------------")
  print(freeze_chunk)

  plt.title('Video Analyze')
  plt.xlabel('Time')
  plt.ylabel('Strength')

  for chk in silence_chunk:
    x = ((chk[1] - chk[0]) / 2 ) + chk[0]
    y = 3
    plt.plot(x, y, marker='o', color='b')


  for chk in freeze_chunk:
    x = chk[2] / 2 + chk[0]
    y = 3.2

    plt.plot(int(x), y, marker='x', color='r')


  plt.xlim(0, 1320)  # Set x-axis limits
  plt.ylim(0, 5)


  plt.show()
  # read_file_volumedetect("../output-text/mine_volumedetect.txt")

if __name__ == "__main__":
  main()
# get data

# plot data
