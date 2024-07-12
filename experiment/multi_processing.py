import multiprocessing as mp
from icecream import ic




if __name__ == "__main__":
  num_processes = mp.cpu_count()
  ic(num_processes)