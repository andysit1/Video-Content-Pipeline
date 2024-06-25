

#rbg to gray scale

def average_method(x_list):
  return sum(x_list) / len(x_list)

def luminosity_method(x_list):
  if len(x_list) != 3:
    raise ValueError("List type is wrong, not enough values (3) found {}".format(len(x_list)))

  return 0.3 * x_list[0] + 0.59 * x_list[1] + 0.11 * x_list[2]

