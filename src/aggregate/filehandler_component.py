

import os


#Note: will add try statements when I look at document to figure what exceptions it raises

class FileHandleComponent:
  def __init__(self, file_path : str) -> None:
        self.file_path = file_path

  #My Methods
  def fix_str_path(self, file_path : str) -> str:
      return file_path.replace('\\', '/')

  def makedirs(self, file_path : str):
      os.makedirs(file_path, exist_ok=True)

      #DOCS: TODO: figure a good way to fix this issue future bug fix
        # except OSE rror:
        # # Cannot rely on checking for EEXIST, since the operating system
        # # could give priority to other errors like EACCES or EROFS
        # if not exist_ok or not path.isdir(name):
        #     raise

  def path_exists(self, file_path : str):
      return os.path.exists(file_path)


  #AI genereated for the sake of just having them. Might remove in future if super redundant to reduce code amount
  def read_file(self) -> str:
      """Read the entire file content and return as a string."""
      with open(self.file_path, 'r') as file:
          return file.read()

  def write_file(self, content: str) -> None:
      """Write the provided content to the file."""
      with open(self.file_path, 'w') as file:
          file.write(content)

  def append_to_file(self, content: str) -> None:
      """Append the provided content to the file."""
      with open(self.file_path, 'a') as file:
          file.write(content)

  def read_lines(self) -> list:
      """Read the file and return a list of lines."""
      with open(self.file_path, 'r') as file:
          return file.readlines()

  def write_lines(self, lines: list) -> None:
      """Write a list of lines to the file."""
      with open(self.file_path, 'w') as file:
          file.writelines(lines)

  def file_exists(self) -> bool:
      """Check if the file exists."""
      return os.path.isfile(self.file_path)

  def delete_file(self) -> None:
      """Delete the file."""
      if self.file_exists():
          os.remove(self.file_path)



class FileMaster(FileHandleComponent):
    def __init__(self) -> None:
        self.community_directory : str = None

    """
        This class will handle the logic for the directory where we store our ouput, frames, debug files
        It will organize by date and name, etc. We can maybe just remove the files after x date. TBD


        Ideally, how it works is that you inherit it to some pipe into the future and it will handle saving files,
        pulling data, file / debugging cache, and whatever.

        Why do we need a sep class? Well one of the pain points was that the folder structure was annoying and it
        was saving all over the place. This likely could have been saved earlier if I did a little bit a teaking.
        Then I thought it would be much easier to just have a class that stored the preset paths that are tested and working
        so I never have to define a path again.

    """