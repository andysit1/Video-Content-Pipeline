

import os
import glob

#Note: will add try statements when I look at document to figure what exceptions it raises

class FileHandleComponent:
    def __init__(self, file_path : str) -> None:
        self.file_path = file_path
        self.input_video_path = "" # self.input_video_path
        self.out_file = ""

    def fix_str_path(self, file_path : str) -> str:
        return file_path.replace('\\', '/')

    def makedirs(self, file_path : str):
      ic("Created dir located {}".format(file_path))
      os.makedirs(file_path, exist_ok=True)

      #DOCS: TODO: figure a good way to fix this issue future bug fix
        # except OSE rror:
        # # Cannot rely on checking for EEXIST, since the operating system
        # # could give priority to other errors like EACCES or EROFS
        # if not exist_ok or not path.isdir(name):
        #     raise

    def path_exists(self, file_path : str):
        return os.path.exists(file_path)


    #Leaving this out for now cause this will probably cause a lot of memory issue or storage issues
    #if we plan to process an entire video

    # def save_out_frames(self, images, pattern: str):
    #     c = 0

    #     for image in images:
    #         out_file = os.path.join(current_dir, "frame_extraction", "out_frame", f"{pattern}{c}.png")
    #         if not cv2.imwrite(out_file, image):
    #             raise Exception("Could not write image")
    #         c += 1

    #my methods
    def remove_all_contents_output_frame(self, path: str):

        if os.path.exists(path):
            frames = glob.glob(self.input_video_path)

            for frame in frames:
                ic("Removed {}".format(frame))
                os.remove(frame)
        else:
            raise FileNotFoundError("Input file not found.")

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



import unittest
from icecream import ic



#testing basic functions
#TODO Add testcases for more
class TestFileHandleComponent(unittest.TestCase):

    def setUp(self):
        self.handler = FileHandleComponent("test_file.txt")
        self.test_content = "This is a test content."
        self.test_lines = ["Line 1\n", "Line 2\n", "Line 3\n"]


    #note: builtin function in testcase to be called after each test
    def tearDown(self):
        if os.path.exists(self.handler.file_path):
            os.remove(self.handler.file_path)

    def test_fix_str_path(self):
        self.assertEqual(self.handler.fix_str_path("C:\\Users\\test"), "C:/Users/test")

    def test_makedirs(self):
        test_dir = "test_dir"
        self.handler.makedirs(test_dir)
        self.assertTrue(os.path.exists(test_dir))
        os.rmdir(test_dir)

    def test_path_exists(self):
        open(self.handler.file_path, 'a').close()
        self.assertTrue(self.handler.path_exists(self.handler.file_path))

    def test_read_file(self):
        with open(self.handler.file_path, 'w') as file:
            file.write(self.test_content)
        self.assertEqual(self.handler.read_file(), self.test_content)

    def test_write_file(self):
        self.handler.write_file(self.test_content)
        with open(self.handler.file_path, 'r') as file:
            self.assertEqual(file.read(), self.test_content)

    def test_append_to_file(self):
        self.handler.write_file(self.test_content)
        self.handler.append_to_file("\nAppended content.")
        with open(self.handler.file_path, 'r') as file:
            self.assertEqual(file.read(), self.test_content + "\nAppended content.")

    def test_read_lines(self):
        with open(self.handler.file_path, 'w') as file:
            file.writelines(self.test_lines)
        self.assertEqual(self.handler.read_lines(), self.test_lines)

    def test_write_lines(self):
        self.handler.write_lines(self.test_lines)
        with open(self.handler.file_path, 'r') as file:
            self.assertEqual(file.readlines(), self.test_lines)

    def test_file_exists(self):
        open(self.handler.file_path, 'a').close()
        self.assertTrue(self.handler.file_exists())

    def test_delete_file(self):
        open(self.handler.file_path, 'a').close()
        self.handler.delete_file()
        self.assertFalse(os.path.exists(self.handler.file_path))

    def test_remove_all_contents_output_frame(self):
        self.handler.makedirs("frames")
        open("frames/frame1.png", 'a').close()
        open("frames/frame2.png", 'a').close()
        self.handler.input_video_path = "frames/*.png"
        self.handler.remove_all_contents_output_frame("frames")
        self.assertFalse(os.path.exists("frames/frame1.png"))
        self.assertFalse(os.path.exists("frames/frame2.png"))
        os.rmdir("frames")

if __name__ == '__main__':
    unittest.main()


