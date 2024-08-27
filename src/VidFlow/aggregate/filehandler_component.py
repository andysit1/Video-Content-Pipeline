

import os
import glob

#Note: will add try statements when I look at document to figure what exceptions it raises

class FileHandleComponent:
    def set_file_path(self, path):
        if self.path_exists(path):
            self.file_path = path

    def fix_str_path(self, file_path : str) -> str:
        return file_path.replace('\\', '/')

    def makedirs(self, file_path : str):
      print(file_path)

      os.makedirs(file_path, exist_ok=True)
      ic("Created dir located {}".format(file_path))

      #DOCS: TODO: figure a good way to fix this issue future bug fix
        # except OSE rror:
        # # Cannot rely on checking for EEXIST, since the operating system
        # # could give priority to other errors like EACCES or EROFS
        # if not exist_ok or not path.isdir(name):
        #     raise

    def path_exists(self, file_path : str):
        return os.path.exists(file_path)

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
    def read_file(self, path: str) -> str:
        """Read the entire file content and return as a string."""
        with open(path, 'r') as file:
            return file.read()

    def write_file(self, path, content: str) -> None:
        """Write the provided content to the file."""
        with open(path, 'w') as file:
            file.write(str(content))

    def append_to_file(self, path, content: str) -> None:
        """Append the provided content to the file."""
        with open(path, 'a+') as file:
            file.write(content)

    def read_lines(self, path) -> list[str]:
        """Read the file and return a list of lines."""
        with open(path, 'r') as file:
            return file.readlines()

    def write_lines(self, path, lines: list) -> None:
        """Write a list of lines to the file."""
        with open(path, 'w') as file:
            file.writelines(str(line) + '\n' for line in lines)

    def file_exists(self, path) -> bool:
        """Check if the file exists."""
        return os.path.isfile(path)

    def delete_file(self, path) -> None:
        """Delete the file."""
        if self.file_exists(path):
            os.remove(path)

class FileMaster(FileHandleComponent):
    def __init__(self, engine) -> None:
        self.engine = engine
        self.origin_dir : str = None
        self.community_dir : str = os.path.join('.', 'output-video', 'community')
        self.private_dir : str = os.path.join('.', 'output-video', 'private')


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

    #MIGHT change the flag/logic for this in the future will be easier to have it like this
    def is_community(self):
        if self.engine.payload['is_community']:
            return True
        else:
            return False

    def start_community_bundle_files(self):
        community_txt_cache = os.path.join(self.community_dir, 'text_cache')
        community_clips = os.path.join(self.community_dir, 'clips')

        ic(community_txt_cache)
        ic(community_clips)

        self.makedirs(community_clips)
        self.makedirs(community_txt_cache)

        self.engine.payload['cache_txt_out'] = community_txt_cache
        self.engine.payload['clips_out'] = community_clips

    def start_private_bundle_files(self):
        private_txt_cache = os.path.join(self.private_dir, self.engine.payload['video_name'], 'text_cache')
        private_clips = os.path.join(self.private_dir, self.engine.payload['video_name'], 'clips')

        ic(private_txt_cache)
        ic(private_clips)

        self.makedirs(private_clips)
        self.makedirs(private_txt_cache)

        self.engine.payload['cache_txt_out'] = private_txt_cache
        self.engine.payload['clips_out'] = private_clips

    #sets up the file given first flag
    def setup(self):

        if self.is_community():
            self.remove_all_contents_output_frame(self.community_dir)
            self.start_community_bundle_files()
        else:
            to_make = os.path.join(self.private_dir, self.engine.payload['video_name'])
            ic(to_make)
            self.makedirs(to_make)
            self.start_private_bundle_files()



"""
    Test cases...
"""



import unittest
from icecream import ic
import unittest
from unittest.mock import MagicMock
import os

class TestFileHandler(unittest.TestCase):

    def setUp(self):
        self.file_handler = FileHandleComponent()
        self.test_file_path = 'test.txt'
        with open(self.test_file_path, 'w') as f:
            f.write('Sample content')

    def tearDown(self):
        import os
        os.remove(self.test_file_path)

    def test_read_file(self):
        content = self.file_handler.read_file(self.test_file_path)
        self.assertEqual(content, 'Sample content')

    def test_write_file(self):
        new_content = 'New content'
        self.file_handler.write_file(self.test_file_path, new_content)
        with open(self.test_file_path, 'r') as f:
            content = f.read()
        self.assertEqual(content, new_content)

    def test_append_file(self):
        append_content = ' Appended content'
        self.file_handler.append_to_file(self.test_file_path, append_content)
        with open(self.test_file_path, 'r') as f:
            content = f.read()
        self.assertEqual(content, 'Sample content' + append_content)

    def test_file_exists(self):
        self.assertTrue(self.file_handler.file_exists(self.test_file_path))
        self.assertFalse(self.file_handler.file_exists('non_existent_file.txt'))


class TestFileMaster(unittest.TestCase):
    def setUp(self):
        # Mocking the engine
        self.mock_engine = MagicMock()
        self.file_master = FileMaster(self.mock_engine)
        self.file_master.community_dir = 'test/community_dir'
        self.file_master.private_dir = 'test/private_dir'

        # Create test directories
        os.makedirs(self.file_master.community_dir, exist_ok=True)
        os.makedirs(self.file_master.private_dir, exist_ok=True)

    def tearDown(self):
        # Clean up test directories
        import shutil
        import time
        time.sleep(1)

        shutil.rmtree('test/community_dir')
        shutil.rmtree('test/private_dir')

    def test_is_community_true(self):
        self.mock_engine.payload = {'is_community': True}
        self.assertTrue(self.file_master.is_community())

    def test_is_community_false(self):
        self.mock_engine.payload = {'is_community': False}
        self.assertFalse(self.file_master.is_community())

    def test_setup_community(self):
        self.mock_engine.payload = {'is_community': True}
        self.file_master.remove_all_contents_output_frame = MagicMock()

        for i in range(3):
            self.file_master.write_file('test/community_dir/text{}.txt'.format(i), str(i))

        self.file_master.setup()
        self.file_master.remove_all_contents_output_frame.assert_called_once_with('test/community_dir')

    def test_setup_private(self):
        video_name = 'test_video'
        self.mock_engine.payload = {'is_community': False, 'video_name': video_name}
        self.file_master.makedirs = MagicMock()

        for i in range(3):
            self.file_master.write_file('test/community_dir/text{}.txt'.format(i), str(i))

        self.file_master.setup()

if __name__ == '__main__':
    unittest.main()
