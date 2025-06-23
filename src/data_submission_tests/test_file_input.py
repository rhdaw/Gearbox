import unittest
from unittest.mock import patch, Mock, mock_open
import pandas as pd
import os
from tests.file_input_check import test_file_size, test_file_vs_meta, fcs_colnames_in_dir_to_df

class TestFileInput(unittest.TestCase):
    def setUp(self):
        # Mock data
        self.mock_directory_files = ['file1.fcs', 'file2.fcs', 'other.txt']
        self.mock_csv_data = pd.DataFrame({
            'Filename': ['file1.fcs', 'file2.fcs', 'file3.fcs']
        })
        
    @patch('os.path.getsize')
    @patch('os.listdir')
    def test_file_size(self, mock_listdir, mock_getsize):
        # Setup mocks
        mock_listdir.return_value = self.mock_directory_files
        mock_getsize.side_effect = [2000, 500]  # file1.fcs = 2000b, file2.fcs = 500b
        
        with patch('builtins.print') as mock_print:
            test_file_size()
            mock_print.assert_called_with("FILE SUBMISSION FAILED")

    @patch('pandas.read_csv')
    @patch('os.listdir')
    def test_file_vs_meta(self, mock_listdir, mock_read_csv):
        # Setup mocks
        mock_listdir.return_value = self.mock_directory_files
        mock_read_csv.return_value = self.mock_csv_data
        
        with patch('builtins.print') as mock_print:
            test_file_vs_meta()
            mock_print.assert_called_with("FILE SUBMISSION FAILED")

    @patch('fcsparser.parse')
    @patch('os.listdir')
    def test_fcs_colnames_in_dir_to_df(self, mock_listdir, mock_parse):
        # Setup mock FCS data
        mock_listdir.return_value = self.mock_directory_files
        mock_meta = Mock()
        mock_data = {'Column1': [1,2], 'Column2': [3,4]}
        mock_parse.return_value = (mock_meta, mock_data)
        
        result = fcs_colnames_in_dir_to_df()
        self.assertIsInstance(result, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()