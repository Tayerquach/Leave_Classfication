# -*- coding: utf-8 -*-
# @author: Boi Mai Quach <quachmaiboi@gmail.com>
################################################

import tarfile
import urllib.request
from tqdm import tqdm

url = "https://sourceforge.net/projects/flavia/files/Leaf%20Image%20Dataset/1.0/Leaves.tar.bz2"
output_path = "data/raw/Leaves.tar.bz2"
outdir = "data/raw"

class DownloadProgressBar(tqdm):
    """
        Create a ProgressBar to update the entire process.
    """
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """
        Download data from website via url to local folder

        Parameters
        ----------
        url: str
            Link website contains dataset
        output_path: None | str
            Location of where to look for the dataset storing location. If None, the current location will store the downloaded dataset
        
        Returns
        -------
        None
    """
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_raw_data(file_path, outdir):
    """
        Extracting tar.bz2 file

        Parameters
        ----------
        file_path: str
            A string representing the path of the tar file.
        outdir: str
            A string representing the output folder after extracting
        
        Returns
        -------
        None
    """
    tar = tarfile.open(file_path, "r:bz2")  
    tar.extractall(outdir)
    tar.close()

if __name__ == '__main__':
    #Download the compressed folder containning dataset from website
    download_url(url, output_path)
    #Extract compressed folder into dataset folder
    extract_raw_data(output_path,outdir)

    