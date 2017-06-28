from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data.dataset import Dataset

class CasiaWebFaceData(Dataset):
  """ImageNet data set."""

  def __init__(self, subset):
    super(CasiaWebFaceData, self).__init__('CASIA-WebFace', subset)

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 10566

  def num_examples_per_epoch(self):
    """Returns the number of examples in the data set."""
    if self.subset == 'train':
      return 491847
    if self.subset == 'validation':
      return 0

  def download_message(self):
    """Instruction to download and extract the tarball from the website."""
    pass
