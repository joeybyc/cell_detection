from yacs.config import CfgNode as CN

_C = CN(new_allowed=True)

_C.sam_model = CN()


_C.sam_model.ckpt = "models/sam_vit_h_4b8939.pth"
_C.sam_model.model_type = "vit_h"

_C.dataset = CN()
# _C.dataset.root = 'data'
# _C.dataset.data_path = 'mini_data'
# _C.dataset.chamber_output_path = 'chamber_results'
# _C.dataset.cell_output_path = 'cell_results'
# _C.dataset.labels = 'label.csv'

_C.cell_classifier = CN()
_C.cell_classifier.size_L = 20
_C.cell_classifier.ckpt = "models/cell_classifier.pth"

cfg = _C    # users can `from config import cfg`
