import dataclasses
import itertools
from pathlib import Path
import typing


SEQ_NUM = int
TRUTH_TYPE = str
IMG_NAME = str


class Annotation:
    def __init__(self, img_path: Path, truth: str, annotations: typing.List[str]):
        self.image_path: Path = Path(img_path)
        self.image_suffix = self.image_path.name[1:]

        self.truth: str = truth
        self.annotations: typing.Dict[str, typing.Optional[Path]] = {}

        seq_folder = self.image_path.parent
        truth_folder = seq_folder.parent / f'{seq_folder.name}_{truth}'

        img_ann_name = 'man_seg' + self.image_suffix
        for ann_name in annotations:
            self.annotations.setdefault(ann_name, None)
            if not (ann_path := truth_folder / ann_name / img_ann_name).exists():
                continue
            self.annotations[ann_name] = ann_path
    
    def __getitem__(self, ann: str) -> typing.Optional[Path]:
        return self.annotations.get(ann, None)
    
    def to_dict(self) -> typing.Dict[str, typing.Optional[Path]]:
        return self.annotations
    
    @property
    def paths(self) -> typing.List[Path]:
        return [maybe_path for maybe_path in self.annotations.values() if maybe_path is not None]
    

@dataclasses.dataclass
class Sample:
    image_path: Path
    gold_annotations: Annotation
    silver_annotations: Annotation
    sequence: 'Sequence'


class CTCFolder:
    """
    Class representing a folder with CTC data.

    Attributes:
        path (pathlib.Path): path to the CTC folder
        sequence_names (list of str): numerical names of sequences present in this CTC dataset
        sequences (list of Sequence): list of Sequence objects
    """
    def __init__(self, path: Path):
        self.path: Path = Path(path)
        self.sequence_names = [_path.name for _path in self.path.glob('*') if _path.is_dir() and _path.name.isdigit()]
        self.sequences = [Sequence(self.path / seq_name, self) for seq_name in self.sequence_names]
    
    @property
    def num_sequences(self) -> int:
        return len(self.sequences)

    @property
    def all_samples(self) -> typing.List[Sample]:
        return list(itertools.chain(*(seq.samples for seq in self.sequences)))


class Sequence:
    def __init__(self, path: Path, ctc_dataset: CTCFolder):
        self.path = Path(path)
        self.dataset: CTCFolder = ctc_dataset
        self.sequence_number: int = int(self.path.name)
        self.sequence_name: str = f'{self.sequence_number:0>2}'

        self.image_paths: typing.List[Path] = list(self.path.glob('*.tif'))

        gold_ann_folder = self.path.parent / f'{self.sequence_name}_GT'
        gold_annotations = [_p.name for _p in gold_ann_folder.glob('*') if _p.is_dir()]

        silver_ann_folder = self.path.parent / f'{self.sequence_name}_ST'
        silver_annotations = [_p.name for _p in silver_ann_folder.glob('*') if _p.is_dir()]

        self.gold_annotations: typing.Dict[str, Annotation] = {img_path.name: Annotation(img_path, 'GT', gold_annotations) for img_path in self.image_paths}
        self.silver_annotations: typing.Dict[str, Annotation] = {img_path.name: Annotation(img_path, 'ST', silver_annotations) for img_path in self.image_paths}
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Sample:
        img_name = self.image_paths[idx].name
        return Sample(self.image_paths[idx],
                      gold_annotations=self.gold_annotations[img_name],
                      silver_annotations=self.silver_annotations[img_name],
                      sequence=self)

    @property
    def samples(self) -> typing.List[Sample]:
        return [self[i] for i in range(len(self))]

