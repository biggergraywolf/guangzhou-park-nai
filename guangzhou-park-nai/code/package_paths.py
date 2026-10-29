"""Locate immutable research inputs and keep all new outputs outside the project."""
from dataclasses import dataclass
import os
from pathlib import Path

@dataclass(frozen=True)
class PackagePaths:
    root: Path
    reproduced: Path

    @property
    def master_table(self): return self.root / 'data/master_features.csv'
    @property
    def station_groups(self): return self.root / 'data/station_groups.csv'
    @property
    def published_validation(self): return self.root / 'results/validation'

    def output(self, stage):
        if stage not in {'comparison','m4_final','model_export','preprocessing','mapping'}:
            raise ValueError('Unknown output stage: '+stage)
        target=self.reproduced/stage
        target.mkdir(parents=True,exist_ok=True)
        return target

def ensure_reproduction_output(output,root):
    root=Path(root).resolve()
    output=Path(output).expanduser()
    output=(output if output.is_absolute() else root/output).resolve()
    if output==root or root in output.parents:
        raise ValueError('New outputs must be outside the frozen research project: '+str(output))
    return output

def get_paths():
    default=Path(__file__).resolve().parent.parent
    root=Path(os.environ.get('NAI_PACKAGE_ROOT',str(default))).expanduser().resolve()
    given=os.environ.get('NAI_REPRODUCED_DIR',str(root.parent/'NAI_reproduced'))
    return PackagePaths(root,ensure_reproduction_output(given,root))
