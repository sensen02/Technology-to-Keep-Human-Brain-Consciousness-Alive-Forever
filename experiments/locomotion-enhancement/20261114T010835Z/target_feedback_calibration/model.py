"""Parent-only EXACT fcbd deployment operations, new targetfeedback training identity.
ONLYCODE preparation never imports this numerical module. No additional parameters.
"""
from pathlib import Path
import copy
import importlib.util
import sys

PARENT=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T230738Z/local_mean_corrective_calibration')
_spec=importlib.util.spec_from_file_location('_targetfeedback_exact_fcbd_model',PARENT/'model.py')
old=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=old;_spec.loader.exec_module(old)
torch=old.torch;nn=old.nn
FORMAT='INDEPENDENT_MOTOR_TARGET_FEEDBACK_V1'
PARENT_FORMAT='INDEPENDENT_MOTOR_LOCAL_MEAN_CORRECTIVE_V1'
PARENT_SHA='fcbd818f9d6caf3431d6c545356ecd03c09369aab9eeae29363a7ccf128ce49b'
OLD_SHA=old.OLD_SHA;STATE_DIMS=old.STATE_DIMS;FEATURE_DIM=old.FEATURE_DIM;ALPHA=old.ALPHA;BOUND=old.BOUND;ADAPTER_SHAPES=old.ADAPTER_SHAPES
state_digest=old.state_digest;ResidualScan=old.ResidualScan


class IndependentMotor(old.IndependentMotor):
    """All production/cache/prime/sequence operations inherited exactly from fcbd."""
    def __init__(self,parent_motor):
        nn.Module.__init__(self);parent_motor.verify_old();identity=parent_motor.identity()
        if identity.get('format_version')!=PARENT_FORMAT or identity.get('state_dims')!=[128,64,64,96] or tuple(parent_motor.encoder.weight.shape)!=(96,412) or tuple(parent_motor.head.weight.shape)!=(42,192):raise ValueError('Strict actualfcbd deployment architecture required')
        self.old=copy.deepcopy(parent_motor.old);self.encoder=copy.deepcopy(parent_motor.encoder);self.gru=copy.deepcopy(parent_motor.gru);self.head=copy.deepcopy(parent_motor.head)
        for layer in (self.encoder,self.gru,self.head):layer.requires_grad_(True)
        self.frozen_old_state_digest=state_digest(self.old.state_dict());self.frozen_old_identity=copy.deepcopy(self.old.identity());self.parent_identity=copy.deepcopy(identity)
        self.validate_architecture()
    def identity(self):
        d=super().identity();d.update(format_version=FORMAT,parent_checkpoint_sha256=PARENT_SHA,parent_identity=self.parent_identity,encoder_initialization='exactfcbd412weightandbias',additional_encoder_parameters=0,head_initialization='exactfcbd192weightandbias',migration_numeric_requirement='allmodeltensors exactcopy; production fixedhistory parity parentqualification pending',deployment_operations='exactfrozenfcbd source inheritance, no targetfeedback operations added to actor',training_only_target_feedback=True,scope='targetfeedback training identity, unchanged bodymotor architecture; not brainneck deployment or behavioral acceptance')
        return d


def make_residual_scan(model,script=True):return old.make_residual_scan(model,script=script)
def make_scan(model):return make_residual_scan(model,script=True)
